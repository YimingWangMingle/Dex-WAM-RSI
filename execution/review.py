"""Proposal review before execution (framework §5.3).

Adopted from SenseNova-RoboRSI PR #175 ``hybrid_host.py`` / ``hybrid_adapter.py``
(commit 60f20ed): a model proposal is shown with per-row bound violations; the
Agent must review it before any movement, citing evidence and the current
subgoal; a stale proposal (its observation is no longer the latest measured
one) cannot execute; an approved prefix containing a violating row cannot be
accepted; rejection leaves the robot unmoved.

First version supports ``accept`` (a prefix) and ``reject``. Switching skills is
a TaskSkillRunner command that rejects the pending proposal first. Rewriting
individual action values (SenseNova ``modify``) is deliberately not supported:
rewritten rows would no longer correspond to the model's prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

from skills.contracts import MeasuredObservation, SkillProposal

Violations = Tuple[Tuple[str, ...], ...]   # one tuple of messages per proposed row
RowChecker = Callable[[SkillProposal, MeasuredObservation], Violations]


class ReviewRejected(ValueError):
    """The review itself is invalid; nothing was executed and the proposal stays pending."""


@dataclass(frozen=True)
class ReviewPolicy:
    min_prefix: int = 1
    max_prefix: Optional[int] = None   # None: up to the full proposal
    default_prefix: Optional[int] = None

    def __post_init__(self):
        if self.min_prefix < 1 or (self.max_prefix is not None and self.max_prefix < self.min_prefix):
            raise ValueError("invalid prefix limits")


@dataclass(frozen=True)
class ReviewDecision:
    proposal_id: str
    decision: str            # "accept" | "reject"
    reason: str
    current_subgoal: str
    approved_steps: Optional[int] = None
    candidate_id: Optional[int] = None


@dataclass(frozen=True)
class ReviewedProposal:
    proposal_id: str
    decision: str
    approved_rows: tuple
    reason: str
    current_subgoal: str
    violations: Violations
    record: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class BoundsChecker:
    """Contract-agnostic row checks on the proposal's own action space.

    ``lower``/``upper``: per-dimension absolute limits (None = unbounded).
    ``max_delta``: per-dimension limit on |row[i] - previous row[i]|, the first
    row compared against ``reference(observation)`` (e.g. the measured joint state).
    A forward-kinematics checker (SenseNova's 5 cm / 0.35 rad EEF bounds) can be
    supplied instead once the RoboDojo action contract is fixed.
    """
    lower: Sequence[Optional[float]] = ()
    upper: Sequence[Optional[float]] = ()
    max_delta: Sequence[Optional[float]] = ()
    reference: Optional[Callable[[MeasuredObservation], Sequence[float]]] = None

    def __call__(self, proposal: SkillProposal, observation: MeasuredObservation) -> Violations:
        previous = list(self.reference(observation)) if self.reference else None
        out = []
        for index, row in enumerate(proposal.actions):
            issues = []
            for i, value in enumerate(row):
                lo = self.lower[i] if i < len(self.lower) else None
                hi = self.upper[i] if i < len(self.upper) else None
                if (lo is not None and value < lo) or (hi is not None and value > hi):
                    issues.append(f"row {index} dim {i}: {value:.4g} outside [{lo}, {hi}]")
                step = self.max_delta[i] if i < len(self.max_delta) else None
                if step is not None and previous is not None and abs(value - previous[i]) > step + 1e-9:
                    issues.append(f"row {index} dim {i}: step {abs(value - previous[i]):.4g} > {step}")
            out.append(tuple(issues))
            previous = list(row)
        return tuple(out)


def no_checks(proposal: SkillProposal, observation: MeasuredObservation) -> Violations:
    return tuple(() for _ in proposal.actions)


def validate(decision: ReviewDecision, proposal: SkillProposal, latest: MeasuredObservation,
             violations: Violations, policy: ReviewPolicy = ReviewPolicy()) -> ReviewedProposal:
    """Check a review against the pending proposal; raise ``ReviewRejected`` if invalid."""
    if decision.proposal_id != proposal.proposal_id:
        raise ReviewRejected("review must name the current pending proposal")
    if not decision.reason.strip():
        raise ReviewRejected("an evidence-based review reason is required")
    if not decision.current_subgoal.strip():
        raise ReviewRejected("assessment of the current subgoal is required")
    if len(violations) != len(proposal.actions):
        raise ReviewRejected("row checks must cover every proposed row")
    base = proposal.request.observation
    record = dict(proposal_id=proposal.proposal_id, skill=proposal.identity.name,
                  decision=decision.decision, reason=decision.reason,
                  current_subgoal=decision.current_subgoal,
                  proposed_steps=len(proposal.actions), candidate_id=decision.candidate_id,
                  violating_rows=[i for i, v in enumerate(violations) if v])
    if decision.decision == "reject":
        return ReviewedProposal(proposal.proposal_id, "reject", (), decision.reason,
                                decision.current_subgoal, violations, dict(record, approved_steps=0))
    if decision.decision != "accept":
        raise ReviewRejected("decision must be accept or reject")
    if (latest.episode_id != base.episode_id or latest.observation_id != base.observation_id
            or latest.control_step != base.control_step):
        raise ReviewRejected("stale proposal: the robot state changed since it was generated")
    span = decision.approved_steps
    if span is None:
        span = policy.default_prefix or len(proposal.actions)
    upper = min(len(proposal.actions), policy.max_prefix or len(proposal.actions))
    if type(span) is not int or not policy.min_prefix <= span <= upper:
        raise ReviewRejected(f"approved_steps must be an integer in [{policy.min_prefix}, {upper}]")
    if any(violations[:span]):
        raise ReviewRejected("cannot accept rows that violate bounds; shorten the prefix or reject")
    rows = proposal.actions[:span]
    if any(not math.isfinite(v) for row in rows for v in row):
        raise ReviewRejected("non-finite approved action")
    return ReviewedProposal(proposal.proposal_id, "accept", rows, decision.reason,
                            decision.current_subgoal, violations, dict(record, approved_steps=span))
