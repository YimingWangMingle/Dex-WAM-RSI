"""Proposal-only interfaces for direct_control, wam_control and vla_control.

Backends never own env.step. A TaskSkillRunner must review proposals, execute
them through a single executor and supply measured feedback. This module does
not implement that runner, a remote model transport, or skill handoff.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Optional, Protocol, Tuple
from uuid import uuid4

from execution.feedback import OperationReport

SkillName = str  # Names are discovered from the registry; built-ins remain compatible.
SKILL_NAMES = ("direct_control", "wam_control", "vla_control")
ActionRows = Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class SkillIdentity:
    name: SkillName
    backend_revision: str
    action_contract: str  # Versioned contract: layout, units, frame, Hz, normalization.
    action_dim: int
    checkpoint: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise ValueError("invalid task skill name")
        if not self.backend_revision or not self.action_contract:
            raise ValueError("backend revision and action contract are required")
        if type(self.action_dim) is not int or self.action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        if self.name in ("wam_control", "vla_control") and not self.checkpoint:
            raise ValueError("model skills require an explicit checkpoint identity")


@dataclass(frozen=True)
class MeasuredObservation:
    episode_id: str
    observation_id: str
    control_step: int
    time_s: float
    payload_ref: str  # Immutable RGB/state bundle; adapter validates its schema.
    source: str = "measured"

    def __post_init__(self):
        if self.source != "measured":
            raise ValueError("predictions cannot be used as measured history")
        if not self.episode_id or not self.observation_id or not self.payload_ref:
            raise ValueError("episode, observation and payload identities are required")
        if type(self.control_step) is not int or self.control_step < 0:
            raise ValueError("control_step must be a nonnegative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("invalid observation timestamp")


@dataclass(frozen=True)
class SkillExperienceContext:
    """Explicit per-skill historical text, never a replacement for measured history."""
    identity_json: str
    icl: Tuple[Tuple[str, str], ...] = ()
    hmr: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self):
        import json
        return dict(identity=json.loads(self.identity_json),
                    icl=[dict(path=p, text=t) for p, t in self.icl],
                    hmr=[dict(path=p, text=t) for p, t in self.hmr])


@dataclass(frozen=True)
class SkillRequest:
    observation: MeasuredObservation
    subgoal: str
    action_contract: str
    max_actions: int
    history: Tuple[MeasuredObservation, ...] = ()
    experience: Optional[SkillExperienceContext] = None

    def __post_init__(self):
        if not self.subgoal.strip() or not self.action_contract:
            raise ValueError("subgoal and action contract are required")
        if type(self.max_actions) is not int or self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        object.__setattr__(self, "history", tuple(self.history))
        previous_step, previous_time = -1, -1.0
        for item in self.history:
            if (item.episode_id != self.observation.episode_id
                    or not previous_step < item.control_step < self.observation.control_step
                    or not previous_time <= item.time_s <= self.observation.time_s):
                raise ValueError("history must be ordered, measured and from this episode's past")
            previous_step, previous_time = item.control_step, item.time_s


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: int
    actions: ActionRows
    prediction_ref: Optional[str] = None


@dataclass(frozen=True)
class BackendProposal:
    actions: ActionRows
    prediction_ref: Optional[str] = None  # Optional for WAM; never required for VLA.
    stage: Optional[str] = None
    candidates: Tuple[ActionCandidate, ...] = ()
    visual_only: bool = False


@dataclass(frozen=True)
class SkillProposal:
    proposal_id: str
    identity: SkillIdentity
    request: SkillRequest
    actions: ActionRows
    prediction_ref: Optional[str]
    stage: Optional[str] = None
    candidates: Tuple[ActionCandidate, ...] = ()
    visual_only: bool = False


@dataclass(frozen=True)
class ExecutionFeedback:
    proposal_id: str
    executed_actions: ActionRows
    final_observation: MeasuredObservation
    terminated: bool
    operation: Optional[OperationReport] = None
    # Native success/score belongs to the recorder/evaluator, not this input.


class SkillStopped(RuntimeError):
    """A backend declined to propose any physical action."""


class ControlBackend(Protocol):
    def reset(self, episode_id: str) -> None: ...
    def propose(self, request: SkillRequest) -> BackendProposal: ...
    def observe(self, feedback: ExecutionFeedback) -> None: ...


class ControlSkill:
    """Checked backend boundary, not an executor or an Agent approval policy.

The caller must finish/cancel in-flight execution before reset or handoff.
Feedback preserves the actually executed actions, which may differ from the
proposal after review. Corresponding WAM predictions then require revalidation.
"""

    def __init__(self, identity: SkillIdentity, backend: ControlBackend):
        self.identity, self.backend = identity, backend
        self._episode: Optional[str] = None
        self._pending: Optional[SkillProposal] = None
        self._last_feedback: Optional[MeasuredObservation] = None

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode identity is required")
        if self._pending is not None:
            raise RuntimeError("resolve the pending proposal before reset/handoff")
        self._episode = None
        self._last_feedback = None
        self.backend.reset(episode_id)
        self._episode = episode_id

    def _actions(self, actions, limit: int, *, allow_empty: bool = False) -> ActionRows:
        rows = tuple(tuple(float(v) for v in row) for row in actions)
        if len(rows) > limit or (not rows and not allow_empty):
            raise ValueError("invalid action horizon")
        if any(len(row) != self.identity.action_dim or
               not all(math.isfinite(v) for v in row) for row in rows):
            raise ValueError("invalid action dimensions or non-finite values")
        return rows

    def propose(self, request: SkillRequest) -> SkillProposal:
        if self._episode is None or request.observation.episode_id != self._episode:
            raise RuntimeError("reset to the request episode before proposing")
        if self._pending is not None:
            raise RuntimeError("resolve the pending proposal first")
        if self._last_feedback is not None and (
                request.observation.control_step < self._last_feedback.control_step
                or request.observation.time_s < self._last_feedback.time_s):
            raise ValueError("request observation predates the last execution feedback")
        if request.action_contract != self.identity.action_contract:
            raise ValueError("action contract mismatch; explicit conversion is required")
        result = self.backend.propose(request)
        candidates = tuple(ActionCandidate(c.candidate_id,
            self._actions(c.actions, request.max_actions), c.prediction_ref)
            for c in result.candidates)
        ids = [c.candidate_id for c in candidates]
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('candidate IDs must be unique nonnegative integers')
        proposal = SkillProposal(uuid4().hex, self.identity, request,
                                 self._actions(result.actions, request.max_actions),
                                 result.prediction_ref, result.stage, candidates, result.visual_only)
        self._pending = proposal
        return proposal

    def candidate(self, candidate_id: Optional[int]) -> SkillProposal:
        proposal = self._pending
        if proposal is None:
            raise ValueError('no pending proposal')
        if not proposal.candidates:
            if candidate_id is not None:
                raise ValueError('single proposal does not accept candidate_id')
            return proposal
        if type(candidate_id) is not int:
            raise ValueError('explicit candidate_id required')
        for candidate in proposal.candidates:
            if candidate.candidate_id == candidate_id:
                return replace(proposal, actions=candidate.actions,
                               prediction_ref=candidate.prediction_ref)
        raise ValueError('unknown candidate_id')

    def prepare_execution(self, candidate_id: Optional[int], rows: ActionRows) -> None:
        proposal = self.candidate(candidate_id)
        if rows != proposal.actions[:len(rows)] or not rows:
            raise ValueError('execution must be an exact candidate prefix')
        # Validate the entire approval before consuming a remote selection.
        prepare = getattr(self.backend, 'prepare_execution', None)
        if prepare is not None:
            prepare(candidate_id, rows)
        self._pending = proposal

    def observe(self, feedback: ExecutionFeedback) -> None:
        proposal = self._pending
        if proposal is None or feedback.proposal_id != proposal.proposal_id:
            raise ValueError("feedback does not match the pending proposal")
        before, after = proposal.request.observation, feedback.final_observation
        rows = self._actions(feedback.executed_actions, len(proposal.actions), allow_empty=True)
        if (after.episode_id != before.episode_id
                or after.control_step != before.control_step + len(rows)
                or after.time_s < before.time_s):
            raise ValueError("feedback must correspond to actual executed control rows")
        if feedback.operation is not None:
            feedback.operation.validate_observations(before, after)
            if proposal.stage is not None and feedback.operation.stage != proposal.stage:
                raise ValueError("operation report does not match proposal stage")
        # Clear only after successful delivery. Failed/uncertain transports must
        # implement idempotent observe using proposal_id before they are deployed.
        self.backend.observe(ExecutionFeedback(feedback.proposal_id, rows, after, feedback.terminated,
                                               feedback.operation))
        self._pending = None
        self._last_feedback = after
        if feedback.terminated:
            self._episode = None


class ModelControlSkill(ControlSkill):
    """An explicitly model-backed skill, regardless of its registered name."""

    def __init__(self, identity: SkillIdentity, backend: ControlBackend):
        if not isinstance(identity.checkpoint, str) or not identity.checkpoint.strip():
            raise ValueError('model skills require an explicit checkpoint identity')
        super().__init__(identity, backend)


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, ControlSkill] = {}

    def register(self, skill: ControlSkill) -> None:
        if skill.identity.name in self._skills:
            raise ValueError("skill already registered")
        self._skills[skill.identity.name] = skill

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def get(self, name: SkillName) -> ControlSkill:
        if name not in self._skills:
            raise LookupError(f"skill {name!r} has no configured backend")
        return self._skills[name]
