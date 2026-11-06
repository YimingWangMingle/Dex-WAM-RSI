"""Observation-bound operation checks, separate from native task success/score.

Verifiers consume allowed measured observations only. A passed check is local
physical evidence, never an oracle task verdict or a model forecast.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

VERDICTS = ("passed", "failed", "unknown")
FAILURE_ROUTES = {
    "geometry_code": "repair_tool",
    "perception": "remeasure",
    "reachability": "replan_motion",
    "contact": "collect_verified_recovery",
    "prediction_alignment": "audit_alignment_before_training",
    "task_semantics": "review_task_conditions",
    "budget": "review_schedule",
    "unknown": "collect_more_evidence",
}


@dataclass(frozen=True)
class ObservationEvidence:
    episode_id: str
    observation_id: str
    control_step: int
    payload_ref: str

    @classmethod
    def from_observation(cls, observation):
        if observation.source != "measured":
            raise ValueError("operation evidence must be measured")
        return cls(observation.episode_id, observation.observation_id,
                   observation.control_step, observation.payload_ref)

    def __post_init__(self):
        if not all((self.episode_id, self.observation_id, self.payload_ref)):
            raise ValueError("evidence identity is required")
        if type(self.control_step) is not int or self.control_step < 0:
            raise ValueError("invalid evidence control step")


@dataclass(frozen=True)
class OperationCheck:
    name: str
    verdict: str
    method: str                 # versioned verifier, e.g. held-object-tracker/v1
    evidence: Tuple[ObservationEvidence, ...] = ()
    detail: str = ""

    def __post_init__(self):
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.name or not self.method or self.verdict not in VERDICTS:
            raise ValueError("invalid operation check")
        if self.verdict != "unknown" and not self.evidence:
            raise ValueError("passed/failed checks require measured evidence")


@dataclass(frozen=True)
class LocalMetric:
    name: str
    value: float
    unit: str
    tolerance: float

    def __post_init__(self):
        if (not self.name or not self.unit or not math.isfinite(self.value)
                or not math.isfinite(self.tolerance) or self.tolerance < 0):
            raise ValueError("local metrics require finite values, units and a nonnegative tolerance")


@dataclass(frozen=True)
class OperationReport:
    stage: str
    motion: str = "unknown"     # passed / failed / unknown; not task success
    checks: Tuple[OperationCheck, ...] = ()
    metrics: Tuple[LocalMetric, ...] = ()
    failure_kind: Optional[str] = None
    failure_detail: str = ""
    recovery_options: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("checks", "metrics", "recovery_options"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.stage or self.motion not in VERDICTS:
            raise ValueError("invalid operation stage/motion")
        if self.failure_kind is not None and self.failure_kind not in FAILURE_ROUTES:
            raise ValueError("unknown failure category")
        if len({c.name for c in self.checks}) != len(self.checks):
            raise ValueError("duplicate operation checks")
        if self.failure_kind is not None and not self.failure_detail.strip():
            raise ValueError("failure hypotheses require a description")

    def validate_observations(self, before, after):
        allowed = {ObservationEvidence.from_observation(o) for o in (before, after)}
        current = ObservationEvidence.from_observation(after)
        for check in self.checks:
            if any(e not in allowed for e in check.evidence):
                raise ValueError("operation evidence is stale, foreign or not bound to execution")
            if check.verdict != "unknown" and current not in check.evidence:
                raise ValueError("operation checks require the final measured observation")

    def permits_next_stage(self, required_checks):
        checks = {c.name: c.verdict for c in self.checks}
        return (self.motion == "passed" and self.failure_kind is None
                and all(c.verdict == "passed" for c in self.checks)
                and all(checks.get(name) == "passed" for name in required_checks))

    def to_dict(self):
        return asdict(self)


def diagnose_operation(report: OperationReport):
    """Return a routing hypothesis, not a causal proof or permission to train."""
    category = report.failure_kind or "unknown"
    return dict(category=category, route=FAILURE_ROUTES[category],
                hypothesis=True, detail=report.failure_detail,
                evidence=[asdict(e) for c in report.checks for e in c.evidence],
                requires_independent_validation=True)


class OperationBlocked(ValueError):
    """Expected refusal before motion; return evidence to the Agent for recovery."""
    def __init__(self, message: str, report: OperationReport):
        super().__init__(message)
        self.report = report
