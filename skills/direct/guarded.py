"""Proposal-only, staged direct backend. All movement remains in the executor.

Supply robot-specific planners and measured-observation verifiers. No geometry,
IK, grasp detector, robot action conversion or task success is fabricated here.
Wire verify_execution into TaskSkillRunner.operation_verifiers['direct_control'].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

from skills.contracts import BackendProposal, SkillRequest
from execution.feedback import OperationBlocked, OperationCheck, OperationReport


@dataclass(frozen=True)
class DirectStage:
    name: str
    plan: Callable[[SkillRequest], Sequence[Sequence[float]]]
    verify: Callable[..., Tuple[OperationCheck, ...]]
    required_checks: Tuple[str, ...]
    precondition: Optional[Callable[[SkillRequest], OperationReport]] = None
    required_preconditions: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "required_checks", tuple(self.required_checks))
        object.__setattr__(self, "required_preconditions", tuple(self.required_preconditions))
        if bool(self.precondition) != bool(self.required_preconditions):
            raise ValueError("precondition verifier and required names must be supplied together")
        if not self.name or not self.required_checks or any(not n for n in self.required_checks):
            raise ValueError("every stage requires named physical postconditions")


class GuardedDirectBackend:
    """One reviewed proposal per stage; only measured checks unlock the next.

Rejected, partial, modified, terminated or unverified execution blocks this
operation. The Agent can switch skills or explicitly select another subgoal;
no hidden retry, queued tail, or automatic motion is permitted.
"""
    def __init__(self, recipes: Mapping[str, Sequence[DirectStage]]):
        self.recipes = {k: tuple(v) for k, v in recipes.items()}
        for name, stages in self.recipes.items():
            if not name or not stages or len({s.name for s in stages}) != len(stages):
                raise ValueError("recipes require unique stage names")
        self.episode_id = None
        self.pending = None
        self.subgoal = None
        self.index = 0
        self.blocked = False
        self._operations = {}

    def reset(self, episode_id):
        if self.pending is not None:
            raise RuntimeError("resolve direct proposal before reset")
        # Handoff is not a new operation. Preserve failures/completion in this episode.
        if episode_id == self.episode_id:
            return
        self._operations.clear()
        self.episode_id, self.subgoal = episode_id, None
        self.index, self.blocked = 0, False

    def propose(self, request):
        if request.observation.episode_id != self.episode_id or self.pending is not None:
            raise RuntimeError("direct backend has wrong episode or unresolved proposal")
        if request.subgoal not in self.recipes:
            raise ValueError("no direct recipe for the explicit subgoal")
        if request.subgoal != self.subgoal:
            if self.subgoal is not None:
                self._operations[self.subgoal] = (self.index, self.blocked)
            self.subgoal = request.subgoal
            self.index, self.blocked = self._operations.get(self.subgoal, (0, False))
        stages = self.recipes[self.subgoal]
        if self.blocked or self.index >= len(stages):
            raise OperationBlocked("direct operation blocked or complete; select recovery/new subgoal",
                                   OperationReport("blocked", recovery_options=("reobserve", "select_recovery")))
        stage = stages[self.index]
        if stage.precondition is not None:
            report = stage.precondition(request)
            report.validate_observations(request.observation, request.observation)
            checks = {c.name: c.verdict for c in report.checks}
            if (report.failure_kind is not None
                    or any(c.verdict != "passed" for c in report.checks)
                    or any(checks.get(n) != "passed" for n in stage.required_preconditions)):
                raise OperationBlocked("direct stage preconditions are not verified", report)
        rows = tuple(tuple(float(v) for v in row) for row in stage.plan(request))
        # Never truncate a stage (e.g. silently drop closure/lift) to fit a budget.
        if not rows or len(rows) > request.max_actions:
            raise OperationBlocked("direct stage does not fit the reviewed action budget",
                                   OperationReport(stage.name, failure_kind="budget",
                                                   failure_detail="stage exceeds requested action budget",
                                                   recovery_options=("replan",)))
        self.pending = (request, rows, stage)
        return BackendProposal(rows, stage=stage.name)

    def verify_execution(self, proposal, result):
        if self.pending is None:
            raise RuntimeError("no direct stage to verify")
        request, rows, stage = self.pending
        if proposal.request != request or proposal.stage != stage.name:
            raise ValueError("direct stage does not match proposal")
        complete = (tuple(map(tuple, result.executed_actions)) == rows
                    and result.stop_reason == "completed" and not result.terminated)
        checks = tuple(stage.verify(request.observation, result.final_observation))
        return OperationReport(stage.name, "passed" if complete else "unknown", checks,
                               recovery_options=() if complete else ("reobserve", "replan"))

    def observe(self, feedback):
        if self.pending is None:
            raise RuntimeError("direct feedback without pending stage")
        request, rows, stage = self.pending
        report = feedback.operation
        if report is not None:
            report.validate_observations(request.observation, feedback.final_observation)
        complete = (feedback.executed_actions == rows and not feedback.terminated
                    and report is not None and report.stage == stage.name
                    and report.permits_next_stage(stage.required_checks))
        self.pending = None
        if complete:
            self.index += 1
        else:
            self.blocked = True
        self._operations[self.subgoal] = (self.index, self.blocked)
