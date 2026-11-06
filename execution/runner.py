"""Task-internal Agent loop: experience, context, proposal review and skill hand-off.

One episode: the Agent sees allowed observations, its context and the published
experience; it calls one of direct_control / wam_control / vla_control with a
subgoal; the proposal is reviewed; an approved prefix is executed by the single
executor (the only owner of env.step); measured feedback returns to the skill
and to the Agent, which decides again.

Hand-off between skills follows framework §5.4 (and R²SI PR #162 lifecycle):

  1. stop the old skill from submitting;
  2. make sure no action is in flight (complete or cancel);
  3. clear unexecuted rows, rejecting the old pending proposal with zero rows;
  4. take a fresh measured observation;
  5. reset the old and the new backend, so the new model rebuilds its history
     from measured observations only (predictions never enter history);
  6. the new skill proposes from that current state.

Task progress, fixed goals and holding state live in ``TaskState`` and survive.

Every step is written to an event sink, keeping the four records apart:
proposed rows -> approved rows -> executed rows -> measured observation.
Native success/score is not visible here; the trial adapter records it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Union

from agents.context import AgentContext
from rsi.selection_policy import SelectionPolicy
from tools.contracts import ToolSession, ToolRequest, ToolLimits, ToolBudgetExceeded
from skills.contracts import (
    ActionCandidate, ActionRows, ControlSkill, ExecutionFeedback, MeasuredObservation, SkillProposal,
    SkillRegistry, SkillRequest, SkillStopped, SkillExperienceContext,
)
from tools.measurement import MeasurementResult, MeasurementTool
from execution.feedback import OperationBlocked, OperationReport, diagnose_operation
from experience.packages import ExperienceError, ExperienceWorkspace
from execution.review import (
    ReviewDecision, ReviewPolicy, ReviewRejected, RowChecker, Violations, no_checks, validate,
)


# ---------------------------------------------------------------- executor ----
@dataclass(frozen=True)
class ExecutionResult:
    executed_actions: ActionRows          # rows actually dispatched, in the skill's contract
    final_observation: MeasuredObservation
    terminated: bool                      # native episode end (success, failure or timeout)
    stop_reason: str                      # "completed" | "terminated" | "interrupted:<why>"
    measured: Mapping[str, Any] = field(default_factory=dict)  # measured state change summary
    operation: Optional[OperationReport] = None


class Executor(Protocol):
    """Single owner of env.step. Executes rows one by one and may stop early."""
    def current_observation(self) -> MeasuredObservation: ...
    def execute(self, rows: ActionRows, *, action_contract: str) -> ExecutionResult: ...
    def in_flight(self) -> bool: ...
    def cancel(self) -> None: ...   # cancel in-flight work and drop queued rows; idempotent


# ---------------------------------------------------------------- commands ----
@dataclass(frozen=True)
class CallSkill:
    skill: str
    subgoal: str
    max_actions: int


@dataclass(frozen=True)
class Review:
    decision: ReviewDecision


@dataclass(frozen=True)
class ReadExperience:
    op: str                    # "list" | "read_text" | "read_image"
    path: str = ""
    offset: int = 0
    limit: Optional[int] = None


@dataclass(frozen=True)
class Measure:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CallTool:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateTaskState:
    progress: Optional[str] = None
    fix_goal: Optional[Mapping[str, Any]] = None
    clear_goal: Optional[str] = None
    holding: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class Stop:
    reason: str


Command = Union[CallSkill, Review, ReadExperience, Measure, CallTool, UpdateTaskState, Stop]


@dataclass(frozen=True)
class AgentView:
    messages: List[Dict[str, Any]]
    latest_observation: MeasuredObservation
    active_skill: Optional[str]
    pending: Optional[Dict[str, Any]]
    available_skills: tuple
    agent_step: int
    available_measurements: tuple = ()
    measurement_interfaces: Mapping[str, Any] = field(default_factory=dict)
    available_tools: tuple = ()
    tool_interfaces: Mapping[str, Any] = field(default_factory=dict)
    tool_usage: Mapping[str, Any] = field(default_factory=dict)
    allow_experience: bool = True


class AgentPolicy(Protocol):
    def next(self, view: AgentView) -> Command: ...


# ------------------------------------------------------------------ events ----
class JsonlEventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0

    def __call__(self, event: Dict[str, Any]) -> None:
        self.seq += 1
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(seq=self.seq, wall_time=time.time(), **event),
                                    ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _obs(o: MeasuredObservation) -> Dict[str, Any]:
    return dict(observation_id=o.observation_id, control_step=o.control_step,
                time_s=o.time_s, payload_ref=o.payload_ref)


# ------------------------------------------------------------------ runner ----
@dataclass
class RunnerConfig:
    max_agent_steps: int = 200
    history_len: int = 8                     # measured observations passed to skills
    review: ReviewPolicy = field(default_factory=ReviewPolicy)


class TaskSkillRunner:
    def __init__(self, registry: SkillRegistry, executor: Executor, agent: AgentPolicy,
                 context: AgentContext, *, sink: Callable[[Dict[str, Any]], None],
                 row_checkers: Optional[Mapping[str, RowChecker]] = None,
                 experience: Optional[ExperienceWorkspace] = None,
                 config: RunnerConfig = RunnerConfig(),
                 operation_verifiers: Optional[Mapping[str, Callable]] = None,
                 measurement_tools: Optional[Mapping[str, Callable | MeasurementTool]] = None,
                 skill_experience: Optional[Mapping[str, Mapping[str, tuple]]] = None,
                 task_tools=None, tool_limits=ToolLimits(), selection_policy=None,
                 required_prediction_skills=()):
        self.registry, self.executor, self.agent = registry, executor, agent
        self.context, self.sink, self.config = context, sink, config
        self.row_checkers = dict(row_checkers or {})
        self.experience = experience
        self.skill_experience = {}
        for name, channels in (skill_experience or {}).items():
            registry.get(name)
            if not isinstance(channels, Mapping) or set(channels) - {'icl', 'hmr'}:
                raise ValueError('skill experience supports only icl/hmr path lists')
            normalized = {}
            for channel, paths in channels.items():
                if not isinstance(paths, (tuple, list)) or len(paths) > 8:
                    raise ValueError('each experience channel requires at most 8 paths')
                if any(not isinstance(p, str) or not p for p in paths):
                    raise ValueError('experience paths must be nonempty strings')
                normalized[channel] = tuple(paths)
            if any(normalized.values()) and experience is None:
                raise ValueError('enabled skill experience requires a workspace or catalog')
            self.skill_experience[name] = normalized
        self.operation_verifiers = dict(operation_verifiers or {})
        self.measurement_tools = dict(measurement_tools or {})
        self.tool_session = ToolSession(task_tools, limits=tool_limits)
        if selection_policy is not None and not isinstance(selection_policy, SelectionPolicy):
            raise TypeError('selection_policy must be a frozen SelectionPolicy')
        self.selection_policy = selection_policy
        capabilities = set(self.tool_session.tools) | {'measurement:'+n for n in self.measurement_tools}
        if experience is not None: capabilities.add('experience')
        if selection_policy is not None:
            if not set(selection_policy.skills) <= set(registry.names()) or not set(selection_policy.tools) <= capabilities:
                raise ValueError('selection policy requests unregistered capabilities')
            if not set(selection_policy.automatic_tools) <= set(self.tool_session.tools):
                raise ValueError('automatic tools must be registered task tools')
            if 'experience' not in selection_policy.tools and any(any(c.values()) for c in self.skill_experience.values()):
                raise ValueError('experience injection conflicts with selection policy')
        self.required_prediction_skills = frozenset(required_prediction_skills)
        if not self.required_prediction_skills <= set(registry.names()):
            raise ValueError('prediction requirements reference unknown skills')
        if self.required_prediction_skills and 'a2v' not in self.tool_session.tools:
            raise ValueError('host requires a registered a2v tool')
        if selection_policy and self.required_prediction_skills.intersection(selection_policy.skills) and 'a2v' not in selection_policy.tools:
            raise ValueError('selection policy disables host-required A2V')
        self.episode_id = ""
        self.active: Optional[str] = None
        self.activated: List[str] = []
        self.pending: Optional[SkillProposal] = None
        self.pending_violations: Violations = ()
        self.pending_review_error = None
        self._visual_audit = []
        self.latest: Optional[MeasuredObservation] = None
        self.measured: List[MeasuredObservation] = []
        self.terminated = False
        self.skill_stop_reason = None
        self.counters = dict(agent_steps=0, proposals=0, accepted=0, rejected=0,
                             invalid_reviews=0, executed_steps=0, handoffs=0, experience_reads=0)

    # -- bookkeeping ------------------------------------------------------
    def _emit(self, event: str, **data) -> None:
        self.sink(dict(event=event, episode_id=self.episode_id, **data))

    def _set_latest(self, observation: MeasuredObservation) -> None:
        if observation.episode_id != self.episode_id:
            raise ValueError("executor returned an observation from another episode")
        if self.latest is not None and observation.control_step < self.latest.control_step:
            raise ValueError("executor observation went back in time")
        self.latest = observation
        if not self.measured or observation.control_step > self.measured[-1].control_step:
            self.measured.append(observation)

    def _history(self) -> tuple:
        past = [o for o in self.measured if o.control_step < self.latest.control_step]
        return tuple(past[-self.config.history_len:]) if self.config.history_len else ()

    def _skill(self, name: str) -> ControlSkill:
        return self.registry.get(name)   # LookupError: no configured backend

    def _tool(self, kind: str, text: str, images: tuple = ()) -> None:
        self.context.add("tool", kind, text, images)

    # -- hand-off ---------------------------------------------------------
    def _resolve_pending(self, reason: str, *, terminated: bool = False) -> None:
        """Close the pending proposal with zero executed rows at the latest state."""
        if self.pending is None:
            return
        proposal = self.pending
        latest = self.executor.current_observation()
        self._set_latest(latest)
        if latest.control_step != proposal.request.observation.control_step:
            raise RuntimeError("robot moved while a proposal was pending")
        self._skill(proposal.identity.name).observe(
            ExecutionFeedback(proposal.proposal_id, (), latest, terminated))
        self.pending, self.pending_violations = None, ()
        self.tool_session.clear()
        self.counters["rejected"] += 1
        self._emit("review", proposal_id=proposal.proposal_id, skill=proposal.identity.name,
                   decision="reject", reason=reason, approved_steps=0, executed_steps=0,
                   automatic=True)

    def _handoff(self, new: str, reason: str) -> None:
        old = self.active
        steps: List[str] = []
        self.active = None                                   # 1. old skill stops submitting
        steps.append("stop_old_submission")
        if self.executor.in_flight():                        # 2. nothing left in flight
            self.executor.cancel()
        steps.append("in_flight_resolved")
        self._resolve_pending("superseded_by_skill_switch")  # 3. unexecuted rows dropped
        self.executor.cancel()
        steps.append("queue_cleared")
        self._set_latest(self.executor.current_observation())  # 4. fresh measured state
        steps.append("fresh_observation")
        if old is not None:                                  # 5. rebuild model histories
            self._skill(old).reset(self.episode_id)
        self._skill(new).reset(self.episode_id)
        steps.append("histories_reset_from_measured")
        self.active = new                                    # 6. new skill proposes next
        if new not in self.activated:
            self.activated.append(new)
        steps.append("new_skill_ready")
        self.counters["handoffs"] += old is not None
        self._emit("handoff", from_skill=old, to_skill=new, reason=reason, steps=steps,
                   observation=_obs(self.latest), task_state=self.context.state.render())

    # -- command handlers -------------------------------------------------
    def _skill_experience(self, name):
        channels = self.skill_experience.get(name, {})
        if not any(channels.values()):
            return None
        # Only explicit published paths; existing readers enforce bounds and log accesses.
        data = {channel: tuple((path, self.experience.read_text(path, limit=4000))
                               for path in channels.get(channel, ()))
                for channel in ('icl', 'hmr')}
        self.counters['experience_reads'] += sum(len(v) for v in data.values())
        return SkillExperienceContext(json.dumps(self.experience.identity(), sort_keys=True),
                                      **data)

    def _call_skill(self, cmd: CallSkill) -> None:
        if self.selection_policy and cmd.skill not in self.selection_policy.skills:
            self._tool('error', 'skill disabled by frozen selection policy')
            return
        try:
            skill = self._skill(cmd.skill)
        except LookupError as error:
            self._tool("error", str(error))
            return
        if self.pending is not None and cmd.skill == self.active:
            self._tool("error", "review the pending proposal before requesting another one")
            return
        if cmd.skill != self.active:
            self._handoff(cmd.skill, reason=f"agent selected {cmd.skill}")
        self._set_latest(self.executor.current_observation())
        request = SkillRequest(self.latest, cmd.subgoal, skill.identity.action_contract,
                               cmd.max_actions, self._history(), self._skill_experience(cmd.skill))
        try:
            proposal = skill.propose(request)
        except SkillStopped as error:
            self.skill_stop_reason = str(error)
            self._emit('skill_stop', skill=cmd.skill, reason=str(error))
            return
        except OperationBlocked as error:
            error.report.validate_observations(self.latest, self.latest)
            self._emit("proposal_blocked", skill=cmd.skill, reason=str(error),
                       operation=error.report.to_dict(), diagnosis=diagnose_operation(error.report))
            self._tool("operation_blocked", json.dumps(dict(reason=str(error),
                       operation=error.report.to_dict()), sort_keys=True))
            return
        except Exception as error:
            # Never turn ambiguous inference into a replay; the trial ends here.
            self._emit("skill_failure", skill=cmd.skill, error=f"{type(error).__name__}: {error}")
            raise
        checker = self.row_checkers.get(cmd.skill, no_checks)
        violations = checker(proposal, self.latest)
        self.pending, self.pending_violations = proposal, violations
        self.pending_review_error = None
        self.counters["proposals"] += 1
        record = dict(proposal_id=proposal.proposal_id, skill=cmd.skill,
                      backend_revision=proposal.identity.backend_revision,
                      checkpoint=proposal.identity.checkpoint,
                      action_contract=proposal.identity.action_contract,
                      subgoal=cmd.subgoal, observation=_obs(self.latest),
                      history=[o.observation_id for o in request.history],
                      skill_experience=request.experience.to_dict() if request.experience else None,
                      proposed_actions=[list(r) for r in proposal.actions],
                      prediction_ref=proposal.prediction_ref, stage=proposal.stage,
                      candidates=[dict(candidate_id=c.candidate_id,
                          actions=[list(r) for r in c.actions], prediction_ref=c.prediction_ref)
                          for c in proposal.candidates],
                      violations=[list(v) for v in violations])
        if proposal.visual_only:
            # Keep exact candidates out of review-time files; flush after all reviews end.
            self._visual_audit.append(record)
            self._emit('proposal_pending', proposal_id=proposal.proposal_id, skill=cmd.skill,
                       visual_only=True, proposed_steps=len(proposal.actions),
                       candidate_ids=[c.candidate_id for c in proposal.candidates])
        else:
            self._emit("proposal", **record)
        self._tool("proposal", json.dumps(self._pending_view(), sort_keys=True))
        for name in self.selection_policy.automatic_tools if self.selection_policy else ():
            self._call_task_tool(CallTool(name, self._prediction_arguments()), automatic=True)

    def _pending_view(self) -> Optional[Dict[str, Any]]:
        if self.pending is None:
            return None
        p = self.pending
        return dict(proposal_id=p.proposal_id, skill=p.identity.name,
                    subgoal=p.request.subgoal, control_step=p.request.observation.control_step,
                    skill_experience=p.request.experience.to_dict() if p.request.experience else None,
                    actions=None if p.visual_only else [list(r) for r in p.actions],
                    prediction_ref=None if p.visual_only else p.prediction_ref, stage=p.stage,
                    review_feedback=self.pending_review_error,
                    tool_predictions=[dict(receipt_id=r['receipt_id'],tool=r['tool'],revision=r['revision'],
                                           data=r['data']) for r in self.tool_session.receipts
                                      if r['binding']['proposal_id']==p.proposal_id],
                    candidates=[dict(candidate_id=c.candidate_id,
                        actions=None if p.visual_only else [list(r) for r in c.actions],
                        prediction_ref=None if p.visual_only else c.prediction_ref) for c in p.candidates],
                    violations=[(["constraint_violation"] if v else []) if p.visual_only else list(v)
                                for v in self.pending_violations])

    def _review(self, cmd: Review) -> None:
        if self.pending is None:
            self._tool("error", "no pending proposal to review")
            return
        proposal = self.pending
        self._set_latest(self.executor.current_observation())
        skill = self._skill(proposal.identity.name)
        try:
            if cmd.decision.decision != 'reject':
                proposal = skill.candidate(cmd.decision.candidate_id)
            violations = self.row_checkers.get(proposal.identity.name, no_checks)(proposal, self.latest)
            reviewed = validate(cmd.decision, proposal, self.latest, violations,
                                self.config.review)
            if reviewed.decision == 'accept':
                automatic = self.selection_policy.automatic_tools if self.selection_policy else ()
                required = proposal.identity.name in self.required_prediction_skills or 'a2v' in automatic
                cid = cmd.decision.candidate_id if proposal.candidates else 0
                if required and not self.tool_session.matching(proposal,self.latest,reviewed.approved_rows,cid):
                    raise ReviewRejected('matching A2V tool prediction required before execution')
        except (ReviewRejected, ValueError) as error:
            # Zero movement: the proposal stays pending and can be reviewed again.
            self.counters["invalid_reviews"] += 1
            self.pending_review_error = dict(error=str(error), robot_moved=False,
                instruction="Shorten the prefix, choose another candidate, or reject this proposal.")
            self._emit("review_invalid", proposal_id=proposal.proposal_id, error=str(error))
            self._tool("error", f"review rejected: {error}")
            return
        skill = self._skill(proposal.identity.name)
        if reviewed.decision == "reject":
            skill.observe(ExecutionFeedback(proposal.proposal_id, (), self.latest, False))
            self.pending, self.pending_violations = None, ()
            self.tool_session.clear()
            self.counters["rejected"] += 1
            self._emit("review", **reviewed.record, executed_steps=0)
            self._tool("review", f"rejected; robot not moved: {reviewed.reason}")
            return
        self._emit("review", **reviewed.record)
        before = self.latest
        skill.prepare_execution(cmd.decision.candidate_id, reviewed.approved_rows)
        self.pending = proposal
        try:
            result = self.executor.execute(reviewed.approved_rows,
                                           action_contract=proposal.identity.action_contract)
        except BaseException:
            # Motion may have occurred. Never turn an uncertain dispatch into zero
            # feedback during cleanup, and never retry it in this episode.
            self.pending, self.pending_violations = None, ()
            self._emit('execution_fault', proposal_id=proposal.proposal_id,
                       outcome='unknown', replay_allowed=False)
            raise
        executed = tuple(tuple(float(v) for v in r) for r in result.executed_actions)
        if len(executed) > len(reviewed.approved_rows):
            raise RuntimeError("executor ran more rows than were approved")
        if result.final_observation.control_step != before.control_step + len(executed):
            raise RuntimeError("executor feedback does not match executed rows")
        self._set_latest(result.final_observation)
        # Persist dispatched rows BEFORE verification/backend delivery: either can
        # fail after the robot moved. Never erase evidence or replay those rows.
        self.counters["accepted"] += 1
        self.counters["executed_steps"] += len(executed)
        self.terminated = result.terminated
        self._emit("execution", proposal_id=proposal.proposal_id, skill=proposal.identity.name,
                   proposed_steps=len(proposal.actions),
                   approved_actions=[list(r) for r in reviewed.approved_rows],
                   executed_actions=[list(r) for r in executed],
                   unexecuted_approved_steps=len(reviewed.approved_rows) - len(executed),
                   before=_obs(before), after=_obs(self.latest), stop_reason=result.stop_reason,
                   terminated=result.terminated, measured=dict(result.measured))
        report = result.operation
        verifier = self.operation_verifiers.get(proposal.identity.name)
        verification_error = None
        try:
            if verifier is not None:
                if report is not None:
                    raise ValueError("configure only one operation report producer")
                report = verifier(proposal, result)
            if report is not None:
                report.validate_observations(before, self.latest)
                if proposal.stage is not None and report.stage != proposal.stage:
                    raise ValueError("operation report has wrong stage")
        except Exception as error:
            # Actual motion still reaches the backend; unknown outcome blocks
            # guarded direct advancement. A bad verifier cannot trigger replay.
            report = OperationReport(proposal.stage or "unclassified",
                                     failure_kind="unknown", failure_detail=str(error) or type(error).__name__,
                                     recovery_options=("reobserve",))
            verification_error = f"{type(error).__name__}: {error}"
        self._emit("operation_verification", proposal_id=proposal.proposal_id,
                   operation=report.to_dict() if report else None,
                   error=verification_error,
                   diagnosis=diagnose_operation(report) if report and
                   (report.failure_kind or any(c.verdict != "passed" for c in report.checks)) else None)
        # Physical execution is settled. Clear runner pending before observe so
        # episode cleanup cannot incorrectly report a zero-action rejection.
        self.pending, self.pending_violations = None, ()
        self.tool_session.clear()
        try:
            skill.observe(ExecutionFeedback(proposal.proposal_id, executed, self.latest,
                                            result.terminated, report))
        except Exception as error:
            self._emit("feedback_delivery_failed", proposal_id=proposal.proposal_id,
                       executed_steps=len(executed), error=f"{type(error).__name__}: {error}")
            raise
        self._tool("execution", json.dumps(dict(
            proposal_id=proposal.proposal_id, executed_steps=len(executed),
            approved_steps=len(reviewed.approved_rows), stop_reason=result.stop_reason,
            observation=_obs(self.latest), terminated=result.terminated, measured=dict(result.measured),
            operation=report.to_dict() if report else None, verification_error=verification_error),
            sort_keys=True), images=(self.latest.payload_ref,))

    def _read_experience(self, cmd: ReadExperience) -> None:
        if not self._tool_allowed('experience'):
            self._tool('error','experience tool disabled by frozen selection policy')
            return
        if self.experience is None:
            self._tool("error", "no published experience is loaded for this task")
            return
        try:
            if cmd.op == "list":
                text = "\n".join(self.experience.list(cmd.path))
                self._tool("experience", text)
            elif cmd.op == "read_text":
                self._tool("experience", self.experience.read_text(cmd.path, cmd.offset, cmd.limit))
            elif cmd.op == "read_image":
                data, mime = self.experience.read_image(cmd.path)
                self._tool("experience", f"{cmd.path} ({mime}, {len(data)} bytes)",
                           images=(f"experience:{cmd.path}",))
            else:
                raise ExperienceError(f"unknown experience operation {cmd.op!r}")
        except ExperienceError as error:
            self._tool("error", str(error))
            return
        self.counters["experience_reads"] += 1

    def _measure(self, cmd: Measure) -> None:
        if not self._tool_allowed('measurement:'+cmd.tool):
            self._tool('error','measurement disabled by frozen selection policy')
            return
        if self.pending is not None:
            self._tool("error", "resolve the pending motion proposal before measuring")
            return
        tool = self.measurement_tools.get(cmd.tool)
        if tool is None:
            self._tool("error", f"unknown measurement tool {cmd.tool!r}")
            return
        self._set_latest(self.executor.current_observation())
        before = self.latest
        # Callback receives only an immutable observation reference, no executor.
        if isinstance(tool, MeasurementTool):
            result = tool.run(before, dict(cmd.arguments))
            if not isinstance(result, MeasurementResult):
                raise TypeError("parameterized measurement must return MeasurementResult")
            report, data = result.report, dict(result.data)
        else:
            if cmd.arguments:
                self._tool("error", "legacy measurement callback does not accept arguments")
                return
            report, data = tool(before), {}
        current = self.executor.current_observation()
        if current.control_step != before.control_step:
            self._emit("measurement_violation", tool=cmd.tool, before=_obs(before), after=_obs(current))
            raise RuntimeError("measurement tool moved the robot outside the executor")
        report.validate_observations(before, before)
        self._emit("measurement", tool=cmd.tool, arguments=dict(cmd.arguments),
                   observation=_obs(before), operation=report.to_dict(), data=data)
        self._tool("measurement", json.dumps(dict(tool=cmd.tool, operation=report.to_dict(), data=data),
                                             sort_keys=True))

    def _tool_allowed(self, name):
        return self.selection_policy is None or name in self.selection_policy.tools

    def _prediction_arguments(self):
        p=self.pending
        return dict(proposal_id=p.proposal_id, observation_id=p.request.observation.observation_id,
                    candidate_ids=[c.candidate_id for c in p.candidates] if p.candidates else [0])

    def _call_task_tool(self, cmd: CallTool, automatic=False):
        if cmd.tool not in self.tool_session.tools or not self._tool_allowed(cmd.tool):
            self._tool('error','task tool unavailable under frozen selection policy')
            return
        p=self.pending
        if p is None:
            self._tool('error','action prediction requires a pending proposal')
            return
        if p.visual_only:
            raise ValueError('independent task tools are not exposed during strict visual review')
        args=dict(cmd.arguments)
        before=self.executor.current_observation()
        if (set(args)!={'proposal_id','observation_id','candidate_ids'} or
                args['proposal_id']!=p.proposal_id or before!=p.request.observation or
                args['observation_id']!=before.observation_id):
            self._tool('error','stale or malformed prediction request')
            return
        ids=args['candidate_ids']
        candidates=p.candidates or (ActionCandidate(0,p.actions,p.prediction_ref),)
        available={c.candidate_id:c for c in candidates}
        if (not isinstance(ids,list) or not ids or any(type(i) is not int for i in ids)
                or len(set(ids))!=len(ids) or not set(ids)<=set(available)):
            self._tool('error','select existing unique prediction candidates')
            return
        request=ToolRequest(before,p.proposal_id,p.identity.action_contract,tuple(available[i] for i in ids))
        self._emit('tool_call',tool=cmd.tool,proposal_id=p.proposal_id,observation=_obs(before),
                   candidate_ids=ids,automatic=automatic)
        try:
            receipt=self.tool_session.invoke(cmd.tool,request)
            if self.executor.current_observation()!=before:
                raise RuntimeError('prediction tool changed the measured observation')
        except BaseException as error:
            self._emit('tool_failure',tool=cmd.tool,error=type(error).__name__,usage=self.tool_session.snapshot())
            raise
        self._emit('tool_result',**receipt,usage=self.tool_session.snapshot())
        self._tool('prediction',json.dumps(dict(receipt_id=receipt['receipt_id'],tool=cmd.tool,
                   proposal_id=p.proposal_id,observation_id=before.observation_id,data=receipt['data']),sort_keys=True))

    def _update_state(self, cmd: UpdateTaskState) -> None:
        state = self.context.state
        try:
            if cmd.progress:
                state.progress.append(cmd.progress)
            if cmd.clear_goal:
                state.clear_goal(cmd.clear_goal)
            for name, value in (cmd.fix_goal or {}).items():
                state.fix_goal(name, value)
            if cmd.holding is not None:
                state.holding = dict(cmd.holding)
        except ValueError as error:
            self._tool("error", str(error))
            return
        self._emit("task_state", **state.render())

    # -- episode ----------------------------------------------------------
    def run(self, episode_id: str) -> Dict[str, Any]:
        if not episode_id:
            raise ValueError("episode identity is required")
        self.episode_id = episode_id
        self._set_latest(self.executor.current_observation())
        if self.experience is not None and self._tool_allowed('experience'):
            self.context.pin(self.experience.instructions())
        self._emit("episode_start", observation=_obs(self.latest),
                   task_state=self.context.state.render(),
                   skills=list(self.registry.names()),
                   experience=self.experience.identity() if self.experience else None,
                   selection_policy=self.selection_policy.identity() if self.selection_policy else None,
                   required_prediction_skills=sorted(self.required_prediction_skills),
                   task_tools={n:t.interface() for n,t in self.tool_session.tools.items()})
        reason = "agent_step_budget"
        try:
            while not self.terminated and self.counters["agent_steps"] < self.config.max_agent_steps:
                self.context.begin_turn()
                self.context.add("observation", "measured",
                                 json.dumps(_obs(self.latest), sort_keys=True),
                                 images=(self.latest.payload_ref,))
                view = AgentView(self.context.messages(), self.latest, self.active,
                                 self._pending_view(),
                                 self.selection_policy.skills if self.selection_policy else self.registry.names(),
                                 self.counters["agent_steps"], tuple(n for n in self.measurement_tools
                                     if self._tool_allowed('measurement:'+n)),
                                 {name: tool.interface() for name, tool in self.measurement_tools.items()
                                  if isinstance(tool, MeasurementTool) and self._tool_allowed('measurement:'+name)},
                                 available_tools=tuple(n for n in self.tool_session.tools if self._tool_allowed(n)),
                                 tool_interfaces={n:t.interface() for n,t in self.tool_session.tools.items() if self._tool_allowed(n)},
                                 tool_usage=self.tool_session.snapshot(),allow_experience=self._tool_allowed('experience'))
                command = self.agent.next(view)
                self.counters["agent_steps"] += 1
                self.context.add("agent", type(command).__name__,
                                 json.dumps(_command(command), sort_keys=True, default=str))
                if isinstance(command, Stop):
                    reason = "agent_stop:" + command.reason
                    break
                if isinstance(command, CallSkill):
                    self._call_skill(command)
                    if self.skill_stop_reason is not None:
                        reason = 'skill_stop:' + self.skill_stop_reason
                        break
                elif isinstance(command, Review):
                    self._review(command)
                elif isinstance(command, ReadExperience):
                    self._read_experience(command)
                elif isinstance(command, CallTool):
                    self._call_task_tool(command)
                elif isinstance(command, Measure):
                    self._measure(command)
                elif isinstance(command, UpdateTaskState):
                    self._update_state(command)
                else:
                    raise TypeError(f"unknown Agent command {command!r}")
            if self.terminated:
                reason = "native_termination"
        except ToolBudgetExceeded:
            reason = 'tool_call_budget_exhausted'
        finally:
            # Stop submissions, close any pending proposal with zero rows; the trial
            # adapter then records native results before the environment reset.
            pending_at_end = self.pending is not None
            try:
                self._resolve_pending("episode_end", terminated=True)
            finally:
                self.active = None
                for record in self._visual_audit:
                    self._emit('proposal', **record, deferred_until_episode_end=True)
                self._visual_audit.clear()
                self._emit("episode_end", reason=reason, counters=dict(self.counters),
                           final_observation=_obs(self.latest), task_state=self.context.state.render(),
                           pending_at_end=pending_at_end)
        return dict(reason=reason, counters=dict(self.counters), final_observation=self.latest,
                    activated_skills=list(self.activated),tool_usage=self.tool_session.snapshot())

    def _configured(self, name: str) -> bool:
        try:
            self.registry.get(name)
            return True
        except LookupError:
            return False


def _command(command: Command) -> Dict[str, Any]:
    out = dict(type=type(command).__name__)
    for key, value in vars(command).items():
        out[key] = vars(value) if hasattr(value, "__dataclass_fields__") else value
    return out
