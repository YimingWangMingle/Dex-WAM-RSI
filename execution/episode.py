"""Simulator-independent assembly of the existing TaskSkillRunner.

The host resets/owns the environment. Factories receive immutable measured
snapshots and one shared ModelBudget, never an execution handle. All model
clients must be wrapped with that budget by their factory.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from agents.context import AgentContext, ContextPolicy, TaskState
from skills.contracts import MeasuredObservation, SkillRegistry
from agents.budget import ModelBudget, ModelBudgetExhausted
from execution.review import BoundsChecker
from execution.runner import ExecutionResult, JsonlEventLog, RunnerConfig, TaskSkillRunner


class MeasuredSession:
    def __init__(self, environment, root, max_steps):
        self.environment, self.root = environment, Path(root)
        self.max_steps, self.episode_id = max_steps, uuid4().hex
        self.records, self.busy, self.faulted = [], False, False
        self.executed_steps = 0
        self.capture()

    def capture(self):
        obs = self.environment.observe()
        contract = self.environment.contract
        if (type(obs.step) is not int or obs.step != self.executed_steps or
                not math.isfinite(obs.time_s) or abs(obs.time_s-obs.step/contract.control_hz)>1e-6):
            raise ValueError('measured observation step/time differs from execution contract')
        if (not isinstance(obs.instruction,str) or not obs.instruction.strip() or
                len(obs.state)!=contract.dimension or not all(math.isfinite(v) for v in obs.state)):
            raise ValueError('measured instruction and finite state required')
        folder = self.root/'observations'/self.episode_id/f'{obs.step:06d}'
        folder.mkdir(parents=True)
        images, hashes = {}, {}
        for i,(name,rgb) in enumerate(obs.images.items()):
            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError('measured uint8 HWC RGB required')
            path = folder/f'view-{i}.png'
            Image.fromarray(rgb).save(path)
            images[name] = str(path)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not images:
            raise ValueError('at least one measured camera required')
        document = dict(instruction=obs.instruction,state=list(obs.state),images=images,
                        image_hashes=hashes,step=obs.step,time_s=obs.time_s)
        data = json.dumps(document,sort_keys=True).encode()
        path = folder/'observation.json'; path.write_bytes(data)
        ref = MeasuredObservation(self.episode_id,hashlib.sha256(data).hexdigest(),
                                  obs.step,obs.time_s,str(path))
        self.records.append(ref)
        return ref

    def load(self, ref):
        if ref.episode_id != self.episode_id or ref not in self.records:
            raise ValueError('foreign measured observation')
        data = Path(ref.payload_ref).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref.observation_id:
            raise ValueError('measured observation modified')
        doc = json.loads(data)
        for path, sha in doc['image_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError('measured image modified')
        return doc


class EnvironmentExecutor:
    def __init__(self, session, sink, execution_check=None):
        self.session, self.sink, self.check = session, sink, execution_check

    def current_observation(self): return self.session.records[-1]
    def in_flight(self): return self.session.busy
    def cancel(self):
        if self.in_flight(): raise RuntimeError('synchronous action is in flight')

    def execute(self, rows, *, action_contract):
        s, env = self.session, self.session.environment
        rows = tuple(tuple(float(x) for x in row) for row in rows)
        if s.busy or s.faulted or env.is_terminal():
            raise RuntimeError('environment executor unavailable')
        if action_contract != env.contract.name or not rows or s.executed_steps+len(rows)>s.max_steps:
            raise ValueError('action contract or execution budget mismatch')
        previous = s.load(self.current_observation())['state']
        for row in rows:
            env.contract.validate(row, previous)
            previous = row
        executed = []
        s.busy = True
        try:
            for row in rows:
                if self.check: self.check()
                # Check against actual measured state immediately before dispatch, too.
                env.contract.validate(row,s.load(self.current_observation())['state'])
                env.execute(row)
                executed.append(row); s.executed_steps += 1
                self.sink(dict(event='native_dispatch',control_step=s.executed_steps,action=row))
                s.capture()
                if env.is_terminal(): break
        except BaseException:
            s.faulted = True
            self.sink(dict(event='native_execution_fault',executed_rows=executed))
            raise
        finally:
            s.busy = False
        terminal = env.is_terminal()
        return ExecutionResult(tuple(executed),self.current_observation(),terminal,
                               'terminated' if terminal else 'completed')


def run_episode(environment, *, root, skill_factories, agent_factory, action_spec=None,
                max_steps=800, max_model_calls=170, execution_check=None,
                experience=None, experience_catalog=None, experience_task=None,
                context_policy=None, context_summarizer=None,
                skill_experience=None, candidate_revision=None,
                measurement_tools=None, operation_verifiers=None, task_tools=None, tool_limits=None,
                selection_policy=None, required_prediction_skills=()):
    """Run an already-reset Environment; caller closes it even on errors.

    Factories: skill_factory(load_snapshot, model_budget) -> ControlSkill;
    agent_factory(load_snapshot, model_budget) -> AgentPolicy. This API does not
    silently adapt the RoboDojo-specific WAM/PI0.5 input/action contracts.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('positive action budget required')
    if action_spec is not None and action_spec != environment.contract:
        raise ValueError('host action contract differs from environment')
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("episode output must be empty")
    root.mkdir(parents=True,exist_ok=True)
    budget = ModelBudget(max_model_calls)
    sink = JsonlEventLog(root/'framework-events.jsonl')
    session = MeasuredSession(environment, root, max_steps)
    if experience_catalog is not None:
        if experience is not None:
            raise ValueError('supply a workspace or catalog, not both')
        task = experience_task or environment.identity().get('task')
        if not task:
            raise ValueError('experience catalog requires an explicit task identity')
        from experience.packages import ExperienceCatalog, materialize
        experience = materialize(ExperienceCatalog.load(experience_catalog),task,
                                 root/session.episode_id,on_access=sink)
    from tools.contracts import ToolLimits
    registry = SkillRegistry()
    for factory in skill_factories:
        skill = factory(session.load,budget)
        if (skill.identity.action_contract != environment.contract.name or
                skill.identity.action_dim != environment.contract.dimension):
            raise ValueError('skill and environment action contracts differ')
        registry.register(skill)
    if not registry.names(): raise ValueError('at least one skill required')
    underlying = agent_factory(session.load,budget)
    class BoundedAgent:
        def next(self, view):
            from execution.runner import Stop, CallSkill
            remaining = max_steps-session.executed_steps
            if environment.is_terminal(): return Stop('terminated')
            if remaining <= 0: return Stop('action_budget_exhausted')
            command = underlying.next(view)
            if isinstance(command,CallSkill) and command.max_actions > remaining:
                return CallSkill(command.skill,command.subgoal,remaining)
            return command
    contract = environment.contract
    checker = BoundsChecker(lower=contract.lower,upper=contract.upper,max_delta=contract.max_delta,
                            reference=lambda ref:session.load(ref)['state'])
    runner = TaskSkillRunner(registry,EnvironmentExecutor(session,sink,execution_check),BoundedAgent(),
        AgentContext(TaskState(session.load(session.records[0])['instruction']),
            policy=context_policy or ContextPolicy(),on_event=sink,
            summarizer=budget.callback(context_summarizer,'context_summary')),sink=sink,
        experience=experience,skill_experience=skill_experience,measurement_tools=measurement_tools,
        task_tools=task_tools,tool_limits=tool_limits or ToolLimits(),selection_policy=selection_policy,
        required_prediction_skills=required_prediction_skills,
        operation_verifiers=operation_verifiers,row_checkers={n:checker for n in registry.names()},
        config=RunnerConfig(max_agent_steps=max_steps*4+20))
    reason = 'error'
    try:
        result = runner.run(session.episode_id)
        reason = result['reason']
        return dict(executed_steps=session.executed_steps,stop_reason=reason,framework=result)
    except ModelBudgetExhausted:
        reason = 'model_budget_exhausted'
        return dict(executed_steps=session.executed_steps,stop_reason=reason)
    finally:
        outcome = environment.outcome()
        if type(outcome.success) not in (bool,type(None)):
            raise ValueError('native success must be bool or unknown')
        summary = dict(framework='TaskSkillRunner/v1',platform=environment.identity(),
            action_contract=asdict(contract),candidate_revision=candidate_revision,
            episode_id=session.episode_id,registered_skills=list(registry.names()),
            native_success=outcome.success,native_score=outcome.score,native_ended=outcome.terminated,
            executed_steps=session.executed_steps,stop_reason=reason,model_budget=budget.snapshot(),
            tool_usage=runner.tool_session.snapshot(),
            selection_policy=selection_policy.identity() if selection_policy else None)
        (root/'episode-summary.json').write_text(json.dumps(summary,indent=2))
