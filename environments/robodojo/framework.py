"""RoboDojo experiments composed inside TaskSkillRunner.

All modes share the same measured store, executor, context, experience and event
sink. Backends propose; only NativeExecutor owns take_action. Native verdicts
never enter AgentView. Model transport selection is not physical execution.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from agents.context import AgentContext, ContextPolicy, TaskState
from environments.robodojo.training import NATIVE_ACTION_CONTRACT
from skills.contracts import (ActionCandidate, BackendProposal, ControlSkill, ModelControlSkill,
    MeasuredObservation, SkillIdentity, SkillRegistry, SkillStopped)
from execution.review import ReviewDecision, BoundsChecker
from execution.runner import (CallSkill, ExecutionResult, JsonlEventLog,
    Review, RunnerConfig, Stop, TaskSkillRunner)
from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect.types import (
    ActionSpace, JOINT_CHANNELS, Observation)


def flat(action):
    """Exact native joint14 conversion; missing channels are errors, never zeros."""
    if isinstance(action, dict):
        parts = []
        for key, size in JOINT_CHANNELS:
            values = np.asarray(action[key], dtype=np.float32).reshape(-1)
            if values.shape != (size,):
                raise ValueError('wrong native channel shape: ' + key)
            parts.extend(float(x) for x in values)
        return tuple(parts)
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (14,):
        raise ValueError('native joint14 required')
    return tuple(float(x) for x in values)


class NativeSession:
    def __init__(self, env, root, *, max_steps, control_hz=25.0, enable_geometry=False):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('positive native step budget required')
        self.env, self.root = env, Path(root)
        self.enable_geometry = enable_geometry
        self.episode_id = uuid4().hex
        self.directory = self.root / 'observations' / self.episode_id
        self.directory.mkdir(parents=True)
        self.limit, self.hz = max_steps, control_hz
        self.records = []
        self.decision_steps = []
        self.faulted = False
        self.trajectory = None
        self.capture()

    def capture(self):
        step = len(self.records)
        source = self.env.get_obs()
        state = source.get('state', {})
        values = flat(state)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite measured state')
        hz = source.get('additional_info', {}).get('frequency')
        if hz is None or float(hz) != self.hz:
            raise ValueError('native observation frequency differs from contract')
        instruction = source.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('explicit native instruction required')
        folder = self.directory / f'{step:06d}'
        folder.mkdir()
        images, hashes = {}, {}
        for index, (key, view) in enumerate(source.get('vision', {}).items()):
            if 'color' not in view:
                continue
            rgb = view['color']
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
                from XPolicyLab.utils.process_data import decode_image_bit
                rgb = np.asarray(decode_image_bit(rgb))
            if rgb.dtype != np.uint8 or rgb.shape[-1] != 3:
                raise ValueError('native uint8 RGB image required')
            path = folder / f'view-{index}.png'
            Image.fromarray(rgb).save(path)
            images[key] = str(path)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not images:
            raise ValueError('measured RGB required')
        # Raw server payload is restricted to measured RGB/proprioception and Hz.
        # Keep state on disk; do not retain every decoded camera frame in memory.
        serial_state = {k: np.asarray(v).tolist() for k, v in state.items()}
        doc = dict(step=step, instruction=instruction, state=serial_state, images=images, image_hashes=hashes)
        if self.enable_geometry:
            from environments.robodojo.runtime.robodojo_local_geometry import LocalGeometry
            geometry, errors = {}, {}
            public = dict(vision={k:dict(color=np.asarray(Image.open(v).convert('RGB'))) for k,v in images.items()})
            for name in images:
                try:
                    frame = LocalGeometry(self.env).camera(name, public, step)
                    geometry[name] = dict(K=frame.K.tolist(), T_world_camera=frame.T_world_camera.tolist(),
                        time_s=frame.time_s, calibration_revision=frame.calibration_revision,
                        rectified_native=frame.rectified_native, calibration_image_hw=list(frame.calibration_image_hw))
                except ValueError as error:
                    errors[name] = str(error)
            doc.update(calibration=geometry, calibration_errors=errors)
        path = folder / 'observation.json'
        data = json.dumps(doc, sort_keys=True).encode()
        path.write_bytes(data)
        ref = MeasuredObservation(self.episode_id, hashlib.sha256(data).hexdigest(),
                                  step, step / self.hz, str(path))
        self.records.append(ref)
        if self.trajectory is not None:
            self.record_measured(ref)
        return ref

    def record_measured(self, ref):
        doc = self.load(ref)
        self.trajectory.write('measured',step=ref.control_step,time_s=ref.time_s,
            state=list(flat(doc['state'])),executed_row=ref.control_step-1 if ref.control_step else None,
            observation_ref=ref.payload_ref, images=doc['images'],
            camera_names=list(doc['images']),
            attachments={f'rgb_{i}':np.asarray(Image.open(path).convert('RGB'))
                         for i,path in enumerate(doc['images'].values())})

    def load(self, ref):
        if ref.episode_id != self.episode_id or self.records[ref.control_step] != ref:
            raise ValueError('foreign observation')
        data = Path(ref.payload_ref).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref.observation_id:
            raise ValueError('measured observation modified')
        doc = json.loads(data)
        for path, expected in doc['image_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('measured RGB modified')
        return doc

    def raw(self, step):
        doc = self.load(self.records[step])
        return dict(instruction=doc['instruction'],
            state={k: np.asarray(v) for k, v in doc['state'].items()},
            vision={k: dict(color=np.asarray(Image.open(p).convert('RGB')))
                    for k, p in doc['images'].items()},
            additional_info=dict(frequency=self.hz), rsi_control_step=step)

    def policy_observation(self, ref):
        raw = self.raw(ref.control_step)
        return Observation(images={k: v['color'] for k, v in raw['vision'].items()},
            state=raw['state'], instruction=raw['instruction'], step=ref.control_step,
            remaining_steps=self.limit-ref.control_step,
            extra=dict(task=getattr(self.env, 'task_name', None)))

    def visual(self, ref):
        doc = self.load(ref)
        history = [dict(step=n, source='measured',
                   images=list(self.load(self.records[n])['images'].values()))
                   for n in self.decision_steps[-3:] if n < ref.control_step]
        return list(doc['images'].values()), history


class NativeExecutor:
    def __init__(self, session, contract, sink, *, after_step=None, execution_check=None):
        self.session, self.contract, self.sink = session, contract, sink
        self.busy = False
        self.after_step = after_step
        self.execution_check = execution_check

    def current_observation(self):
        return self.session.records[-1]

    def in_flight(self):
        return self.busy

    def cancel(self):
        if self.busy:
            raise RuntimeError('synchronous native action is in flight')

    def execute(self, rows, *, action_contract):
        s = self.session
        if s.faulted or self.busy or s.env.is_episode_end():
            raise RuntimeError('native executor not available')
        values = np.asarray(rows, dtype=np.float32)
        if (action_contract != self.contract or values.ndim != 2 or values.shape[1] != 14
                or not len(values) or not np.isfinite(values).all()
                or len(s.records)-1+len(values) > s.limit
                or ((values[:, [6,13]] < 0) | (values[:, [6,13]] > 1)).any()):
            raise ValueError('invalid approved native prefix')
        s.decision_steps.append(len(s.records)-1)
        if s.trajectory is not None:
            from execution.trajectory import action_hash
            s.trajectory.write('command',decision_step=len(s.records)-1,
                hash=action_hash(values),source_hash=action_hash(values),
                postprocess='framework_exact_prefix',attachments=dict(actions=values))
        executed = []
        interrupted = False
        self.busy = True
        try:
            for original, row in zip(rows, values):
                if self.execution_check is not None:
                    self.execution_check()
                s.env.take_action(dict(ActionSpace().decode(row).data))
                executed.append(tuple(original))
                # Persist command evidence before observation transport can fail.
                self.sink(dict(event='native_dispatch', episode_id=s.episode_id,
                               control_step=len(s.records)-1, action=list(original)))
                s.capture()
                if s.env.is_episode_end():
                    break
                if self.after_step is not None and self.after_step(self.current_observation()):
                    interrupted = True
                    break
        except BaseException:
            s.faulted = True
            self.sink(dict(event='native_execution_fault', executed_rows=executed))
            raise
        finally:
            self.busy = False
        terminated = bool(s.env.is_episode_end())
        return ExecutionResult(tuple(executed), self.current_observation(), terminated,
                               'terminated' if terminated else 'event_interrupted' if interrupted else 'completed')


class EEFBackend:
    """Existing RoboProbe planning becomes a proposal-only direct skill."""
    def __init__(self, policy, session):
        self.policy, self.session = policy, session
        self.started = False
        self.view = None
        self.reason = ''

    def reset(self, episode_id):
        if episode_id != self.session.episode_id:
            raise ValueError('foreign direct episode')
        # Keep task conversation across skill hand-offs, but no pending motion.
        if not self.started:
            self.policy.reset()
            self.started = True

    def propose(self, request):
        observation = replace(self.session.policy_observation(request.observation), instruction=request.subgoal)
        chunk = self.policy.act(observation)
        if any(action.meta.get('request_stop') for action in chunk.actions):
            self.policy.confirm_executed(0)
            raise SkillStopped(str(chunk.reasoning or 'EEF controller stopped'))
        rows = tuple(flat(dict(action.data)) for action in chunk.actions)
        if len(rows) > request.max_actions:
            self.policy.confirm_executed(0)
            raise SkillStopped('complete direct plan exceeds remaining budget')
        self.reason = chunk.reasoning or 'EEF tool request and native planner accepted this complete path'
        return BackendProposal(rows, stage='eef_plan')

    def observe(self, feedback):
        self.policy.confirm_executed(len(feedback.executed_actions))


class WAMCandidates:
    """Transport adapter; framework owns selection approval and physical motion."""
    def __init__(self, client, session, *, mode3=True, predict=True, candidate_count=None, action_only=False):
        self.client, self.session = client, session
        self.action_only = action_only
        self.mode3, self.predict = mode3, predict
        self.count = candidate_count if candidate_count is not None else (3 if mode3 else 1)
        self.preview = None
        self.selected = False
        self.synced = -1
        self.revision_plans = None
        self.recovery_feedback = None
        self.sampling_step = None
        self.sampling_round = 0
        self.identity = client.call('framework_identity')
        if (self.identity.get('action_contract') != NATIVE_ACTION_CONTRACT
                or self.identity.get('action_dim') != 14 or self.identity.get('control_hz') != session.hz):
            raise ValueError('WAM/native action contract mismatch')

    def reset(self, episode_id):
        if self.preview is not None:
            raise RuntimeError('resolve candidate set before handoff')
        self.client.call('reset')
        self.synced = -1
        self.revision_plans = self.recovery_feedback = None
        self.sampling_step = None
        self.sampling_round = 0

    def sync_observation(self, observation):
        # Rebuild from measured rows, never from predictions or unexecuted plans.
        for step in range(self.synced+1, observation.control_step+1):
            self.client.call('framework_observe', self.session.raw(step))
            self.synced = step

    def propose(self, request):
        self.sync_observation(request.observation)
        if request.observation.control_step != self.sampling_step:
            self.sampling_step = request.observation.control_step
            self.sampling_round = 0
            self.recovery_feedback = None
        request_packet = dict(mode3=self.mode3, predict=self.predict, count=self.count)
        if self.revision_plans is not None:
            request_packet['plans'] = self.revision_plans
        elif self.sampling_round and not self.mode3:
            request_packet['seed_offset'] = self.sampling_round * self.count
        result = self.client.call('framework_preview', request_packet)
        self.preview = result['preview']
        self.revision_plans = None
        self.sampling_round += 1
        self.selected = False
        limit = request.max_actions if (self.mode3 or self.action_only or getattr(self,'tool_horizon',False)) else min(request.max_actions,self.preview['execute_steps'])
        self.offered_steps = min(limit, min(len(c['actions']) for c in result['candidates'])) if result['candidates'] else 0
        candidates = tuple(ActionCandidate(c['candidate_id'],
            tuple(flat(row) for row in c['actions'][:limit]),
            None if self.action_only else json.dumps(c.get('forecast'), sort_keys=True)) for c in result['candidates'])
        self.offered_candidates = candidates
        if not candidates:
            raise ValueError('empty WAM candidate set')
        return BackendProposal(candidates[0].actions, candidates[0].prediction_ref,
                               candidates=candidates, visual_only=self.mode3)

    def packet(self, ref):
        from environments.robodojo.runtime.robodojo_a2v import mode3_astra_input
        images, history = self.session.visual(ref)
        if self.action_only:
            packet = dict(actor='wam', review_mode='action_only', current_rgb=images, history=history,
                instruction=self.session.load(ref)['instruction'], observation_id=ref.observation_id,
                decision_step=ref.control_step, candidates=[dict(candidate_id=c.candidate_id,
                    actions=[list(row) for row in c.actions]) for c in self.offered_candidates])
        else:
            packet = mode3_astra_input(self.preview, images=images, history=history,
                instruction=self.session.load(ref)['instruction'])
        lengths = range(1, self.offered_steps+1) if getattr(self,'recovery_review',False) and not self.mode3 else (24,16,8,4)
        packet['allowed_execute_steps'] = [n for n in lengths
                                          if n <= min(self.offered_steps,self.session.limit-ref.control_step)]
        packet['recovery_feedback'] = self.recovery_feedback
        return packet

    def revise(self, choice):
        from environments.robodojo.action_revision import revised_plans
        if self.mode3:
            raise ValueError('strict visual mode cannot revise actions')
        if choice['operation'] == 'edit':
            self.revision_plans = revised_plans(self.preview['candidates'], choice['candidate_id'], choice['edits'])
        self.recovery_feedback = dict(operation=choice['operation'], reason=choice['reason'],
            edits=copy.deepcopy(choice['edits']), candidate_id=choice['candidate_id'],
            at_step=self.sampling_step, sampling_round=self.sampling_round,
            robot_moved=False, requires_fresh_review=True)

    def prepare_execution(self, candidate_id, rows):
        request = dict(review_id=self.preview['review_id'], candidate_id=candidate_id,
                       execute_steps=len(rows))
        # Never retry an uncertain selection; no physical movement occurs here.
        self.selected = True
        selected = self.client.call('framework_select', request)
        actual = tuple(flat(row) for row in selected)
        if actual != rows:
            raise ValueError('server selection differs from framework-approved rows')

    def sync_execution(self, ref):
        for step in range(self.synced+1, ref.control_step+1):
            self.client.call('update_obs', self.session.raw(step))
            self.synced = step

    def observe(self, feedback):
        if self.preview is None:
            return
        if not self.selected:
            self.client.call('discard_candidates', {'review_id': self.preview['review_id']})
        else:
            self.sync_execution(feedback.final_observation)
            self.client.call('framework_feedback', dict(
                executed_actions=[list(r) for r in feedback.executed_actions],
                control_step=feedback.final_observation.control_step))
        self.preview = None


class ExperimentAgent:
    """Mode policy inside the common runner; a selector can switch real skills."""
    def __init__(self, session, *, direct=None, wam=None, reviewer=None, selector=None, experience=None, vla=None, vla_reviewer=None, skill_reviewers=None, max_review_attempts=3):
        self.session, self.direct, self.wam = session, direct, wam
        self.vla, self.vla_reviewer = vla, vla_reviewer
        self.skill_reviewers = dict(skill_reviewers or {})
        self.max_review_attempts = max_review_attempts
        self.review_attempts = {}
        self.reviewer, self.selector = reviewer, selector
        self.view = None
        self.experience = experience
        self.experience_reads = list(experience.selection.entry_points) if experience else []

    def _with_context(self, packet):
        # Share bounded task evidence, never the pending proposal's hidden rows.
        packet = copy.deepcopy(packet)
        packet['review_feedback'] = copy.deepcopy(
            (self.view.pending or {}).get('review_feedback')) if self.view else None
        packet['skill_experience'] = copy.deepcopy(
            (self.view.pending or {}).get('skill_experience')) if self.view else None
        allowed = {'pinned', 'task_state', 'summary', 'experience', 'measurement', 'execution', 'prediction'}
        packet['tool_predictions'] = copy.deepcopy((self.view.pending or {}).get('tool_predictions', [])) if self.view else []
        packet['framework_context'] = [copy.deepcopy(m) for m in self.view.messages
                                       if m.get('kind') in allowed] if self.view else []
        return packet

    def _prefix_review(self, skill):
        if skill == 'wam_control':
            return self.wam is not None and (self.wam.mode3 or self.wam.action_only)
        return skill == 'vla_control' and self.vla_reviewer is not None

    def _review(self, proposal, observation):
        skill = proposal['skill']
        if skill in self.skill_reviewers:
            decision = self.skill_reviewers[skill](copy.deepcopy(proposal), copy.deepcopy(self.view))
            if not isinstance(decision, ReviewDecision):
                raise ValueError('registered skill reviewer must return ReviewDecision')
            return Review(decision)
        if skill == 'direct_control':
            cid, count, reason = None, len(proposal['actions']), getattr(self.direct, "reason", "Configured direct stage")
        else:
            backend, reviewer = ((self.wam, self.reviewer) if skill == 'wam_control'
                                 else (self.vla, self.vla_reviewer))
            if (getattr(self,'review_protocol','standard') == 'experiment-v3' and
                    ((skill == 'wam_control' and not backend.mode3) or skill == 'vla_control') and reviewer is not None):
                from environments.robodojo.action_revision import validate_operation
                packet = self._with_context(backend.packet(observation) if skill == 'vla_control' or backend.action_only else backend.preview)
                packet['allowed_execute_steps'] = list(range(1, min(backend.offered_steps, self.session.limit-observation.control_step)+1))
                packet['recovery_feedback'] = backend.recovery_feedback
                packet['review_operations'] = ['execute', 'resample', 'edit', 'stop']
                choice = reviewer(packet)
                if isinstance(choice, dict) and 'operation' in choice:
                    operation = validate_operation(choice)
                    if operation == 'stop':
                        return Stop('reviewer_stop:' + choice['reason'])
                    if operation in ('resample', 'edit'):
                        backend.revise(choice)
                        return Review(ReviewDecision(proposal['proposal_id'], 'reject',
                            choice['reason'], proposal['subgoal'], 0, None))
                    cid, count = choice['candidate_id'], choice['execute_steps']
                elif type(choice) is int:
                    cid, count = choice, max(packet['allowed_execute_steps'])
                elif choice is None:
                    return Stop('reviewer_stopped')  # legacy protocol; explicit resample opts in
                else:
                    if not isinstance(choice, dict) or set(choice) != {'candidate_id','execute_steps'}:
                        raise ValueError('review requires candidate_id and execute_steps')
                    cid, count = choice['candidate_id'], choice['execute_steps']
                if cid is None and count == 0:
                    return Stop('reviewer_stopped')
                if type(count) is not int or count not in packet['allowed_execute_steps']:
                    raise ValueError('unsupported execution length')
            elif self._prefix_review(skill):
                from environments.robodojo.runtime.robodojo_mode3 import validate_choice
                packet = self._with_context(backend.packet(observation))
                choice = validate_choice(reviewer(packet), packet['allowed_execute_steps'])
                cid, count = (choice['candidate_id'], choice['execute_steps']) if choice else (None, 0)
            elif reviewer is None:
                cid, count = 0, None
            else:
                packet = self._with_context(backend.preview)
                packet['allowed_execute_steps'] = [n for n in (24,16,8,4)
                    if n <= min(len(c['actions']) for c in proposal['candidates'])]
                choice = reviewer(packet)
                if isinstance(choice, dict):
                    if set(choice) != {'candidate_id','execute_steps'}:
                        raise ValueError('review requires candidate_id and execute_steps')
                    cid, count = choice['candidate_id'], choice['execute_steps']
                    if not (cid is None and type(count) is int and count == 0):
                        if type(count) is not int or count not in packet['allowed_execute_steps']:
                            raise ValueError('unsupported execution length')
                else:
                    cid, count = choice, None
            if cid is None:
                return Stop('pi05_reviewer_stopped' if skill == 'vla_control' else 'reviewer_stopped')
            candidate = next((c for c in proposal['candidates'] if c['candidate_id'] == cid), None)
            if type(cid) is not int or candidate is None:
                raise ValueError('reviewer must select an existing integer candidate ID')
            count = len(candidate['actions']) if count is None else count
            reason = 'Configured reviewer selected candidate prefix' if reviewer else 'Configured single-candidate baseline'
        return Review(ReviewDecision(proposal['proposal_id'], 'accept', reason,
                                     proposal['subgoal'], count, cid))

    def next(self, view):
        self.view = view
        if self.session.execution_check is not None:
            self.session.execution_check()
        s = self.session
        if self.experience_reads and view.allow_experience and not view.pending:
            from execution.runner import ReadExperience
            return ReadExperience('read_text',self.experience_reads.pop(0))
        if s.env.is_episode_end() or view.latest_observation.control_step >= s.limit:
            return Stop('native_step_budget_or_end')
        if view.pending:
            proposal_id = view.pending['proposal_id']
            attempts = self.review_attempts.get(proposal_id, 0)
            if attempts >= self.max_review_attempts:
                return Stop('review_attempt_budget_exhausted')
            if getattr(self, 'agent_review', False):
                command = self.selector(view)
                if isinstance(command, Review):
                    self.review_attempts[proposal_id] = attempts + 1
                return command
            self.review_attempts[proposal_id] = attempts + 1
            return self._review(view.pending, view.latest_observation)
        command = self.selector(view) if self.selector else (
            next((n for n in ('wam_control','vla_control','direct_control') if n in view.available_skills),view.available_skills[0]))
        remaining = s.limit-view.latest_observation.control_step
        if isinstance(command, str):
            command = CallSkill(command, s.load(view.latest_observation)['instruction'], remaining)
        if not isinstance(command, CallSkill):
            return command  # Measure/ReadExperience/UpdateTaskState/Stop.
        count = min(command.max_actions, remaining)
        if command.skill in ('wam_control', 'vla_control'):
            count = min(count, 24)
            if self._prefix_review(command.skill) and count < 4:
                return Stop('remaining_budget_less_than_pi05_review_prefix' if command.skill == 'vla_control'
                            else 'remaining_budget_less_than_mode3_prefix')
        return CallSkill(command.skill, command.subgoal, count)


class ContextClient:
    """The framework state/experience is visible to the existing EEF tool policy."""
    def __init__(self, client):
        self.client, self.agent = client, None

    def complete(self, messages, tools):
        context = [] if self.agent is None or self.agent.view is None else self.agent.view.messages
        text = json.dumps(context, ensure_ascii=False)
        return self.client.complete([messages[0],
            dict(role='user', content='Framework task context and execution evidence:\n'+text),
            *messages[1:]], tools)


def run_framework_episode(env, *, root, client=None, reviewer=None, direct_policy=None,
                          mode3=True, predict=True, max_steps=800, selector=None,
                          experience=None, measurement_tools=None, context_client=None,
                          row_checkers=None, operation_verifiers=None,
                          experience_catalog=None, candidate_revision=None, candidate_count=None,
                          action_spec=None, enable_geometry=False, supervisor_client=None,
                          event_gate=None, bowl_perception=None, pi05_client=None,
                          pi05_candidates=1, pi05_seed=0, pi05_a2v_client=None,
                          vla_reviewer=None, wam_review_mode=None, vla_review_mode=None,
                          direct_backend_factory=None, context_policy=None, context_summarizer=None,
                          execution_check=None, skill_factories=None, skill_reviewers=None,
                          skill_experience=None, max_review_attempts=3, max_model_calls=170,
                          task_tools=None, tool_limits=None, selection_policy=None,
                          required_prediction_skills=(), a2v_tool_client=None, prediction_loader=None, review_protocol="standard"):
    from agents.budget import ModelBudget, ModelBudgetExhausted
    if review_protocol not in ("standard", "experiment-v3"):
        raise ValueError("unknown review protocol")
    from tools.contracts import ToolLimits
    agent_review = selection_policy is not None and selection_policy.decision_mode == 'agent'
    if agent_review and (selector is None and supervisor_client is None):
        raise ValueError('agent selection requires selector or supervisor_client')
    if agent_review and mode3 and client is not None:
        raise ValueError('agent tool selection requires non-strict review mode')
    model_budget = ModelBudget(max_model_calls)
    if type(max_review_attempts) is not int or max_review_attempts < 1:
        raise ValueError('positive review attempt budget required')
    if direct_policy is not None and direct_backend_factory is not None:
        raise ValueError('supply direct_policy or direct_backend_factory, not both')
    for value in (wam_review_mode, vla_review_mode):
        if value not in (None, "action_only"):
            raise ValueError("review mode must be None (existing behavior) or action_only")
    if wam_review_mode == "action_only":
        if client is None or reviewer is None or event_gate is not None:
            raise ValueError("WAM action-only review requires client/reviewer and no prediction event gate")
        if candidate_count is not None and (type(candidate_count) is not int or candidate_count not in (1, 3)):
            raise ValueError("action-only review supports 1 or 3 candidates")
        mode3, predict = False, False
    if client is not None:
        effective_count = candidate_count if candidate_count is not None else (3 if mode3 else 1)
        if type(effective_count) is not int or not 1 <= effective_count <= 8 or (mode3 and effective_count != 3):
            raise ValueError('invalid WAM candidate count')
        if not mode3 and reviewer is None and not agent_review and (predict or effective_count != 1):
            raise ValueError('WAM candidate/prediction review requires an explicit reviewer')
    if vla_review_mode == "action_only":
        if pi05_client is None or vla_reviewer is None or pi05_a2v_client is not None:
            raise ValueError("VLA action-only review requires PI0.5/reviewer and no A2V client")
    if pi05_client is None and (pi05_a2v_client is not None or vla_reviewer is not None):
        raise ValueError('PI0.5 must be explicitly configured')
    if pi05_client is not None and (pi05_candidates != 1 or pi05_a2v_client is not None) and vla_reviewer is None and not agent_review:
        raise ValueError('PI0.5 candidate/A2V review requires an explicit reviewer')
    root = Path(root)
    session = NativeSession(env, root, max_steps=max_steps, enable_geometry=enable_geometry)
    session.execution_check = execution_check
    log = JsonlEventLog(root / 'framework-events.jsonl')
    executed_events = []
    def sink(event):
        log(event)
        if event.get('event') == 'execution':
            executed_events.append(dict(skill=event['skill'],steps=len(event['executed_actions']),
                stop_reason=event['stop_reason'],control_step=event['after']['control_step']))
    def summarize(_prompt, max_chars):
        # Structured measured evidence only; no generated semantic claims.
        recent = executed_events[-12:]
        summary = dict(executed_steps=sum(e['steps'] for e in executed_events),
                       recent_execution=recent,full_evidence=str(log.path))
        while len(json.dumps(summary)) > max_chars and recent:
            recent.pop(0)
        return json.dumps(summary)
    if experience_catalog is not None:
        if experience is not None:
            raise ValueError('supply a workspace or a catalog, not both')
        from experience.packages import ExperienceCatalog, materialize
        experience = materialize(ExperienceCatalog.load(experience_catalog),
            getattr(env,'task_name',''),root/session.episode_id,on_access=sink)
    wam = WAMCandidates(client, session, mode3=mode3, predict=predict, candidate_count=candidate_count, action_only=wam_review_mode == "action_only") if client is not None else None
    if wam is not None:
        wam.recovery_review = review_protocol == "experiment-v3"
        wam.tool_horizon = a2v_tool_client is not None or 'a2v' in (task_tools or {})
    contract = NATIVE_ACTION_CONTRACT
    vla = None
    if pi05_client is not None:
        from skills.pi05.backend import PI05Backend, CONTRACT
        if contract != CONTRACT:
            raise ValueError('PI0.5 and executor action contracts differ')
        vla = PI05Backend(pi05_client,session,candidate_count=pi05_candidates,seed=pi05_seed,
            a2v_client=pi05_a2v_client,sink=sink,action_only=vla_review_mode == "action_only")
    if vla is not None:
        vla.recovery_review = review_protocol == "experiment-v3"
    registry = SkillRegistry()
    direct = EEFBackend(direct_policy, session) if direct_policy is not None else None
    if direct_backend_factory is not None:
        # Give recipes read-only snapshot loading, never the environment/executor.
        direct = direct_backend_factory(session.load)
        verifier = getattr(direct, 'verify_execution', None)
        if verifier is None:
            raise ValueError('direct backend factory must provide operation verification')
        operation_verifiers = dict(operation_verifiers or {})
        if 'direct_control' in operation_verifiers:
            raise ValueError('direct backend already owns operation verification')
        operation_verifiers['direct_control'] = verifier
    if direct:
        registry.register(ControlSkill(SkillIdentity('direct_control','guarded-direct/framework-v1' if direct_backend_factory else 'roboprobe-eef/framework-v1',
                                                     contract,14),direct))
    if wam:
        registry.register(ModelControlSkill(SkillIdentity('wam_control','robodojo-candidates/framework-v1',
            contract,14,wam.identity['checkpoint']),wam))
    if vla:
        registry.register(ModelControlSkill(SkillIdentity('vla_control','pi05-http-npz/v1',
            contract,14,vla.identity['checkpoint']),vla))
    extensions = {}
    for factory in skill_factories or ():
        skill = factory(session.load)
        if not isinstance(skill, ControlSkill):
            raise TypeError('skill factory must return ControlSkill')
        if skill.identity.name in ('direct_control','wam_control','vla_control'):
            raise ValueError('built-in skills use their existing explicit adapters')
        if skill.identity.action_contract != contract or skill.identity.action_dim != 14:
            raise ValueError('extension skill differs from native executor action contract')
        if not agent_review and not callable((skill_reviewers or {}).get(skill.identity.name)):
            raise ValueError('extension skill requires an explicit review callback')
        registry.register(skill)
        extensions[skill.identity.name] = dict(backend_revision=skill.identity.backend_revision,
            checkpoint=skill.identity.checkpoint)
    if (not set(skill_reviewers or {}) <= set(extensions) or
            (not agent_review and set(skill_reviewers or {}) != set(extensions))):
        raise ValueError('review callbacks must exactly match extension skills')
    if not registry.names():
        raise ValueError('register a real backend')
    bridge_root = None
    if wam and mode3 and reviewer is None:
        from environments.robodojo.runtime.robodojo_mode3 import Mode3FileReviewer
        bridge_root = root.parent/(root.name+'-review')
        reviewer = Mode3FileReviewer(bridge_root)
    if event_gate is not None:
        if wam is None or not mode3:
            raise ValueError('event gate requires mode3 WAM')
        event_gate.bind(session, wam, reviewer, sink)
        reviewer = event_gate.review
    if supervisor_client is not None:
        supervisor_client = model_budget.client(supervisor_client, 'supervisor')
        if selector is not None:
            raise ValueError('supply one supervisor')
        import base64
        from agents.deepseek import DeepSeekAgent
        def load_public(ref):
            doc = session.load(ref)
            return dict(observation_id=ref.observation_id, state=doc['state'],
                images=[dict(camera=k,data_url='data:image/png;base64,'+
                    base64.b64encode(Path(v).read_bytes()).decode()) for k,v in doc['images'].items()])
        supervisor = DeepSeekAgent(supervisor_client, load_public,
            instruction=session.load(session.records[0])['instruction'],max_actions=max_steps,
            allow_experience=experience is not None,prediction_loader=prediction_loader)
        selector = supervisor.next
    if getattr(vla_reviewer, 'bind_runtime', None):
        vla_reviewer.bind_runtime(vla, model_budget)
    reviewer = model_budget.callback(reviewer, 'wam_review')
    vla_reviewer = model_budget.callback(vla_reviewer, 'vla_review')
    skill_reviewers = {name:model_budget.callback(fn, 'review:'+name)
                       for name,fn in (skill_reviewers or {}).items()}
    context_summarizer = model_budget.callback(context_summarizer, 'context_summary')
    agent = ExperimentAgent(session,direct=direct,wam=wam,reviewer=reviewer,selector=selector,experience=experience,vla=vla,vla_reviewer=vla_reviewer,skill_reviewers=skill_reviewers,max_review_attempts=max_review_attempts)
    agent.review_protocol = review_protocol
    agent.agent_review = agent_review
    if context_client is not None:
        context_client.agent = agent
    context = AgentContext(TaskState(session.load(session.records[0])['instruction']),
                           policy=context_policy or ContextPolicy(),
                           on_event=sink,summarizer=context_summarizer or summarize)
    from execution.trajectory import TrajectoryLogger
    versions = dict(checkpoint=wam.identity['checkpoint'] if wam else vla.identity['checkpoint'] if vla else 'none:direct_control',
        action_stats_version=wam.identity.get('action_stats_version','native_unscaled') if wam else 'native_unscaled',
        camera_transform='native_calibration_snapshot' if enable_geometry else 'native_rgb_no_calibration_claim',runtime='TaskSkillRunner/robodojo-v1',
        candidate_revision=candidate_revision, pi05_identity=vla.identity if vla else None,
        extension_skills=extensions)
    if enable_geometry:
        from tools.perception.tools import GeometryTools, CalibratedObservation
        from tools.perception.geometry import CameraFrame, GeometryUnavailable
        def calibrated(ref):
            doc = session.load(ref)
            if doc.get('calibration_errors') or not doc.get('calibration'):
                raise GeometryUnavailable('native_calibration_unavailable: '+json.dumps(doc.get('calibration_errors')))
            cameras = {name:CameraFrame(np.asarray(Image.open(doc['images'][name]).convert('RGB')),
                **record) for name,record in doc['calibration'].items()}
            return CalibratedObservation(ref,cameras,'robodojo_world',{})
        registered = GeometryTools(calibrated).registry()
        if set(registered) & set(measurement_tools or {}):
            raise ValueError('duplicate geometry tools')
        if bowl_perception is not None:
            from tools.measurement import MeasurementTool, MeasurementResult
            from execution.feedback import OperationReport
            def perceive(ref,args):
                if args:
                    raise ValueError('perceive_bowls accepts no arguments')
                data = bowl_perception.observe(dict(calibrated(ref).cameras),ref.control_step)
                return MeasurementResult(OperationReport('rgb_bowl_perception'),data)
            registered['perceive_bowls'] = MeasurementTool('perceive_bowls','rgb-rim-multiview/v1',
                'RGB bowl detection, instance tracking and calibrated rim fitting; uncertain geometry stays unknown.',{},perceive)
        measurement_tools = dict(registered, **(measurement_tools or {}))
    elif bowl_perception is not None:
        raise ValueError('bowl perception requires calibrated RGB snapshots')
    if action_spec is not None:
        if action_spec.control_hz != session.hz:
            raise ValueError('action spec frequency mismatch')
        checker = BoundsChecker(lower=action_spec.low,upper=action_spec.high,
            max_delta=action_spec.max_step or (),reference=lambda ref:flat(session.load(ref)['state']))
        checks = {name:checker for name in registry.names()}
        checks.update(row_checkers or {})
        row_checkers = checks
    executor = NativeExecutor(session,contract,sink,
        after_step=event_gate.after_step if event_gate else None, execution_check=execution_check)
    task_tools = dict(task_tools or {})
    if a2v_tool_client is not None:
        if 'a2v' in task_tools or a2v_tool_client is client or a2v_tool_client is pi05_a2v_client:
            raise ValueError('supply a dedicated A2V tool client, without duplicate tool registration')
        from environments.robodojo.prediction_tool import create_a2v_tool
        task_tools['a2v'] = create_a2v_tool(a2v_tool_client,session)
    runner = TaskSkillRunner(registry,executor,agent,context,sink=sink,experience=experience,
        task_tools=task_tools,tool_limits=tool_limits or ToolLimits(),selection_policy=selection_policy,
        required_prediction_skills=required_prediction_skills,
        measurement_tools=measurement_tools,operation_verifiers=operation_verifiers,
        skill_experience=skill_experience,
        row_checkers=row_checkers,config=RunnerConfig(max_agent_steps=max_steps*4+20))
    session.trajectory = TrajectoryLogger(root/'trajectories',session.episode_id,versions)
    session.record_measured(session.records[0])
    reason, result = 'error', None
    original_direct_client = getattr(direct_policy, '_client', None)
    if original_direct_client is not None:
        direct_policy._client = model_budget.client(original_direct_client, 'direct')
    try:
        result = runner.run(session.episode_id)
        reason = result['reason']
        return dict(executed_steps=result['counters']['executed_steps'],stop_reason=reason,
                    framework=result)
    except ModelBudgetExhausted:
        reason = 'model_budget_exhausted'
        return dict(executed_steps=len(session.records)-1, stop_reason=reason,
                    model_budget=model_budget.snapshot())
    finally:
        if original_direct_client is not None:
            direct_policy._client = original_direct_client
        try:
            if a2v_tool_client is not None:
                a2v_tool_client.call('finish_episode')
            if client is not None:
                client.call('finish_episode')
            if pi05_a2v_client is not None and pi05_a2v_client is not client:
                pi05_a2v_client.call('finish_episode')
        finally:
            summary = dict(framework='TaskSkillRunner/v1',mode='hybrid' if len(registry.names())>1 else
                ('mode3_a2v_visual_review' if wam and mode3 else 'wam' if wam else 'pi05_vla' if vla else 'direct_eef' if direct else registry.names()[0]),
                registered_skills=list(registry.names()),
                wam_review_mode=wam_review_mode, vla_review_mode=vla_review_mode,review_protocol=review_protocol,
                episode_id=session.episode_id,stop_reason=reason,
                model_budget=model_budget.snapshot(),tool_usage=runner.tool_session.snapshot(),
                selection_policy=selection_policy.identity() if selection_policy else None,
                executed_steps=len(session.records)-1,counters=runner.counters,
                native_success=bool(env.success[0] and env.end_flag[0])
                    if hasattr(env,'success') and hasattr(env,'end_flag') else None,
                native_success_flag=bool(env.success[0]) if hasattr(env,'success') else None,
                native_ended=bool(env.end_flag[0]) if hasattr(env,'end_flag') else None)
            root.mkdir(parents=True,exist_ok=True)
            (root/'episode-summary.json').write_text(json.dumps(summary,indent=2))
            session.trajectory.write('outcome',outcome='success' if summary['native_success'] else
                'failure' if summary['native_success'] is False else 'uncertain',
                native_result={k:v for k,v in summary.items() if k.startswith('native_')})
            session.trajectory.close()
            if bridge_root is not None and bridge_root.exists():
                # Archive the public request/response only after no further review is possible.
                import shutil
                shutil.copytree(bridge_root, root/'review-artifacts')
