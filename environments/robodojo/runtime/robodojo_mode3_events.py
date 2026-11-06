"""Opt-in simulation-only event review. Geometry stays outside visual reviewer input."""
import numpy as np
from environments.robodojo.runtime.robodojo_a2v import Mode3ReviewMixin
from environments.robodojo.runtime.robodojo_mode3 import _save

# Conservative experiment thresholds, not calibrated physical safety guarantees.
DISTANCE_M = 0.30
HEIGHT_M = 0.15
ARM_SEPARATION_M = 0.35
TRACKING_RAD = 0.15
JOINT_STEP_RAD = 0.10
GRIPPER_DELTA = 0.02
MAX_AUTO_BLOCKS = 2


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def geometry_status(ee, bowls):
    ee, bowls = np.asarray(ee, dtype=float), np.asarray(bowls, dtype=float)
    if ee.shape != (2, 3) or bowls.shape != (3, 3) or not np.isfinite(ee).all() or not np.isfinite(bowls).all():
        return dict(clear=False, reason='geometry_missing_or_invalid')
    distance = float(np.linalg.norm(ee[:, None, :] - bowls[None, :, :], axis=-1).min())
    height = float(ee[:, 2].min() - bowls[:, 2].max())
    separation = float(np.linalg.norm(ee[0] - ee[1]))
    clear = distance > DISTANCE_M and height > HEIGHT_M and separation > ARM_SEPARATION_M
    return dict(clear=clear, reason='clear' if clear else 'near_bowl_or_low_clearance',
                min_bowl_distance_m=distance, height_above_bowls_m=height, arm_separation_m=separation)


def simulation_geometry(env):
    try:
        manager = env.robot_manager
        ee = [manager.get_real_endpose(r, env_idx_list=[0], is_relative=True)[0][:3]
              for r in manager.robot_list if r.type == 'target']
        bowls = [env.reward_manager.func_parser.get_label_pose(f'bowl{i}')[0][0] for i in range(3)]
        return geometry_status(ee, bowls)
    except Exception as error:
        return dict(clear=False, reason='geometry_unavailable', error=type(error).__name__)


def plan_status(plan, state):
    plan, state = array(plan), array(state)
    if plan.shape != (24, 14) or state.shape != (14,) or not np.isfinite(plan).all() or not np.isfinite(state).all():
        return dict(clear=False, reason='invalid_plan_state')
    grip = float(np.max(np.abs(plan[:, [6, 13]] - state[[6, 13]])))
    joint = np.r_[0:6, 7:13]
    delta = float(np.max(np.abs(np.diff(np.vstack([state, plan])[:, joint], axis=0))))
    # Only empty/open-hand transit may bypass review, never carrying a bowl.
    open_hands = bool(np.min(state[[6, 13]]) >= 0.95 and np.min(plan[:, [6, 13]]) >= 0.95)
    return dict(clear=grip <= GRIPPER_DELTA and delta <= JOINT_STEP_RAD and open_hands,
                reason='stable_open_grippers' if grip <= GRIPPER_DELTA and open_hands else 'gripper_change_or_closed',
                gripper_delta=grip, max_joint_step_rad=delta, open_hands=open_hands)


class EventMode3Mixin(Mode3ReviewMixin):
    def reset(self):
        self._event_gripper_delta = None
        return super().reset()

    def update_obs(self, obs):
        previous = None if self.observation is None else array(self.observation.state).copy()
        result = super().update_obs(obs)
        self._event_gripper_delta = (None if previous is None else
            float(np.max(np.abs(array(self.observation.state)[[6, 13]] - previous[[6, 13]]))))
        return result

    def event_status(self):
        if self.observation is None:
            return dict(clear=False, reason='no_observation')
        error = None
        if self.last_command is not None:
            start, commands = self.last_command
            row = self.observation.step - start - 1
            if 0 <= row < len(commands):
                error = float(np.max(np.abs(array(self.observation.state)[self.joint_indices] - commands[row, self.joint_indices])))
        delta = self._event_gripper_delta
        clear = error is not None and np.isfinite(error) and error <= TRACKING_RAD and delta is not None and np.isfinite(delta) and delta <= GRIPPER_DELTA
        return dict(clear=bool(clear), tracking_error_rad=error, measured_gripper_delta=delta,
                    step=self.observation.step)

    def candidate_auto_check(self, request):
        self._at_boundary()
        if (not isinstance(request, dict) or set(request) != {'review_id'} or self._review is None
                or request['review_id'] != self._review[0]['review_id']):
            raise ValueError('auto check requires current review id')
        # Deterministic baseline candidate; this gate does not rank hidden action values.
        return dict(candidate_id=0, plan=plan_status(self._review[1][0][0], self.observation.state),
                    measured=self.event_status())

    def interrupt_execution(self, request):
        if (not isinstance(request, dict) or set(request) != {'step'} or type(request['step']) is not int
                or self.observation is None or request['step'] != self.observation.step
                or self._review is not None or self.last_command is None):
            raise ValueError('interrupt requires latest measured execution step')
        start, commands = self.last_command
        step = request['step']
        if not start <= step < self.next_decision:
            raise ValueError('no unfinished command prefix')
        self._record(dict(event='execution_interrupted', step=step, issued_until=self.next_decision,
                          discarded_steps=self.next_decision-step))
        self.last_command = (start, commands[:step-start].copy())
        self.next_decision = step
        return dict(interrupted=True, step=step)


class FrameworkEventGate:
    """Event experiments participate in the same approval and execution path."""
    def bind(self, session, wam, reviewer, sink):
        self.session, self.wam, self.reviewer, self.sink = session, wam, reviewer, sink
        self.force_review, self.automatic, self.auto_streak = True, False, 0
        self.last_status = None
        self.audit, self.interrupts = [], []

    def review(self, packet):
        geometry = simulation_geometry(self.session.env)
        check = self.wam.client.call('candidate_auto_check', {'review_id':self.wam.preview['review_id']})
        measured = self.last_status or check['measured']
        auto = (not self.force_review and self.auto_streak < MAX_AUTO_BLOCKS
                and geometry['clear'] and check['plan']['clear'] and measured['clear'])
        record = dict(event='simulation_event_gate', step=self.session.records[-1].control_step,
            geometry_source='simulator_privileged_state', geometry=geometry, check=check,
            latest_execution_status=measured, automatic=bool(auto))
        self.sink(record);self.audit.append(record)
        _save(self.session.root/'event-gates.json',self.audit)
        self.automatic, self.force_review = bool(auto), False
        if auto:
            self.auto_streak += 1
            return dict(candidate_id=0, execute_steps=packet['allowed_execute_steps'][0])
        self.auto_streak = 0
        return self.reviewer(packet)

    def after_step(self, observation):
        # Publish each real row before checking tracking; feedback will skip these
        # already-synchronized rows and resolve the exact interrupted prefix.
        if self.wam.preview is None or not self.wam.selected:
            self.force_review, self.last_status = True, None
            return False
        self.wam.sync_execution(observation)
        status = self.wam.client.call('event_status')
        self.last_status = status
        error = status.get('tracking_error_rad')
        tracking_bad = error is None or not np.isfinite(error) or error > TRACKING_RAD
        geometry = simulation_geometry(self.session.env) if self.automatic or tracking_bad else None
        interrupt = tracking_bad or (self.automatic and (not geometry['clear'] or not status['clear']))
        if interrupt:
            self.force_review = True
            record = dict(event='simulation_event_interrupt',step=observation.control_step,
                          geometry=geometry,status=status)
            self.sink(record);self.interrupts.append(record)
            _save(self.session.root/'event-interrupts.json',self.interrupts)
        return bool(interrupt)


def eval_mode3_episode(task_env, client, *, root, reviewer=None, max_steps=800):
    from environments.robodojo.framework import run_framework_episode
    if type(max_steps) is not int or max_steps <= 0 or max_steps % 4:
        raise ValueError('mode 3 action budget must be a positive multiple of 4')
    return run_framework_episode(task_env,root=root,client=client,reviewer=reviewer,
        max_steps=max_steps,mode3=True,event_gate=FrameworkEventGate())
