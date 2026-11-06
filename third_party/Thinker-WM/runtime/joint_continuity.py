"""Measured-state anchored continuity for native ARX-X5 joint14 proposals.

These are simulation tuning limits, not certified hardware limits. The retained
plan is advisory and must never be inserted into the measured history buffer.
"""
from dataclasses import dataclass
import numpy as np

JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = [6, 13]


@dataclass(frozen=True)
class PlanningHorizon:
    num_blocks: int = 1
    execute_steps: int = 8
    actions_per_block: int = 8
    max_blocks: int = 3

    def __post_init__(self):
        if any(type(x) is not int for x in (self.num_blocks, self.execute_steps,
                                            self.actions_per_block, self.max_blocks)):
            raise ValueError('planning horizons must be integers')
        if not 1 <= self.num_blocks <= self.max_blocks or self.actions_per_block < 1:
            raise ValueError('prediction horizon exceeds the trained block budget')
        if not 1 <= self.execute_steps <= self.predict_steps:
            raise ValueError('execution horizon must fit prediction horizon')

    @property
    def predict_steps(self):
        return self.num_blocks * self.actions_per_block


class JointContinuity:
    def __init__(self, *, control_hz, max_velocity_rad_s, max_acceleration_rad_s2,
                 ema_alpha=.5, overlap_steps=4, bridge_steps=4, stale_error_rad=.05):
        self.hz = float(control_hz)
        self.vmax = np.asarray(max_velocity_rad_s, dtype=np.float64)
        self.amax = np.asarray(max_acceleration_rad_s2, dtype=np.float64)
        if (self.vmax.shape != (12,) or self.amax.shape != (12,) or
                not np.isfinite(self.vmax).all() or not np.isfinite(self.amax).all() or
                (self.vmax <= 0).any() or (self.amax <= 0).any() or
                not np.isfinite(self.hz) or self.hz <= 0):
            raise ValueError('finite positive per-joint limits and frequency required')
        if not 0 < ema_alpha <= 1 or not np.isfinite(stale_error_rad) or stale_error_rad <= 0:
            raise ValueError('invalid continuity tuning')
        if any(type(x) is not int or x < 0 for x in (overlap_steps, bridge_steps)):
            raise ValueError('overlap/bridge lengths must be nonnegative integers')
        self.alpha, self.overlap, self.bridge = ema_alpha, overlap_steps, bridge_steps
        self.stale_error = stale_error_rad
        self.reset()

    def reset(self):
        self.previous_plan = None
        self.previous_step = None
        self.issued_steps = 0

    def process(self, raw, state, step, execute_steps, measured_velocity=None):
        raw, state = np.asarray(raw, np.float64), np.asarray(state, np.float64)
        if (raw.ndim != 2 or raw.shape[1] != 14 or state.shape != (14,) or
                not np.isfinite(raw).all() or not np.isfinite(state).all() or
                type(step) is not int or step < 0 or type(execute_steps) is not int or
                not 1 <= execute_steps <= len(raw)):
            raise ValueError('invalid native joint proposal or measured anchor')
        if self.previous_step is not None and step <= self.previous_step:
            raise ValueError('replanning must advance measured time')
        velocity = np.zeros(12) if measured_velocity is None else np.asarray(measured_velocity, np.float64)
        if velocity.shape != (12,) or not np.isfinite(velocity).all():
            raise ValueError('invalid measured velocity')
        smooth = raw.copy()
        for i in range(1, len(raw)):
            smooth[i, JOINTS] = self.alpha * raw[i, JOINTS] + (1-self.alpha) * smooth[i-1, JOINTS]
        fused = smooth.copy()
        overlap_rows, consistency, tracking, stale = 0, None, None, False
        if self.previous_plan is not None:
            shift = step - self.previous_step
            if shift <= self.issued_steps:
                tracking = float(np.max(np.abs(state[JOINTS] - self.previous_plan[shift-1, JOINTS])))
                stale = tracking > self.stale_error
                overlap_rows = min(self.overlap, len(raw), len(self.previous_plan)-shift)
                if overlap_rows > 0:
                    old = self.previous_plan[shift:shift+overlap_rows, JOINTS]
                    consistency = float(np.sqrt(np.mean((raw[:overlap_rows, JOINTS]-old)**2)))
                    if not stale:
                        weights = .5 * (1-np.arange(overlap_rows)/overlap_rows)
                        fused[:overlap_rows, JOINTS] = (weights[:, None] * old +
                            (1-weights[:, None]) * smooth[:overlap_rows, JOINTS])
            else:
                stale = True
        count = min(self.bridge, len(raw))
        if count:
            weights = ((np.arange(count)+1)/count)[:, None]
            fused[:count, JOINTS] = state[JOINTS] + weights * (fused[:count, JOINTS]-state[JOINTS])
        plan = fused.copy()
        position = state[JOINTS].copy()
        projected_velocity = bool((np.abs(velocity) > self.vmax).any())
        velocity = np.clip(velocity, -self.vmax, self.vmax)
        clipped = 0
        for i in range(len(plan)):
            desired = (fused[i, JOINTS]-position) * self.hz
            low = np.maximum(-self.vmax, velocity-self.amax/self.hz)
            high = np.minimum(self.vmax, velocity+self.amax/self.hz)
            next_velocity = np.clip(desired, low, high)
            clipped += int(np.count_nonzero(np.abs(next_velocity-desired) > 1e-7))
            position = position + next_velocity/self.hz
            plan[i, JOINTS], velocity = position, next_velocity
        # Gripper events keep model timing; never average them with arm poses.
        plan[:, GRIPPERS] = raw[:, GRIPPERS].clip(0., 1.)
        self.previous_plan, self.previous_step = plan.copy(), step
        self.issued_steps = execute_steps
        diagnostics = dict(post_ema=smooth.tolist(), fused=fused.tolist(),
            retained_tail=plan[execute_steps:].tolist(), overlap_rows=overlap_rows,
            overlap_used=bool(overlap_rows and not stale), stale_tail=stale,
            same_future_joint_rmse_rad=consistency, measured_command_max_error_rad=tracking,
            joint_limit_clipped_elements=clipped, initial_velocity_projected=projected_velocity,
            gripper_clipped_elements=int(np.count_nonzero(raw[:, GRIPPERS] != plan[:, GRIPPERS])),
            raw_boundary_max_rad=float(np.max(np.abs(raw[0, JOINTS]-state[JOINTS]))),
            sent_boundary_max_rad=float(np.max(np.abs(plan[0, JOINTS]-state[JOINTS]))),
            forecast_matches_commands=bool(np.array_equal(raw[:execute_steps], plan[:execute_steps])))
        return plan.astype(np.float32), diagnostics


class JointEMA:
    """Ablation: within-chunk arm EMA only, with no cross-chunk state.

The first row stays unchanged. Grippers retain their timing and only receive
native [0, 1] clipping. No bridge, overlap blending, or rate limits are applied.
"""
    def __init__(self, *, ema_alpha: float = .5):
        if not np.isfinite(ema_alpha) or not 0 < ema_alpha <= 1:
            raise ValueError('EMA alpha must be finite in (0, 1]')
        self.alpha = float(ema_alpha)

    def reset(self) -> None:
        pass

    def process(self, raw, state, step, execute_steps, measured_velocity=None):
        raw, state = np.asarray(raw, np.float64), np.asarray(state, np.float64)
        if (raw.ndim != 2 or raw.shape[1] != 14 or state.shape != (14,) or
                not np.isfinite(raw).all() or not np.isfinite(state).all() or
                type(step) is not int or step < 0 or type(execute_steps) is not int or
                not 1 <= execute_steps <= len(raw)):
            raise ValueError('invalid native joint proposal or measured anchor')
        plan = raw.copy()
        for i in range(1, len(plan)):
            plan[i, JOINTS] = self.alpha * raw[i, JOINTS] + (1-self.alpha) * plan[i-1, JOINTS]
        smoothed = plan.copy()
        plan[:, GRIPPERS] = raw[:, GRIPPERS].clip(0., 1.)
        result = plan.astype(np.float32)
        return result, dict(mode='ema', ema_alpha=self.alpha, post_ema=smoothed.tolist(),
            fused=smoothed.tolist(), retained_tail=plan[execute_steps:].tolist(),
            overlap_rows=0, overlap_used=False, stale_tail=False,
            same_future_joint_rmse_rad=None, measured_command_max_error_rad=None,
            joint_limit_clipped_elements=0, initial_velocity_projected=False,
            gripper_clipped_elements=int(np.count_nonzero(raw[:, GRIPPERS] != plan[:, GRIPPERS])),
            raw_boundary_max_rad=float(np.max(np.abs(raw[0, JOINTS]-state[JOINTS]))),
            sent_boundary_max_rad=float(np.max(np.abs(plan[0, JOINTS]-state[JOINTS]))),
            forecast_matches_commands=bool(np.array_equal(raw[:execute_steps].astype(np.float32), result[:execute_steps])))


def build_joint_postprocessor(config, *, control_hz: float):
    """Explicit ablation selection; omission preserves legacy raw joint output."""
    if config is None:
        return None
    options = dict(config)
    mode = options.pop('mode', 'full')
    if mode == 'ema':
        return JointEMA(**options)
    if mode == 'full':
        return JointContinuity(control_hz=control_hz, **options)
    raise ValueError(f'unknown joint postprocessing mode: {mode!r}')
