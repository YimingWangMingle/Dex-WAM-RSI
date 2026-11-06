"""Measured observation storage and synchronous single-owner RoboDojo execution.

The XPL caller supplies an explicit observation binding and native step function.
Native scores are persisted separately, never attached to Agent observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Mapping
import numpy as np
import torch

from thinker_wm.models.inference_api import Observation, action_hash
from skills.contracts import MeasuredObservation
from environments.robodojo.training import RoboDojoProfile, sha256
from execution.runner import ExecutionResult, JsonlEventLog
from execution.trajectory import TrajectoryLogger, read_trajectory


class RoboDojoObservationStore:
    def __init__(self, root: str | Path, episode_id: str, adapter: Any, profile: RoboDojoProfile):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', episode_id):
            raise ValueError('episode_id must be a safe nonempty directory name')
        if (adapter.robot_name != profile.robot or adapter.fm.digest != profile.field_map_digest
                or tuple(adapter.model_hw) != profile.model_hw or adapter.resize_mode != profile.resize_mode):
            raise ValueError('adapter differs from trained camera/robot contract')
        if adapter.binding.depth_keys:
            raise ValueError('this RGB-only policy does not consume observed depth')
        self.root = Path(root) / episode_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.episode_id, self.adapter, self.profile = episode_id, adapter, profile
        self.camera_transform = adapter.camera_transform
        self._records: dict[int, tuple] = {}
        self._prediction_count = 0

    def add(self, live: Mapping, step: int) -> MeasuredObservation:
        if type(step) is not int or step != len(self._records):
            raise ValueError('record every executed step exactly once, starting from zero')
        obs, _ = self.adapter.observation(live, step)
        if (obs.state.shape != (self.profile.action_dim,) or obs.robot_name != self.profile.robot
                or obs.source != 'measured' or not torch.isfinite(obs.state).all()
                or tuple(obs.rgb.shape) != (3, *self.profile.model_hw) or not torch.isfinite(obs.rgb).all()):
            raise ValueError('invalid native measured observation')
        file = self.root / f'observation-{step:06d}.npz'
        with file.open('xb') as stream:
            np.savez_compressed(stream, rgb=obs.rgb.detach().cpu().numpy(), state=obs.state.detach().cpu().numpy())
        signature = sha256(file)
        ref = MeasuredObservation(self.episode_id, f'{step}:{signature}', step, obs.time_s, str(file.resolve()))
        (self.root / f'observation-{step:06d}.json').write_text(json.dumps(asdict(ref), sort_keys=True) + '\n')
        self._records[step] = (ref, signature)
        return ref

    def load(self, reference: MeasuredObservation) -> Observation:
        record = self._records.get(reference.control_step)
        if record is None or record[0] != reference or reference.episode_id != self.episode_id:
            raise ValueError('foreign or unknown observation reference')
        file = Path(reference.payload_ref)
        if sha256(file) != record[1]:
            raise ValueError('measured observation changed after recording')
        with np.load(file, allow_pickle=False) as data:
            return Observation(torch.from_numpy(data['rgb'].copy()), torch.from_numpy(data['state'].copy()),
                               reference.control_step, reference.time_s, self.profile.robot)

    def since(self, episode_id: str, after: int | None, through: int) -> Iterator[Observation]:
        if episode_id != self.episode_id:
            raise ValueError('foreign history episode')
        start = 0 if after is None else after + 1
        for step in range(start, through + 1):
            if step not in self._records:
                raise ValueError('missing intermediate executed observation')
            yield self.load(self._records[step][0])

    def save_prediction(self, reference: MeasuredObservation, forecast: Any) -> str:
        self.load(reference)
        if forecast.decision_step != reference.control_step:
            raise ValueError('forecast refers to a different decision')
        path = self.root / f'prediction-{self._prediction_count:06d}.json'
        self._prediction_count += 1
        arrays = dict(actions=forecast.actions)
        for name in ('video_latent', 'depth_latent', 'frames', 'depth'):
            value = getattr(forecast, name)
            if value is not None:
                arrays[name] = value.detach().cpu().numpy()
        with path.with_suffix('.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with path.open('x') as stream:
            json.dump(dict(observation=asdict(reference), forecast=forecast.meta(),
                           arrays=path.with_suffix('.npz').name, arrays_sha256=sha256(path.with_suffix('.npz'))), stream)
        return str(path.resolve())


@dataclass(frozen=True)
class NativeStep:
    live: Mapping[str, Any]
    terminated: bool
    native_result: Mapping[str, Any]  # consumed only by recorder/evaluator


class RoboDojoExecutor:
    """step_fn must dispatch exactly the supplied native joint row once.

    No queue runs beyond execute(); no automatic retry follows an exception.
    A dispatch with missing feedback is preserved as ambiguous and faults this
    executor. Its caller must end the trial rather than replay the command.
    """
    def __init__(self, observations: RoboDojoObservationStore, initial_live: Mapping,
                 step_fn: Callable[[np.ndarray], NativeStep], *, max_control_steps: int,
                 trajectory: TrajectoryLogger | None = None):
        if type(max_control_steps) is not int or max_control_steps < 1:
            raise ValueError('positive explicit episode action budget required')
        if trajectory is not None:
            header = read_trajectory(trajectory.dir)[0]
            versions = header['versions']
            if (header['episode_id'] != observations.episode_id
                    or versions['action_stats_version'] != observations.profile.action_stats_version
                    or versions['camera_transform'] != observations.camera_transform):
                raise ValueError('trajectory identity differs from observation contract')
        self.trajectory = trajectory
        self.store, self.step_fn, self.limit = observations, step_fn, max_control_steps
        self.latest = observations.add(initial_live, 0)
        self.terminated, self.faulted, self._busy = False, False, False
        self._cancel = False
        self.native_result: dict = {}
        self.log = JsonlEventLog(observations.root / 'executor.jsonl')
        self._record_measured(None)

    def _record_measured(self, executed_row: int | None) -> None:
        if self.trajectory is None:
            return
        obs = self.store.load(self.latest)
        self.trajectory.write('measured', step=obs.step, time_s=obs.time_s,
                              state=obs.state.numpy(), executed_row=executed_row,
                              attachments=dict(rgb=obs.rgb.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()))

    def current_observation(self) -> MeasuredObservation:
        return self.latest

    def in_flight(self) -> bool:
        return self._busy

    def cancel(self) -> None:
        self._cancel = True

    def execute(self, rows: tuple, *, action_contract: str) -> ExecutionResult:
        if self.faulted or self.terminated or self._busy:
            raise RuntimeError('executor faulted, terminated or already executing')
        if action_contract != self.store.profile.action_contract:
            raise ValueError('native action contract mismatch')
        actions = np.asarray(rows, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all() or not len(actions):
            raise ValueError('finite native joint14 commands required')
        # Reject the whole requested prefix instead of truncating a direct stage.
        if self.latest.control_step + len(actions) > self.limit:
            raise ValueError('approved prefix exceeds the episode action budget')
        if np.any((actions[:, [6, 13]] < 0) | (actions[:, [6, 13]] > 1)):
            raise ValueError('native grippers must be in [0,1]; refusing silent clipping')
        if self.trajectory is not None:
            self.trajectory.write('command', decision_step=self.latest.control_step,
                                  hash=action_hash(actions), source_hash=action_hash(actions),
                                  postprocess='native_joint14_identity', forecast_valid=False,
                                  invalid_reason='executor does not bind a forecast',
                                  attachments=dict(actions=actions))
        self._busy, self._cancel = True, False
        executed = []
        reason = 'completed'
        try:
            for row in actions:
                if self._cancel:
                    reason = 'interrupted:cancelled'
                    break
                before = self.latest.control_step
                self.log(dict(event='dispatch', control_step=before, command=row.tolist()))
                native = self.step_fn(row.copy())
                if not isinstance(native, NativeStep) or type(native.terminated) is not bool:
                    raise TypeError('native step must return NativeStep with a boolean termination')
                # Persist native verdict even if allowed observation adaptation fails.
                self.native_result = dict(native.native_result)
                self.log(dict(event='native_feedback', control_step=before + 1,
                              terminated=native.terminated, native_result=self.native_result))
                self.latest = self.store.add(native.live, before + 1)
                executed.append(tuple(float(v) for v in row))
                self._record_measured(before)
                self.terminated = native.terminated or self.latest.control_step == self.limit
                self.log(dict(event='measured', observation=asdict(self.latest), command=row.tolist()))
                if self.terminated:
                    reason = 'terminated' if native.terminated else 'budget_exhausted'
                    break
        except Exception as error:
            self.faulted = True
            self.log(dict(event='execution_fault', measured_step=self.latest.control_step,
                          error=type(error).__name__ + ': ' + str(error), replay_allowed=False))
            raise
        finally:
            self._busy = False
        if self.trajectory is not None and self.terminated:
            success = self.native_result.get('success')
            verdict = 'success' if success is True else 'failure' if success is False else 'uncertain'
            self.trajectory.write('outcome', outcome=verdict, reason=reason,
                                  native_result=self.native_result)
        return ExecutionResult(tuple(executed), self.latest, self.terminated, reason)
