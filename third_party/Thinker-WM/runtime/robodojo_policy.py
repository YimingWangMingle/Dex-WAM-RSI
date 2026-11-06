"""Strict RoboDojo WAM policy; native joint commands and measured history.

The XPolicyLab transport owns RGB decoding. This policy consumes decoded RGB,
never privileged object state, and separates the generated prediction horizon from the executed action prefix.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np

LOG = logging.getLogger(__name__)


def load_runtime(profile, runtime_config=None):
    """Select the audited default without changing existing AR deployments."""
    from thinker_wm.runtime.joint_continuity import PlanningHorizon
    runtime_path = Path(runtime_config) if runtime_config else (
        Path(__file__).resolve().parents[1] / 'configs/runtime' / (
            'robodojo_joint24_v1.json' if profile.actions_per_block == 24
            else 'robodojo_default.json'))
    runtime = json.loads(runtime_path.read_text())
    if set(runtime) - {'num_blocks', 'execute_steps', 'continuity', 'prefix_cache_mode',
                       'prefix_cache_max_mb', 'async_lead_steps'}:
        raise ValueError('unknown runtime configuration keys')
    horizon = PlanningHorizon(num_blocks=runtime.get('num_blocks', 1),
        execute_steps=runtime.get('execute_steps', profile.actions_per_block),
        actions_per_block=profile.actions_per_block,
        max_blocks=profile.training_actions // profile.actions_per_block)
    if runtime.get('prefix_cache_mode', 'off') not in ('off', 'fixed'):
        raise ValueError('prefix_cache_mode must be off or fixed')
    budget = runtime.get('prefix_cache_max_mb', 1024.0)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError('prefix_cache_max_mb must be finite and nonnegative')
    lead = runtime.get('async_lead_steps', 0)
    if type(lead) is not int or not 0 <= lead < horizon.execute_steps or lead > horizon.predict_steps - horizon.execute_steps:
        raise ValueError('async lead must fit both execution interval and unused prediction horizon')
    return runtime, horizon


class RoboDojoPolicy:
    def __init__(self, usr_args):
        import torch
        from omegaconf import OmegaConf
        from thinker_wm.models.inference_api import GenerationSettings, WAMInference, make_message_builder
        from thinker_wm.runtime.history_buffer import HistoryBuffer
        from thinker_wm.runtime.robodojo_adapter import ObservationBinding, RoboDojoObservationAdapter
        from environments.robodojo.training import inspect_checkpoint, load_training
        from thinker_wm.scripts.eval_video_ckpt import create_eval_model, _load_model_weights
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        a = dict(usr_args)
        cfg, fm, profile, _ = load_training(a['config_path'], a['field_map_path'])
        cfg = OmegaConf.create(cfg)
        from thinker_wm.runtime.joint_continuity import build_joint_postprocessor, JOINTS
        self.joint_indices = JOINTS
        runtime, self.horizon = load_runtime(profile, a.get('runtime_config'))
        from thinker_wm.runtime.async_replan import AsyncReplan
        self.async_lead_steps = runtime.get('async_lead_steps', 0)
        self.async_replan = AsyncReplan() if self.async_lead_steps else None
        continuity = runtime.get('continuity')
        self.continuity = build_joint_postprocessor(continuity, control_hz=profile.control_hz)
        self.measured_velocity = None
        self.last_command = None
        checkpoint = inspect_checkpoint(a['checkpoint'])
        self.dims = get_robot_action_dim_info(a.get('env_cfg_type', 'arx_x5'))
        if self.dims['arm_dim'] != [6, 6] or self.dims['ee_dim'] != [1, 1]:
            raise ValueError('simulator dimensions differ from trained ARX-X5 joint14')
        self.log_dir = Path(a['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        kernel_options = None
        if torch.cuda.get_device_capability(0) == (8, 6):
            # Ampere workstation GPUs have less shared memory than the training
            # accelerators. Keep the same compiled FlexAttention and mask;
            # select tile sizes that fit the A6000's per-block hardware limit.
            from functools import partial
            from thinker_wm.models import temporal_attention
            if temporal_attention.ATTN_IMPL == 'flex':
                kernel_options = {'BLOCK_M': 32, 'BLOCK_N': 32}
                temporal_attention._flex = partial(temporal_attention._flex_attention(),
                                                   kernel_options=kernel_options)
        LOG.info('Building RoboDojo model; strict loading %s', checkpoint['weights'])
        model = create_eval_model(cfg, load_pretrained_backbones=False)
        load_info = _load_model_weights(model, Path(checkpoint['weights']), strict=True, mmap=True)
        model.to(torch.device('cuda:0')).eval()
        self.adapter = RoboDojoObservationAdapter(fm, ObservationBinding(
            rgb_keys={'head': 'vision.cam_head.color',
                      'left_wrist': 'vision.cam_left_wrist.color',
                      'right_wrist': 'vision.cam_right_wrist.color'},
            state_keys=('state.left_arm_joint_state', 'state.left_ee_joint_state',
                        'state.right_arm_joint_state', 'state.right_ee_joint_state')),
            model_hw=profile.model_hw, resize_mode=profile.resize_mode, robot_name=profile.robot)
        inf = cfg.model.inference
        settings = GenerationSettings(
            num_inference_steps=int(inf.num_inference_timesteps),
            action_num_inference_steps=int(inf.action_num_inference_timesteps),
            guidance_scale=float(inf.guidance_scale),
            action_guidance_scale=float(inf.action_guidance_scale),
            seed=int(a.get('seed', 0)), decode_outputs=False, return_depth=True,
            num_blocks=self.horizon.num_blocks,
            prefix_cache_mode=runtime.get('prefix_cache_mode', 'off'),
            prefix_cache_max_mb=runtime.get('prefix_cache_max_mb', 1024.0))
        self.inf = WAMInference(model, checkpoint=checkpoint['path'],
            action_stats_version=profile.action_stats_version,
            camera_transform=self.adapter.camera_transform, settings=settings,
            message_builder=make_message_builder(cfg, 'robodojo'))
        self.history = HistoryBuffer.from_config(cfg, profile.robot)
        self.block_size = self.horizon.execute_steps
        self.profile = profile
        self.control_hz = profile.control_hz
        self.episode_file = None
        self.observation = None
        self.next_decision = 0
        self.instruction = None
        receipt = dict(checkpoint=checkpoint, strict_load=load_info,
                       memory_mapped_checkpoint=True, flex_kernel_options=kernel_options,
                       profile=asdict(profile), settings=asdict(settings),
                       camera_transform=self.adapter.camera_transform,
                       device=torch.cuda.get_device_name(0),
                       allocated_bytes=torch.cuda.memory_allocated(0),
                       mode='WAM closed loop; no Agent or recovery', runtime=runtime,
                       horizon=asdict(self.horizon),
                       gripper_postprocess='clip to [0,1], same as native simulator')
        (self.log_dir / 'strict-load.json').write_text(json.dumps(receipt, indent=2))
        LOG.info('Strict load passed: %s', receipt)

    def _record(self, data):
        if self.episode_file is not None:
            self.episode_file.write(json.dumps(data, allow_nan=False) + '\n')
            self.episode_file.flush()

    def reset(self):
        self.finish_episode()
        self.episode_id = uuid4().hex
        self.episode_file = (self.log_dir / f'{self.episode_id}.jsonl').open('w')
        self.history.reset(self.episode_id)
        if self.continuity is not None:
            self.continuity.reset()
        self.measured_velocity = None
        self.last_command = None
        self.observation = None
        self.instruction = None
        self.next_decision = 0
        self._record(dict(event='reset', episode_id=self.episode_id))

    def update_obs(self, obs):
        if self.episode_file is None:
            raise RuntimeError('reset must precede observations')
        step = obs.get('rsi_control_step')
        expected = 0 if self.observation is None else self.observation.step + 1
        if type(step) is not int or step != expected:
            raise ValueError(f'expected measured control step {expected}, got {step!r}')
        if step > self.next_decision:
            raise ValueError('received a step not covered by the issued action block')
        frequency = obs.get('additional_info', {}).get('frequency')
        if frequency is None or float(frequency) != self.control_hz:
            raise ValueError(f'expected {self.control_hz} Hz observations, got {frequency}')
        instruction = obs.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('native task instruction must be a nonempty string')
        if self.instruction is not None and instruction != self.instruction:
            raise ValueError('task instruction changed within episode')
        measured, _ = self.adapter.observation(obs, step)
        if self.observation is not None:
            self.measured_velocity = ((measured.state[self.joint_indices] -
                self.observation.state[self.joint_indices]) * self.control_hz)
        tracking_error = None
        if self.last_command is not None:
            command_step, commands = self.last_command
            row = step - command_step - 1
            if 0 <= row < len(commands):
                tracking_error = (np.asarray(measured.state)[self.joint_indices] -
                    commands[row, self.joint_indices]).tolist()
        self.history.add(measured)
        self.instruction = instruction
        self.observation = measured
        self._record(dict(event='observation', step=step,
                          state=measured.state.tolist(), joint_tracking_error_rad=tracking_error, instruction=instruction if step == 0 else None,
                          episode_meta=obs.get('rsi_episode_meta') if step == 0 else None))

        if (self.async_replan is not None and step > 0 and
                step == self.next_decision - self.async_lead_steps):
            from thinker_wm.runtime.async_replan import snapshot_history
            snapshot = snapshot_history(self.history)
            self.async_replan.start(step, self.inf.generate_action,
                                    measured, snapshot, instruction)

    def get_action(self):
        from XPolicyLab.utils.process_data import unpack_robot_state
        if self.observation is None or self.observation.step != self.next_decision:
            raise RuntimeError('every executed row must be observed before replanning')
        start = time.monotonic()
        source_step = self.observation.step
        if getattr(self, 'async_replan', None) is not None and self.async_replan.pending:
            source_step, output = self.async_replan.take()
        else:
            output = self.inf.generate_action(self.observation, self.history, self.instruction)
        offset = self.observation.step - source_step
        raw = np.asarray(output.actions, dtype=np.float32)
        if raw.shape != (self.horizon.predict_steps, 14) or not np.isfinite(raw).all():
            raise ValueError(f'invalid model action block {raw.shape}')
        full_raw = raw
        if not 0 <= offset <= self.horizon.predict_steps - self.block_size:
            raise RuntimeError('asynchronous prediction cannot cover the next execution interval')
        raw = raw[offset:]
        if self.continuity is None:
            plan, diagnostics = raw.copy(), {}
            plan[:, [6, 13]] = plan[:, [6, 13]].clip(0., 1.)
        else:
            plan, diagnostics = self.continuity.process(raw, self.observation.state,
                self.observation.step, self.block_size, self.measured_velocity)
        commanded = plan[:self.block_size].copy()
        self.last_command = (self.observation.step, commanded)
        # The raw latent forecast has not been conditioned on filtered commands.
        diagnostics['forecast_matches_commands'] = bool(np.array_equal(
            raw[:self.block_size], commanded))
        latency = time.monotonic() - start
        self._record(dict(event='action_block', step=self.observation.step,
                          latency_s=latency, raw=full_raw.tolist(), commanded=commanded.tolist(),
                          source_step=source_step, skipped_elapsed_actions=offset,
                          cache_stats=getattr(getattr(output, 'forecast', None), 'cache_stats', None),
                          history=self.history.describe(), predicted_steps=len(raw),
                          executed_steps=len(commanded), continuity=diagnostics))
        LOG.info('episode=%s step=%d block latency=%.3fs', self.episode_id,
                 self.observation.step, latency)
        self.next_decision += self.block_size
        return unpack_robot_state(commanded, 'joint', self.dims, source_type='obs')

    def finish_episode(self):
        if getattr(self, 'async_replan', None) is not None:
            self.async_replan.close()
        if self.episode_file is not None:
            self._record(dict(event='finish', measured_last_step=(
                None if self.observation is None else self.observation.step)))
            self.episode_file.close()
            self.episode_file = None

    def update_obs_batch(self, obs):
        raise NotImplementedError('this evaluated baseline uses one simulator environment')

    def get_action_batch(self, obs=None):
        raise NotImplementedError('this evaluated baseline uses one simulator environment')
