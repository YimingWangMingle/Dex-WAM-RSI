"""Native stateless ThinkerWAM generation behind the proposal-only RSI boundary.

ThinkerWAM re-encodes measured history, so it does not pretend to implement WSP's
internal commit API. Only the observation store can populate physical history.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from thinker_wm.runtime.joint_continuity import PlanningHorizon
from skills.contracts import BackendProposal, ControlSkill, ModelControlSkill, ExecutionFeedback, SkillIdentity, SkillRequest
from environments.robodojo.training import RoboDojoProfile, verify_bundle


class ThinkerWAMBackend:
    def __init__(self, inference: Any, history: Any, observations: Any, profile: RoboDojoProfile,
                 *, execute_steps: int = 8, continuity: Any = None):
        self.inference, self.history, self.observations, self.profile = inference, history, observations, profile
        steps = inference.num_frames * inference.ratio
        if steps % profile.actions_per_block:
            raise ValueError('prediction must contain whole trained blocks')
        self.horizon = PlanningHorizon(steps // profile.actions_per_block, execute_steps,
                                      profile.actions_per_block,
                                      profile.training_actions // profile.actions_per_block)
        self.continuity = continuity
        self.previous_measured = None
        self.measured_velocity = None
        self.last_diagnostics = {}
        self.last_subgoal = None
        self.episode_id = None
        self.pending = None
        self.last_prediction = None
        self.prediction_valid_steps = 0

    def reset(self, episode_id: str) -> None:
        if self.pending is not None:
            raise RuntimeError('resolve pending ThinkerWAM feedback before reset')
        self.episode_id = episode_id
        self.history.reset(episode_id)
        if self.continuity is not None:
            self.continuity.reset()
        self.previous_measured, self.measured_velocity = None, None
        self.last_diagnostics = {}
        self.last_subgoal = None
        self.last_prediction, self.prediction_valid_steps = None, 0

    def _sync(self, reference: Any) -> Any:
        if reference.episode_id != self.episode_id:
            raise ValueError('foreign observation episode')
        latest = self.observations.load(reference)
        last = self.history.last_step
        for obs in self.observations.since(self.episode_id, last, reference.control_step):
            if self.previous_measured is not None:
                from thinker_wm.runtime.joint_continuity import JOINTS
                delta = obs.step - self.previous_measured.step
                if delta != 1:
                    raise ValueError('joint continuity requires every measured control row')
                self.measured_velocity = (np.asarray(obs.state)[JOINTS] -
                    np.asarray(self.previous_measured.state)[JOINTS]) * self.profile.control_hz
            self.history.add(obs)
            self.previous_measured = obs
        if self.history.last_step != latest.step:
            raise ValueError('measured history does not reach the requested observation')
        return latest

    def propose(self, request: SkillRequest) -> BackendProposal:
        if self.pending is not None:
            raise RuntimeError('pending ThinkerWAM proposal')
        if request.action_contract != self.profile.action_contract:
            raise ValueError('RoboDojo action contract mismatch')
        obs = self._sync(request.observation)
        if request.subgoal != self.last_subgoal:
            if self.continuity is not None:
                self.continuity.reset()
            self.last_diagnostics = {}
            self.last_subgoal = request.subgoal
        output = self.inference.generate_action(obs, self.history, request.subgoal)
        actions = np.asarray(output.actions, dtype=np.float32)
        if actions.shape != (self.horizon.predict_steps, self.profile.action_dim) or not np.isfinite(actions).all():
            raise ValueError('ThinkerWAM must return finite native actions matching its prediction horizon')
        execute_steps = min(request.max_actions, self.horizon.execute_steps)
        raw = actions.copy()
        if self.continuity is not None:
            actions, self.last_diagnostics = self.continuity.process(
                actions, obs.state, obs.step, execute_steps, self.measured_velocity)
        rows = tuple(tuple(float(v) for v in row) for row in actions[:execute_steps])
        self.pending = (request, rows)
        unchanged = np.array_equal(raw[:execute_steps], actions[:execute_steps])
        self.last_prediction = output.forecast if unchanged else None
        self.prediction_valid_steps = 0
        reference = (self.observations.save_prediction(request.observation, output.forecast)
                     if unchanged else None)
        return BackendProposal(rows, reference)

    def observe(self, feedback: ExecutionFeedback) -> None:
        if self.pending is None:
            raise RuntimeError('no ThinkerWAM proposal awaits feedback')
        request, rows = self.pending
        if feedback.final_observation.control_step != request.observation.control_step + len(feedback.executed_actions):
            raise ValueError('feedback step/action count mismatch')
        self._sync(feedback.final_observation)
        matches = (len(feedback.executed_actions) <= len(rows) and
                   feedback.executed_actions == rows[:len(feedback.executed_actions)] and
                   self.last_prediction is not None)
        self.prediction_valid_steps = len(feedback.executed_actions) if matches else 0
        if not matches or not feedback.executed_actions:
            self.last_prediction = None
        # No generated latent or unexecuted rows are inserted into measured history.
        if self.continuity is not None and (not feedback.executed_actions or
                feedback.executed_actions != rows[:len(feedback.executed_actions)]):
            self.continuity.reset()
        self.pending = None


def load_robodojo_skill(bundle_path: str | Path, observations: Any, *, device: str = 'cuda',
                        seed: int = 0, num_blocks: int = 1, execute_steps: int = 8,
                        continuity: Any = None, model_adapter: Any = None,
                        adapter_revision: str | None = None) -> ControlSkill:
    """Load a sealed candidate strictly; does not execute any robot actions."""
    if (model_adapter is None) != (adapter_revision is None):
        raise ValueError('task model adapter and immutable revision must be supplied together')
    if model_adapter is not None and (not callable(model_adapter) or not adapter_revision.strip()):
        raise ValueError('callable task adapter and nonempty revision required')
    PlanningHorizon(num_blocks, execute_steps)
    import torch
    from omegaconf import OmegaConf
    from thinker_wm.models.inference_api import GenerationSettings, WAMInference, make_message_builder
    from thinker_wm.models.action_stats import ACTION_STATS
    from thinker_wm.data_prepare.robodojo_core import FieldMap
    from thinker_wm.runtime.history_buffer import HistoryBuffer
    from thinker_wm.scripts.eval_video_ckpt import create_eval_model, _load_model_weights
    document = verify_bundle(bundle_path)
    cfg = OmegaConf.create(document['config'])
    fm = FieldMap.from_dict(document['field_map'], digest=document['profile']['field_map_digest'])
    profile = RoboDojoProfile.from_config(document['config'], fm, ACTION_STATS)
    if observations.profile != profile:
        raise ValueError('observation contract differs from the sealed training contract')
    model = create_eval_model(cfg, load_pretrained_backbones=False)
    _load_model_weights(model, Path(document['checkpoint']['weights']), strict=True)
    if model_adapter is not None:
        # Explicit extension point: caller owns compatibility/strict adapter loading.
        # No automatic LoRA injection, task switching or implicit checkpoint fallback.
        model = model_adapter(model, profile)
    model.to(torch.device(device)).eval()
    inf = cfg.model.inference
    settings = GenerationSettings(num_inference_steps=int(inf.num_inference_timesteps),
        action_num_inference_steps=int(inf.action_num_inference_timesteps),
        guidance_scale=float(inf.guidance_scale), action_guidance_scale=float(inf.action_guidance_scale),
        return_depth=True, decode_outputs=False, num_blocks=num_blocks, seed=int(seed))
    checkpoint_id = 'sha256:' + document['checkpoint']['weights_sha256']
    if adapter_revision is not None:
        checkpoint_id += ':adapter:' + adapter_revision
    inference = WAMInference(model, checkpoint=checkpoint_id,
        action_stats_version=profile.action_stats_version, camera_transform=observations.camera_transform,
        settings=settings, message_builder=make_message_builder(cfg, 'robodojo'))
    backend = ThinkerWAMBackend(inference, HistoryBuffer.from_config(cfg, profile.robot), observations, profile,
                                execute_steps=execute_steps, continuity=continuity)
    return ModelControlSkill(SkillIdentity('wam_control', 'thinkerwam-robodojo/v2:' + document['bundle_id'] + (':' + adapter_revision if adapter_revision else ''),
        profile.action_contract, profile.action_dim, inference.checkpoint), backend)
