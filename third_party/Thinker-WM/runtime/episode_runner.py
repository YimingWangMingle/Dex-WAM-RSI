"""Chunked closed-loop control: the single owner of per-step logic.

``ChunkedController.act(observation)`` returns exactly one final command row
per environment step. Both ``EpisodeRunner`` (local rollouts) and the
PhysicalRSI System-1 adapter (``skills.wam.core``) call it, so there is one
implementation of: history update -> error recording -> recovery decision ->
(re)plan -> post-process once -> bind forecast -> dispatch.

Time base: the chunk generated at decision row d has command rows d..d+A-1;
row k is dispatched at step d+k and the next observation is row d+k+1. Video
frame n of the forecast is row d+n*ratio; latent f (>=1) summarises frames
4f-3..4f and is compared at row d+4f*ratio, on decoded pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch

from thinker_wm.models.inference_api import Observation
from thinker_wm.runtime.action_postprocess import ActionPostprocessor, Command
from thinker_wm.runtime.history_buffer import HistoryBuffer
from thinker_wm.runtime.prediction_error import PredictionErrorTracker
from thinker_wm.runtime.prediction_store import PredictionStore
from thinker_wm.runtime.progress_checker import ProgressChecker
from thinker_wm.runtime.recovery_controller import CONTINUE, STOP, RecoveryController
from execution.trajectory import TrajectoryLogger


class StopEpisode(RuntimeError):
    pass


@dataclass(frozen=True)
class ControllerConfig:
    execute_steps: int = 16
    compare_visual: bool = True          # needs decode_outputs=True in GenerationSettings
    compare_depth: bool = False
    reforecast_rewritten: bool = True    # predict_outcome on the final command if post-processing changed it
    condition_on_recovery: bool = False  # pass error features to generate_action (needs recovery-trained model)
    depth_z_near: float = 0.1
    depth_z_far: float = 10.0
    depth_mode: str = "log"
    log_measured_rgb: bool = False       # needed when rollouts feed Data-RSI


class ChunkedController:
    def __init__(self, inference, postprocessor: ActionPostprocessor, instruction: str, *,
                 config: ControllerConfig = ControllerConfig(), history: Optional[HistoryBuffer] = None,
                 tracking_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                 recovery: Optional[RecoveryController] = None,
                 progress: Optional[ProgressChecker] = None,
                 logger: Optional[TrajectoryLogger] = None):
        self.inf, self.post, self.instruction, self.cfg = inference, postprocessor, instruction, config
        self.ratio, self.num_frames = int(inference.ratio), int(inference.num_frames)
        self.chunk = self.ratio * self.num_frames
        if not 0 < config.execute_steps <= self.chunk:
            raise ValueError(f"execute_steps must be in (0, {self.chunk}]")
        settings = getattr(inference, "settings", None)
        if config.compare_visual and settings is not None and not settings.decode_outputs:
            raise ValueError("compare_visual needs GenerationSettings(decode_outputs=True); "
                             "otherwise set ControllerConfig(compare_visual=False)")
        self.history = history
        self.store = PredictionStore()
        self.tracker = PredictionErrorTracker(config.execute_steps, self.num_frames // 4, tracking_fn)
        self.recovery = recovery or RecoveryController()
        self.progress = progress or ProgressChecker()
        self.logger = logger
        self.command: Optional[Command] = None
        self.k = 0
        self.decisions = []
        self.last_chunk_errors = []
        self.started = False

    # -- episode lifecycle -------------------------------------------------
    def begin(self, episode_id: str = "") -> None:
        if self.history is not None:
            self.history.reset(episode_id)
        self.store.reset()
        self.tracker.reset()
        self.recovery.reset()
        self.progress.reset()
        self.command, self.k, self.decisions, self.started = None, 0, [], True
        self.dispatched = 0

    def end(self) -> Dict[str, Any]:
        if self.command is not None and self.k < self.cfg.execute_steps:
            self.store.interrupt("episode ended")
        elif self.command is not None:
            self.store.complete()
        self.started = False
        return dict(decisions=list(self.decisions), chunks=len(self.decisions),
                    dispatched_steps=self.dispatched)

    # -- helpers ------------------------------------------------------------
    def _log(self, role, **kw):
        if self.logger is not None:
            self.logger.write(role, **kw)

    def _compare(self, obs: Observation, depth_mm: Optional[torch.Tensor]) -> None:
        if self.command is None or not self.cfg.compare_visual:
            return
        d = self.command.decision_step
        fc = self.store.usable(d, obs.step)
        if fc is None or obs.step not in fc.latent_steps or fc.frames is None:
            return
        n = (obs.step - d) // self.ratio
        pred = fc.frames[0, :, n - 1]
        pdepth = None
        if self.cfg.compare_depth and fc.depth is not None and depth_mm is not None:
            from thinker_wm.models.depth_codec import decode_to_depth_m
            m, ok = decode_to_depth_m(fc.depth[0, :, n - 1], z_near=self.cfg.depth_z_near,
                                      z_far=self.cfg.depth_z_far, mode=self.cfg.depth_mode, in_range="01")
            pdepth = torch.where(ok, m * 1000.0, torch.zeros_like(m))
        self.tracker.record_visual(obs.step, pred, obs.rgb, pdepth, depth_mm,
                                   depth_content_mask=fc.depth_content_mask)

    def _plan(self, obs: Observation, decision) -> None:
        rec = None
        if self.cfg.condition_on_recovery and self.command is not None:
            rec = dict(features=self.tracker.features(obs.time_s).tensor()[None],
                       valid=torch.tensor([True]))
        if self.history is not None:
            self._log("history", step=obs.step, **self.history.metrics())
        out = self.inf.generate_action(obs, self.history, self.instruction, recovery=rec)
        cmd = self.post.finalize(out.actions, obs.step)
        b = self.store.bind(cmd, out.forecast)
        if b.forecast is None and self.cfg.reforecast_rewritten and self.cfg.compare_visual:
            self.store.attach_forecast(cmd.decision_step,
                                       self.inf.predict_outcome(obs, self.history, self.instruction,
                                                                cmd.model_actions))
        self.command, self.k = cmd, 0
        self.last_chunk_errors = list(self.tracker.log)
        self.tracker.reset(obs.step, obs.time_s)
        self.decisions.append(dict(step=obs.step, decision=decision.action, reason=decision.reason))
        self._log("command", decision_step=obs.step, hash=cmd.hash, source_hash=cmd.source_hash,
                  postprocess=cmd.postprocess_revision, forecast_valid=self.store.binding(obs.step).forecast is not None,
                  invalid_reason=self.store.binding(obs.step).invalid_reason,
                  attachments=dict(actions=cmd.actions))
        fc = self.store.binding(obs.step).forecast
        if fc is not None:
            self._log("forecast", **fc.meta())

    # -- per-step entry points ---------------------------------------------
    def observe(self, obs: Observation, depth_mm: Optional[torch.Tensor] = None,
                info: Optional[Mapping[str, Any]] = None):
        """Take a measured observation without planning or dispatching. Used
        directly for the final feedback after the last executed row."""
        if not self.started:
            raise RuntimeError("call begin() first")
        if self.history is not None:
            self.history.add(obs)
        self.store.observe(obs.step, obs.time_s)
        rgb = None
        if self.cfg.log_measured_rgb:
            rgb = dict(rgb=obs.rgb.clamp(0, 1).mul(255).round().to(torch.uint8).numpy())
        # This observation is the feedback of the previously dispatched row.
        confirmed = self.command is not None and self.k > 0
        self._log("measured", step=obs.step, time_s=obs.time_s, state=obs.state.numpy(),
                  executed_row=obs.step - 1 if confirmed else None, attachments=rgb)
        self._log("progress", **self.progress.update(obs.step, obs.state.numpy(), dict(info or {})))
        if confirmed:
            self.tracker.record_step(obs.step, self.command.model_actions[self.k - 1], obs.state.numpy())
            self._compare(obs, depth_mm)
        feats = self.tracker.features(obs.time_s)
        if self.command is not None:
            self._log("error", step=obs.step, features=feats.values)
        return feats

    def act(self, obs: Observation, depth_mm: Optional[torch.Tensor] = None,
            info: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        feats = self.observe(obs, depth_mm, info)
        decision = self.recovery.decide(feats)
        if self.command is not None:
            self._log("decision", step=obs.step, action=decision.action, reason=decision.reason)
        if decision.action == STOP:
            self.store.interrupt("recovery stop")
            raise StopEpisode(decision.reason)
        if self.command is None or self.k >= self.cfg.execute_steps or decision.action != CONTINUE:
            if self.command is not None:
                (self.store.complete() if self.k >= self.cfg.execute_steps
                 else self.store.interrupt(decision.reason))
                self.store.expire_before(obs.step)
            self._plan(obs, decision)
        if self.command.decision_step + self.k != obs.step:
            raise RuntimeError(f"observation step {obs.step} does not follow command row {self.k}")
        row = self.command.row(self.k)
        self.store.dispatch(obs.step, self.k)
        self.k += 1
        self.dispatched += 1
        return row


class EpisodeRunner:
    """Local rollout: ``env.reset() -> live obs``; ``env.step(action) -> (live obs, done, info)``."""

    def __init__(self, controller: ChunkedController, adapter, max_steps: int):
        self.ctl, self.adapter, self.max_steps = controller, adapter, int(max_steps)

    def run(self, env, episode_id: str) -> Dict[str, Any]:
        live = env.reset()
        self.ctl.begin(episode_id)
        status, info = "budget", {}
        try:
            for step in range(self.max_steps):
                obs, depth = self.adapter.observation(live, step)
                action = self.ctl.act(obs, depth, info)
                live, done, info = env.step(action)
                if done:
                    status = "terminated"
                    break
            if self.max_steps > 0:
                # Feedback of the last executed row; no further planning.
                obs, depth = self.adapter.observation(live, step + 1)
                self.ctl.observe(obs, depth, info)
        except StopEpisode as e:
            status, info = "stopped", dict(reason=str(e))
        summary = self.ctl.end()
        summary.update(status=status, steps=summary["dispatched_steps"],
                       info={k: v for k, v in (info or {}).items() if isinstance(v, (bool, int, float, str))})
        if self.ctl.logger is not None:
            self.ctl.logger.write("outcome", **summary)
        return summary
