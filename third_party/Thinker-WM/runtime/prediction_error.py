"""Fixed-size prediction-error features (input of RecoveryEncoder / controller).

Comparisons are always made in pixel space on decoded predicted frames vs.
measured frames after the same camera transform - never temporal-VAE latents
against single-frame latents. Action tracking is checked every control step,
visual deviation once per predicted latent (its last frame row).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

FEATURE_NAMES = (
    "track_mean", "track_max", "track_last",
    "visual_mse_mean", "visual_mse_max", "visual_mse_last",
    "depth_abs_mm_mean", "depth_valid_frac",
    "track_count_frac", "visual_count_frac", "forecast_age_s", "chunk_progress",
    "valid_track", "valid_visual", "valid_depth", "valid_forecast",
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class PredictionErrorFeatures:
    values: Dict[str, float]

    def vector(self) -> np.ndarray:
        return np.array([float(self.values.get(k, 0.0)) for k in FEATURE_NAMES], dtype=np.float32)

    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.vector())


def frame_mse(pred: torch.Tensor, measured: torch.Tensor) -> float:
    if pred.shape != measured.shape:
        raise ValueError(f"frame shapes differ {tuple(pred.shape)} vs {tuple(measured.shape)}")
    return float(((pred.float().clamp(0, 1) - measured.float().clamp(0, 1)) ** 2).mean())


def depth_abs_mm(pred_mm: torch.Tensor, measured_mm: torch.Tensor, content_mask=None):
    if pred_mm.shape != measured_mm.shape:
        raise ValueError("predicted and measured depth shapes differ")
    valid = torch.isfinite(measured_mm) & (measured_mm > 0)
    region = torch.ones_like(valid)
    if content_mask is not None:
        if content_mask.dtype != torch.bool or content_mask.shape != measured_mm.shape:
            raise ValueError("depth content mask must be bool HW matching measured depth")
        region = content_mask.to(valid.device)
        valid = valid & region
    if not bool(torch.isfinite(pred_mm[valid]).all()):
        raise ValueError("nonfinite prediction inside valid measured depth")
    if not bool(valid.any()):
        return None, 0.0
    return float((pred_mm[valid] - measured_mm[valid]).abs().mean()), float(valid.sum() / region.sum().clamp_min(1))


class PredictionErrorTracker:
    """Accumulates per-chunk errors. ``tracking_fn(command_row, measured_state)``
    returns a scalar error and must come from the audited action semantics."""

    def __init__(self, chunk_steps: int, latents: int,
                 tracking_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None):
        self.chunk_steps, self.latents, self.tracking_fn = int(chunk_steps), int(latents), tracking_fn
        self.reset()

    def reset(self, decision_step: Optional[int] = None, forecast_time_s: Optional[float] = None):
        self.decision_step, self.forecast_time_s = decision_step, forecast_time_s
        self.track: List[float] = []
        self.visual: List[float] = []
        self.depth: List[float] = []
        self.depth_frac: List[float] = []
        self.has_forecast = False
        self.executed = 0
        self.log: List[dict] = []

    def record_step(self, step: int, command_row: np.ndarray, measured_state: np.ndarray) -> None:
        self.executed += 1
        if self.tracking_fn is None:
            return
        e = float(self.tracking_fn(np.asarray(command_row), np.asarray(measured_state)))
        self.track.append(e)
        self.log.append(dict(step=int(step), kind="track", value=e))

    def record_visual(self, step: int, pred_rgb, measured_rgb, pred_depth_mm=None, measured_depth_mm=None, depth_content_mask=None):
        self.has_forecast = True
        m = frame_mse(pred_rgb, measured_rgb)
        self.visual.append(m)
        row = dict(step=int(step), kind="visual", value=m)
        if pred_depth_mm is not None and measured_depth_mm is not None:
            err, frac = depth_abs_mm(pred_depth_mm, measured_depth_mm, depth_content_mask)
            if err is not None:
                self.depth.append(err)
                self.depth_frac.append(frac)
                row.update(depth_abs_mm=err, depth_valid_frac=frac)
        self.log.append(row)

    def features(self, now_s: Optional[float] = None) -> PredictionErrorFeatures:
        v: Dict[str, float] = {}
        if self.track:
            v.update(track_mean=np.mean(self.track), track_max=np.max(self.track),
                     track_last=self.track[-1], valid_track=1.0)
        if self.visual:
            v.update(visual_mse_mean=np.mean(self.visual), visual_mse_max=np.max(self.visual),
                     visual_mse_last=self.visual[-1], valid_visual=1.0)
        if self.depth:
            v.update(depth_abs_mm_mean=np.mean(self.depth), depth_valid_frac=np.mean(self.depth_frac),
                     valid_depth=1.0)
        v["track_count_frac"] = len(self.track) / max(self.chunk_steps, 1)
        v["visual_count_frac"] = len(self.visual) / max(self.latents, 1)
        v["chunk_progress"] = self.executed / max(self.chunk_steps, 1)
        if self.has_forecast and now_s is not None and self.forecast_time_s is not None:
            v.update(forecast_age_s=float(now_s - self.forecast_time_s), valid_forecast=1.0)
        return PredictionErrorFeatures({k: float(x) for k, x in v.items()})
