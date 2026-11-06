"""Single place that turns model actions into the command the controller receives.

Order: optional EMA smoothing -> per-dim clipping -> gripper binarisation
(optional, explicit dims/threshold) -> unit scale. Applied exactly once per
chunk; the resulting ``Command`` carries the hash that predictions bind to.
Limits and gripper semantics must come from the audited embodiment, so there
are no implicit defaults beyond "do nothing".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from thinker_wm.models.inference_api import action_hash


@dataclass(frozen=True)
class PostprocessConfig:
    smooth_alpha: float = 0.0
    low: Optional[Tuple[float, ...]] = None
    high: Optional[Tuple[float, ...]] = None
    gripper_dims: Tuple[int, ...] = ()
    gripper_threshold: Optional[float] = None
    gripper_open_close: Tuple[float, float] = (1.0, 0.0)
    unit_scale: Optional[Tuple[float, ...]] = None
    revision: str = "postprocess-v1"


@dataclass(frozen=True)
class Command:
    actions: np.ndarray            # [A, Da] final controller units
    source_hash: str               # hash of model actions before post-processing
    hash: str                      # hash of final actions
    decision_step: int
    postprocess_revision: str
    model_actions: np.ndarray      # final actions in model units (before unit_scale)
    model_hash: str                # what conditioned forecasts (predict_outcome) bind to
    finalized: bool = field(default=True)

    def row(self, k: int) -> np.ndarray:
        return self.actions[k]


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1.0 - alpha) * y[t - 1]
    return y


class ActionPostprocessor:
    def __init__(self, cfg: PostprocessConfig):
        if not 0.0 <= cfg.smooth_alpha < 1.0:
            raise ValueError("smooth_alpha in [0, 1)")
        if cfg.gripper_dims and cfg.gripper_threshold is None:
            raise ValueError("gripper_dims require an explicit gripper_threshold")
        self.cfg = cfg

    def finalize(self, actions: np.ndarray, decision_step: int) -> Command:
        if isinstance(actions, Command):
            raise ValueError("actions already post-processed")
        a = np.asarray(actions, dtype=np.float64).copy()
        src = action_hash(actions)
        c = self.cfg
        if c.smooth_alpha > 0:
            a = _ema(a, c.smooth_alpha)
        if c.low is not None or c.high is not None:
            lo = -np.inf if c.low is None else np.asarray(c.low)
            hi = np.inf if c.high is None else np.asarray(c.high)
            a = np.clip(a, lo, hi)
        for d in c.gripper_dims:
            a[:, d] = np.where(a[:, d] >= c.gripper_threshold, c.gripper_open_close[0],
                               c.gripper_open_close[1])
        model = a.astype(np.float32)
        if c.unit_scale is not None:
            a = a * np.asarray(c.unit_scale)
        out = a.astype(np.float32)
        return Command(out, src, action_hash(out), int(decision_step), c.revision,
                       model, action_hash(model))

    def is_identity(self) -> bool:
        c = self.cfg
        return (c.smooth_alpha == 0 and c.low is None and c.high is None and not c.gripper_dims
                and c.unit_scale is None)
