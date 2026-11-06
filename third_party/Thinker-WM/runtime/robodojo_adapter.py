"""Live RoboDojo observation -> model ``Observation`` with the training transform.

The binding from live simulator keys to field-map views is explicit (no key
guessing). Views are tiled with ``robodojo_core.compose`` (bilinear RGB,
nearest depth) and resized with ``pad_or_resize`` exactly like the dataset
reader, so ``camera_transform`` identifies one transform for train and deploy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from thinker_wm.data_prepare.robodojo_core import FieldMap, compose, pad_or_resize
from thinker_wm.models.inference_api import Observation


def _get(obs: Mapping[str, Any], dotted: str):
    cur: Any = obs
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(f"live observation lacks {dotted!r}")
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class ObservationBinding:
    rgb_keys: Dict[str, str]                       # field-map view -> live key (dotted)
    state_key: str = ""
    state_columns: Optional[Tuple[int, int]] = None
    depth_keys: Dict[str, str] = field(default_factory=dict)
    depth_unit_to_mm: Optional[float] = None
    rgb_channel_order: str = "rgb"                 # "rgb" | "bgr"
    state_keys: Tuple[str, ...] = ()                # explicitly ordered arm/gripper fields

    def __post_init__(self):
        if bool(self.state_key) == bool(self.state_keys):
            raise ValueError("supply one state vector key OR ordered state_keys")
        if self.rgb_channel_order not in ("rgb", "bgr"):
            raise ValueError("rgb_channel_order must be rgb or bgr")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class RoboDojoObservationAdapter:
    def __init__(self, field_map: FieldMap, binding: ObservationBinding, *, model_hw: Tuple[int, int],
                 resize_mode: str, robot_name: str):
        missing = [v for v in field_map.layout.views if v not in binding.rgb_keys]
        if missing:
            raise ValueError(f"binding lacks rgb keys for layout views {missing}")
        if binding.depth_keys and binding.depth_unit_to_mm is None:
            raise ValueError("depth_keys require depth_unit_to_mm")
        if resize_mode not in ("pad", "interpolate"):
            raise ValueError("resize_mode must be 'pad' or 'interpolate'")
        self.fm, self.binding = field_map, binding
        self.model_hw, self.resize_mode, self.robot_name = tuple(model_hw), resize_mode, robot_name

    @property
    def camera_transform(self) -> str:
        return (f"robodojo/fm={self.fm.digest}/bind={self.binding.digest()}/"
                f"{self.resize_mode}/{self.model_hw[0]}x{self.model_hw[1]}")

    def rgb(self, live: Mapping[str, Any]) -> torch.Tensor:
        views = {}
        for v in self.fm.layout.views:
            x = np.asarray(_get(live, self.binding.rgb_keys[v]))
            if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3:
                raise ValueError(f"live view {v!r} is {x.dtype}{x.shape}, expected uint8 HW3")
            if self.binding.rgb_channel_order == "bgr":
                x = x[..., ::-1]
            views[v] = np.ascontiguousarray(x)[None]
        chw = torch.from_numpy(compose(views, self.fm.layout, nearest=False)).permute(0, 3, 1, 2)
        if self.resize_mode == "pad":
            out = pad_or_resize(chw, self.model_hw, "pad", nearest=False).float() / 255.0
        else:
            out = pad_or_resize(chw.float() / 255.0, self.model_hw, "interpolate", nearest=False)
        return out[0].clamp(0, 1)

    def depth_mm(self, live: Mapping[str, Any]) -> Optional[torch.Tensor]:
        if not self.binding.depth_keys:
            return None
        views = {}
        for v in self.fm.layout.views:
            d = np.asarray(_get(live, self.binding.depth_keys[v]), dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            bad = ~np.isfinite(d) | (d <= 0)
            for iv in self.fm.depth_invalid_values:
                bad |= d == iv
            d = np.where(bad, 0.0, d * self.binding.depth_unit_to_mm)
            views[v] = d[None]
        t = torch.from_numpy(compose(views, self.fm.layout, nearest=True)).unsqueeze(1)
        return pad_or_resize(t, self.model_hw, self.resize_mode, nearest=True)[0, 0]

    def state(self, live: Mapping[str, Any]) -> torch.Tensor:
        keys = self.binding.state_keys or (self.binding.state_key,)
        s = np.concatenate([np.asarray(_get(live, key), dtype=np.float32).reshape(-1) for key in keys])
        if self.binding.state_columns is not None:
            a, b = self.binding.state_columns
            s = s[a:b]
        if not np.isfinite(s).all():
            raise ValueError("robot state must be finite")
        if self.fm.state_names and len(s) != len(self.fm.state_names):
            raise ValueError("live state dimension differs from field map")
        return torch.from_numpy(s.copy())

    def observation(self, live: Mapping[str, Any], step: int) -> Tuple[Observation, Optional[torch.Tensor]]:
        obs = Observation(rgb=self.rgb(live), state=self.state(live), step=int(step),
                          time_s=int(step) / self.fm.control_hz, robot_name=self.robot_name)
        return obs, self.depth_mm(live)
