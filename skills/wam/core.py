"""WAM as a versioned PhysicalRSI System-1 skill.

The upstream ``ExperimentRuntime`` owns ``env.step``; this policy returns one
JSON action per ``act`` call from ``thinker_wm.runtime.episode_runner.ChunkedController``
(the same controller local rollouts use). Prediction (``predict_outcome``)
only produces forecasts for checking; it never emits a control command.

Observations cross the boundary as JSON with image references (the upstream
runtime freezes and persists every observation in ``trajectory.json``).
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from thinker_wm.models.inference_api import Observation
from thinker_wm.runtime.episode_runner import ChunkedController, StopEpisode

OBS_SCHEMA = "thinkerwam.observation/v1"
ACTION_SCHEMA = "thinkerwam.command/v1"


class ImageStore:
    """Content-addressed npz store shared by environment and policy."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, **arrays) -> Dict[str, str]:
        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
        data = buf.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        rel = f"{sha[:2]}/{sha}.npz"
        p = self.root / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
        return dict(path=rel, sha256=sha)

    def get(self, ref: Mapping[str, str]) -> Dict[str, np.ndarray]:
        data = (self.root / ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError(f"image reference changed: {ref['path']}")
        with np.load(io.BytesIO(data)) as z:
            return {k: z[k] for k in z.files}


def encode_observation(store: ImageStore, obs: Observation, depth_mm: Optional[torch.Tensor],
                       camera_transform: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rgb = obs.rgb.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    out = dict(schema=OBS_SCHEMA, step=int(obs.step), time_s=float(obs.time_s), robot=obs.robot_name,
               state=[float(x) for x in obs.state.tolist()], camera_transform=camera_transform,
               rgb=store.put(rgb=rgb),
               depth=None if depth_mm is None else store.put(depth_mm=depth_mm.numpy().astype(np.float32)))
    if extra:
        out.update(extra)
    return out


def decode_observation(store: ImageStore, j: Mapping[str, Any], camera_transform: str):
    if j.get("schema") != OBS_SCHEMA:
        raise ValueError(f"unexpected observation schema {j.get('schema')!r}")
    if j["camera_transform"] != camera_transform:
        raise ValueError("observation camera transform differs from the policy's")
    rgb = torch.from_numpy(store.get(j["rgb"])["rgb"]).float().div(255)
    depth = None if j.get("depth") is None else torch.from_numpy(store.get(j["depth"])["depth_mm"])
    obs = Observation(rgb=rgb, state=torch.tensor(j["state"], dtype=torch.float32), step=int(j["step"]),
                      time_s=float(j["time_s"]), robot_name=None if j["robot"] is None else str(j["robot"]))
    return obs, depth


class WAMSkill:
    """System-1 policy. ``controller_factory(instruction) -> ChunkedController``."""

    def __init__(self, *, name: str, revision: str, controller_factory: Callable[[str], ChunkedController],
                 store: ImageStore, camera_transform: str, observation_contract, action_contract,
                 timing, dependencies: Sequence = (), system1_cls=None, identity_extra=None):
        self.name, self.revision = name, revision
        self.factory, self.store, self.camera_transform = controller_factory, store, camera_transform
        self._obs_contract, self._act_contract, self._timing = observation_contract, action_contract, timing
        self._deps = tuple(dependencies)
        self._system1_cls = system1_cls
        self._identity_extra = dict(identity_extra or {})
        self.ctl: Optional[ChunkedController] = None
        self.episodes = 0
        self._last_step, self._pending, self._stopped = None, None, False

    def identity(self) -> Dict[str, Any]:
        return dict(kind="thinkerwam.wam-skill/v1", name=self.name, revision=self.revision,
                    camera_transform=self.camera_transform, **self._identity_extra)

    def describe(self):
        return self._system1_cls(self.name, self.revision, self._obs_contract, self._act_contract,
                                 self._deps, self._timing)

    def begin_episode(self, task: str, case: dict, observation, context) -> None:
        instruction = case.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("case must carry the public task instruction")
        self.ctl = self.factory(instruction)
        episode = hashlib.sha256(repr((task, sorted(case.items()))).encode()).hexdigest()[:16]
        self.ctl.begin(episode)
        self.episodes += 1
        self._last_step, self._pending, self._stopped = None, None, False
        context.check()

    def act(self, observation, context):
        if self.ctl is None:
            raise RuntimeError("begin_episode was not called")
        context.check()
        obs, depth = decode_observation(self.store, observation, self.camera_transform)
        self._last_step, self._pending = obs.step, None
        try:
            row = self.ctl.act(obs, depth, observation.get("task") or {})
        except StopEpisode as e:
            self._stopped = True
            return dict(schema=ACTION_SCHEMA, stop=True, reason=str(e))
        context.check()
        return dict(schema=ACTION_SCHEMA, command=[float(x) for x in np.asarray(row).reshape(-1)])

    def on_environment_observation(self, observation) -> None:
        """Compatibility hook for cores that do not pass the terminal observation
        to the policy: the environment adapter reports every post-step observation."""
        if self.ctl is not None and observation is not None:
            self._pending = observation

    def observe_final(self, observation, context) -> None:
        """Feedback of the last executed row: measured and compared, no planning.
        Ignored after a policy stop or for an already observed step."""
        if self.ctl is None or self._stopped or observation is None:
            return
        context.check()
        obs, depth = decode_observation(self.store, observation, self.camera_transform)
        if self._last_step is not None and obs.step <= self._last_step:
            return
        self._last_step, self._pending = obs.step, None
        self.ctl.observe(obs, depth, observation.get("task") or {})

    def end_episode(self, context) -> Dict[str, Any]:
        if self._pending is not None:
            self.observe_final(self._pending, context)
        summary = self.ctl.end() if self.ctl is not None else dict(chunks=0)
        self.ctl = None
        return dict(stopped=True, chunks=int(summary["chunks"]))
