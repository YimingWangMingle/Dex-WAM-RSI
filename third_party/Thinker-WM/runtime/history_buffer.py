"""Episode history of *measured* observations for policy / world-model conditioning.

Row selection reuses ``data_prepare.temporal_windows.select_window`` (same
recent clip / sparse slots / episode-start slot as training) with jitter off
and no future requirement. Predicted frames can never enter: ``add`` refuses
any observation whose source is not "measured".

Rows the runtime never observed are handled conservatively: a recent-clip
frame is marked invalid (the clip must stay uniformly spaced for the causal
VAE); a sparse slot falls back to the latest stored row within ``ratio`` rows
before it, with its real time offset, else it is invalid.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from thinker_wm.data_prepare.temporal_windows import WindowSpec, select_window
from thinker_wm.data_prepare.history_memory import (EventSpec, EventDetector, select_event_history,
                                         pad_recent_pixels, history_metrics, history_window_spec)


class HistoryBuffer:
    def __init__(self, spec: WindowSpec, robot_name: str, episode_id: str = "",
                 events: Optional[EventSpec] = None):
        self.spec = replace(spec, require_full_future=False)
        self.events = events or EventSpec()
        self.detector = EventDetector(self.events, self.spec.control_hz)
        self._event_rgb, self._event_state, self._event_times = {}, {}, {}
        self._times = {}
        self.robot_name = robot_name
        self.episode_id = episode_id
        r, hz = self.spec.video_action_freq_ratio, self.spec.control_hz
        span_s = self.spec.recent_frames * self.spec.frame_dt + \
            (self.spec.sparse_slots + 1) * self.spec.sparse_stride_s
        self.retention_rows = int(math.ceil(span_s * hz)) + 2 * r
        self._rgb: Dict[int, np.ndarray] = {}
        self._state: Dict[int, np.ndarray] = {}
        self._cache_key: Optional[Tuple] = None
        self._cache_val: Any = None
        self.last_step: Optional[int] = None
        self.enable_sparse_vae_cache = False
        self._sparse_vae_cache = None
        self._encoding_identity = None

    @classmethod
    def from_config(cls, config, robot_name, episode_id=""):
        h = config.get("history", {})
        return cls(history_window_spec(config), robot_name, episode_id,
                   events=EventSpec.from_config(h.get("events")))

    def reset(self, episode_id: str = "") -> None:
        self.detector.reset()
        self._times.clear()
        self._event_times.clear()
        self._event_rgb.clear()
        self._event_state.clear()
        self._rgb.clear()
        self._state.clear()
        self._cache_key, self._cache_val = None, None
        self.last_step = None
        self.episode_id = episode_id
        self._encoding_identity = None
        if self._sparse_vae_cache is not None:
            self._sparse_vae_cache.clear()

    def __len__(self) -> int:
        return len(self._rgb)

    def add(self, observation) -> None:
        if getattr(observation, "source", None) != "measured":
            raise ValueError("history accepts measured observations only")
        if observation.robot_name != self.robot_name:
            raise ValueError("observation robot differs from the history buffer robot")
        s = int(observation.step)
        if self.last_step is not None and s <= self.last_step:
            raise ValueError(f"history steps must increase ({s} after {self.last_step})")
        rgb = observation.rgb.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
        state = observation.state.detach().float().cpu().numpy()
        now = float(observation.time_s)
        if not math.isfinite(now) or (self.last_step is not None and now <= self._times[self.last_step]):
            raise ValueError("measured timestamps must be finite and strictly increase")
        # The detector validates before mutating, so a rejected frame leaves no state.
        self.detector.update(s, state, now)
        self._rgb[s], self._state[s], self._times[s] = rgb.cpu().numpy(), state, now
        keep = self.detector.retained_rows()
        for k in keep:
            if k in self._rgb:
                self._event_rgb[k], self._event_state[k] = self._rgb[k], self._state[k]
                self._event_times[k] = self._times[k]
        for k in list(self._event_rgb):
            if k not in keep:
                del self._event_rgb[k], self._event_state[k], self._event_times[k]
        self.last_step = s
        cutoff = s - self.retention_rows
        for k in [k for k in self._rgb if 0 < k < cutoff]:
            del self._rgb[k], self._state[k], self._times[k]

    def _resolve(self, w, d: int):
        r, hz = self.spec.video_action_freq_ratio, self.spec.control_hz
        rec_valid = w.recent_valid.copy()
        for i, row in enumerate(w.recent_rows):
            if rec_valid[i] and int(row) not in self._rgb:
                rec_valid[i] = False
        sp_rows, sp_times, sp_valid = w.sparse_rows.copy(), w.sparse_times.copy(), w.sparse_valid.copy()
        stored = np.array(sorted(self._rgb), dtype=np.int64)
        for i, row in enumerate(sp_rows):
            if not sp_valid[i] or int(row) in self._rgb or int(row) in self._event_rgb:
                continue
            cand = stored[(stored <= row) & (stored > row - r)]
            if cand.size:
                sp_rows[i] = int(cand.max())
                sp_times[i] = (sp_rows[i] - d) / hz
            else:
                sp_valid[i] = False
        # A missing sparse sample may resolve onto an already selected event
        # image. Deduplicate again after resolution, keeping event pairs first.
        seen = set()
        order = sorted(range(len(sp_rows)), key=lambda i: (w.sparse_kind[i] < 3, i))
        for i in order:
            if sp_valid[i]:
                row = int(sp_rows[i])
                if row in seen:
                    sp_valid[i] = False
                else:
                    seen.add(row)
        return rec_valid, sp_rows, sp_times, sp_valid

    def window(self, step: int):
        d = int(step)
        if self.last_step is None or d != self.last_step:
            raise ValueError("history is queried at the latest measured step only")
        w = select_window(self.spec, d + 1, d, episode_id=self.episode_id, jitter=False)
        w = select_event_history(w, self.spec, self.events, self.detector.records)
        rv, sr, st, sv = self._resolve(w, d)
        current = self._times[d]
        rt = w.recent_times.copy()
        valid = [i for i in range(len(w.recent_rows)) if rv[i]]
        if valid:
            # A continuous VAE clip needs the configured cadence. Off-grid
            # measurements are holes, not fabricated continuous motion. Cadence
            # is checked within the clip (anchor = frame most others agree with),
            # so a late current frame only shifts the real offsets.
            tol = .5 / self.spec.control_hz + 1e-8
            ts = {i: self._times[int(w.recent_rows[i])] for i in valid}
            nom = {i: float(rt[i]) for i in valid}
            on_grid = lambda a, i: abs((ts[i] - ts[a]) - (nom[i] - nom[a])) <= tol  # noqa: E731
            a = max(valid, key=lambda a: (sum(on_grid(a, i) for i in valid), nom[a]))
            for i in valid:
                if not on_grid(a, i):
                    rv[i] = False
                rt[i] = ts[i] - current
        for i, row in enumerate(sr):
            if sv[i]:
                st[i] = (self._times[int(row)] if int(row) in self._times
                         else self._event_times[int(row)]) - current
        # Fallback rows can move a sample before its neighbours; restore time order.
        kind = w.sparse_kind.copy()
        p = np.flatnonzero(sv)
        o = p[np.argsort(st[p], kind="stable")]
        sr[p], st[p], kind[p] = sr[o], st[o], kind[o]
        return replace(w, recent_times=rt, sparse_kind=kind), rv, sr, st, sv

    def history_inputs(self, step: int) -> Optional[Dict[str, torch.Tensor]]:
        w, rec_valid, sp_rows, sp_times, sp_valid = self.window(step)
        if not rec_valid.any() and not sp_valid.any():
            return None
        some = next(iter(self._rgb.values()))
        blank = np.zeros_like(some)
        blank_s = np.zeros_like(next(iter(self._state.values())))

        def pix(rows, valid):
            arr = np.stack([(self._rgb[int(r)] if int(r) in self._rgb else self._event_rgb[int(r)]) if v else blank for r, v in zip(rows, valid)])
            return torch.from_numpy(arr).float().div(255).unsqueeze(0)

        def st(rows, valid):
            arr = np.stack([(self._state[int(r)] if int(r) in self._state else self._event_state[int(r)]) if v else blank_s for r, v in zip(rows, valid)])
            return torch.from_numpy(arr).float().unsqueeze(0)

        out: Dict[str, torch.Tensor] = {}
        if len(sp_rows):
            out.update(history_sparse_rgb=pix(sp_rows, sp_valid),
                       history_sparse_times=torch.tensor(sp_times, dtype=torch.float32)[None],
                       history_sparse_valid=torch.tensor(sp_valid)[None],
                       history_sparse_kind=torch.tensor(w.sparse_kind)[None],
                       history_sparse_state=st(sp_rows, sp_valid))
        if len(w.recent_rows):
            out.update(history_recent_rgb=pad_recent_pixels(pix(w.recent_rows, rec_valid)[0], rec_valid)[None],
                       history_recent_times=torch.tensor(w.recent_times, dtype=torch.float32)[None],
                       history_recent_valid=torch.tensor(rec_valid)[None],
                       history_recent_state=st(w.recent_rows, rec_valid))
        return out

    def cache_key(self, step: int, model_id: str) -> Tuple:
        w, rec_valid, sp_rows, sp_times, sp_valid = self.window(step)
        return (model_id, self.robot_name, tuple(int(x) for x in w.recent_rows), tuple(w.recent_times.tolist()), tuple(rec_valid.tolist()),
                tuple(int(x) for x in sp_rows), tuple(sp_valid.tolist()), tuple(w.sparse_kind.tolist()), tuple(sp_times.tolist()), int(step))

    def context_for(self, model, step: int, model_id: str):
        """Encoded history dict for ``inference_step(history_context=...)``, cached."""
        identity = (id(model), model_id, str(model.device), str(model.dtype),
                    bool(getattr(model.config, "history_encode_valid_only", False)),
                    bool(self.enable_sparse_vae_cache), bool(model.training))
        if identity != self._encoding_identity:
            self._cache_key, self._cache_val = None, None
            if self._sparse_vae_cache is not None:
                self._sparse_vae_cache.clear()
            self._encoding_identity = identity
        key = self.cache_key(step, identity)
        if key == self._cache_key:
            return self._cache_val
        inputs = self.history_inputs(step)
        with torch.no_grad():
            names = None if self.robot_name is None else [self.robot_name]
            extra = {}
            if (self.enable_sparse_vae_cache and inputs is not None
                    and "history_sparse_rgb" in inputs):
                from thinker_wm.runtime.sparse_vae_cache import SparseVAECache
                if self._sparse_vae_cache is None:
                    self._sparse_vae_cache = SparseVAECache(max(32, self.spec.history_slots))
                _, _, rows, _, _ = self.window(step)
                extra["sparse_latent"] = self._sparse_vae_cache.encode(
                    model, inputs["history_sparse_rgb"], inputs["history_sparse_valid"], rows)
            val = None if inputs is None else model._prepare_history(inputs, names, **extra)
        self._cache_key, self._cache_val = key, val
        return val

    def sparse_cache_stats(self):
        return {} if self._sparse_vae_cache is None else self._sparse_vae_cache.stats()

    def metrics(self):
        if self.last_step is None:
            return history_metrics(None)
        w, rv, rows, times, sv = self.window(self.last_step)
        return history_metrics(dict(history_recent_valid=torch.tensor(rv)[None],
                                    history_sparse_valid=torch.tensor(sv)[None],
                                    history_sparse_kind=torch.tensor(w.sparse_kind)[None]))

    def describe(self) -> Dict[str, Any]:
        return dict(spec=self.spec.__dict__, robot=self.robot_name, stored=len(self._rgb),
                    retention_rows=self.retention_rows, last_step=self.last_step,
                    event_records=len(self.detector.records), event_images=len(self._event_rgb),
                    events=self.events.__dict__, **self.metrics())
