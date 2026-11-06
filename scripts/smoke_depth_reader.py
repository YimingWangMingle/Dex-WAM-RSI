#!/usr/bin/env python3
"""Smoke test for the depth sidecar reader + batch-contract attach, on real data.

Proves step-2 end to end without the full motus pipeline / VAE:
  - reader resolves real EgoScale clips to uint16-mm frames,
  - DepthAttacher mirrors the RGB stream (depth_first/depth_frames in [0,1]),
  - present flag is True for dumped clips, False for unknown episodes,
  - nearest-resize path works,
  - decoded median depth is physically sane (roughly 0.5-8 m).

  python scripts/smoke_depth_reader.py
"""
from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.data_prepare.depth_attach import DepthAttacher  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.models.depth_reader import load_depth_selections  # noqa: E402

EGO_ROOT = "/data/xiongwei.zhao/egoscale_ffs_depth/v1"
N = 48  # num_video_frames
H0, W0 = 480, 640  # sidecar grid


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _fake_batch(episode_ids, frame_indices, hw):
    h, w = hw
    b = len(episode_ids)
    return {
        "video_frames": torch.zeros((b, N, 3, h, w), dtype=torch.float32),
        "first_frame": torch.zeros((b, 3, h, w), dtype=torch.float32),
        "episode_ids": list(episode_ids),
        "frame_indices": list(frame_indices),
    }


def main() -> int:
    index = load_depth_selections([EGO_ROOT])
    if not index:
        _fail(f"no prod_keep selections under {EGO_ROOT}")
    print(f"indexed prod_keep episodes: {len(index)}")

    picks = list(islice(index.items(), 4))
    episode_ids = [eid for eid, _ in picks]
    frame_indices = [list(entry.clips[0]) for _, entry in picks]
    for eid, clip in zip(episode_ids, frame_indices):
        if len(clip) != N + 1:
            _fail(f"{eid}: clip has {len(clip)} frames, expected {N + 1}")

    attacher = DepthAttacher([EGO_ROOT])  # log / 0.5 / 10 defaults

    # 1) native grid, all present
    batch = _fake_batch(episode_ids, frame_indices, (H0, W0))
    out = attacher.attach(batch)
    df, dfr, pres = out["depth_first"], out["depth_frames"], out["depth_present"]
    if tuple(df.shape) != (len(picks), 3, H0, W0):
        _fail(f"depth_first shape {tuple(df.shape)}")
    if tuple(dfr.shape) != (len(picks), N, 3, H0, W0):
        _fail(f"depth_frames shape {tuple(dfr.shape)}")
    if not bool(pres.all()):
        _fail(f"present={pres.tolist()} expected all True")
    if float(dfr.min()) < -1e-6 or float(dfr.max()) > 1.0 + 1e-6:
        _fail(f"depth_frames range [{float(dfr.min())},{float(dfr.max())}] not [0,1]")
    if not torch.allclose(dfr[:, :, 0], dfr[:, :, 1]) or not torch.allclose(
        dfr[:, :, 0], dfr[:, :, 2]
    ):
        _fail("depth 3 channels not identical")
    print("ok  shapes / range / 3-channel / present(all)")

    # decoded median sanity per sample (first future frame)
    z, valid = decode_to_depth_m(dfr[:, 0], in_range="01")  # [B, H, W]
    print(f"  {'episode':52} {'valid%':>7} {'med_m':>7}")
    for i, eid in enumerate(episode_ids):
        v = valid[i]
        med = float(z[i][v].median()) if bool(v.any()) else float("nan")
        vp = 100.0 * float(v.float().mean())
        short = eid if len(eid) <= 52 else eid[:49] + "..."
        print(f"  {short:52} {vp:7.1f} {med:7.2f}")
        if bool(v.any()) and not (0.3 <= med <= 9.9):
            _fail(f"{eid}: median depth {med:.2f} m out of plausible range")
    print("ok  decoded median depth physically sane")

    # 2) unknown episode -> present False, zeros
    miss = _fake_batch(["egoscale:__does_not_exist__"], [frame_indices[0]], (H0, W0))
    mout = attacher.attach(miss)
    if bool(mout["depth_present"][0]):
        _fail("unknown episode reported present=True")
    if float(mout["depth_frames"].abs().max()) != 0.0:
        _fail("unknown episode depth not zero-filled")
    print("ok  missing source -> present=False, zero-filled")

    # 3) nearest-resize path (target grid != sidecar grid)
    rb = _fake_batch(episode_ids[:1], frame_indices[:1], (64, 64))
    rout = attacher.attach(rb)
    if tuple(rout["depth_frames"].shape) != (1, N, 3, 64, 64):
        _fail(f"resized depth_frames shape {tuple(rout['depth_frames'].shape)}")
    if not bool(rout["depth_present"][0]):
        _fail("resize path lost present")
    print("ok  nearest-resize to 64x64")

    # 4) npz must not accumulate full volumes (V4/V5 RSS leak)
    npz_cached = [
        k
        for k, v in attacher.reader._array_cache.items()
        if isinstance(v, tuple) and v[0] == "npz"
    ]
    if npz_cached:
        _fail(f"npz full arrays cached after attach: {npz_cached}")
    print("ok  npz not retained in _array_cache")

    print("\nsmoke_depth_reader OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
