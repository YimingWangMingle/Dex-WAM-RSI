#!/usr/bin/env python3
"""Smoke: DA3 metric depth on one AgiBot-Beta window, then time extrapolate.

Same env/weights as T4W dump (motus + local DA3, no pip):
  /home/fengodchen/miniconda3/envs/motus/bin/python
  /code1/zxw_data/checkpoint/DA3NESTED-GIANT-LARGE-1.1

  CUDA_VISIBLE_DEVICES=0 python scripts/smoke_da3_depth.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

T4W_ROOT = "/code1/zxw_data/pretrain/third_party/track4world"
DA3_EXT = os.path.join(T4W_ROOT, "track4world/nets/external")
DA3_LOCAL = "/code1/zxw_data/checkpoint/DA3NESTED-GIANT-LARGE-1.1"
CATALOG = (
    "/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/"
    "agibot_beta_parallel_gripper.20260805_full_v1.episode_catalog.jsonl"
)
RANGES = (
    "/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/"
    "agibot_beta_parallel_gripper.train.ranges.sqlite"
)
OUT = Path("/data/xiongwei.zhao/agibot_da3_depth/smoke")
TRAIN_H, TRAIN_W = 480, 640
N_FRAMES = 49
TARGET_KEEP = 5000


def _stub() -> None:
    sys.modules.setdefault("moviepy", types.ModuleType("moviepy"))
    sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
    evo_traj = types.ModuleType("evo.core.trajectory")
    evo_traj.PosePath3D = type("PosePath3D", (), {"__init__": lambda *a, **k: None})
    sys.modules.setdefault("evo", types.ModuleType("evo"))
    sys.modules.setdefault("evo.core", types.ModuleType("evo.core"))
    sys.modules["evo.core.trajectory"] = evo_traj
    export_mod = types.ModuleType("depth_anything_3.utils.export")
    export_mod.export = lambda *a, **k: None
    sys.modules["depth_anything_3.utils.export"] = export_mod
    if "addict" not in sys.modules:
        addict = types.ModuleType("addict")

        class Dict(dict):
            def __getattr__(self, name):
                if name.startswith("_"):
                    raise AttributeError(name)
                try:
                    return self[name]
                except KeyError:
                    missing = Dict()
                    self[name] = missing
                    return missing

            def __setattr__(self, name, value):
                self[name] = value

            def __delattr__(self, name):
                try:
                    del self[name]
                except KeyError:
                    pass

        addict.Dict = Dict
        sys.modules["addict"] = addict


def extract_tar_member(path: str, offset: int, size: int, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(path, "rb") as src, open(dest, "wb") as dst:
        src.seek(offset)
        left = size
        while left > 0:
            chunk = src.read(min(8 * 1024 * 1024, left))
            if not chunk:
                raise IOError(f"short read {path}@{offset}")
            dst.write(chunk)
            left -= len(chunk)


def pick_first_clip() -> dict:
    with open(CATALOG) as f:
        rec = json.loads(next(f))
    st = rec["video"]["storage"]
    return {
        "episode_id": rec["episode_id"],
        "tar_path": st["path"],
        "offset": int(st["offset"]),
        "size": int(st["size"]),
        "n_src": int(rec["video"]["count"]),
        "fps": float(rec["video"].get("fps", 30.0)),
    }


def decode_first_n(mp4: str, n: int) -> list:
    import cv2
    import imageio_ffmpeg
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="da3_ff_") as td:
        pattern = os.path.join(td, "frame_%06d.png")
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", mp4, "-vf", f"select=lt(n\\,{n})", "-vsync", "0",
            "-frames:v", str(n), pattern,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {p.stderr[-500:]}")
        paths = sorted(Path(td).glob("frame_*.png"))
        if len(paths) < n:
            raise RuntimeError(f"got {len(paths)} frames, want {n}")
        frames = []
        for fp in paths[:n]:
            bgr = cv2.imread(str(fp))
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return frames


def to_uint16_mm(depth_m, h: int, w: int):
    import cv2
    import numpy as np

    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim == 3:
        d = d[0]
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    bad = ~np.isfinite(d) | (d <= 0)
    mm = np.clip(np.rint(d * 1000.0), 0, 65535).astype(np.uint16)
    mm[bad] = 0
    return mm


def colorize_mm(mm):
    import cv2
    import numpy as np

    valid = mm > 0
    if not valid.any():
        return np.zeros((*mm.shape, 3), np.uint8)
    lo, hi = np.percentile(mm[valid], [2, 98])
    x = np.clip((mm.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    x = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[:, :, ::-1]


def catalog_len_stats() -> dict:
    import sqlite3

    con = sqlite3.connect(f"file:{RANGES}?mode=ro", uri=True)
    rows = [int(r[0]) for r in con.execute("SELECT episode_len FROM episode_ranges")]
    con.close()
    rows.sort()
    n = len(rows)
    def q(p):
        return rows[int((n - 1) * p)]
    return {
        "n": n,
        "min": rows[0],
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "mean": sum(rows) / n,
        "max": rows[-1],
    }


def hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def main() -> int:
    import cv2
    import numpy as np
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    stats = catalog_len_stats()
    print(
        f"ranges n={stats['n']}  mean={stats['mean']:.0f}  "
        f"p50={stats['p50']} p90={stats['p90']} max={stats['max']}",
        flush=True,
    )

    clip = pick_first_clip()
    print(f"smoke eid={clip['episode_id']} n_src={clip['n_src']} fps={clip['fps']}", flush=True)
    raw = str(OUT / "_tmp_head.mp4")
    t0 = time.time()
    extract_tar_member(clip["tar_path"], clip["offset"], clip["size"], raw)
    t_extract = time.time() - t0
    frames = decode_first_n(raw, N_FRAMES)
    t_decode = time.time() - t0 - t_extract
    h0, w0 = frames[0].shape[:2]
    print(f"extract {t_extract:.1f}s  decode49 {t_decode:.1f}s  rgb={h0}x{w0}", flush=True)

    _stub()
    sys.path[:0] = [T4W_ROOT, DA3_EXT]
    from depth_anything_3.api import DepthAnything3

    t1 = time.time()
    model = DepthAnything3.from_pretrained(DA3_LOCAL, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = torch.device("cuda:0")
    model.to(device)
    t_load = time.time() - t1
    print(f"load DA3 {t_load:.1f}s  vram={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)

    depths_mm = []
    pred0 = None
    t_infer = 0.0
    with torch.no_grad():
        for i, rgb in enumerate(frames):
            t = time.time()
            pred = model.inference([rgb], export_feat_layers=[])
            torch.cuda.synchronize()
            dt = time.time() - t
            t_infer += dt
            if i == 0:
                pred0 = pred
            mm = to_uint16_mm(pred.depth, TRAIN_H, TRAIN_W)
            depths_mm.append(mm)
            if i < 3 or i == N_FRAMES - 1:
                d = np.asarray(pred.depth, dtype=np.float32)
                if d.ndim == 3:
                    d = d[0]
                print(
                    f"  frame {i:02d} {dt:.3f}s  da3_hw={d.shape}  "
                    f"is_metric={pred.is_metric}  "
                    f"d_m=[{np.nanpercentile(d,2):.3f},{np.nanpercentile(d,98):.3f}]  "
                    f"valid={float((mm>0).mean()):.3f}",
                    flush=True,
                )

    stack = np.stack(depths_mm, 0)
    np.save(OUT / "depth_uint16_mm.npy", stack)
    dcol = cv2.resize(colorize_mm(stack[0]), (w0, h0), interpolation=cv2.INTER_NEAREST)
    viz = np.concatenate([frames[0], dcol], axis=1)
    cv2.imwrite(str(OUT / "rgb_depth.png"), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    meta = {
        "episode_id": clip["episode_id"],
        "n_frames": N_FRAMES,
        "rgb_hw": [h0, w0],
        "depth_hw": [TRAIN_H, TRAIN_W],
        "unit": "uint16_mm",
        "teacher": "DA3NESTED-GIANT-LARGE-1.1",
        "is_metric": int(pred0.is_metric) if pred0 is not None else None,
        "sec_extract": t_extract,
        "sec_decode49": t_decode,
        "sec_load": t_load,
        "sec_infer49": t_infer,
        "sec_per_frame": t_infer / N_FRAMES,
        "ranges": stats,
    }
    (OUT / "smoke.json").write_text(json.dumps(meta, indent=2))
    try:
        os.remove(raw)
    except OSError:
        pass

    per = t_infer / N_FRAMES
    mean_t = stats["mean"]
    p50_t = stats["p50"]
    print("", flush=True)
    print(f"SMOKE OK  {N_FRAMES} frames  infer={t_infer:.1f}s  {per:.3f}s/frame", flush=True)
    print(f"wrote {OUT / 'depth_uint16_mm.npy'}  shape={tuple(stack.shape)}", flush=True)
    for label, tlen, n_ep in (
        ("1 ep mean-len", mean_t, 1),
        ("5k keep @ p50", p50_t, TARGET_KEEP),
        ("5k keep @ mean", mean_t, TARGET_KEEP),
        ("7k candidates @ mean", mean_t, 7000),
    ):
        sec = per * tlen * n_ep
        print(
            f"  {label:24s}  1GPU {hours(sec):>8s}   8GPU {hours(sec/8):>8s}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
