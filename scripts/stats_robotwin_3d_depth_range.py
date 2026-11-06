#!/usr/bin/env python3
"""Measure robotwin_3d metric depth so the shared log [z_near, z_far] is data-driven."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare.robotwin_3d import CAMS, decode_depth_mm_frame_map, robotwin_3d_paths  # noqa: E402

BIN_W = 10  # mm
BIN_MAX = 4000


def _hist() -> np.ndarray:
    return np.zeros(BIN_MAX // BIN_W, dtype=np.int64)


def _add(hist: np.ndarray, mm: np.ndarray) -> None:
    valid = mm[mm > 0]
    if valid.size == 0:
        return
    clipped = np.minimum(valid.astype(np.int64), BIN_MAX - 1)
    idx = clipped // BIN_W
    np.add.at(hist, idx, 1)


def _pct_from_hist(hist: np.ndarray, qs: list[float]) -> dict[str, float]:
    c = np.cumsum(hist)
    total = int(c[-1])
    if total == 0:
        return {f"p{int(q*100):02d}": 0.0 for q in qs}
    out = {}
    for q in qs:
        t = q * (total - 1)
        i = int(np.searchsorted(c, t, side="left"))
        out[f"p{int(q * 100):02d}"] = float(i * BIN_W + BIN_W / 2)
    return out


def _frac_below(hist: np.ndarray, mm: float) -> float:
    total = int(hist.sum())
    if total == 0:
        return 0.0
    cut = min(int(mm) // BIN_W, hist.size)
    return float(hist[:cut].sum() / total)


def _frac_above(hist: np.ndarray, mm: float) -> float:
    total = int(hist.sum())
    if total == 0:
        return 0.0
    cut = min(int(mm) // BIN_W, hist.size)
    return float(hist[cut:].sum() / total)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/xiongwei.zhao/robotwin_3d")
    p.add_argument(
        "--out",
        default=str(_REPO / "tmp/inspect_robotwin_3d/depth_log_range.json"),
    )
    args = p.parse_args()
    root = Path(args.root)
    tm = json.loads((root / "task_episode_map.json").read_text())
    Lmap = {
        json.loads(line)["episode_index"]: json.loads(line)["length"]
        for line in open(root / "meta/episodes.jsonl")
    }

    hists = {cam: _hist() for cam in CAMS}
    mins = {cam: [] for cam in CAMS}
    maxs = {cam: [] for cam in CAMS}
    n_frames = 0
    for name, spec in sorted(tm["tasks"].items(), key=lambda kv: kv[1]["episode_start"]):
        start = int(spec["episode_start"])
        for ep_i in (start + 20, start + 200):
            T = Lmap[ep_i]
            paths = robotwin_3d_paths(root, ep_i)
            for frac in (0.3, 0.7):
                fi = int(round(frac * (T - 1)))
                for cam in CAMS:
                    mm = decode_depth_mm_frame_map(paths[f"depth_{cam}"], [fi])[fi]
                    valid = mm[mm > 0]
                    if valid.size:
                        mins[cam].append(float(valid.min()))
                        maxs[cam].append(float(valid.max()))
                    _add(hists[cam], mm)
                n_frames += 1

    qs = [0.001, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 0.999]
    per_cam = {}
    for cam in CAMS:
        pct = _pct_from_hist(hists[cam], qs)
        per_cam[cam] = {
            **pct,
            "n_valid": int(hists[cam].sum()),
            "frame_min_median": float(np.median(mins[cam])) if mins[cam] else 0,
            "frame_min_p05": float(np.percentile(mins[cam], 5)) if mins[cam] else 0,
            "frame_max_p95": float(np.percentile(maxs[cam], 95)) if maxs[cam] else 0,
            "frame_max_max": float(np.max(maxs[cam])) if maxs[cam] else 0,
            "lt_30mm": _frac_below(hists[cam], 30),
            "lt_50mm": _frac_below(hists[cam], 50),
            "lt_80mm": _frac_below(hists[cam], 80),
            "lt_150mm": _frac_below(hists[cam], 150),
            "gt_1000mm": _frac_above(hists[cam], 1000),
            "gt_1400mm": _frac_above(hists[cam], 1400),
            "gt_2000mm": _frac_above(hists[cam], 2000),
        }

    all_h = sum(hists[c] for c in CAMS)
    all_pct = _pct_from_hist(all_h, qs)
    candidates = [
        {"name": "v9 body-cam", "z_near_m": 0.15, "z_far_m": 4.0},
        {"name": "current draft", "z_near_m": 0.05, "z_far_m": 2.0},
        {"name": "tight wrist", "z_near_m": 0.03, "z_far_m": 1.6},
        {"name": "conservative", "z_near_m": 0.03, "z_far_m": 2.0},
        {"name": "head-biased", "z_near_m": 0.08, "z_far_m": 1.6},
    ]
    for cand in candidates:
        near_mm = cand["z_near_m"] * 1000
        far_mm = cand["z_far_m"] * 1000
        cand["clip_near"] = {
            cam: _frac_below(hists[cam], near_mm) for cam in CAMS
        }
        cand["clip_far"] = {
            cam: _frac_above(hists[cam], far_mm) for cam in CAMS
        }
        cand["clip_near_all"] = _frac_below(all_h, near_mm)
        cand["clip_far_all"] = _frac_above(all_h, far_mm)
        # log octaves and where typical depths land
        import math

        ln, lf = math.log(cand["z_near_m"]), math.log(cand["z_far_m"])
        span = lf - ln

        def v_of(mm: float) -> float:
            z = max(cand["z_near_m"], min(cand["z_far_m"], mm / 1000.0))
            return 1.0 - (math.log(z) - ln) / span

        cand["v_at"] = {
            "50mm": v_of(50),
            "90mm": v_of(90),
            "300mm": v_of(300),
            "700mm": v_of(700),
            "1000mm": v_of(1000),
            "1400mm": v_of(1400),
        }
        cand["octaves"] = span / math.log(2.0)

    # coarse hist for plot (50 mm bins)
    coarse_w = 50
    coarse = {}
    for cam in (*CAMS, "all"):
        src = all_h if cam == "all" else hists[cam]
        n = BIN_MAX // coarse_w
        c = np.zeros(n, dtype=np.int64)
        for i, v in enumerate(src):
            c[(i * BIN_W) // coarse_w] += int(v)
        total = int(c.sum()) or 1
        coarse[cam] = {
            "edges_mm": [i * coarse_w for i in range(n + 1)],
            "density": [float(x / total) for x in c.tolist()],
        }

    report = {
        "root": str(root),
        "n_decoded_triplets": n_frames,
        "n_tasks": len(tm["tasks"]),
        "sample": "2 episodes/task (clean+rand) x 2 frames (30%,70%) x 3 cams",
        "per_cam": per_cam,
        "all": {
            **all_pct,
            "n_valid": int(all_h.sum()),
            "lt_30mm": _frac_below(all_h, 30),
            "lt_50mm": _frac_below(all_h, 50),
            "lt_80mm": _frac_below(all_h, 80),
            "lt_150mm": _frac_below(all_h, 150),
            "gt_1000mm": _frac_above(all_h, 1000),
            "gt_1400mm": _frac_above(all_h, 1400),
            "gt_2000mm": _frac_above(all_h, 2000),
        },
        "candidates": candidates,
        "hist_50mm": coarse,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_decoded_triplets", "per_cam", "all", "candidates")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
