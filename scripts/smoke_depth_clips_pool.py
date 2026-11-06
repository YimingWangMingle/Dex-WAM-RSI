#!/usr/bin/env python3
"""Validate the depth_clips SamplePool against the real RGB catalogs.

For each (dataset, depth_root) pair: resolve the frozen RGB EpisodeRepository via
releases.yaml, build the depth_clips pool, and report how many depth clips
reconstruct into aligned motus selections (video_frame_indices == stored clip).
No model / GPU needed.

    python scripts/smoke_depth_clips_pool.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/libs" / "motus-data-factory"))

from motus_data.indexed_pipeline import open_dataset  # noqa: E402
from thinker_wm.data_prepare.depth_clips import DepthClipsPool, _episode_in_split  # noqa: E402
from thinker_wm.models.depth_reader import load_depth_selections  # noqa: E402

NUM_VIDEO_FRAMES = 48
PAIRS = [
    ("agibot_beta_dexterous_hand", "/data/xiongwei.zhao/agibot_da3_depth/v1"),
    ("egoscale_second1700h_batch02", "/data/xiongwei.zhao/egoscale_ffs_depth/v1"),
]


def main() -> int:
    ok = True
    for dataset, root in PAIRS:
        print(f"\n=== {dataset}  root={root} ===")
        repo = open_dataset(dataset)
        pool = DepthClipsPool(
            repo, roots=[root], num_video_frames=NUM_VIDEO_FRAMES, prod_keep_only=True,
        )
        n_sel = len(pool)
        n_eps = len(pool.episode_groups)
        print(f"  pool_id={pool.pool_id}  selections={n_sel}  episodes={n_eps}")
        # 抽一条 selection，确认 video_frame_indices 为 49 帧、stride 6
        sel = pool.selections[0]
        vfi = [int(i) for i in sel["video_frame_indices"]]
        stride = vfi[1] - vfi[0]
        print(f"  sample selection: frames={len(vfi)} stride={stride} "
              f"first5={vfi[:5]} episode={sel.get('episode_id')}")
        if n_sel == 0:
            print("  FAIL: no aligned selections")
            ok = False
        if len(vfi) != NUM_VIDEO_FRAMES + 1:
            print(f"  FAIL: expected {NUM_VIDEO_FRAMES + 1} frames")
            ok = False

        # episode 级 holdout 切分：对**单次快照**的 episode 集验证
        # train ∪ holdout == all 且不相交、比例≈target。用快照而非重建
        # 池（depth dump 正在并发追加 depth_selections，重复读文件会漂移）。
        frac = 0.1
        eps = sorted(load_depth_selections([root], prod_keep_only=True).keys())
        te = [e for e in eps if _episode_in_split(e, holdout_fraction=frac, split="train", holdout_seed=20260824)]
        he = [e for e in eps if _episode_in_split(e, holdout_fraction=frac, split="holdout", holdout_seed=20260824)]
        overlap = set(te) & set(he)
        union_ok = (set(te) | set(he)) == set(eps)
        hold_frac = len(he) / max(1, len(eps))
        print(f"  split(frac={frac}, snapshot eps={len(eps)}): train={len(te)} "
              f"holdout={len(he)} overlap={len(overlap)} union_eq_all={union_ok} "
              f"holdout_ratio={hold_frac:.3f}")
        if overlap or not union_ok:
            print("  FAIL: split not a clean partition")
            ok = False
        if not (0.05 <= hold_frac <= 0.16):
            print(f"  WARN: holdout ratio {hold_frac:.3f} off target ~0.10")

    print("\nALL_DEPTH_CLIPS_POOL_OK" if ok else "\nDEPTH_CLIPS_POOL_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
