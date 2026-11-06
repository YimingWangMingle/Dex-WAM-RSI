#!/usr/bin/env python3
"""Show Motus T compose + shared-log depth vs naive per-view normalize."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare.robotwin_3d import (  # noqa: E402
    CAMS,
    DEFAULT_Z_FAR_M,
    DEFAULT_Z_NEAR_M,
    compose_motus_t_depth_mm,
    compose_motus_t_rgb,
    decode_depth_mm_frame_map,
    pad_depth_mm_to_size,
    pad_rgb_to_size,
    robotwin_3d_paths,
)
from thinker_wm.models.depth_codec import encode_depth_mm  # noqa: E402
from motus_data.datasets.agibot_common import decode_video_frame_map_ffmpeg  # noqa: E402


def _per_view_norm(mm: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mm, dtype=np.float32)
    for y0, y1, x0, x1 in (
        (0, 240, 0, 320),
        (240, 360, 0, 160),
        (240, 360, 160, 320),
    ):
        patch = mm[y0:y1, x0:x1].astype(np.float32)
        valid = patch > 0
        if not valid.any():
            continue
        lo, hi = float(patch[valid].min()), float(patch[valid].max())
        if hi <= lo:
            out[y0:y1, x0:x1][valid] = 1.0
        else:
            out[y0:y1, x0:x1] = np.where(valid, (hi - patch) / (hi - lo), 0.0)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/xiongwei.zhao/robotwin_3d")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--out",
        default=str(_REPO / "tmp/inspect_robotwin_3d/compose_shared_vs_perview.png"),
    )
    args = p.parse_args()
    root = Path(args.root)
    L = None
    for line in open(root / "meta/episodes.jsonl"):
        rec = json.loads(line)
        if rec["episode_index"] == args.episode:
            L = int(rec["length"])
            break
    if L is None:
        raise SystemExit(f"episode {args.episode} not in episodes.jsonl")
    fi = L // 2
    paths = robotwin_3d_paths(root, args.episode)
    rgb = {
        cam: decode_video_frame_map_ffmpeg(str(paths[f"rgb_{cam}"]), [fi])[fi]
        for cam in CAMS
    }
    depth = {
        cam: decode_depth_mm_frame_map(paths[f"depth_{cam}"], [fi])[fi] for cam in CAMS
    }
    rgb_t = compose_motus_t_rgb(rgb["cam_high"], rgb["cam_left_wrist"], rgb["cam_right_wrist"])
    mm_t = compose_motus_t_depth_mm(
        depth["cam_high"], depth["cam_left_wrist"], depth["cam_right_wrist"]
    )
    rgb_pad = pad_rgb_to_size(rgb_t[None], (480, 640))[0]
    mm_pad = pad_depth_mm_to_size(mm_t[None], (480, 640))[0]
    shared = encode_depth_mm(
        torch.from_numpy(mm_pad.astype(np.int32)),
        z_near=DEFAULT_Z_NEAR_M,
        z_far=DEFAULT_Z_FAR_M,
        mode="log",
        invalid_to="far",
        out_range="01",
    )[0].numpy()
    per_view = _per_view_norm(mm_t)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), dpi=140)
    fig.suptitle(
        f"episode_{args.episode:06d} f{fi}/{L}  |  keep metric mm, one log curve "
        f"[{DEFAULT_Z_NEAR_M*1000:.0f}, {DEFAULT_Z_FAR_M*1000:.0f}] mm",
        fontsize=11,
    )
    axes[0, 0].imshow(rgb_t)
    axes[0, 0].set_title("RGB Motus T 360×320")
    axes[0, 1].imshow(np.ma.masked_where(mm_t == 0, mm_t), cmap="turbo", vmin=50, vmax=1400)
    axes[0, 1].set_title("Depth mm (shared metric)")
    axes[0, 2].imshow(per_view, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("WRONG: per-view min-max")
    axes[1, 0].imshow(rgb_pad)
    axes[1, 0].set_title("RGB pad 480×640")
    axes[1, 1].imshow(shared, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Shared log encode (near=bright)")
    axes[1, 2].imshow(np.ma.masked_where(mm_pad == 0, mm_pad), cmap="turbo", vmin=50, vmax=1400)
    axes[1, 2].set_title("Padded mm (0=invalid bars)")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
