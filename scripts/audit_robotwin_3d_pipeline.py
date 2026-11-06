#!/usr/bin/env python3
"""Data-side readiness audit for robotwin_3d: RGB / depth / action through the
real training pipeline (indexed_mix + DepthAttacher), not a hand-rolled reader.

Canvas contract checked here: the Motus T (360x320) is aspect-preserving
upscaled to fit 480x640 -> 480x426 content, pillarbox pad on the left/right.
RGB and depth must land in the exact same box, and depth pad must be 0 (invalid).

  python scripts/audit_robotwin_3d_pipeline.py --batches 2 --batch-size 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare import robotwin_3d as _robotwin_3d  # noqa: E402,F401
from thinker_wm.data_prepare.depth_attach import DepthAttacher  # noqa: E402
from thinker_wm.data_prepare.factory import (  # noqa: E402
    create_eval_data_pipeline,
    create_training_data_pipeline,
)
from thinker_wm.models.action_stats import get_stats  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402

FAILS: list[str] = []
# Motus T: head 240 of 360 canvas rows, wrists the bottom 120.
HEAD_FRAC = 240.0 / 360.0


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def _content_box(mask2d: torch.Tensor) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1) half-open bounds of the nonzero region."""
    rows = (mask2d.sum(dim=1) > 0).nonzero().flatten()
    cols = (mask2d.sum(dim=0) > 0).nonzero().flatten()
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def audit_batch(batch, cfg, tag: str) -> None:
    video = batch["video_frames"]
    first = batch["first_frame"]
    B = int(video.shape[0])
    H, W = int(video.shape[-2]), int(video.shape[-1])
    n_frames = int(cfg.common.num_video_frames)

    print(f"\n--- {tag}: B={B} T={int(video.shape[1])} HxW={H}x{W} ---")
    check(f"{tag} rgb shape",
          tuple(video.shape) == (B, n_frames, 3, int(cfg.common.video_height), int(cfg.common.video_width)),
          f"{tuple(video.shape)}")
    check(f"{tag} rgb dtype/range",
          video.dtype == torch.float32 and float(video.min()) >= 0.0 and float(video.max()) <= 1.0,
          f"dtype={video.dtype} range=[{float(video.min()):.3f}, {float(video.max()):.3f}]")
    check(f"{tag} rgb finite", bool(torch.isfinite(video).all()) and bool(torch.isfinite(first).all()))
    check(f"{tag} first_frame shape", tuple(first.shape) == (B, 3, H, W), f"{tuple(first.shape)}")

    # ---- depth ----
    dframes = batch.get("depth_frames")
    present = batch.get("depth_present")
    check(f"{tag} depth_present all True", present is not None and bool(present.all()),
          f"{present.tolist() if present is not None else None}")
    check(f"{tag} depth shape mirrors rgb",
          dframes is not None and tuple(dframes.shape) == tuple(video.shape),
          f"{tuple(dframes.shape) if dframes is not None else None}")

    # ---- pillarbox geometry, shared by RGB and depth ----
    rgb_occ = (video[:, :, 0].abs() > 1e-6).float().sum(dim=(0, 1))
    r0, r1, c0, c1 = _content_box(rgb_occ)
    check(f"{tag} rgb content box is 480x426 pillarbox",
          (r1 - r0, c1 - c0) == (480, 426) and c0 == (W - 426) // 2,
          f"rows[{r0},{r1}) cols[{c0},{c1}) size={(r1 - r0)}x{(c1 - c0)}")
    check(f"{tag} rgb pillarbox bars are 0",
          float(video[..., :c0].abs().max()) == 0.0 and float(video[..., c1:].abs().max()) == 0.0,
          f"left={float(video[..., :c0].abs().max()):.4f} right={float(video[..., c1:].abs().max()):.4f}")
    check(f"{tag} rgb content non-trivial", float(video[..., c0:c1].std()) > 0.02,
          f"std={float(video[..., c0:c1].std()):.4f}")

    if dframes is not None:
        check(f"{tag} depth range [0,1]",
              float(dframes.min()) >= 0.0 and float(dframes.max()) <= 1.0,
              f"[{float(dframes.min()):.4f}, {float(dframes.max()):.4f}]")
        check(f"{tag} depth finite", bool(torch.isfinite(dframes).all()))
        check(f"{tag} depth 3 channels identical",
              float((dframes[:, :, 0] - dframes[:, :, 1]).abs().max()) == 0.0)
        dep_occ = (dframes[:, :, 0] > 0).float().sum(dim=(0, 1))
        dr0, dr1, dc0, dc1 = _content_box(dep_occ)
        check(f"{tag} depth shares the rgb content box", (dr0, dr1, dc0, dc1) == (r0, r1, c0, c1),
              f"depth rows[{dr0},{dr1}) cols[{dc0},{dc1})")
        check(f"{tag} depth pillarbox bars are 0 (invalid)",
              float(dframes[..., :c0].abs().max()) == 0.0 and float(dframes[..., c1:].abs().max()) == 0.0,
              f"left={float(dframes[..., :c0].abs().max()):.4f}")
        dcontent = dframes[..., r0:r1, c0:c1]
        valid = float((dcontent[:, :, 0] > 0).float().mean())
        check(f"{tag} depth valid ratio in content > 0.9", valid > 0.9, f"valid={valid:.4f}")
        z_m, vmask = decode_to_depth_m(
            dcontent, z_near=float(cfg.depth.z_near), z_far=float(cfg.depth.z_far),
            mode=str(cfg.depth.mode), in_range="01",
        )
        zv = z_m[vmask]
        check(f"{tag} decoded depth within [z_near, z_far]",
              float(zv.min()) >= float(cfg.depth.z_near) - 1e-3
              and float(zv.max()) <= float(cfg.depth.z_far) + 1e-3,
              f"[{float(zv.min()):.3f}, {float(zv.max()):.3f}]m median={float(zv.median()):.3f}m")
        split = r0 + int(round((r1 - r0) * HEAD_FRAC))
        head, wrist = z_m[..., : split - r0, :], z_m[..., split - r0 :, :]
        hm, wm = float(head[head > 0].median()), float(wrist[wrist > 0].median())
        check(f"{tag} head farther than wrists", hm > wm, f"head={hm:.3f}m wrist={wm:.3f}m")

    # ---- action / state (motus_model_batch_v1 keys) ----
    action = batch.get("action_sequence")
    state = batch.get("initial_state")
    check(f"{tag} action_sequence shape",
          action is not None and tuple(action.shape) == (B, n_frames, 14),
          f"{tuple(action.shape) if action is not None else None}")
    check(f"{tag} initial_state shape",
          state is not None and tuple(state.shape) == (B, 14),
          f"{tuple(state.shape) if state is not None else None}")
    check(f"{tag} action/state marked valid",
          bool(batch["action_sequence_valid"].all()) and bool(batch["initial_state_valid"].all()))
    if action is not None and state is not None:
        check(f"{tag} action finite", bool(torch.isfinite(action).all()) and bool(torch.isfinite(state).all()))
        mean, std = get_stats("aloha")
        z = (action - mean) / std
        check(f"{tag} action z within +-6", float(z.abs().max()) < 6.0,
              f"z_absmax={float(z.abs().max()):.2f} z_mean={float(z.mean()):.3f} z_std={float(z.std()):.3f}")
        grip = action[..., [6, 13]]
        check(f"{tag} gripper in [0,1]",
              float(grip.min()) >= -1e-4 and float(grip.max()) <= 1.0 + 1e-4,
              f"[{float(grip.min()):.3f}, {float(grip.max()):.3f}]")
        d0 = float((action[:, 0] - state).abs().max())
        check(f"{tag} action[0] close to initial_state (absolute qpos)", d0 < 0.5, f"max|d|={d0:.4f}")
        step = float((action[:, 1:] - action[:, :-1]).abs().max())
        check(f"{tag} action step continuity (<0.5 rad/frame)", step < 0.5, f"max|step|={step:.4f}")

    # ---- bookkeeping ----
    fi = batch.get("frame_indices")
    if fi is not None:
        check(f"{tag} frame_indices len == T+1", {len(x) for x in fi} == {n_frames + 1},
              f"{ {len(x) for x in fi} }")
        check(f"{tag} frame_indices consecutive",
              all(list(x) == list(range(x[0], x[0] + len(x))) for x in fi))
    srcs = batch.get("source_datasets")
    check(f"{tag} source is robotwin_3d",
          srcs is not None and set(map(str, srcs)) == {"robotwin_3d"}, f"{set(map(str, srcs)) if srcs else None}")
    rn = batch.get("robot_names")
    check(f"{tag} robot_names registered in ACTION_STATS",
          rn is not None and all(get_stats(str(n)) is not None for n in rn),
          f"{set(map(str, rn)) if rn else None}")
    check(f"{tag} ratio is 1 (video:action)",
          all(int(r) == 1 for r in batch.get("action_video_fps_ratios", [1])))
    txt = batch.get("t5_input_messages")
    check(f"{tag} instruction text present", txt is not None and all(len(str(t)) > 5 for t in txt),
          f"{str(txt[0])[:60] if txt else None}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_REPO / "third_party/Thinker-WM/configs/robotwin_3d_rgbd.yaml"))
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "training.batch_size", int(args.batch_size), force_add=True)
    OmegaConf.update(cfg, "system.num_workers", 0, force_add=True)
    OmegaConf.update(cfg, "system.ready_batch_queue_enabled", False, force_add=True)

    attacher = DepthAttacher(
        [],
        z_near=float(cfg.depth.z_near),
        z_far=float(cfg.depth.z_far),
        mode=str(cfg.depth.mode),
        invalid_to=str(cfg.depth.invalid_to),
        prod_keep_only=bool(cfg.depth.prod_keep_only),
        robotwin_3d_root=str(cfg.depth.robotwin_3d_root),
    )

    for tag, builder in (("train", create_training_data_pipeline), ("eval", create_eval_data_pipeline)):
        pipeline = builder(cfg, rank=0, world_size=1, seed=0)
        with pipeline:
            for i in range(int(args.batches)):
                batch, _stats = pipeline.next()
                batch = attacher.attach(batch)
                audit_batch(batch, cfg, f"{tag}[{i}]")

    print()
    if FAILS:
        print(f"AUDIT_FAIL {len(FAILS)} checks failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("AUDIT_OK all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
