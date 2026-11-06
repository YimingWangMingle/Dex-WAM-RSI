#!/usr/bin/env python3
"""Compare motion-mask variants on the 30-clip gallery (Ego camera-motion leak).

variants
  base  : current _depth_motion_mask
  warp  : compensate global camera motion (LK + RANSAC homography) before diff
  warpcc: warp + drop connected components smaller than --min-area
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(REPO / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import _build_depth_attacher  # noqa: E402
from eval_video_ckpt import load_config  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.periodic_eval.runner import _depth_motion_mask  # noqa: E402

CAT = Path(
    "/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000"
    "/motion_mask_30seq/catalog.json"
)
THRESH_M = 0.10
RESIDUAL_M = 0.05
BLUR_KS = 41
RGB_THRESH = 0.05


def _est_homography(a8: np.ndarray, b8: np.ndarray) -> np.ndarray | None:
    """Global frame-to-frame warp a->b, RANSAC-robust to moving foreground."""
    p0 = cv2.goodFeaturesToTrack(a8, maxCorners=800, qualityLevel=0.01, minDistance=8)
    if p0 is None or len(p0) < 40:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a8, b8, p0, None)
    if p1 is None:
        return None
    st = st.reshape(-1).astype(bool)
    if int(st.sum()) < 40:
        return None
    H, inl = cv2.findHomography(p0[st], p1[st], cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < 30:
        return None
    return H


def _residual_gate(dz: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(dz).unsqueeze(0).unsqueeze(0)
    pad = BLUR_KS // 2
    lf = F.avg_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=BLUR_KS, stride=1
    )
    return (x - lf).squeeze().numpy() > RESIDUAL_M


def _mask_warp(z: np.ndarray, valid: np.ndarray, luma: np.ndarray) -> np.ndarray:
    """Same three gates as base, but on camera-motion-compensated frame pairs."""
    n, h, w = z.shape
    out = np.zeros((n, h, w), dtype=bool)
    l8 = (luma * 255).clip(0, 255).astype(np.uint8)
    for t in range(n - 1):
        H = _est_homography(l8[t], l8[t + 1])
        if H is None:
            za, va, la = z[t], valid[t], luma[t]
        else:
            za = cv2.warpPerspective(z[t], H, (w, h), flags=cv2.INTER_NEAREST)
            va = (
                cv2.warpPerspective(
                    valid[t].astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST
                )
                > 0
            )
            la = cv2.warpPerspective(luma[t], H, (w, h), flags=cv2.INTER_NEAREST)
        dz = np.abs(z[t + 1] - za)
        moving = (valid[t + 1] & va) & (dz > THRESH_M) & _residual_gate(dz)
        moving &= np.abs(luma[t + 1] - la) > RGB_THRESH
        out[t + 1] |= moving
        out[t] |= moving
    return out


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for t in range(mask.shape[0]):
        lab, k = ndimage.label(mask[t])
        if k == 0:
            continue
        sizes = ndimage.sum(mask[t], lab, range(1, k + 1))
        keep = np.zeros(k + 1, dtype=bool)
        keep[1:] = sizes >= min_area
        out[t] = keep[lab]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v6_1.yaml"))
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--out", default="/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000/ego_mask_exp")
    p.add_argument("--viz", default="0,3,5,6,7,8,9")
    p.add_argument("--only", default="", help="comma list of catalog i to process")
    args = p.parse_args()

    clips = json.loads(CAT.read_text())
    if args.only.strip():
        keep = {int(v) for v in args.only.split(",") if v.strip()}
        clips = [c for c in clips if int(c["i"]) in keep]
    config = load_config(args.config)
    attacher = _build_depth_attacher(config)
    z_near = float(getattr(config.depth, "z_near", 0.15))
    z_far = float(getattr(config.depth, "z_far", 4.0))
    mode = str(getattr(config.depth, "mode", "log"))
    seed = int(config.periodic_eval.seed)
    viz_ids = {int(v) for v in args.viz.split(",") if v.strip()}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip in clips:
        batch = create_diverse_eval_batch(
            config,
            clip["source"],
            batch_size=1,
            shard_ordinal=int(clip["shard"]),
            seed=seed,
            use_eval_pool=(clip["split"] == "holdout"),
            episode_id=clip.get("episode_id"),
        )
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        rgb = torch.cat(
            [batch["first_frame"][0].float().unsqueeze(0), batch["video_frames"][0].float()], 0
        )
        gt = torch.cat(
            [batch["depth_first"][0].float().unsqueeze(0), batch["depth_frames"][0].float()], 0
        )
        n = min(rgb.shape[0], gt.shape[0])
        rgb, gt = rgb[:n], gt[:n]
        zt, vt = decode_to_depth_m(gt, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
        base = (vt & _depth_motion_mask(zt, vt, thresh_m=THRESH_M, rgb=rgb)).numpy()
        z = zt.numpy()
        valid = vt.numpy()
        luma = rgb.mean(dim=1).numpy()
        warp = _mask_warp(z, valid, luma) & valid
        warpcc = _drop_small(warp, args.min_area) & valid
        den = max(int(valid.sum()), 1)
        rec = {
            "i": clip["i"],
            "family": clip["family"],
            "split": clip["split"],
            "episode_id": clip["episode_id"],
            "base": float(base.sum() / den),
            "warp": float(warp.sum() / den),
            "warpcc": float(warpcc.sum() / den),
        }
        rows.append(rec)
        print(
            f"{rec['i']:02d} {rec['family']} base={rec['base']:.3f} "
            f"warp={rec['warp']:.3f} warpcc={rec['warpcc']:.3f} {clip['episode_id'][:38]}",
            flush=True,
        )

        if int(clip["i"]) in viz_ids:
            from PIL import Image, ImageDraw, ImageFont

            per_t = base.reshape(n, -1).mean(axis=1)
            ts = np.sort(np.argsort(-per_t)[:2])
            rgb_np = rgb.permute(0, 2, 3, 1).numpy().clip(0, 1)
            tint = np.array([1.0, 0.25, 0.08], dtype=np.float32)

            def ov(t, m):
                o = rgb_np[t].copy()
                o[m[t]] = 0.35 * o[m[t]] + 0.65 * tint
                return o.clip(0, 1)

            font_p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font = (
                ImageFont.truetype(font_p, 20)
                if Path(font_p).is_file()
                else ImageFont.load_default()
            )
            tiles = []
            for t in ts:
                t = int(t)
                row = np.concatenate(
                    [rgb_np[t], ov(t, base), ov(t, warp), ov(t, warpcc)], axis=1
                )
                img = Image.fromarray((row * 255).astype(np.uint8))
                d = ImageDraw.Draw(img)
                d.rectangle((0, 0, img.width, 28), fill=(255, 255, 255))
                d.text(
                    (6, 3),
                    f"{clip['family']} {clip['episode_id'][:38]} t={t}  "
                    f"RGB | base {rec['base']:.1%} | warp {rec['warp']:.1%} | warp+cc {rec['warpcc']:.1%}",
                    fill=(20, 20, 20),
                    font=font,
                )
                tiles.append(np.asarray(img))
            sheet = Image.fromarray(np.concatenate(tiles, axis=0))
            sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.Resampling.BILINEAR)
            sheet.save(out_dir / f"cmp_{int(clip['i']):02d}_{clip['family']}.jpg", quality=88)

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for fam in ("agi", "ego"):
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        print(
            f"{fam} n={len(sub)} "
            + "  ".join(
                f"{k}: med={np.median([r[k] for r in sub]):.3f} mean={np.mean([r[k] for r in sub]):.3f}"
                for k in ("base", "warp", "warpcc")
            ),
            flush=True,
        )
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
