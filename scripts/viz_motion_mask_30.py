#!/usr/bin/env python3
"""Dump ~30 GT frames: RGB | depth | motion overlay. CPU only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(REPO / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import _build_depth_attacher  # noqa: E402
from eval_video_ckpt import load_config  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.periodic_eval.runner import _depth_motion_mask  # noqa: E402


def _turbo(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from matplotlib import colormaps

    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = colormaps["turbo"](t)[..., :3].astype(np.float32)
    rgb[np.isnan(z)] = 1.0
    return rgb


def _overlay(rgb: np.ndarray, mot: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    tint = np.array([1.0, 0.25, 0.08], dtype=np.float32)
    out[mot] = 0.35 * out[mot] + 0.65 * tint
    return out.clip(0, 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v6_1.yaml"))
    p.add_argument("--clips", default=str(REPO / "scripts/family_holdout_18_v5.json"))
    p.add_argument("--clip-index", type=int, default=0)
    p.add_argument("--n-show", type=int, default=30)
    p.add_argument("--out", default="/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000/motion_mask_30")
    args = p.parse_args()

    clips = json.loads(Path(args.clips).read_text())
    clip = clips[int(args.clip_index)]
    config = load_config(args.config)
    attacher = _build_depth_attacher(config)
    batch = create_diverse_eval_batch(
        config,
        clip["source"],
        batch_size=1,
        shard_ordinal=int(clip["shard"]),
        seed=int(config.periodic_eval.seed),
        use_eval_pool=(clip["split"] == "holdout"),
        episode_id=clip.get("episode_id"),
    )
    if attacher is not None:
        batch = dict(attacher.attach(batch))

    rgb = torch.cat(
        [batch["first_frame"][0].float().unsqueeze(0), batch["video_frames"][0].float()],
        dim=0,
    )
    gt = torch.cat(
        [batch["depth_first"][0].float().unsqueeze(0), batch["depth_frames"][0].float()],
        dim=0,
    )
    n = min(rgb.shape[0], gt.shape[0])
    rgb, gt = rgb[:n], gt[:n]
    z_near = float(getattr(config.depth, "z_near", 0.15))
    z_far = float(getattr(config.depth, "z_far", 4.0))
    mode = str(getattr(config.depth, "mode", "log"))
    zx, vx = decode_to_depth_m(gt, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
    mot = _depth_motion_mask(zx, vx, thresh_m=0.10, rgb=rgb)
    mot = vx & mot
    frac = float(mot.sum() / vx.sum().clamp_min(1))
    per_t = mot.reshape(n, -1).float().mean(dim=1).cpu().numpy()

    n_show = min(int(args.n_show), n)
    if n_show >= n:
        show = np.arange(n)
    else:
        show = np.unique(np.round(np.linspace(0, n - 1, n_show)).astype(int))

    rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1)
    z = np.where(vx.numpy(), zx.numpy(), np.nan)
    mot_np = mot.numpy()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in show:
        row = np.concatenate(
            [
                rgb_np[t],
                _turbo(z[t], 0.15, 2.5),
                _overlay(rgb_np[t], mot_np[t]),
            ],
            axis=1,
        )
        frames.append((row * 255).clip(0, 255).astype(np.uint8))
        print(f"t={int(t):02d} motion_frac={per_t[t]:.3f}", flush=True)

    from PIL import Image, ImageDraw, ImageFont

    eid = clip.get("episode_id", "")
    font_p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_p, 22) if Path(font_p).is_file() else ImageFont.load_default()
    labeled = []
    for t, fr in zip(show, frames):
        img = Image.fromarray(fr)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 32), fill=(255, 255, 255))
        draw.text(
            (8, 4),
            f"{eid}  t={int(t)}/{n-1}  pix={per_t[t]:.1%}  clip={frac:.1%}   RGB | GT depth | mask",
            fill=(20, 20, 20),
            font=font,
        )
        labeled.append(np.asarray(img))
        Image.fromarray(fr).save(out_dir / f"t{int(t):02d}.png")

    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(out_dir / "motion_mask_30.mp4"), labeled, fps=5, codec="libx264")
    except Exception:
        import imageio.v2 as imageio

        imageio.mimsave(str(out_dir / "motion_mask_30.gif"), labeled, fps=5)

    # 6x5 contact sheet if we have ~30
    h, w = labeled[0].shape[:2]
    cols = 5
    rows = int(np.ceil(len(labeled) / cols))
    sheet = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    for i, fr in enumerate(labeled):
        r, c = divmod(i, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = fr
    # downsample for a readable sheet
    sheet_img = Image.fromarray(sheet)
    sheet_img = sheet_img.resize((sheet_img.width // 4, sheet_img.height // 4), Image.Resampling.BILINEAR)
    sheet_img.save(out_dir / "contact_sheet.jpg", quality=85)
    print(f"wrote {out_dir} n={len(show)} clip_motion_frac={frac:.3f} eid={eid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
