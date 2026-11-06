#!/usr/bin/env python3
"""10 full-clip videos: RGB | motion overlay. CPU only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(REPO / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import _build_depth_attacher  # noqa: E402
from eval_video_ckpt import load_config  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.periodic_eval.runner import _depth_motion_mask  # noqa: E402

CAT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000/motion_mask_30seq/catalog.json")
OUT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000/motion_mask_10vid")
CFG = REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v6_1.yaml"
# typical / high / empty, 5 agi + 5 ego
PICK = [0, 5, 7, 18, 4, 8, 10, 12, 13, 20]


def _overlay(rgb: np.ndarray, mot: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    tint = np.array([1.0, 0.25, 0.08], dtype=np.float32)
    out[mot] = 0.35 * out[mot] + 0.65 * tint
    return out.clip(0, 1)


def main() -> int:
    catalog = json.loads(CAT.read_text())
    by_i = {int(c["i"]): c for c in catalog}
    clips = [by_i[i] for i in PICK]
    config = load_config(str(CFG))
    attacher = _build_depth_attacher(config)
    z_near = float(getattr(config.depth, "z_near", 0.15))
    z_far = float(getattr(config.depth, "z_far", 4.0))
    mode = str(getattr(config.depth, "mode", "log"))
    seed = int(config.periodic_eval.seed)
    font_p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_p, 22) if Path(font_p).is_file() else ImageFont.load_default()
    OUT.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    for k, clip in enumerate(clips):
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
        zx, vx = decode_to_depth_m(gt, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
        mot = vx & _depth_motion_mask(zx, vx, thresh_m=0.10, rgb=rgb)
        frac = float(mot.sum() / vx.sum().clamp_min(1))
        per_t = mot.reshape(n, -1).float().mean(dim=1).cpu().numpy()
        rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1)
        mot_np = mot.numpy()
        frames = []
        for t in range(n):
            row = np.concatenate([rgb_np[t], _overlay(rgb_np[t], mot_np[t])], axis=1)
            img = Image.fromarray((row * 255).clip(0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, img.width, 30), fill=(255, 255, 255))
            draw.text(
                (8, 4),
                f"{clip['family']} {clip['episode_id'][:42]}  t={t}/{n-1}  frame={per_t[t]:.1%}  clip={frac:.1%}",
                fill=(20, 20, 20),
                font=font,
            )
            frames.append(np.asarray(img))
        name = f"{k:02d}_{clip['family']}_{clip['episode_id'].replace('/', '_')[:48]}.mp4"
        path = OUT / name
        try:
            imageio.mimsave(str(path), frames, fps=5, codec="libx264")
        except Exception:
            imageio.mimsave(str(path.with_suffix(".gif")), frames, fps=5)
            path = path.with_suffix(".gif")
        print(f"wrote {path.name} n={n} mot={frac:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
