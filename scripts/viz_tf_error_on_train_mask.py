#!/usr/bin/env python3
"""Overlay the training 3-gate motion mask on V6.1@3000 TF 3-panel videos.

Does not open AgiBot observation tars. GT depth comes from DepthAttacher
(DA3 sidecar). Mask is models.depth_motion.depth_motion_mask (no min_area,
no pose). Panel error is abs-diff of the colorized pred vs GT panels — not
metric AbsRel.

  python scripts/viz_tf_error_on_train_mask.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
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
from thinker_wm.models.depth_motion import depth_motion_mask  # noqa: E402

CFG = REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v7.yaml"
CLIPS = REPO / "scripts/family_holdout_18_v5.json"
TF_ROOT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000")
OUT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/gt_quality_motion_mask/viz")
HOLDOUT_DA3 = "/data/xiongwei.zhao/agibot_da3_depth/holdout_v1"
TINT = np.array([1.0, 0.25, 0.08], dtype=np.float32)
CONTOUR_RGB = (0, 255, 255)


def _eid_short(eid: str) -> str:
    parts = str(eid).split(":")
    return "_".join(parts[-2:]) if len(parts) >= 2 else str(eid).replace(":", "_")


def _tf_dir(idx: int, eid: str) -> Path:
    safe = str(eid).replace(":", "_")
    return TF_ROOT / f"{idx:02d}_holdout_{safe}"


def _read_tf_panels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import imageio.v2 as imageio

    frames = np.stack(imageio.mimread(str(path), memtest=False), 0)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected tf video shape {frames.shape} in {path}")
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if w % 3 != 0:
        raise ValueError(f"cannot infer 3-panel width from {frames.shape}")
    pw = w // 3
    rgb = frames[:, :, :pw]
    pred = frames[:, :, pw : 2 * pw]
    gt = frames[:, :, 2 * pw :]
    return rgb, pred, gt


def _luma_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _overlay(rgb01: np.ndarray, mot: np.ndarray) -> np.ndarray:
    out = rgb01.copy()
    out[mot] = 0.35 * out[mot] + 0.65 * TINT
    return out.clip(0, 1)


def _contours(img_u8: np.ndarray, mot: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(img_u8.copy())
    if not mot.any():
        return out
    cnts, _ = cv2.findContours(
        mot.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, cnts, -1, CONTOUR_RGB, 1)
    return out


def _hot(err01: np.ndarray) -> np.ndarray:
    u8 = (np.clip(err01, 0, 1) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _label_row(img_u8: np.ndarray, text: str, font) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(img_u8)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, 26), fill=(255, 255, 255))
    d.text((6, 3), text, fill=(20, 20, 20), font=font)
    return np.asarray(img)


def main() -> int:
    clips = [
        c
        for c in json.loads(CLIPS.read_text())
        if c["split"] == "holdout" and str(c["episode_id"]).startswith("agibot")
    ]
    index = json.loads((TF_ROOT / "index.json").read_text())
    by_eid = {str(r["episode_id"]): i for i, r in enumerate(index)}

    config = load_config(str(CFG))
    roots = list(config.depth.roots)
    if HOLDOUT_DA3 not in [str(r) for r in roots]:
        roots.append(HOLDOUT_DA3)
        config.depth.roots = roots
    attacher = _build_depth_attacher(config)
    z_near = float(getattr(config.depth, "z_near", 0.15))
    z_far = float(getattr(config.depth, "z_far", 4.0))
    mode = str(getattr(config.depth, "mode", "log"))
    seed = int(config.periodic_eval.seed)

    from PIL import Image, ImageDraw, ImageFont

    font_p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_p, 16) if Path(font_p).is_file() else ImageFont.load_default()
    import imageio.v2 as imageio

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []

    for clip in clips:
        eid = str(clip["episode_id"])
        idx = int(by_eid.get(eid, len(summary)))
        tf_path = _tf_dir(idx, eid) / "tf_rgb_preddepth_gtdepth.mp4"
        if not tf_path.is_file():
            print(f"skip {eid}: missing {tf_path}", flush=True)
            continue

        print(f"load {idx:02d} {eid}", flush=True)
        batch = create_diverse_eval_batch(
            config,
            clip["source"],
            batch_size=1,
            shard_ordinal=int(clip["shard"]),
            seed=seed,
            use_eval_pool=True,
            episode_id=eid,
        )
        batch = dict(attacher.attach(batch))
        present = batch.get("depth_present")
        if present is not None and not bool(present.reshape(-1)[0].item()):
            print(f"skip {eid}: depth_present=False", flush=True)
            continue

        rgb = torch.cat(
            [batch["first_frame"][0].float().unsqueeze(0), batch["video_frames"][0].float()], 0
        )
        gt01 = torch.cat(
            [batch["depth_first"][0].float().unsqueeze(0), batch["depth_frames"][0].float()], 0
        )
        n_full = min(int(rgb.shape[0]), int(gt01.shape[0]))
        rgb, gt01 = rgb[:n_full], gt01[:n_full]
        depth_m, valid_t = decode_to_depth_m(
            gt01, z_near=z_near, z_far=z_far, mode=mode, in_range="01"
        )
        mask_full = depth_motion_mask(
            depth_m.unsqueeze(0), valid_t.unsqueeze(0), rgb=rgb.unsqueeze(0)
        )[0]
        # TF mp4 is video_frames only (no first_frame) — align mask[1:]
        mask_np = mask_full[1:].numpy().astype(bool)
        valid_np = valid_t[1:].numpy().astype(bool)

        tf_rgb, tf_pred, tf_gt = _read_tf_panels(tf_path)
        n = min(int(tf_rgb.shape[0]), int(mask_np.shape[0]), int(valid_np.shape[0]))
        if tf_rgb.shape[1] != mask_np.shape[1] or tf_rgb.shape[2] != mask_np.shape[2]:
            mh, mw = int(tf_rgb.shape[1]), int(tf_rgb.shape[2])
            mask_r = np.empty((n, mh, mw), dtype=bool)
            valid_r = np.empty((n, mh, mw), dtype=bool)
            for t in range(n):
                mask_r[t] = cv2.resize(
                    mask_np[t].astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                valid_r[t] = cv2.resize(
                    valid_np[t].astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            mask_np, valid_np = mask_r, valid_r
        else:
            mask_np, valid_np = mask_np[:n], valid_np[:n]
        tf_rgb, tf_pred, tf_gt = tf_rgb[:n], tf_pred[:n], tf_gt[:n]

        err = np.abs(tf_pred.astype(np.float32) - tf_gt.astype(np.float32))
        err_luma = _luma_u8(err)
        p98 = float(np.percentile(err_luma, 98))
        err01 = np.clip(err_luma / max(p98, 1e-6), 0.0, 1.0)

        look = valid_np
        den = max(int(look.sum()), 1)
        mask_frac = float((mask_np & look).sum() / den)
        inm = look & mask_np
        outm = look & ~mask_np
        err_in = float(err01[inm].mean()) if inm.any() else float("nan")
        err_out = float(err01[outm].mean()) if outm.any() else float("nan")
        err_in_raw = float(err_luma[inm].mean()) if inm.any() else float("nan")
        err_out_raw = float(err_luma[outm].mean()) if outm.any() else float("nan")
        hi = look & (err01 >= 0.90)
        hi_in_mask = float((hi & mask_np).sum() / max(int(hi.sum()), 1)) if hi.any() else float("nan")
        concentrates = bool(
            inm.any()
            and outm.any()
            and err_in > err_out
            and (not np.isnan(hi_in_mask) and hi_in_mask > mask_frac)
        )
        rec = {
            "idx": idx,
            "episode_id": eid,
            "source": clip["source"],
            "n_frames": n,
            "panel_w": int(tf_rgb.shape[2]),
            "mask_frac": mask_frac,
            "error_in": err_in,
            "error_out": err_out,
            "error_in_raw": err_in_raw,
            "error_out_raw": err_out_raw,
            "error_ratio": (err_in / err_out) if (err_out and err_out > 0) else float("nan"),
            "high_err_in_mask_frac": hi_in_mask,
            "high_error_concentrates_in_mask": concentrates,
            "p98_panel_luma": p98,
            "n_valid": int(look.sum()),
            "n_in": int(inm.sum()),
            "n_out": int(outm.sum()),
            "error_note": "panel Δ (colorized pred vs GT), per-clip p98-normalized luma; not AbsRel",
            "tf_mp4": str(tf_path),
        }
        summary.append(rec)
        print(
            f"  mask_frac={mask_frac:.3f} err_in={err_in:.3f} err_out={err_out:.3f} "
            f"ratio={rec['error_ratio']:.2f} hi_in_mask={hi_in_mask:.3f} "
            f"concentrates={concentrates}",
            flush=True,
        )

        frames = []
        for t in range(n):
            rgb01 = (tf_rgb[t].astype(np.float32) / 255.0).clip(0, 1)
            overlay = _overlay(rgb01, mask_np[t])
            pred_c = _contours(tf_pred[t], mask_np[t])
            heat = _contours(_hot(err01[t]), mask_np[t])
            top = np.concatenate(
                [(rgb01 * 255).astype(np.uint8), (overlay * 255).clip(0, 255).astype(np.uint8)],
                axis=1,
            )
            bot = np.concatenate([pred_c, heat], axis=1)
            row = np.concatenate([top, bot], axis=0)
            img = _label_row(
                row,
                f"{eid} t={t}/{n - 1}  RGB | RGB+train 3-gate  |  pred depth+contour | "
                f"panel Δ (colorized pred vs GT) HOT  mask={mask_frac:.1%}",
                font,
            )
            frames.append(img)

        name = f"{idx:02d}_{_eid_short(eid)}.mp4"
        path = OUT / name
        try:
            imageio.mimsave(str(path), frames, fps=6, codec="libx264")
        except Exception:
            path = path.with_suffix(".gif")
            imageio.mimsave(str(path), frames, fps=6)
        rec["video"] = str(path)
        print(f"  wrote {path}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT / 'summary.json'} n={len(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
