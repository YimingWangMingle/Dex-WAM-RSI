#!/usr/bin/env python3
"""Visualize GT-depth frame-diff motion masks on the family-holdout 18 clips.

Reloads the same batches as eval (source + shard + seed), no GPU.
Writes a 18-row mosaic: RGB | GT depth | |Δz| | motion overlay.

  python scripts/viz_eval_motion_masks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(REPO / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import _build_depth_attacher  # noqa: E402
from eval_video_ckpt import load_config  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.periodic_eval.runner import _depth_motion_mask, _depth_motion_mask_raw  # noqa: E402
from publish_eval_to_monitor import _meta_first  # noqa: E402

META_ROOT = Path(
    "/data/xiongwei.zhao/thinkerwam-depth/motus_monitor_data/runs"
    "/depth_ablation_v5_step20k_motion_baseline/eval/step_0020000"
)
CFG = REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v5.yaml"
OUT_DIR = REPO / "tmp/depth_viz_gallery/motion_mask_18"
THRESH_M = 0.10
Z_LO, Z_HI = 0.15, 2.5
DZ_HI = 0.30
N_SHOW = 4


def _turbo(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from matplotlib import colormaps

    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = colormaps["turbo"](t)[..., :3].astype(np.float32)
    rgb[np.isnan(z)] = 1.0
    return rgb


def _gray(d: np.ndarray, hi: float) -> np.ndarray:
    g = np.clip(d / max(hi, 1e-6), 0.0, 1.0).astype(np.float32)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[np.isnan(d)] = 1.0
    return rgb


def _hcat(imgs: list[np.ndarray], gap: int = 2) -> np.ndarray:
    h = imgs[0].shape[0]
    bar = np.ones((h, gap, 3), dtype=np.float32)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.concatenate([out, bar, im], axis=1)
    return out


def _vcat(imgs: list[np.ndarray], gap: int = 6) -> np.ndarray:
    w = imgs[0].shape[1]
    bar = np.ones((gap, w, 3), dtype=np.float32)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.concatenate([out, bar, im], axis=0)
    return out


def _overlay(rgb: np.ndarray, mot: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    tint = np.array([1.0, 0.25, 0.08], dtype=np.float32)
    out[mot] = 0.35 * out[mot] + 0.65 * tint
    return out.clip(0, 1)


def main() -> int:
    metas = sorted(META_ROOT.glob("rank*/sample_*/meta.json"))
    if not metas:
        print(f"no metas under {META_ROOT}")
        return 1
    config = load_config(str(CFG))
    OmegaConf.set_struct(config, False)
    attacher = _build_depth_attacher(config)
    depth_cfg = config.depth
    codec = {
        "z_near": float(getattr(depth_cfg, "z_near", 0.1)),
        "z_far": float(getattr(depth_cfg, "z_far", 10.0)),
        "mode": str(getattr(depth_cfg, "mode", "log")),
        "in_range": "01",
    }
    seed = int(config.periodic_eval.seed)
    rows = []
    catalog = []
    for path in metas:
        payload = json.loads(path.read_text())
        source = payload["source"]
        split = payload["split"]
        shard = int(payload["shard_ordinal"])
        batch = create_diverse_eval_batch(
            config,
            source,
            batch_size=1,
            shard_ordinal=shard,
            seed=seed,
            use_eval_pool=(split == "holdout"),
        )
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        eid = str(_meta_first(batch, "episode_ids") or payload.get("episode_id") or "")
        rgb = batch["video_frames"][0].float()  # [T,3,H,W] 0-1
        gt_d = batch["depth_frames"][0].float()
        n = min(rgb.shape[0], gt_d.shape[0])
        rgb, gt_d = rgb[:n], gt_d[:n]
        zx, vx = decode_to_depth_m(gt_d, **codec)
        motion_raw = _depth_motion_mask_raw(zx, vx, THRESH_M)
        motion = _depth_motion_mask(zx, vx, THRESH_M, rgb=rgb)
        both = vx
        mot = both & motion
        mot_raw = both & motion_raw
        frac = float(mot.sum() / both.sum().clamp_min(1))
        frac_raw = float(mot_raw.sum() / both.sum().clamp_min(1))
        per_t = mot_raw.reshape(n, -1).sum(dim=1).cpu().numpy()
        show = np.sort(np.argsort(-per_t)[:N_SHOW])
        rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy()
        z = np.where(vx.numpy(), zx.numpy(), np.nan)
        dz = np.full(zx.shape, np.nan, dtype=np.float32)
        if n >= 2:
            d = (zx[1:] - zx[:-1]).abs().numpy()
            ok = (vx[1:] & vx[:-1]).numpy()
            dz[1:] = np.where(ok, d, np.nan)
        mot_np = mot.numpy()
        mot_raw_np = mot_raw.numpy()
        tiles = []
        for t in show:
            tiles.append(
                _hcat(
                    [
                        rgb_np[t].clip(0, 1),
                        _turbo(z[t], Z_LO, Z_HI),
                        _gray(dz[t], DZ_HI),
                        _overlay(rgb_np[t].clip(0, 1), mot_raw_np[t]),
                        _overlay(rgb_np[t].clip(0, 1), mot_np[t]),
                    ]
                )
            )
        body = _vcat(tiles, gap=2)
        fam = "agi" if "agibot" in source else "ego"
        label = (
            f"{fam}_{split}  {eid}  raw={frac_raw:.1%}  filtered={frac:.1%}  "
            f"t={list(map(int, show))}"
        )
        catalog.append(
            {
                "source": source,
                "split": split,
                "episode_id": eid,
                "shard": shard,
                "motion_frac_raw": frac_raw,
                "motion_frac": frac,
                "show_t": [int(t) for t in show],
                "metrics": payload.get("metrics") or {},
            }
        )
        header = np.ones((40, body.shape[1], 3), dtype=np.float32)
        rows.append((frac, fam, split, label, np.concatenate([header, body], axis=0)))
        print(label, flush=True)

    rows.sort(key=lambda x: (x[1], x[2], -x[0]))
    mosaic = _vcat([b for *_, b in rows], gap=16)

    from PIL import Image, ImageDraw, ImageFont

    font_p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_p, 28) if Path(font_p).is_file() else ImageFont.load_default()
    left, top, bot = 16, 56, 20
    canvas = np.ones((mosaic.shape[0] + top + bot, mosaic.shape[1] + 2 * left, 3), dtype=np.float32)
    canvas[top : top + mosaic.shape[0], left : left + mosaic.shape[1]] = mosaic
    img = Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text(
        (left, 12),
        f"GT depth frame-diff  |Δz|>{THRESH_M*100:.0f}cm   "
        "RGB | depth | |Δz| | old |Δz|>10cm | new (RGB+local residual)",
        fill=(20, 20, 20),
        font=font,
    )
    y = top
    gap = 16
    for _, _, _, label, body in rows:
        draw.text((left + 4, y + 6), label, fill=(20, 20, 20), font=font)
        y += body.shape[0] + gap
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "motion_mask_18.png"
    img.save(out)
    (OUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"wrote {out} size={img.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
