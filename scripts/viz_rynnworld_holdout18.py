#!/usr/bin/env python3
"""Mosaic: RGB | GT depth | RynnWorld pred depth for the 18 clips."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp/rynnworld4d_holdout18"
LAT = OUT / "latents"
INFER = OUT / "infer_joint-on"
CATALOG = OUT / "catalog.json"
N_SHOW = 4


def _turbo_u8(gray: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps

    t = np.clip(gray.astype(np.float32) / 255.0, 0, 1)
    rgb = (colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)
    return rgb


def _read_mp4(path: Path) -> np.ndarray:
    frames = imageio.mimread(path, memtest=False)
    return np.stack(frames, axis=0)


def _hcat(imgs: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = min(im.shape[0] for im in imgs)
    imgs = [im[:h] for im in imgs]
    bar = np.full((h, gap, 3), 255, dtype=np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.concatenate([out, bar, im], axis=1)
    return out


def _vcat(imgs: list[np.ndarray], gap: int = 8) -> np.ndarray:
    w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[1] < w:
            pad = np.full((im.shape[0], w - im.shape[1], 3), 255, dtype=np.uint8)
            im = np.concatenate([im, pad], axis=1)
        padded.append(im)
    bar = np.full((gap, w, 3), 255, dtype=np.uint8)
    out = padded[0]
    for im in padded[1:]:
        out = np.concatenate([out, bar, im], axis=0)
    return out


def main() -> int:
    catalog = json.loads(CATALOG.read_text())
    rows = []
    for item in catalog:
        i = int(item["index"])
        source = item["source"]
        split = item["split"]
        shard = item["shard"]
        stem = f"{i:02d}_{source}_{split}_{shard}"
        rgb = np.load(LAT / f"{stem}_gt_rgb.npy")  # [T,3,H,W] u8
        depth = np.load(LAT / f"{stem}_gt_depth.npy")
        pred_p = INFER / f"holdout18_{i}_depth.mp4"
        if not pred_p.exists():
            print(f"missing {pred_p}")
            continue
        pred = _read_mp4(pred_p)
        t = min(rgb.shape[0], depth.shape[0], pred.shape[0])
        show = np.linspace(0, t - 1, N_SHOW).round().astype(int)
        tiles = []
        for f in show:
            rgb_f = np.transpose(rgb[f], (1, 2, 0))
            gt_f = _turbo_u8(depth[f].mean(axis=0))
            pred_f = pred[f]
            if pred_f.shape[:2] != rgb_f.shape[:2]:
                from PIL import Image

                pred_f = np.array(Image.fromarray(pred_f).resize((rgb_f.shape[1], rgb_f.shape[0])))
            tiles.append(_hcat([rgb_f, gt_f, pred_f]))
        row = _vcat(tiles, gap=2)
        # label bar
        bar = np.full((28, row.shape[1], 3), 255, dtype=np.uint8)
        row = np.concatenate([bar, row], axis=0)
        rows.append(row)
        print(f"ok {i} {item['episode_id']}")
    if not rows:
        print("no rows")
        return 1
    mosaic = _vcat(rows, gap=10)
    out = OUT / "holdout18_rgb_gt_pred.png"
    imageio.imwrite(out, mosaic)
    print(f"wrote {out} shape={mosaic.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
