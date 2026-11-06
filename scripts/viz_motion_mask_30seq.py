#!/usr/bin/env python3
"""30-sequence GT motion-mask gallery: AgiBot + Ego. CPU only."""
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

CFG = REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v6_1.yaml"
PERI = REPO / "scripts/family_peri_26_v61.json"
OUT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000/motion_mask_30seq")
EXTRA = [
    {"source": "agibot_beta_parallel_gripper", "split": "holdout", "shard": 4},
    {"source": "agibot_beta_dexterous_hand", "split": "holdout", "shard": 4},
    {"source": "egoscale_first500h_remaining90pct", "split": "holdout", "shard": 3},
    {"source": "egoscale_second1700h_batch01", "split": "holdout", "shard": 3},
]
N_PEAK = 3
SCALE = 0.35


def _overlay(rgb: np.ndarray, mot: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    tint = np.array([1.0, 0.25, 0.08], dtype=np.float32)
    out[mot] = 0.35 * out[mot] + 0.65 * tint
    return out.clip(0, 1)


def _hcat(imgs: list[np.ndarray], gap: int = 2) -> np.ndarray:
    h = imgs[0].shape[0]
    bar = np.ones((h, gap, 3), dtype=np.float32)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.concatenate([out, bar, im], axis=1)
    return out


def _vcat(imgs: list[np.ndarray], gap: int = 4) -> np.ndarray:
    w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[1] < w:
            pad = np.ones((im.shape[0], w - im.shape[1], 3), dtype=np.float32)
            im = np.concatenate([im, pad], axis=1)
        padded.append(im)
    bar = np.ones((gap, w, 3), dtype=np.float32)
    out = padded[0]
    for im in padded[1:]:
        out = np.concatenate([out, bar, im], axis=0)
    return out


def _short(eid: str) -> str:
    if eid.startswith("agibot_beta:"):
        return eid.replace("agibot_beta:", "agi:")
    if eid.startswith("egoscale:"):
        return "ego:" + eid.split(":")[1][:18]
    return eid[:28]


def main() -> int:
    clips = json.loads(PERI.read_text()) + EXTRA
    config = load_config(str(CFG))
    attacher = _build_depth_attacher(config)
    z_near = float(getattr(config.depth, "z_near", 0.15))
    z_far = float(getattr(config.depth, "z_far", 4.0))
    mode = str(getattr(config.depth, "mode", "log"))
    seed = int(config.periodic_eval.seed)
    font_p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_p, 16) if Path(font_p).is_file() else ImageFont.load_default()

    OUT.mkdir(parents=True, exist_ok=True)
    catalog = []
    agi_rows, ego_rows = [], []

    for i, clip in enumerate(clips):
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
        present = batch.get("depth_present")
        if present is not None and not bool(present.reshape(-1)[0].item()):
            print(f"skip {i} no depth {clip}", flush=True)
            continue
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
        show = np.sort(np.argsort(-per_t)[:N_PEAK])
        rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1)
        mot_np = mot.numpy()
        tiles = []
        for t in show:
            pair = _hcat([rgb_np[int(t)], _overlay(rgb_np[int(t)], mot_np[int(t)])], gap=1)
            tiles.append(pair)
        body = _hcat(tiles, gap=6)
        h, w = body.shape[:2]
        body = np.array(
            Image.fromarray((body * 255).astype(np.uint8)).resize(
                (max(1, int(w * SCALE)), max(1, int(h * SCALE))), Image.Resampling.BILINEAR
            )
        ).astype(np.float32) / 255.0
        eid = str(clip.get("episode_id") or batch.get("episode_ids", [""])[0] or f"{clip['source']}:{clip['shard']}")
        fam = "agi" if "agibot" in clip["source"] else "ego"
        label = f"{i:02d} {fam} {clip['split'][:4]} {_short(eid)}  mot={frac:.1%}  t={list(map(int, show))}"
        header = np.ones((22, body.shape[1], 3), dtype=np.float32)
        row = np.concatenate([header, body], axis=0)
        img = Image.fromarray((row * 255).clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(img).text((4, 2), label, fill=(20, 20, 20), font=font)
        row = np.asarray(img).astype(np.float32) / 255.0
        (agi_rows if fam == "agi" else ego_rows).append(row)
        catalog.append(
            {
                "i": i,
                "family": fam,
                "source": clip["source"],
                "split": clip["split"],
                "shard": int(clip["shard"]),
                "episode_id": eid,
                "motion_frac": frac,
                "peak_t": [int(t) for t in show],
                "peak_frac": [float(per_t[int(t)]) for t in show],
            }
        )
        print(label, flush=True)

    def _save(name: str, rows: list[np.ndarray]) -> None:
        if not rows:
            return
        mosaic = _vcat(rows, gap=3)
        img = Image.fromarray((mosaic * 255).clip(0, 255).astype(np.uint8))
        img.save(OUT / name)
        print(f"wrote {OUT / name} size={img.size} n={len(rows)}", flush=True)

    _save("agibot.png", agi_rows)
    _save("ego.png", ego_rows)
    _save("all30.png", agi_rows + ego_rows)
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    agi_f = [c["motion_frac"] for c in catalog if c["family"] == "agi"]
    ego_f = [c["motion_frac"] for c in catalog if c["family"] == "ego"]
    print(
        f"n={len(catalog)} agi={len(agi_f)} mean={np.mean(agi_f):.3f} "
        f"ego={len(ego_f)} mean={np.mean(ego_f):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
