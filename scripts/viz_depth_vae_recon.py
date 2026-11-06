#!/usr/bin/env python3
"""Frozen Wan VAE round-trip on log-codec depth: turbo-m input | recon | |Δz|.

T=49 (1 cond + 48 pred, Wan stride T % 4 == 1), 480x640, same encode path as
training. AbsRel is reported on all valid pixels and on the motion region
(GT depth frame-diff mask). A T=5 encode of the highest-motion 5-frame slice
is the spatial-only control.

Writes tmp/depth_viz_gallery/codec_vae_recon.png plus codec_vae_recon.json.

  CUDA_VISIBLE_DEVICES=0 python scripts/viz_depth_vae_recon.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.models.depth_codec import decode_to_depth_m, encode_depth_mm  # noqa: E402
from thinker_wm.models.depth_reader import DepthSidecarReader  # noqa: E402

EGO_ROOT = Path("/data/xiongwei.zhao/egoscale_ffs_depth/v1")
AGI_ROOT = Path("/data/xiongwei.zhao/agibot_da3_depth/v1")
VAE_PTH = Path(
    "/code1/llm_team/chen.liao/Motus/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
)
OUT = Path("/code1/zxw_data/pretrain/ThinkerWAM-Depth/tmp/depth_viz_gallery/codec_vae_recon.png")
OUT_JSON = OUT.with_suffix(".json")
H, W = 480, 640
T_FRAMES = 49  # 1 cond + num_video_frames=48; Wan VAE: T % 4 == 1
T_SPATIAL = 5  # spatial-only control (still T % 4 == 1)
SCAN_MAX = 196  # scan at most this many clip frames to pick a motion window
N_SHOW = 4
Z_NEAR, Z_FAR = 0.1, 10.0
MOTION_M = 0.10  # viz + default mask: |Δz| > 10 cm (2 cm is DA3 flicker)
MOTION_THRESHOLDS_M = (0.02, 0.10, 0.20)

PICKS = [
    (
        "EgoScale FFS",
        "egoscale:session_wUUR1LPnOrhHQQ_00001_task_0001_lerobot_v3",
        EGO_ROOT,
    ),
    ("AgiBot DA3", "agibot_beta:660:902793", AGI_ROOT),
]


def _resize_mm(mm: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(mm.shape[-2:]) == hw:
        return mm
    return F.interpolate(mm.unsqueeze(1).float(), size=hw, mode="nearest").squeeze(1)


def _clip_indices(reader: DepthSidecarReader, eid: str, n: int) -> list[int]:
    clip = reader.episodes[eid].clips[0]
    if len(clip) < n:
        raise RuntimeError(f"{eid}: clip length {len(clip)} < {n}")
    return list(clip[: min(len(clip), max(n, SCAN_MAX))])


def _motion_mask(zx: torch.Tensor, vx: torch.Tensor, thresh_m: float = MOTION_M) -> torch.Tensor:
    """[T,H,W] bool. Pixel is motion if either adjacent pair exceeds thresh_m."""
    pair_ok = vx[1:] & vx[:-1]
    moving = pair_ok & ((zx[1:] - zx[:-1]).abs() > thresh_m)
    out = torch.zeros_like(vx)
    out[1:] |= moving
    out[:-1] |= moving
    return out


def _absrel(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if not bool(mask.any()):
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    rel = (pred[mask] - gt[mask]).abs() / gt[mask].clamp_min(1e-6)
    return {
        "mean": float(rel.mean()),
        "median": float(rel.median()),
        "n": int(mask.sum()),
    }


def _best_window(zx: torch.Tensor, vx: torch.Tensor, t: int) -> int:
    """Start index of the length-t window with the most motion pixels."""
    motion = _motion_mask(zx, vx)
    counts = motion.reshape(motion.shape[0], -1).sum(dim=1)
    if zx.shape[0] == t:
        return 0
    window = counts.unfold(0, t, 1).sum(dim=1)
    return int(window.argmax())


@torch.no_grad()
def _roundtrip(vae, x01: torch.Tensor, device: str) -> torch.Tensor:
    """x01 [T,3,H,W] in [0,1] → recon same layout."""
    x = (x01 * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
    x = x.to(device=device, dtype=torch.bfloat16)
    latent = vae.encode(x)
    rec = vae.decode([latent[0]])[0].float().clamp(-1, 1)
    return ((rec + 1.0) / 2.0).permute(1, 0, 2, 3).cpu()  # [T,3,H,W]


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if not VAE_PTH.is_file():
        print(f"SKIP: missing {VAE_PTH}")
        return 0
    from thinker_wm.wan.modules.vae2_2 import Wan2_2_VAE

    device = "cuda"
    print(f"loading Wan 2.2 VAE from {VAE_PTH}", flush=True)
    vae = Wan2_2_VAE(vae_pth=str(VAE_PTH), device=device)

    rows = []
    reports = []
    for label, eid, root in PICKS:
        reader = DepthSidecarReader([root])
        if eid not in reader.episodes:
            print(f"SKIP {label}: {eid} not in index")
            continue
        scan_idxs = _clip_indices(reader, eid, T_FRAMES)
        mm, ok = reader.read_mm(eid, scan_idxs)
        if not ok or mm is None:
            print(f"SKIP {label}: read failed")
            continue
        mm = _resize_mm(mm, (H, W))
        z_scan = mm.float() / 1000.0
        v_scan = mm > 0
        start = _best_window(z_scan, v_scan, T_FRAMES)
        idxs = scan_idxs[start : start + T_FRAMES]
        mm = mm[start : start + T_FRAMES]
        x01 = encode_depth_mm(mm, Z_NEAR, Z_FAR, mode="log", out_range="01")
        r01 = _roundtrip(vae, x01, device)
        mse = float(((x01 - r01) ** 2).mean())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        mae01 = float((x01 - r01).abs().mean())
        zx, vx = decode_to_depth_m(x01, Z_NEAR, Z_FAR, in_range="01")
        zr, vr = decode_to_depth_m(r01, Z_NEAR, Z_FAR, in_range="01")
        both = vx & vr
        z_gt = mm.float() / 1000.0
        v_gt = mm > 0
        motion = _motion_mask(z_gt, v_gt, MOTION_M)
        static = both & ~motion
        all_m = _absrel(zr, zx, both)
        mot_m = _absrel(zr, zx, both & motion)
        sta_m = _absrel(zr, zx, static)
        mot_frac = float((both & motion).sum() / both.sum().clamp_min(1))

        by_thr = {}
        for thr in MOTION_THRESHOLDS_M:
            m = _motion_mask(z_gt, v_gt, thr)
            by_thr[f"{thr:.2f}"] = {
                "frac": float((both & m).sum() / both.sum().clamp_min(1)),
                **_absrel(zr, zx, both & m),
            }

        # Spatial-only control: T=5 encode of the highest-motion 5-frame slice.
        t5_off = _best_window(z_gt, v_gt, T_SPATIAL)
        x5 = x01[t5_off : t5_off + T_SPATIAL]
        r5 = _roundtrip(vae, x5, device)
        z5, v5 = decode_to_depth_m(x5, Z_NEAR, Z_FAR, in_range="01")
        zr5, vr5 = decode_to_depth_m(r5, Z_NEAR, Z_FAR, in_range="01")
        both5 = v5 & vr5
        mot5 = _motion_mask(
            z_gt[t5_off : t5_off + T_SPATIAL],
            v_gt[t5_off : t5_off + T_SPATIAL],
            MOTION_M,
        )
        t5_all = _absrel(zr5, z5, both5)
        t5_mot = _absrel(zr5, z5, both5 & mot5)

        report = {
            "label": label,
            "episode_id": eid,
            "native_indices": idxs,
            "window_start_in_scan": start,
            "t5_offset_in_window": t5_off,
            "psnr": psnr,
            "mae01": mae01,
            "motion_frac": mot_frac,
            "motion_thresh_m": MOTION_M,
            "t49_absrel_all_mean": all_m["mean"],
            "t49_absrel_all_median": all_m["median"],
            "t49_absrel_motion_mean": mot_m["mean"],
            "t49_absrel_motion_median": mot_m["median"],
            "t49_absrel_static_mean": sta_m["mean"],
            "t49_absrel_static_median": sta_m["median"],
            "t49_n_valid": all_m["n"],
            "t49_n_motion": mot_m["n"],
            "t49_absrel_by_motion_thresh": by_thr,
            "t5_absrel_all_mean": t5_all["mean"],
            "t5_absrel_motion_mean": t5_mot["mean"],
            "t5_n_motion": t5_mot["n"],
        }
        reports.append(report)
        thr_txt = "  ".join(
            f">{k}m {v['mean']:.4f}({v['frac']:.1%})" for k, v in by_thr.items()
        )
        print(
            f"{label} {eid} n={idxs[0]}–{idxs[-1]}  PSNR={psnr:.1f}  "
            f"AbsRel T49 all/static/motion@10cm={all_m['mean']:.4f}/"
            f"{sta_m['mean']:.4f}/{mot_m['mean']:.4f}  "
            f"motion={mot_frac:.2%} (n={mot_m['n']})  "
            f"T5-control all/motion={t5_all['mean']:.4f}/{t5_mot['mean']:.4f}  "
            f"by-thr {thr_txt}",
            flush=True,
        )
        rows.append((label, idxs, x01, r01, motion, mae01, psnr, all_m, mot_m))

    if not rows:
        print("no rows")
        return 1

    from matplotlib import colormaps

    turbo = colormaps["turbo"]

    def _colorize(z: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        t = np.clip((z - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        rgb = turbo(t)[..., :3]
        rgb[np.isnan(z)] = 1.0
        return rgb

    def _diff_gray(d: np.ndarray, vmax: float) -> np.ndarray:
        g = np.clip(d / max(vmax, 1e-6), 0.0, 1.0)
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

    def _vcat(imgs: list[np.ndarray], gap: int = 2) -> np.ndarray:
        w = imgs[0].shape[1]
        bar = np.ones((gap, w, 3), dtype=np.float32)
        out = imgs[0]
        for im in imgs[1:]:
            out = np.concatenate([out, bar, im], axis=0)
        return out

    z_lo, z_hi = 0.15, 2.5
    d_hi = 0.03  # 3 cm; gray residual, not magma
    blocks = []
    headers = []
    show_ts_all = []
    for label, idxs, x01, r01, motion, mae01, psnr, all_m, mot_m in rows:
        zx, vx = decode_to_depth_m(x01, Z_NEAR, Z_FAR, in_range="01")
        zr, vr = decode_to_depth_m(r01, Z_NEAR, Z_FAR, in_range="01")
        zin = np.where(vx.numpy(), zx.numpy(), np.nan)
        zrec = np.where(vr.numpy(), zr.numpy(), np.nan)
        dz = np.where(vx.numpy() & vr.numpy(), np.abs(zr.numpy() - zx.numpy()), np.nan)
        mot_np = motion.numpy()
        per_t = mot_np.reshape(mot_np.shape[0], -1).sum(axis=1)
        show_ts = np.argsort(-per_t)[:N_SHOW]
        show_ts = np.sort(show_ts).tolist()
        show_ts_all.append(show_ts)
        row_in = _hcat([_colorize(zin[t], z_lo, z_hi) for t in show_ts])
        row_rec = _hcat([_colorize(zrec[t], z_lo, z_hi) for t in show_ts])
        row_d = _hcat([_diff_gray(dz[t], d_hi) for t in show_ts])
        row_m = _hcat([
            np.where(mot_np[t][..., None], np.array([1.0, 0.35, 0.1]), np.array([0.15, 0.15, 0.15]))
            for t in show_ts
        ])
        body = _vcat([row_in, row_rec, row_d, row_m], gap=2)
        header_h = 112
        band = np.ones((header_h, body.shape[1], 3), dtype=np.float32)
        blocks.append(np.concatenate([band, body], axis=0))
        headers.append(
            f"{label}    PSNR {psnr:.1f}    AbsRel all {all_m['mean']:.3f}    "
            f"motion {mot_m['mean']:.3f}    n={idxs[0]}–{idxs[-1]}"
        )
    mosaic = _vcat(blocks, gap=80)

    from PIL import Image, ImageDraw, ImageFont

    FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(FONT_B, 52)
    font_h = ImageFont.truetype(FONT_B, 64)
    font_s = ImageFont.truetype(FONT_B, 56)

    left, top, right, bot = 230, 110, 200, 72
    mh, mw = mosaic.shape[:2]
    canvas = np.ones((mh + top + bot, mw + left + right, 3), dtype=np.float32)
    canvas[top : top + mh, left : left + mw] = mosaic
    img = Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text(
        (left, 20),
        "Frozen Wan 2.2 VAE T=49   ·   turbo 0.15–2.5 m   ·   |Δz| gray 3 cm   ·   motion |Δz|>2cm",
        fill=(20, 20, 20),
        font=font,
    )
    header_h = 112
    body_h = H * 4 + 3 * 2
    block_h = header_h + body_h + 80
    for b, header in enumerate(headers):
        y0 = top + b * block_h
        draw.text((left, y0 + 24), header, fill=(20, 20, 20), font=font_h)
        body0 = y0 + header_h
        for k, name in enumerate(("in", "recon", "|Δz|", "mot")):
            draw.text((18, body0 + k * (H + 2) + H // 2 - 28), name, fill=(40, 40, 40), font=font_s)
        for i, t in enumerate(show_ts_all[b]):
            draw.text(
                (left + i * (W + 2) + 16, body0 + 8),
                f"t={t}",
                fill=(20, 20, 20),
                font=font_s,
            )

    # colorbars: turbo + gray, painted as thin strips
    strip_h = int(mh * 0.42)
    ys = np.linspace(z_hi, z_lo, strip_h)
    turbo_strip = turbo(np.clip((ys - z_lo) / (z_hi - z_lo), 0, 1))[..., :3]
    turbo_img = (np.repeat(turbo_strip[:, None, :], 12, axis=1) * 255).astype(np.uint8)
    img.paste(Image.fromarray(turbo_img), (left + mw + 10, top + 8))
    draw.text((left + mw + 28, top + 4), "m", fill=(20, 20, 20), font=font_s)
    draw.text((left + mw + 28, top + 8 + strip_h - 20), f"{z_lo}", fill=(20, 20, 20), font=font_s)
    draw.text((left + mw + 28, top + 8), f"{z_hi}", fill=(20, 20, 20), font=font_s)
    gs = np.linspace(1.0, 0.0, int(mh * 0.22))
    gray_strip = np.stack([gs, gs, gs], axis=-1)
    gray_img = (np.repeat(gray_strip[:, None, :], 12, axis=1) * 255).astype(np.uint8)
    gy = top + mh - gray_img.shape[0] - 8
    img.paste(Image.fromarray(gray_img), (left + mw + 10, gy))
    draw.text((left + mw + 28, gy), f"{d_hi}", fill=(20, 20, 20), font=font_s)
    draw.text((left + mw + 28, gy + gray_img.shape[0] - 28), "0", fill=(20, 20, 20), font=font_s)
    draw.text((left + mw + 28, gy + gray_img.shape[0] // 2 - 12), "|Δz|", fill=(20, 20, 20), font=font_s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    OUT_JSON.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) size={img.size}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
