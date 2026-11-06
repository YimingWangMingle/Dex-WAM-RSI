#!/usr/bin/env python3
"""Frozen Wan VAE round-trip on robotwin_3d depth AND rgb, at the exact
training tensor (Motus T composed, pillarbox padded to 480x640, T=49).

Answers two things before committing to a full run:
  1. Does the log-codec depth survive the frozen VAE at the locked
     z_near/z_far, and how large is the metric error per camera?
  2. Does the z_near choice matter? Sweeps a few z_near values on the
     same pixels so the comparison is apples-to-apples.

  CUDA_VISIBLE_DEVICES=0 python scripts/vae_roundtrip_robotwin_3d.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare import robotwin_3d as _robotwin_3d  # noqa: E402,F401
from thinker_wm.data_prepare.depth_attach import DepthAttacher  # noqa: E402
from thinker_wm.data_prepare.factory import create_training_data_pipeline  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m, encode_depth_mm  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

VAE_PTH = Path(
    "/code1/llm_team/chen.liao/Motus/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
)
# Motus T inside the pillarbox: head is the top 240/360 of the canvas.
HEAD_FRAC = 240.0 / 360.0
Z_NEAR_SWEEP = (0.01, 0.05, 0.10, 0.15)


@torch.no_grad()
def roundtrip(vae, x01: torch.Tensor, device: str) -> torch.Tensor:
    """[T,3,H,W] in [0,1] -> reconstruction, same layout, on cpu."""
    x = (x01 * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
    x = x.to(device=device, dtype=torch.bfloat16)
    latent = vae.encode(x)
    rec = vae.decode([latent[0]])[0].float().clamp(-1, 1)
    return ((rec + 1.0) / 2.0).permute(1, 0, 2, 3).cpu()


def absrel(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
    if int(mask.sum()) == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    r = ((pred[mask] - gt[mask]).abs() / gt[mask].clamp_min(1e-6))
    return {"mean": float(r.mean()), "median": float(r.median()), "n": int(mask.sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_REPO / "third_party/Thinker-WM/configs/robotwin_3d_rgbd.yaml"))
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument(
        "--out", default=str(_REPO / "tmp/qc_robotwin_3d/vae_roundtrip.json")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if not VAE_PTH.is_file():
        print(f"SKIP: missing {VAE_PTH}")
        return 0

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "training.batch_size", 1, force_add=True)
    OmegaConf.update(cfg, "system.num_workers", 0, force_add=True)
    OmegaConf.update(cfg, "system.ready_batch_queue_enabled", False, force_add=True)
    z_far = float(cfg.depth.z_far)
    z_near_cfg = float(cfg.depth.z_near)

    # Raw millimetres straight from the inline reader, so the z_near sweep can
    # re-encode the same pixels instead of inverting an already-lossy code.
    inline = _robotwin_3d.Robotwin3dInlineDepthReader(str(cfg.depth.robotwin_3d_root))
    attacher = DepthAttacher(
        [], z_near=z_near_cfg, z_far=z_far, mode=str(cfg.depth.mode),
        invalid_to=str(cfg.depth.invalid_to), prod_keep_only=bool(cfg.depth.prod_keep_only),
        robotwin_3d_root=str(cfg.depth.robotwin_3d_root),
    )

    from thinker_wm.wan.modules.vae2_2 import Wan2_2_VAE

    device = "cuda"
    print(f"loading Wan 2.2 VAE from {VAE_PTH}", flush=True)
    vae = Wan2_2_VAE(vae_pth=str(VAE_PTH), device=device)

    reports = []
    pipeline = create_training_data_pipeline(cfg, rank=0, world_size=1, seed=0)
    with pipeline:
        for i in range(int(args.clips)):
            batch, _ = pipeline.next()
            eid = str(batch["episode_ids"][0])
            indices = list(batch["frame_indices"][0])
            video = batch["video_frames"][0]  # [T,3,H,W] in [0,1]
            first = batch["first_frame"][0].unsqueeze(0)
            rgb49 = torch.cat([first, video], dim=0)  # [49,3,H,W]
            H, W = int(rgb49.shape[-2]), int(rgb49.shape[-1])

            mm, ok = inline.read_mm(eid, indices, target_hw=(H, W))
            if not ok or mm is None:
                print(f"SKIP {eid}: depth read failed")
                continue
            mm = mm[: rgb49.shape[0]]

            occ = (mm > 0).float().sum(dim=0)
            rows = (occ.sum(dim=1) > 0).nonzero().flatten()
            cols = (occ.sum(dim=0) > 0).nonzero().flatten()
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            c0, c1 = int(cols.min()), int(cols.max()) + 1
            split = r0 + int(round((r1 - r0) * HEAD_FRAC))

            rec_rgb = roundtrip(vae, rgb49, device)
            rgb_mse = float(((rgb49 - rec_rgb) ** 2).mean())
            rgb_psnr = 10.0 * math.log10(1.0 / max(rgb_mse, 1e-12))

            entry = {
                "episode_id": eid,
                "frame_start": int(indices[0]),
                "T": int(rgb49.shape[0]),
                "content_box": [r0, r1, c0, c1],
                "rgb_psnr": rgb_psnr,
                "depth": {},
            }

            for z_near in Z_NEAR_SWEEP:
                x01 = encode_depth_mm(
                    mm, z_near=z_near, z_far=z_far, mode="log",
                    invalid_to=str(cfg.depth.invalid_to), out_range="01",
                )
                r01 = roundtrip(vae, x01, device)
                mse = float(((x01 - r01) ** 2).mean())
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))

                zx, vx = decode_to_depth_m(x01, z_near, z_far, in_range="01")
                zr, vr = decode_to_depth_m(r01, z_near, z_far, in_range="01")
                gt_valid = mm > 0
                both = vx & vr & gt_valid

                head_m = torch.zeros_like(both)
                head_m[:, r0:split, c0:c1] = True
                wrist_m = torch.zeros_like(both)
                wrist_m[:, split:r1, c0:c1] = True

                z_gt = mm.float() / 1000.0
                entry["depth"][f"{z_near:.2f}"] = {
                    "code_psnr": psnr,
                    "code_span": [float(x01.min()), float(x01.max())],
                    "absrel_all": absrel(zr, z_gt, both),
                    "absrel_head": absrel(zr, z_gt, both & head_m),
                    "absrel_wrist": absrel(zr, z_gt, both & wrist_m),
                    "mae_mm_head": float(
                        ((zr - z_gt).abs()[both & head_m]).mean() * 1000
                    ),
                    "mae_mm_wrist": float(
                        ((zr - z_gt).abs()[both & wrist_m]).mean() * 1000
                    ),
                    "pad_leak_max": float(r01[:, :, :, :c0].max()),
                }

            reports.append(entry)
            cur = entry["depth"][f"{z_near_cfg:.2f}"]
            print(
                f"[{i}] {eid} start={indices[0]} rgb_psnr={rgb_psnr:.2f}dB | "
                f"z_near={z_near_cfg}: code_psnr={cur['code_psnr']:.2f}dB "
                f"AbsRel all={cur['absrel_all']['mean']:.4f} "
                f"head={cur['absrel_head']['mean']:.4f} ({cur['mae_mm_head']:.1f}mm) "
                f"wrist={cur['absrel_wrist']['mean']:.4f} ({cur['mae_mm_wrist']:.1f}mm) "
                f"pad_leak={cur['pad_leak_max']:.3f}",
                flush=True,
            )

    if not reports:
        print("no clips processed")
        return 1

    print("\n=== z_near sweep (mean over clips) ===")
    print(f"{'z_near':>7} {'codePSNR':>9} {'AbsRel_all':>11} {'AbsRel_head':>12} "
          f"{'mae_head_mm':>12} {'AbsRel_wrist':>13} {'mae_wrist_mm':>13}")
    summary = {}
    for z_near in Z_NEAR_SWEEP:
        k = f"{z_near:.2f}"
        vals = [r["depth"][k] for r in reports]
        row = {
            "code_psnr": sum(v["code_psnr"] for v in vals) / len(vals),
            "absrel_all": sum(v["absrel_all"]["mean"] for v in vals) / len(vals),
            "absrel_head": sum(v["absrel_head"]["mean"] for v in vals) / len(vals),
            "mae_mm_head": sum(v["mae_mm_head"] for v in vals) / len(vals),
            "absrel_wrist": sum(v["absrel_wrist"]["mean"] for v in vals) / len(vals),
            "mae_mm_wrist": sum(v["mae_mm_wrist"] for v in vals) / len(vals),
        }
        summary[k] = row
        mark = "  <- config" if abs(z_near - z_near_cfg) < 1e-9 else ""
        print(f"{k:>7} {row['code_psnr']:>9.2f} {row['absrel_all']:>11.4f} "
              f"{row['absrel_head']:>12.4f} {row['mae_mm_head']:>12.1f} "
              f"{row['absrel_wrist']:>13.4f} {row['mae_mm_wrist']:>13.1f}{mark}")
    rgb_psnr = sum(r["rgb_psnr"] for r in reports) / len(reports)
    print(f"\nRGB round-trip PSNR (same canvas): {rgb_psnr:.2f} dB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"z_near_config": z_near_cfg, "z_far": z_far, "rgb_psnr": rgb_psnr,
             "summary": summary, "clips": reports},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
