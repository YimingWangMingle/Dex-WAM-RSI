#!/usr/bin/env python3
"""Thinker pred -> metres, score vs sidecar DA3 on native 480x640 AgiBot peri clips.

Eval-only. Does not change training or Rynn inference.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import (  # noqa: E402
    _build_depth_attacher,
    _depth_metrics,
    _rewrite_vlm,
)
from eval_video_ckpt import (  # noqa: E402
    _load_model_weights,
    _run_one_sample,
    create_eval_model,
    load_config,
)
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.models.depth_reader import DepthSidecarReader  # noqa: E402
from publish_eval_to_monitor import _meta_first  # noqa: E402

LOG = logging.getLogger("score_thinker_da3")

FAMILY = PROJECT_ROOT / "scripts/family_peri_26_v61.json"
OUT_DEFAULT = Path("/code1/zxw_data/pretrain/ThinkerWAM-Depth/tmp/thinker_agibot_da3_align")
CONFIG = PROJECT_ROOT / "third_party/Thinker-WM/configs/depth_ablation_treatment_v6_1.yaml"
CKPT = Path(
    "/data/xiongwei.zhao/thinkerwam-depth/checkpoints/"
    "depth_ablation_treatment_v6_1/depth_ablation_treatment_v6_1/"
    "depth_ablation_treatment_v6_1/checkpoint_step_3000"
)
Z_NEAR, Z_FAR = 0.15, 4.0
D_MAX = 5.0
SKIP = 1


def _agibot_clips() -> list[dict]:
    raw = json.loads(FAMILY.read_text())
    clips = []
    for peri, item in enumerate(raw):
        if not str(item["source"]).startswith("agibot"):
            continue
        row = dict(item)
        row["peri"] = peri
        clips.append(row)
    return clips


def _absrel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"absrel": None, "absrel_medscale": None, "n": 0}
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    rel = np.abs(pred - gt) / np.clip(gt, 1e-6, None)
    scale = float(np.median(gt[mask] / np.clip(pred[mask], 1e-6, None)))
    rel_s = np.abs(pred * scale - gt) / np.clip(gt, 1e-6, None)
    return {
        "absrel": float(rel[mask].mean()),
        "absrel_medscale": float(rel_s[mask].mean()),
        "scale": scale,
        "n": int(mask.sum()),
        "pred_p50": float(np.median(pred[mask])),
        "gt_p50": float(np.median(gt[mask])),
    }


def _slice_score(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, sl: slice) -> dict:
    return _absrel(pred[sl], gt[sl], mask[sl])


def _score_pair(pred_m: np.ndarray, da3_m: np.ndarray, log_m: np.ndarray, log_ok: np.ndarray) -> dict:
    t = min(pred_m.shape[0], da3_m.shape[0], log_m.shape[0])
    pred_m, da3_m, log_m, log_ok = pred_m[:t], da3_m[:t], log_m[:t], log_ok[:t]
    da3_mask = np.isfinite(da3_m) & (da3_m > 0.05) & (da3_m < D_MAX)
    log_mask = log_ok & np.isfinite(log_m) & (log_m > Z_NEAR) & (log_m < Z_FAR)
    sl_all = slice(0, t)
    sl_skip = slice(SKIP, t)
    sl_f25 = slice(SKIP, min(25, t))
    return {
        "t": t,
        "hw": [int(pred_m.shape[-2]), int(pred_m.shape[-1])],
        "vs_da3": _slice_score(pred_m, da3_m, da3_mask, sl_skip),
        "vs_da3_all": _slice_score(pred_m, da3_m, da3_mask, sl_all),
        "vs_da3_f25": _slice_score(pred_m, da3_m, da3_mask, sl_f25),
        "vs_loggt": _slice_score(pred_m, log_m, log_mask, sl_skip),
        "vs_loggt_all": _slice_score(pred_m, log_m, log_mask, sl_all),
        "codec_roundtrip": _slice_score(log_m, da3_m, da3_mask, sl_skip),
    }


def _load_da3_m(reader: DepthSidecarReader, batch: dict, hw: tuple[int, int]) -> np.ndarray | None:
    eid = str(_meta_first(batch, "episode_ids"))
    indices = batch.get("frame_indices")
    if indices is None:
        return None
    idx = indices[0]
    mm, ok = reader.read_mm(eid, idx)
    if not ok or mm is None:
        return None
    mm = mm.float()
    if tuple(mm.shape[-2:]) != hw:
        mm = torch.nn.functional.interpolate(mm.unsqueeze(1), size=hw, mode="nearest").squeeze(1)
    z = mm.numpy() / 1000.0
    z = np.where(mm.numpy() > 0, z, np.nan)
    # frame_indices is first + T video frames; pred aligns to video frames.
    return z[1:] if z.shape[0] == int(batch["video_frames"].shape[1]) + 1 else z


def run_rank(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clips = _agibot_clips()
    mine = [c for i, c in enumerate(clips) if i % int(args.world_size) == int(args.rank)]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    LOG.info("rank %d/%d clips=%s", args.rank, args.world_size, [c["peri"] for c in mine])

    config = load_config(args.config)
    OmegaConf.set_struct(config, False)
    config.model.wm.depth_read_by_action = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), device.index)
    dtype = torch.bfloat16
    model = create_eval_model(config)
    model.to(device)
    model.eval()
    _load_model_weights(model, Path(args.checkpoint), strict=True)
    model.to(device)
    model.eval()
    attacher = _build_depth_attacher(config)
    reader = DepthSidecarReader(OmegaConf.to_object(config.depth.roots), prod_keep_only=True)
    inf = config.model.inference
    action_params = {
        "action_guidance_scale": float(getattr(inf, "action_guidance_scale", 1.0)),
        "action_num_inference_steps": (
            int(inf.action_num_inference_timesteps)
            if getattr(inf, "action_num_inference_timesteps", None) is not None
            else None
        ),
        "action_smooth_alpha": float(getattr(inf, "action_temporal_smooth_alpha", 0.0)),
        "depth_smooth_alpha": float(getattr(inf, "depth_temporal_smooth_alpha", 0.0)),
    }
    infer_steps = int(args.num_inference_steps)
    guidance = float(getattr(config.model.inference, "guidance_scale", 1.0))

    for clip in mine:
        peri = int(clip["peri"])
        dest = out_root / f"peri{peri:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        batch = create_diverse_eval_batch(
            config,
            clip["source"],
            batch_size=1,
            shard_ordinal=int(clip["shard"]),
            seed=int(config.periodic_eval.seed),
            use_eval_pool=(clip["split"] == "holdout"),
            episode_id=clip.get("episode_id"),
        )
        batch = _rewrite_vlm(config, dict(batch))
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        eid = _meta_first(batch, "episode_ids")
        out = _run_one_sample(
            model=model,
            batch=batch,
            device=device,
            dtype=dtype,
            num_inference_steps=infer_steps,
            guidance_scale=guidance,
            action_inference_params=action_params,
            return_depth=True,
        )
        _pred_rgb, _pred_act, predicted_depth = out
        if predicted_depth is None:
            raise RuntimeError(f"peri{peri:02d} missing pred depth")
        pred_d = predicted_depth[0].detach().cpu().permute(1, 0, 2, 3).float()
        gt_d = batch["depth_frames"][0].detach().cpu().float()
        n = min(pred_d.shape[0], gt_d.shape[0])
        pred_d, gt_d = pred_d[:n], gt_d[:n]
        pred_m, _ = decode_to_depth_m(pred_d, z_near=Z_NEAR, z_far=Z_FAR, mode="log", in_range="01")
        log_m, log_ok = decode_to_depth_m(gt_d, z_near=Z_NEAR, z_far=Z_FAR, mode="log", in_range="01")
        pred_m_np = pred_m.numpy()
        log_m_np = log_m.numpy()
        log_ok_np = log_ok.numpy()
        da3 = _load_da3_m(reader, batch, tuple(pred_m_np.shape[-2:]))
        if da3 is None:
            raise RuntimeError(f"peri{peri:02d} missing DA3 sidecar for {eid}")
        da3 = da3[:n]
        old = _depth_metrics(pred_d, gt_d, config, rgb_tchw=batch["video_frames"][0].detach().cpu().float()[:n])
        scores = _score_pair(pred_m_np, da3, log_m_np, log_ok_np)
        payload = {
            "peri": peri,
            "rank": int(clip["rank"]),
            "sample_index": int(clip["sample_index"]),
            "source": clip["source"],
            "split": clip["split"],
            "shard": int(clip["shard"]),
            "episode_id": str(eid),
            "seconds": time.time() - t0,
            "old_loggt_absrel": old.get("depth_absrel"),
            **scores,
        }
        np.save(dest / "pred_m.npy", pred_m_np.astype(np.float32))
        np.save(dest / "da3_m.npy", da3.astype(np.float32))
        np.save(dest / "loggt_m.npy", log_m_np.astype(np.float32))
        (dest / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        LOG.info(
            "peri%02d %s vs_da3=%.3f vs_loggt=%.3f old=%.3f (%.1fs)",
            peri,
            eid,
            scores["vs_da3"]["absrel"],
            scores["vs_loggt"]["absrel"],
            old.get("depth_absrel") or -1,
            time.time() - t0,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def gather(args) -> int:
    out_root = Path(args.out)
    rows = []
    for path in sorted(out_root.glob("peri*/metrics.json")):
        rows.append(json.loads(path.read_text()))
    if len(rows) != 14:
        raise SystemExit(f"expected 14 clip metrics, got {len(rows)} under {out_root}")

    def pack(field: str) -> dict:
        vals = [r[field] for r in rows]
        by = {}
        for split in ("holdout", "train"):
            sub = [r[field] for r in rows if r["split"] == split]
            by[f"agibot_{split}"] = {
                "n": len(sub),
                "absrel": _mean(sub, "absrel"),
                "absrel_medscale": _mean(sub, "absrel_medscale"),
            }
        return {
            "n": len(vals),
            "absrel": _mean(vals, "absrel"),
            "absrel_medscale": _mean(vals, "absrel_medscale"),
            "by_group": by,
        }

    summary = {
        "note": (
            "Thinker v6.1 step3000 pred decoded to metres vs sidecar DA3 "
            "on native 480x640. skip_f0=1 matches Rynn official scoring."
        ),
        "n": len(rows),
        "old_loggt_absrel": _mean(rows, "old_loggt_absrel"),
        "vs_da3": pack("vs_da3"),
        "vs_da3_all": pack("vs_da3_all"),
        "vs_da3_f25": pack("vs_da3_f25"),
        "vs_loggt": pack("vs_loggt"),
        "vs_loggt_all": pack("vs_loggt_all"),
        "codec_roundtrip": pack("codec_roundtrip"),
        "per_clip": rows,
    }
    dest = out_root / "summary.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in summary if k != "per_clip"}, indent=2))
    print(f"wrote {dest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather", action="store_true")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--checkpoint", default=str(CKPT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    args = parser.parse_args()
    if args.gather:
        return gather(args)
    return run_rank(args)


if __name__ == "__main__":
    raise SystemExit(main())
