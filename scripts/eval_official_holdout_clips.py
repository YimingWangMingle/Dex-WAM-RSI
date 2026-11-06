#!/usr/bin/env python3
"""Official-holdout clip eval for the depth ablation (preview or Gate 2 slice).

Does not touch train v1 sidecars. Reads only holdout_v1 roots. Same seed +
allocation → Control / Treatment see the same clips.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "third_party/Thinker-WM/train") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from thinker_wm.data_prepare.depth_attach import DepthAttacher  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.periodic_eval.runner import (  # noqa: E402
    _depth_motion_mask,
    _depth_structure_metrics,
)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_video_ckpt import (  # noqa: E402
    _compute_video_metrics,
    _image_to_uint8,
    _json_dump,
    _load_model_weights,
    _make_compare_frames,
    _run_one_sample,
    _sample_meta,
    _tensor_to_uint8_frames,
    _write_video,
    create_eval_model,
    load_config,
)
from thinker_wm.utils.vlm_utils import (  # noqa: E402
    PROMPT_MODE_TEXT_ONLY,
    build_vlm_rewrite_message,
    build_vlm_task_message,
    resolve_prompt_mode,
    resolve_vlm_prompt_style,
)


LOG = logging.getLogger("eval_official_holdout")

HOLDOUT_AGIBOT = ["/data/xiongwei.zhao/agibot_da3_depth/holdout_v1"]
HOLDOUT_EGO = ["/data/xiongwei.zhao/egoscale_ffs_depth/holdout_v1"]
TRAIN_AGIBOT = [
    "/data/xiongwei.zhao/agibot_da3_depth/v1",
    "/data/xiongwei.zhao/agibot_da3_depth_fulltimeline/v1",
]
TRAIN_EGO = ["/data/xiongwei.zhao/egoscale_ffs_depth/v1"]
ALLOCATION_HOLDOUT: List[Tuple[str, int]] = [
    ("egoscale_second1700h_batch01", 3),
    ("egoscale_second1700h_batch02", 2),
    ("egoscale_first500h_remaining90pct", 1),
    ("agibot_beta_parallel_gripper", 1),
    ("agibot_beta_dexterous_hand", 1),
]
ALLOCATION_TRAIN: List[Tuple[str, int]] = [
    ("egoscale_second1700h_batch01", 1),
    ("egoscale_second1700h_batch02", 1),
    ("egoscale_first500h_remaining90pct", 1),
    ("agibot_beta_parallel_gripper", 1),
]
FALLBACK_SOURCE = "agibot_beta_parallel_gripper"


def apply_pool(config, pool: str) -> List[Tuple[str, int]]:
    """official_holdout = holdout_v1; train = ablation train v1 (90% split)."""
    if pool == "official_holdout":
        agibot, ego = HOLDOUT_AGIBOT, HOLDOUT_EGO
        fraction, split = 0.0, "all"
        allocation = ALLOCATION_HOLDOUT
    elif pool == "train":
        agibot, ego = TRAIN_AGIBOT, TRAIN_EGO
        fraction, split = 0.1, "train"
        allocation = ALLOCATION_TRAIN
    else:
        raise ValueError(f"unknown pool {pool!r}")
    for src in config.dataset.sources:
        name = str(src.name)
        roots = agibot if name.startswith("agibot") else ego
        src.eval_pool.roots = list(roots)
        src.eval_pool.holdout_fraction = float(fraction)
        src.eval_pool.split = split
    if config.get("depth") is None:
        config.depth = {}
    config.depth.roots = list(agibot) + list(ego)
    return allocation


def _build_depth_attacher(config) -> Optional[DepthAttacher]:
    if not bool(getattr(config.model.wm, "enable_depth", False)):
        return None
    depth_cfg = config.get("depth")
    raw = (
        OmegaConf.to_container(depth_cfg, resolve=True) if depth_cfg is not None else {}
    ) or {}
    roots = list(raw.get("roots") or [])
    inline = raw.get("robotwin_3d_root")
    inline = str(inline) if inline else None
    if not roots and not inline:
        return None
    return DepthAttacher(
        roots,
        z_near=float(raw.get("z_near", 0.1)),
        z_far=float(raw.get("z_far", 10.0)),
        mode=str(raw.get("mode", "log")),
        invalid_to=str(raw.get("invalid_to", "far")),
        prod_keep_only=bool(raw.get("prod_keep_only", True)),
        robotwin_3d_root=inline,
    )


def _rewrite_vlm(config, batch: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(getattr(config.model, "unified_vlm_task_prompt", False)):
        return batch
    if not batch.get("t5_input_messages"):
        return batch
    batch = dict(batch)
    if resolve_prompt_mode(config, batch.get("source_datasets")) == PROMPT_MODE_TEXT_ONLY:
        batch["vlm_input_messages"] = [str(text).strip() for text in batch["t5_input_messages"]]
        batch["first_frame_pils"] = None
    elif bool(getattr(config.model, "vlm_text_rewrite", False)):
        source_key = (batch.get("source_datasets") or [None])[0]
        batch["vlm_input_messages"] = [
            build_vlm_rewrite_message(text, source_key) for text in batch["t5_input_messages"]
        ]
    else:
        batch["vlm_input_messages"] = [
            build_vlm_task_message(text, style=resolve_vlm_prompt_style(config))
            for text in batch["t5_input_messages"]
        ]
    return batch


def _collect_batches(config, seed: int, allocation: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    fallback_shard = 1
    for source, count in allocation:
        ok = 0
        for shard in range(int(count)):
            try:
                batch = create_diverse_eval_batch(
                    config, source, batch_size=1, shard_ordinal=shard, seed=seed
                )
            except Exception as exc:
                LOG.warning("source %s shard %d failed: %s; fallback %s", source, shard, exc, FALLBACK_SOURCE)
                batch = create_diverse_eval_batch(
                    config,
                    FALLBACK_SOURCE,
                    batch_size=1,
                    shard_ordinal=fallback_shard,
                    seed=seed,
                )
                source = FALLBACK_SOURCE
                fallback_shard += 1
            items.append({"source": source, "shard": shard, "batch": batch})
            ok += 1
        LOG.info("collected %d clip(s) from %s", ok, source)
    return items


def _action_mse(pred: Optional[torch.Tensor], gt: Optional[torch.Tensor]) -> Optional[float]:
    if pred is None or gt is None:
        return None
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()
    if pred.ndim == 3:
        pred = pred[0]
    if gt.ndim == 3:
        gt = gt[0]
    t = min(pred.shape[0], gt.shape[1] if gt.ndim == 2 and pred.shape[-1] != gt.shape[-1] else gt.shape[0])
    # pred [T, D], gt [B, T, D] or [T, D]
    if gt.ndim == 3:
        gt = gt[0]
    t = min(pred.shape[0], gt.shape[0])
    d = min(pred.shape[-1], gt.shape[-1])
    return float(((pred[:t, :d] - gt[:t, :d]) ** 2).mean().item())


def _depth_to_luma3(x: torch.Tensor) -> torch.Tensor:
    """Collapse a depth tensor's channel axis to its mean, kept at 3 channels.

    Depth is grayscale by construction; only the rendering path needs this, so
    metrics keep using the raw prediction.
    """
    if x.dim() < 3 or x.shape[-3] != 3:
        return x
    return x.mean(dim=-3, keepdim=True).expand_as(x)


def _depth_metrics(pred_tchw: torch.Tensor, gt_tchw: torch.Tensor, config, rgb_tchw: torch.Tensor | None = None) -> Dict[str, Any]:
    depth_cfg = config.depth
    z_near = float(getattr(depth_cfg, "z_near", 0.1))
    z_far = float(getattr(depth_cfg, "z_far", 10.0))
    mode = str(getattr(depth_cfg, "mode", "log"))
    pred_m, _ = decode_to_depth_m(pred_tchw, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
    gt_m, valid = decode_to_depth_m(gt_tchw, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
    mask = valid
    gray_mse = float(((pred_tchw - gt_tchw) ** 2).mean().item())
    if int(mask.sum()) == 0:
        return {"depth_gray_mse": gray_mse, "depth_absrel": None, "depth_rmse_m": None, "depth_valid_frac": 0.0}
    diff = pred_m - gt_m
    rel = diff.abs() / gt_m.clamp_min(1e-6)
    absrel = float(rel[mask].mean().item())
    rmse = float((diff[mask] ** 2).mean().sqrt().item())
    motion = _depth_motion_mask(gt_m, valid, thresh_m=0.10, rgb=rgb_tchw)
    mot = mask & motion
    bg = mask & ~motion
    out = {
        "depth_gray_mse": gray_mse,
        "depth_absrel": absrel,
        "depth_rmse_m": rmse,
        "depth_valid_frac": float(mask.float().mean().item()),
        "depth_motion_frac": float(mot.sum() / mask.sum().clamp_min(1)),
    }
    if bool(mot.any()):
        out["depth_absrel_motion"] = float(rel[mot].mean().item())
    if bool(bg.any()):
        out["depth_absrel_background"] = float(rel[bg].mean().item())
    out.update(
        _depth_structure_metrics(
            pred_m, gt_m, mask, bg, z_near=z_near, z_far=z_far
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation ckpt on official holdout clips.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--no-strict-load", action="store_true")
    parser.add_argument(
        "--pool",
        default="official_holdout",
        choices=["official_holdout", "train"],
        help="official_holdout=holdout_v1; train=ablation train v1 90% split",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    allocation = apply_pool(config, args.pool)
    config.training.batch_size = 1

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    LOG.info("Creating model enable_depth=%s", bool(getattr(config.model.wm, "enable_depth", False)))
    model = create_eval_model(config)
    load_info = _load_model_weights(model, Path(args.checkpoint), strict=not args.no_strict_load)
    model.to(device)
    model.eval()

    attacher = _build_depth_attacher(config)
    items = _collect_batches(config, int(args.seed), allocation)
    manifest = []
    for i, item in enumerate(items):
        batch = item["batch"]
        eid = batch.get("episode_ids")
        if isinstance(eid, (list, tuple)):
            eid = eid[0]
        manifest.append(
            {
                "index": i,
                "source": item["source"],
                "shard": item["shard"],
                "episode_id": str(eid) if eid is not None else None,
            }
        )
    _json_dump(
        output_dir / "manifest.json",
        {"seed": int(args.seed), "pool": args.pool, "allocation": allocation, "clips": manifest},
    )
    LOG.info("manifest: %s", json.dumps(manifest, ensure_ascii=False))

    guidance_scale = float(getattr(config.model.inference, "guidance_scale", 1.0))
    _inf = config.model.inference
    _action_steps = getattr(_inf, "action_num_inference_timesteps", None)
    action_params = {
        "action_guidance_scale": float(getattr(_inf, "action_guidance_scale", 1.0)),
        "action_num_inference_steps": int(_action_steps) if _action_steps is not None else None,
        "action_smooth_alpha": float(getattr(_inf, "action_temporal_smooth_alpha", 0.0)),
        "depth_smooth_alpha": float(getattr(_inf, "depth_temporal_smooth_alpha", 0.0)),
    }
    want_depth = attacher is not None

    records: List[Dict[str, Any]] = []
    metrics_path = output_dir / "metrics.jsonl"
    start = time.time()
    with metrics_path.open("w", encoding="utf-8") as metrics_fp:
        for sample_idx, item in enumerate(items):
            batch = _rewrite_vlm(config, dict(item["batch"]))
            if attacher is not None:
                batch = dict(attacher.attach(batch))
            sample_dir = output_dir / f"sample_{sample_idx:02d}_{item['source']}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            meta = _sample_meta(batch, sample_idx, sample_idx)
            meta["source"] = item["source"]
            meta["official_holdout"] = args.pool == "official_holdout"
            meta["pool"] = args.pool
            eval_start = time.time()
            try:
                out = _run_one_sample(
                    model=model,
                    batch=batch,
                    device=device,
                    dtype=dtype,
                    num_inference_steps=int(args.num_inference_steps),
                    guidance_scale=guidance_scale,
                    action_inference_params=action_params,
                    return_depth=want_depth,
                )
                if want_depth:
                    predicted_frames_cthw, predicted_actions, predicted_depth = out
                else:
                    predicted_frames_cthw, predicted_actions = out
                    predicted_depth = None
                if predicted_frames_cthw is None:
                    raise ValueError("inference_step returned predicted_frames=None")
                pred_tchw = predicted_frames_cthw[0].detach().cpu().permute(1, 0, 2, 3).float()
                gt_tchw = batch["video_frames"][0].detach().cpu().float()
                first_chw = batch["first_frame"][0].detach().cpu().float()
                min_t = min(pred_tchw.shape[0], gt_tchw.shape[0])
                pred_tchw = pred_tchw[:min_t]
                gt_tchw = gt_tchw[:min_t]
                metrics = _compute_video_metrics(pred_tchw, gt_tchw)
                metrics["action_mse"] = _action_mse(predicted_actions, batch.get("action_sequence"))

                first_uint8 = _image_to_uint8(first_chw)
                gt_uint8 = _tensor_to_uint8_frames(gt_tchw)
                pred_uint8 = _tensor_to_uint8_frames(pred_tchw)
                compare_uint8 = _make_compare_frames(first_uint8, gt_uint8, pred_uint8, include_diff=True)
                Image.fromarray(first_uint8).save(sample_dir / "condition.png")
                _write_video(sample_dir / "gt.mp4", gt_uint8, fps=int(args.fps))
                _write_video(sample_dir / "pred.mp4", pred_uint8, fps=int(args.fps))
                _write_video(sample_dir / "compare.mp4", compare_uint8, fps=int(args.fps))

                if predicted_depth is not None and bool(batch.get("depth_present", torch.tensor([False])).reshape(-1)[0]):
                    pred_d = predicted_depth[0].detach().cpu().permute(1, 0, 2, 3).float()
                    gt_d = batch["depth_frames"][0].detach().cpu().float()
                    min_td = min(pred_d.shape[0], gt_d.shape[0])
                    pred_d = pred_d[:min_td]
                    gt_d = gt_d[:min_td]
                    metrics.update(_depth_metrics(pred_d, gt_d, config))
                    first_d = batch["depth_first"][0].detach().cpu().float()
                    # 深度是灰度语义（编码时 3 通道完全相同），但生成输出的三
                    # 通道会有几个百分点的分歧；直接当 RGB 播会把它渲染成刺眼
                    # 的彩色噪点，严重夸大实际误差。落盘前压成亮度，眼睛看到的
                    # 才是指标真正衡量的那个量。
                    first_d_u8 = _image_to_uint8(_depth_to_luma3(first_d))
                    gt_d_u8 = _tensor_to_uint8_frames(_depth_to_luma3(gt_d))
                    pred_d_u8 = _tensor_to_uint8_frames(_depth_to_luma3(pred_d))
                    Image.fromarray(first_d_u8).save(sample_dir / "depth_condition.png")
                    _write_video(sample_dir / "depth_gt.mp4", gt_d_u8, fps=int(args.fps))
                    _write_video(sample_dir / "depth_pred.mp4", pred_d_u8, fps=int(args.fps))
                    _write_video(
                        sample_dir / "depth_compare.mp4",
                        _make_compare_frames(first_d_u8, gt_d_u8, pred_d_u8, include_diff=True),
                        fps=int(args.fps),
                    )
                    metrics["depth_present"] = True
                else:
                    metrics["depth_present"] = False

                record = {
                    **meta,
                    "ok": True,
                    "seconds": time.time() - eval_start,
                    "num_inference_steps": int(args.num_inference_steps),
                    **metrics,
                }
            except Exception as exc:
                LOG.exception("sample %d failed", sample_idx)
                record = {**meta, "ok": False, "seconds": time.time() - eval_start, "error": repr(exc)}
            _json_dump(sample_dir / "meta.json", record)
            metrics_fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            metrics_fp.flush()
            records.append(record)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ok = [r for r in records if r.get("ok")]
    summary: Dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output": str(output_dir.resolve()),
        "official_holdout": args.pool == "official_holdout",
        "pool": args.pool,
        "seed": int(args.seed),
        "num_clips": len(records),
        "num_ok": len(ok),
        "load_info": load_info,
        "elapsed_seconds": time.time() - start,
        "clips": manifest,
    }
    for key in ("mse", "psnr", "action_mse", "depth_absrel", "depth_rmse_m", "depth_gray_mse"):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
    _json_dump(output_dir / "summary.json", summary)
    LOG.info("Done. %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
