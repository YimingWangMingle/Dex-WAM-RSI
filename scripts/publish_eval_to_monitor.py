#!/usr/bin/env python3
"""Offline ckpt eval → existing motus_monitor run (eval tab).

Uses the same 8 periodic_eval clips as V2/V3/V4 (seed 20260805, shard 0,
4 holdout + 4 train) and stacks depth under RGB in compare.mp4 so the Web
UI needs no change. Later eval_sample writes for the same (step, rank,
sample_index) win in the frontend.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    _action_mse,
    _build_depth_attacher,
    _depth_metrics,
    _depth_to_luma3,
    _rewrite_vlm,
)
from eval_video_ckpt import (  # noqa: E402
    _compute_video_metrics,
    _load_model_weights,
    _run_one_sample,
    create_eval_model,
    load_config,
)
from motus_monitor.store import ChunkedJsonlWriter  # noqa: E402
from thinker_wm.periodic_eval import render  # noqa: E402

LOG = logging.getLogger("publish_eval_to_monitor")

# Default 8-clip set used by in-training periodic_eval (shard 0).
CLIPS = [
    {"source": "agibot_beta_parallel_gripper", "split": "holdout", "shard": 0},
    {"source": "agibot_beta_dexterous_hand", "split": "holdout", "shard": 0},
    {"source": "egoscale_first500h_remaining90pct", "split": "holdout", "shard": 0},
    {"source": "egoscale_second1700h_batch01", "split": "holdout", "shard": 0},
    {"source": "agibot_beta_parallel_gripper", "split": "train", "shard": 0},
    {"source": "agibot_beta_dexterous_hand", "split": "train", "shard": 0},
    {"source": "egoscale_first500h_remaining90pct", "split": "train", "shard": 0},
    {"source": "egoscale_second1700h_batch01", "split": "train", "shard": 0},
]


def _load_clips(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return list(CLIPS)
    raw = json.loads(Path(path).read_text())
    clips = []
    for item in raw:
        clips.append(
            {
                "source": str(item["source"]),
                "split": str(item.get("split", "holdout")),
                "shard": int(item.get("shard", 0)),
                "note": item.get("note"),
            }
        )
    return clips


def _meta_first(batch: Dict[str, Any], key: str) -> Any:
    value = batch.get(key)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _tensor_list(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        if tensor.ndim >= 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        return tensor.tolist()
    return value


def _mean_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = set()
    for row in rows:
        keys.update(k for k, v in row.items() if isinstance(v, (int, float)))
    out: Dict[str, float] = {}
    for key in sorted(keys):
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--clips", default=None, help="JSON list of {source,split,shard}")
    parser.add_argument("--start-rank", type=int, default=0)
    parser.add_argument("--backup-existing", action="store_true", default=False)
    parser.add_argument("--write-round", action="store_true", default=False)
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.9,
        help="CUDA per-process memory fraction. 0.45 when sharing a card with training.",
    )
    args = parser.parse_args()
    clips = _load_clips(Path(args.clips) if args.clips else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(args.run_dir)
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"not a monitor run dir: {run_dir}")

    eval_root = run_dir / "eval"
    step_dir = eval_root / f"step_{int(args.step):07d}"
    if args.backup_existing and step_dir.is_dir():
        backup = eval_root / f"step_{int(args.step):07d}_orig_rgb_only"
        if not backup.exists():
            LOG.info("backing up %s -> %s", step_dir, backup)
            shutil.copytree(step_dir, backup)

    config = load_config(args.config)
    OmegaConf.set_struct(config, False)
    share = bool(getattr(config.model.wm, "depth_share_video_noise", False))
    LOG.info("depth_share_video_noise=%s z_near=%s z_far=%s", share, config.depth.z_near, config.depth.z_far)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), device.index)
    dtype = torch.bfloat16
    model = create_eval_model(config)
    _load_model_weights(model, Path(args.checkpoint), strict=True)
    model.to(device)
    model.eval()
    attacher = _build_depth_attacher(config)
    guidance_scale = float(getattr(config.model.inference, "guidance_scale", 1.0))
    _inf = config.model.inference
    _action_steps = getattr(_inf, "action_num_inference_timesteps", None)
    action_params = {
        "action_guidance_scale": float(getattr(_inf, "action_guidance_scale", 1.0)),
        "action_num_inference_steps": int(_action_steps) if _action_steps is not None else None,
        "action_smooth_alpha": float(getattr(_inf, "action_temporal_smooth_alpha", 0.0)),
        "depth_smooth_alpha": float(getattr(_inf, "depth_temporal_smooth_alpha", 0.0)),
    }

    writer = ChunkedJsonlWriter(eval_root)
    records: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    t_round = time.time()

    for idx, clip in enumerate(clips):
        rank = int(args.start_rank) + idx
        source = clip["source"]
        split = clip["split"]
        shard = int(clip["shard"])
        t0 = time.time()
        batch = create_diverse_eval_batch(
            config,
            source,
            batch_size=1,
            shard_ordinal=shard,
            seed=int(args.seed),
            use_eval_pool=(split == "holdout"),
        )
        batch = _rewrite_vlm(config, dict(batch))
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        eid = _meta_first(batch, "episode_ids")
        LOG.info("rank %d %s %s shard=%d episode=%s", rank, split, source, shard, eid)

        out = _run_one_sample(
            model=model,
            batch=batch,
            device=device,
            dtype=dtype,
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=guidance_scale,
            action_inference_params=action_params,
            return_depth=True,
        )
        predicted_frames_cthw, predicted_actions, predicted_depth = out
        pred_tchw = predicted_frames_cthw[0].detach().cpu().permute(1, 0, 2, 3).float()
        gt_tchw = batch["video_frames"][0].detach().cpu().float()
        first_chw = batch["first_frame"][0].detach().cpu().float()
        n = min(pred_tchw.shape[0], gt_tchw.shape[0])
        metrics = _compute_video_metrics(pred_tchw[:n], gt_tchw[:n])
        metrics["action_mse"] = _action_mse(predicted_actions, batch.get("action_sequence"))

        sample_rel = Path("eval") / f"step_{int(args.step):07d}" / f"rank{rank}" / "sample_0"
        sample_dir = run_dir / sample_rel
        sample_dir.mkdir(parents=True, exist_ok=True)
        first_u8 = render.image_to_uint8(first_chw)
        render.save_png(sample_dir / "first_frame.png", first_u8)

        compare_frames = render.make_compare_frames(
            first_u8,
            render.tensor_to_uint8_frames(gt_tchw[:n]),
            render.tensor_to_uint8_frames(pred_tchw[:n]),
        )
        depth_present = bool(batch.get("depth_present", torch.tensor([False])).reshape(-1)[0])
        if predicted_depth is not None and depth_present:
            pred_d = predicted_depth[0].detach().cpu().permute(1, 0, 2, 3).float()
            gt_d = batch["depth_frames"][0].detach().cpu().float()
            nd = min(pred_d.shape[0], gt_d.shape[0], n)
            pred_d, gt_d = pred_d[:nd], gt_d[:nd]
            metrics.update(_depth_metrics(pred_d, gt_d, config))
            depth_frames = render.make_compare_frames(
                render.image_to_uint8(_depth_to_luma3(batch["depth_first"][0].detach().cpu().float())),
                render.tensor_to_uint8_frames(_depth_to_luma3(gt_d)),
                render.tensor_to_uint8_frames(_depth_to_luma3(pred_d)),
            )
            m = min(len(compare_frames), len(depth_frames))
            compare_frames = np.concatenate([compare_frames[:m], depth_frames[:m]], axis=1)

        render.write_video(sample_dir / "compare.mp4", compare_frames, fps=int(args.fps))

        robot_name = _meta_first(batch, "robot_names")
        task_text = _meta_first(batch, "t5_input_messages")
        vlm_prompt = _meta_first(batch, "vlm_input_messages")
        ts = datetime.now().isoformat(timespec="milliseconds")
        meta_payload = {
            "step": int(args.step),
            "rank": rank,
            "sample_index": 0,
            "source": source,
            "kind": "indexed",
            "split": split,
            "shard_ordinal": shard,
            "video_only": False,
            "robot_name": robot_name,
            "episode_id": str(eid) if eid is not None else None,
            "manifest_id": _meta_first(batch, "manifest_ids"),
            "task_text": task_text,
            "vlm_prompt": vlm_prompt,
            "vlm_output": None,
            "initial_state": _tensor_list(batch.get("initial_state")),
            "gt_actions": _tensor_list(batch.get("action_sequence")),
            "predicted_actions": _tensor_list(predicted_actions),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": guidance_scale,
            "metrics": metrics,
            "offline_republish": True,
        }
        render.write_meta_json(sample_dir / "meta.json", meta_payload)

        sample = {
            "type": "eval_sample",
            "step": int(args.step),
            "ts": ts,
            "rank": rank,
            "sample_index": 0,
            "source": source,
            "dataset": source,
            "split": split,
            "robot_name": robot_name,
            "episode_id": meta_payload["episode_id"],
            "manifest_id": meta_payload["manifest_id"],
            "video_only": False,
            "task_text": task_text,
            "vlm_prompt": vlm_prompt,
            "vlm_output": None,
            "initial_state": meta_payload["initial_state"],
            "metrics": metrics,
            "offline_republish": True,
            "files": {
                "first_frame": str(sample_rel / "first_frame.png"),
                "compare_video": str(sample_rel / "compare.mp4"),
                "gt_video": None,
                "pred_video": None,
                "actions_plot": None,
                "meta": str(sample_rel / "meta.json"),
            },
        }
        writer.write(sample)
        records.append(sample)
        source_counts[source] = source_counts.get(source, 0) + 1
        LOG.info(
            "done rank%d absrel=%s gray_mse=%s psnr=%s (%.1fs)",
            rank,
            metrics.get("depth_absrel"),
            metrics.get("depth_gray_mse"),
            metrics.get("video_psnr") or metrics.get("psnr"),
            time.time() - t0,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "step": int(args.step),
        "num_samples": len(records),
        "sources": source_counts,
        "metrics": _mean_metrics([r["metrics"] for r in records]),
        "offline_republish": True,
    }
    if args.write_round:
        round_rec = {
            "type": "eval_round",
            "step": int(args.step),
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "seconds": time.time() - t_round,
            "num_samples": len(records),
            "ready_ranks": len(records),
            "world_size": int(args.start_rank) + len(records),
            "sources": source_counts,
            "metrics": summary["metrics"],
            "skipped": [],
            "offline_republish": True,
        }
        writer.write(round_rec)
        summary = round_rec
    writer.close()
    LOG.info("published step %d to %s metrics=%s", args.step, run_dir, summary["metrics"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
