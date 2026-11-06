#!/usr/bin/env python3
"""8-rank offline eval of the family holdout set (AgiBot 8 + EgoScale 6 + train 4).

Each process owns one rank, loads the version's model once, then walks that
version's checkpoints. After all ranks finish a step, call --gather.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from thinker_wm.data_prepare.robotwin_3d import motus_t_view_slots  # noqa: E402
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
from thinker_wm.periodic_eval.config import (  # noqa: E402
    assign_sources,
    parse_periodic_eval_config,
    resolve_source_weights,
)
from thinker_wm.periodic_eval.runner import _group_metrics_by_family_split  # noqa: E402
from publish_eval_to_monitor import _mean_metrics, _meta_first, _tensor_list  # noqa: E402

LOG = logging.getLogger("eval_family_holdout")
DEFAULT_EVAL_SET = "robotwin_holdout_100"


def _load_pinned_clips(path: Path, rank: int) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    clips = [dict(item) for item in raw if int(item["rank"]) == int(rank)]
    clips.sort(key=lambda item: int(item.get("sample_index", 0)))
    if not clips:
        raise SystemExit(f"no pinned clips for rank {rank} in {path}")
    return clips


def _rank_clips(config, rank: int, world_size: int) -> List[Dict[str, Any]]:
    pe = parse_periodic_eval_config(config)
    weights = resolve_source_weights(pe, config)
    assignment = assign_sources(
        weights,
        world_size,
        pe.train_pool_fraction,
        holdout_family_counts=pe.holdout_family_counts,
    )[int(rank)]
    n = int(assignment.batch_size) if assignment.batch_size is not None else 1
    if bool(getattr(assignment, "shard_as_start", False)):
        base = int(assignment.shard_ordinal)
    else:
        base = int(assignment.shard_ordinal) * n
    return [
        {
            "source": assignment.source_name,
            "split": assignment.split,
            "shard": base + i,
        }
        for i in range(n)
    ]


def _eval_one_clip(model, config, attacher, clip, device, dtype, infer_steps, guidance, action_params, fps, run_dir, step, rank, sample_idx, teacher_force_depth=False, eval_set=DEFAULT_EVAL_SET):
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
    robotwin_clip = (
        str(clip.get("source")) == "robotwin_3d"
        or str(eid or "").startswith("robotwin_3d:")
    )
    if robotwin_clip:
        present = batch.get("depth_present")
        ok = present is not None and bool(torch.as_tensor(present).reshape(-1)[0])
        if attacher is None or not ok:
            raise RuntimeError(
                f"robotwin_3d eval clip has no depth (source={clip.get('source')} "
                f"eid={eid} attacher={'none' if attacher is None else 'ok'})"
            )
    t0 = time.time()
    out = _run_one_sample(
        model=model,
        batch=batch,
        device=device,
        dtype=dtype,
        num_inference_steps=int(infer_steps),
        guidance_scale=guidance,
        action_inference_params=action_params,
        return_depth=True,
        teacher_force_depth=bool(teacher_force_depth),
    )
    predicted_frames_cthw, predicted_actions, predicted_depth = out
    pred_tchw = predicted_frames_cthw[0].detach().cpu().permute(1, 0, 2, 3).float()
    gt_tchw = batch["video_frames"][0].detach().cpu().float()
    first_chw = batch["first_frame"][0].detach().cpu().float()
    n = min(pred_tchw.shape[0], gt_tchw.shape[0])
    slots = None
    if robotwin_clip:
        slots = motus_t_view_slots(
            (int(pred_tchw.shape[-2]), int(pred_tchw.shape[-1]))
        )
    metrics = _compute_video_metrics(pred_tchw[:n], gt_tchw[:n], view_slots=slots)
    metrics["action_mse"] = _action_mse(predicted_actions, batch.get("action_sequence"))

    sample_rel = Path("eval") / f"step_{int(step):07d}" / f"rank{rank}" / f"sample_{sample_idx}"
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
        rgb_tchw = batch["video_frames"][0].detach().cpu().float()[:nd]
        metrics.update(_depth_metrics(pred_d, gt_d, config, rgb_tchw=rgb_tchw))
        depth_frames = render.make_compare_frames(
            render.image_to_uint8(_depth_to_luma3(batch["depth_first"][0].detach().cpu().float())),
            render.tensor_to_uint8_frames(_depth_to_luma3(gt_d)),
            render.tensor_to_uint8_frames(_depth_to_luma3(pred_d)),
        )
        m = min(len(compare_frames), len(depth_frames))
        compare_frames = np.concatenate([compare_frames[:m], depth_frames[:m]], axis=1)
    render.write_video(sample_dir / "compare.mp4", compare_frames, fps=int(fps))

    robot_name = _meta_first(batch, "robot_names")
    task_text = _meta_first(batch, "t5_input_messages")
    vlm_prompt = _meta_first(batch, "vlm_input_messages")
    ts = datetime.now().isoformat(timespec="milliseconds")
    meta_payload = {
        "step": int(step),
        "rank": int(rank),
        "sample_index": int(sample_idx),
        "source": clip["source"],
        "kind": "indexed",
        "split": clip["split"],
        "shard_ordinal": int(clip["shard"]),
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
        "num_inference_steps": int(infer_steps),
        "guidance_scale": guidance,
        "action_smooth_alpha": float(action_params.get("action_smooth_alpha", 0.0)),
        "depth_smooth_alpha": float(action_params.get("depth_smooth_alpha", 0.0)),
        "depth_read_by_action": bool(getattr(config.model.wm, "depth_read_by_action", False)),
        "depth_read_by_video": bool(getattr(config.model.wm, "depth_read_by_video", False)),
        "depth_read_by_und": bool(getattr(config.model.wm, "depth_read_by_und", False)),
        "video_read_by_action": bool(getattr(config.model.wm, "video_read_by_action", True)),
        "teacher_force_depth": bool(teacher_force_depth),
        "metrics": metrics,
        "eval_set": str(eval_set),
    }
    render.write_meta_json(sample_dir / "meta.json", meta_payload)
    sample = {
        "type": "eval_sample",
        "step": int(step),
        "ts": ts,
        "rank": int(rank),
        "sample_index": int(sample_idx),
        "source": clip["source"],
        "dataset": clip["source"],
        "split": clip["split"],
        "robot_name": robot_name,
        "episode_id": meta_payload["episode_id"],
        "manifest_id": meta_payload["manifest_id"],
        "video_only": False,
        "task_text": task_text,
        "vlm_prompt": vlm_prompt,
        "vlm_output": None,
        "initial_state": meta_payload["initial_state"],
        "metrics": metrics,
        "eval_set": str(eval_set),
        "files": {
            "first_frame": str(sample_rel / "first_frame.png"),
            "compare_video": str(sample_rel / "compare.mp4"),
            "gt_video": None,
            "pred_video": None,
            "actions_plot": None,
            "meta": str(sample_rel / "meta.json"),
        },
    }
    LOG.info(
        "rank%d sample%d %s %s shard=%s absrel=%s (%.1fs) %s",
        rank,
        sample_idx,
        clip["split"],
        clip["source"],
        clip["shard"],
        metrics.get("depth_absrel"),
        time.time() - t0,
        eid,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sample


def run_rank(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Running A→B→C wrappers already have C in argv; this file is the off switch.
    skip_c = Path(
        "/data/xiongwei.zhao/thinkerwam-depth/eval/v10_robotwin_holdout100_abc5000/SKIP_C"
    )
    if bool(getattr(args, "teacher_force_depth", False)) and skip_c.is_file():
        LOG.info("skip teacher-force C (%s)", skip_c)
        return 0
    config = load_config(args.config)
    OmegaConf.set_struct(config, False)
    if args.depth_read_by_action is not None:
        config.model.wm.depth_read_by_action = bool(args.depth_read_by_action)
        LOG.info("override depth_read_by_action=%s", config.model.wm.depth_read_by_action)
    if args.depth_read_by_video is not None:
        config.model.wm.depth_read_by_video = bool(args.depth_read_by_video)
        LOG.info("override depth_read_by_video=%s", config.model.wm.depth_read_by_video)
    if args.depth_read_by_und is not None:
        config.model.wm.depth_read_by_und = bool(args.depth_read_by_und)
        LOG.info("override depth_read_by_und=%s", config.model.wm.depth_read_by_und)
    if getattr(args, "video_read_by_action", None) is not None:
        config.model.wm.video_read_by_action = bool(args.video_read_by_action)
        LOG.info("override video_read_by_action=%s", config.model.wm.video_read_by_action)
    if args.clips:
        clips = _load_pinned_clips(Path(args.clips), args.rank)
    else:
        clips = _rank_clips(config, args.rank, args.world_size)
    LOG.info("rank %d/%d clips=%s", args.rank, args.world_size, clips)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), device.index)
    dtype = torch.bfloat16
    model = create_eval_model(config)
    model.to(device)
    model.eval()
    attacher = _build_depth_attacher(config)
    guidance = float(getattr(config.model.inference, "guidance_scale", 1.0))
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
    LOG.info(
        "inference params=%s depth_read_by_action=%s depth_read_by_video=%s depth_read_by_und=%s video_read_by_action=%s teacher_force_depth=%s",
        action_params,
        bool(getattr(config.model.wm, "depth_read_by_action", False)),
        bool(getattr(config.model.wm, "depth_read_by_video", False)),
        bool(getattr(config.model.wm, "depth_read_by_und", False)),
        bool(getattr(config.model.wm, "video_read_by_action", True)),
        bool(getattr(args, "teacher_force_depth", False)),
    )
    LOG.info("video extras: psnr+per-view only (ssim/lpips off)")
    infer_steps = int(args.num_inference_steps)
    run_dir = Path(args.run_dir)
    steps = [int(s) for s in args.steps]
    ckpts = [Path(p) for p in args.checkpoints]
    if len(steps) != len(ckpts):
        raise SystemExit("steps and checkpoints length mismatch")

    writer = ChunkedJsonlWriter(run_dir / "eval")
    for step, ckpt in zip(steps, ckpts):
        LOG.info("load %s step=%d", ckpt, step)
        _load_model_weights(model, ckpt, strict=True)
        model.to(device)
        model.eval()
        for sample_idx, clip in enumerate(clips):
            sample = _eval_one_clip(
                model, config, attacher, clip, device, dtype, infer_steps,
                guidance, action_params, args.fps, run_dir, step, args.rank, sample_idx,
                teacher_force_depth=bool(getattr(args, "teacher_force_depth", False)),
                eval_set=str(getattr(args, "eval_set", DEFAULT_EVAL_SET)),
            )
            writer.write(sample)
        done = run_dir / "eval" / f"step_{int(step):07d}" / f"rank{args.rank}" / ".rank_done"
        done.write_text(json.dumps({"rank": args.rank, "n": len(clips), "step": step}) + "\n")
    writer.close()
    return 0


def gather(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(args.run_dir)
    step = int(args.step)
    step_dir = run_dir / "eval" / f"step_{step:07d}"
    eval_set = str(getattr(args, "eval_set", DEFAULT_EVAL_SET))
    records = []
    for meta in sorted(step_dir.glob("rank*/sample_*/meta.json")):
        payload = json.loads(meta.read_text())
        if payload.get("eval_set") != eval_set:
            continue
        records.append(
            {
                "source": payload.get("source"),
                "split": payload.get("split"),
                "metrics": payload.get("metrics") or {},
                "episode_id": payload.get("episode_id"),
                "rank": payload.get("rank"),
                "sample_index": payload.get("sample_index"),
                "action_smooth_alpha": payload.get("action_smooth_alpha"),
                "depth_smooth_alpha": payload.get("depth_smooth_alpha"),
                "depth_read_by_action": payload.get("depth_read_by_action"),
                "depth_read_by_video": payload.get("depth_read_by_video"),
                "depth_read_by_und": payload.get("depth_read_by_und"),
                "video_read_by_action": payload.get("video_read_by_action"),
                "teacher_force_depth": payload.get("teacher_force_depth"),
            }
        )
    if not records:
        raise SystemExit(f"no {eval_set} metas under {step_dir}")
    grouped = _group_metrics_by_family_split(records)
    summary = {
        "type": "eval_round",
        "step": step,
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "num_samples": len(records),
        "ready_ranks": len({r["rank"] for r in records}),
        "world_size": int(args.world_size),
        "sources": {},
        "metrics": _mean_metrics([r["metrics"] for r in records]),
        "metrics_by_group": grouped,
        "eval_set": eval_set,
        "action_smooth_alpha": records[0].get("action_smooth_alpha"),
        "depth_smooth_alpha": records[0].get("depth_smooth_alpha"),
        "depth_read_by_action": records[0].get("depth_read_by_action"),
        "depth_read_by_video": records[0].get("depth_read_by_video"),
        "depth_read_by_und": records[0].get("depth_read_by_und"),
        "video_read_by_action": records[0].get("video_read_by_action"),
        "teacher_force_depth": records[0].get("teacher_force_depth"),
        "skipped": [],
    }
    for rec in records:
        summary["sources"][rec["source"]] = summary["sources"].get(rec["source"], 0) + 1
    writer = ChunkedJsonlWriter(run_dir / "eval")
    writer.write(summary)
    writer.close()
    out = run_dir / "eval" / f"step_{step:07d}" / f"{eval_set}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOG.info("gathered step %d n=%d metrics=%s groups=%s", step, len(records), summary["metrics"], grouped)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--steps", nargs="*")
    parser.add_argument("--step", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument(
        "--clips",
        default=None,
        help="Pinned clip JSON (rank/source/split/shard/episode_id). "
        "Use this to freeze the V5 family-holdout 18 set.",
    )
    parser.add_argument(
        "--depth-read-by-action",
        dest="depth_read_by_action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval-only override of model.wm.depth_read_by_action. "
        "Omit to keep the yaml. --no-depth-read-by-action = action 不读 depth.",
    )
    parser.add_argument(
        "--depth-read-by-video",
        dest="depth_read_by_video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval-only override of model.wm.depth_read_by_video. "
        "Omit to keep the yaml. --no-depth-read-by-video = RGB 不读 depth.",
    )
    parser.add_argument(
        "--depth-read-by-und",
        dest="depth_read_by_und",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval-only override of model.wm.depth_read_by_und. "
        "Omit to keep the yaml. --no-depth-read-by-und = und 不读 depth.",
    )
    parser.add_argument(
        "--video-read-by-action",
        dest="video_read_by_action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval-only override of model.wm.video_read_by_action. "
        "Omit to keep the default True. --no-video-read-by-action = action 不读 RGB.",
    )
    parser.add_argument(
        "--teacher-force-depth",
        dest="teacher_force_depth",
        action="store_true",
        default=False,
        help="Pin the full GT depth latent (t_depth=0). Action still denoises. "
        "Use with --depth-read-by-action to test an oracle-depth upper bound.",
    )
    args = parser.parse_args()
    if args.gather:
        return gather(args)
    if not args.config or not args.checkpoints or not args.steps:
        parser.error("--config --checkpoints --steps required unless --gather")
    return run_rank(args)


if __name__ == "__main__":
    raise SystemExit(main())
