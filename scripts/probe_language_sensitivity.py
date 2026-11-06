#!/usr/bin/env python3
"""Language-sensitivity probe for a parked V13 ckpt. Do not run on the V13 train node.

Same holdout window, swap only the instruction. Report action MSE spread across
that episode's official seen paraphrases vs official unseen templates.

  CUDA_VISIBLE_DEVICES=0 python scripts/probe_language_sensitivity.py \
      --config configs/robotwin_3d_full_v13_rgbd.yaml \
      --ckpt /data/xiongwei.zhao/thinkerwam-depth/checkpoints/v13_robotwin_full/robotwin_3d_full_v13_rgbd/V13_full/checkpoint_step_10000 \
      --output-dir tmp/lang_probe/step_10000

Interpretation (action MSE vs GT on the same window):
  seen_cv < 0.05 and unseen_mean ≈ catalog_mse
      wording is almost ignored → mixing 100 sentences mid-run will not move
      closed-loop success much.
  unseen_mean − catalog_mse is large, seen_cv small
      the model locked onto tasks[0] wording; paraphrase sampling is worth a
      V14 change, not a V13 mid-run swap.
  distractor_mse ≈ seen_mean
      VLM is unused; language recipe is the wrong lever.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import (  # noqa: E402
    _action_mse,
    _build_depth_attacher,
    _rewrite_vlm,
)
from eval_video_ckpt import (  # noqa: E402
    _load_model_weights,
    _run_one_sample,
    create_eval_model,
    load_config,
)
from publish_eval_to_monitor import _meta_first  # noqa: E402

LOG = logging.getLogger("lang_probe")

DEFAULT_CONFIG = PROJECT_ROOT / "third_party/Thinker-WM/configs/robotwin_3d_full_v13_rgbd.yaml"
DEFAULT_EPISODES = Path("/data/xiongwei.zhao/robotwin_3d/meta/episodes.jsonl")
DEFAULT_TASK_MAP = Path("/data/xiongwei.zhao/robotwin_3d/task_episode_map.json")
DEFAULT_INSTRUCTIONS = Path(
    "/code1/zxw_data/pretrain/world-motion-action-model/code/"
    "flex-pi-main/third_party/RoboTwin/description/task_instruction"
)
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "cv": None}
    mean = statistics.fmean(nums)
    std = statistics.pstdev(nums) if len(nums) > 1 else 0.0
    return {
        "n": len(nums),
        "mean": mean,
        "std": std,
        "min": min(nums),
        "max": max(nums),
        "cv": (std / mean) if mean > 1e-12 else None,
    }


def _load_jsonl_tasks(path: Path) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            ep_i = int(rec["episode_index"])
            texts = []
            seen = set()
            for item in rec.get("tasks") or []:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
            out[ep_i] = texts
    return out


def _task_of(task_map: Dict[str, Any], episode_index: int) -> str:
    tasks = task_map.get("tasks") or {}
    for name, spec in tasks.items():
        if int(spec["episode_start"]) <= episode_index <= int(spec["episode_end"]):
            return str(name)
    return "unknown"


def _match_template(template: str, filled: str) -> Optional[Dict[str, str]]:
    keys = _PLACEHOLDER.findall(template)
    if not keys:
        return {} if template.strip() == filled.strip() else None
    pattern = "^" + re.escape(template)
    for key in keys:
        pattern = pattern.replace(re.escape("{" + key + "}"), r"(.+?)", 1)
    pattern += "$"
    match = re.match(pattern, filled.strip())
    if not match:
        return None
    return {key: match.group(i + 1) for i, key in enumerate(keys)}


def _recover_slots(filled_seen: Sequence[str], templates: Sequence[str]) -> Dict[str, str]:
    """Recover {A}/{a}/… by matching official *seen* templates to filled lines.

    Merge across matches: some templates omit the arm slot.
    """
    merged: Dict[str, str] = {}
    votes: Dict[str, Dict[str, int]] = {}
    for filled in filled_seen:
        for template in templates:
            slots = _match_template(str(template), str(filled))
            if not slots:
                continue
            for key, value in slots.items():
                value = value.strip()
                if not value:
                    continue
                votes.setdefault(key, {})
                votes[key][value] = votes[key].get(value, 0) + 1
    for key, counts in votes.items():
        merged[key] = max(counts.items(), key=lambda item: item[1])[0]
    if "a" not in merged:
        arm = re.search(r"\bthe (left|right) arm\b", " ".join(filled_seen), flags=re.I)
        if arm:
            merged["a"] = f"the {arm.group(1).lower()} arm"
    return merged


def _fill(template: str, slots: Dict[str, str]) -> Optional[str]:
    text = str(template)
    for key, value in slots.items():
        text = text.replace("{" + key + "}", value)
    if _PLACEHOLDER.search(text):
        return None
    return text.strip() or None


def _pick_texts(
    *,
    catalog_text: str,
    seen: Sequence[str],
    seen_templates: Sequence[str],
    unseen_templates: Sequence[str],
    distractors: Sequence[str],
    n_seen: int,
    n_unseen: int,
    n_distractor: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = [
        {"split": "catalog", "text": catalog_text},
    ]
    extras = [s for s in seen if s != catalog_text]
    rng.shuffle(extras)
    for text in extras[: max(0, n_seen)]:
        prompts.append({"split": "seen", "text": text})
    slots = _recover_slots(seen, seen_templates)
    filled_unseen: List[str] = []
    if slots:
        for template in unseen_templates:
            text = _fill(str(template), slots)
            if text and text not in filled_unseen and text != catalog_text:
                filled_unseen.append(text)
    rng.shuffle(filled_unseen)
    for text in filled_unseen[: max(0, n_unseen)]:
        prompts.append({"split": "unseen", "text": text})
    pool = [t for t in distractors if t and t != catalog_text]
    rng.shuffle(pool)
    for text in pool[: max(0, n_distractor)]:
        prompts.append({"split": "distractor", "text": text})
    return prompts


def _episode_index(episode_id: str) -> int:
    # robotwin_3d:000123
    return int(str(episode_id).rsplit(":", 1)[-1])


def _swap_text(config, batch: Dict[str, Any], text: str) -> Dict[str, Any]:
    batch = dict(batch)
    batch["t5_input_messages"] = [str(text)]
    batch["language_embedding"] = None
    return _rewrite_vlm(config, batch)


@torch.no_grad()
def _eval_prompt(model, config, attacher, batch, text, device, dtype, infer_steps, guidance, action_params):
    tagged = _swap_text(config, batch, text)
    if attacher is not None:
        # Depth is already on the original batch; do not re-attach.
        tagged = dict(tagged)
        for key in (
            "depth_first",
            "depth_frames",
            "depth_present",
        ):
            if key in batch and key not in tagged:
                tagged[key] = batch[key]
    predicted_frames, predicted_actions, _predicted_depth = _run_one_sample(
        model=model,
        batch=tagged,
        device=device,
        dtype=dtype,
        num_inference_steps=int(infer_steps),
        guidance_scale=guidance,
        action_inference_params=action_params,
        return_depth=True,
        teacher_force_depth=False,
    )
    del predicted_frames
    return _action_mse(predicted_actions, batch.get("action_sequence"))


def _report_from_rows(
    *,
    rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    ckpt: str,
    config: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    by_split: Dict[str, List[float]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(float(row["action_mse"]))
    report: Dict[str, Any] = {
        "ckpt": ckpt,
        "config": config,
        "n_windows": len(summaries),
        "n_prompts": len(rows),
        "overall": {split: _stats(vals) for split, vals in by_split.items()},
        "windows": summaries,
        "verdict_hint": _verdict_from_windows(summaries),
    }
    if extra:
        report.update(extra)
    return report


def _merge_worker_dirs(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    sources: List[str] = []
    for raw in args.merge_from:
        src = Path(raw)
        sources.append(str(src))
        rows_path = src / "rows.jsonl"
        report_path = src / "report.json"
        if rows_path.exists():
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    all_rows.append(json.loads(line))
        if report_path.exists():
            summaries.extend(json.loads(report_path.read_text(encoding="utf-8")).get("windows") or [])
    summaries.sort(key=lambda item: int(item.get("shard", 0)))
    all_rows.sort(key=lambda item: (int(item.get("shard", 0)), str(item.get("split")), str(item.get("text"))))
    report = _report_from_rows(
        rows=all_rows,
        summaries=summaries,
        ckpt=str(args.ckpt),
        config=str(args.config),
        extra={"merged_from": sources},
    )
    (out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.info("merged %d rows / %d windows -> %s", len(all_rows), len(summaries), out_dir / "report.json")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(report["verdict_hint"], ensure_ascii=False, indent=2))
    return 0


def _summarize_window(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, List[float]] = {}
    catalog = None
    for row in rows:
        by_split.setdefault(row["split"], []).append(float(row["action_mse"]))
        if row["split"] == "catalog":
            catalog = float(row["action_mse"])
    out = {split: _stats(vals) for split, vals in by_split.items()}
    out["catalog_mse"] = catalog
    seen_mean = out.get("seen", {}).get("mean")
    unseen_mean = out.get("unseen", {}).get("mean")
    if catalog is not None and unseen_mean is not None:
        out["unseen_minus_catalog"] = unseen_mean - catalog
    if seen_mean is not None and unseen_mean is not None:
        out["unseen_minus_seen"] = unseen_mean - seen_mean
    return out


def _verdict_from_windows(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-window ratios. Do not compute CV on MSEs mixed across tasks."""
    seen_cvs = [
        float(item["seen"]["cv"])
        for item in summaries
        if isinstance(item.get("seen"), dict) and item["seen"].get("cv") is not None
    ]
    unseen_deltas = [
        float(item["unseen_minus_catalog"])
        for item in summaries
        if item.get("unseen_minus_catalog") is not None
    ]
    unseen_ratios = []
    distractor_deltas = []
    for item in summaries:
        catalog = item.get("catalog_mse")
        unseen_mean = (item.get("unseen") or {}).get("mean")
        distractor_mean = (item.get("distractor") or {}).get("mean")
        if catalog and unseen_mean is not None and catalog > 1e-12:
            unseen_ratios.append(float(unseen_mean) / float(catalog))
        if catalog is not None and distractor_mean is not None:
            distractor_deltas.append(float(distractor_mean) - float(catalog))
    return {
        "n_windows": len(summaries),
        "mean_seen_cv": statistics.fmean(seen_cvs) if seen_cvs else None,
        "mean_unseen_minus_catalog": (
            statistics.fmean(unseen_deltas) if unseen_deltas else None
        ),
        "mean_unseen_over_catalog": (
            statistics.fmean(unseen_ratios) if unseen_ratios else None
        ),
        "mean_distractor_minus_catalog": (
            statistics.fmean(distractor_deltas) if distractor_deltas else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes-jsonl", default=str(DEFAULT_EPISODES))
    parser.add_argument("--task-map", default=str(DEFAULT_TASK_MAP))
    parser.add_argument("--instruction-dir", default=str(DEFAULT_INSTRUCTIONS))
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate only these window shard ids. Plan is still built for all windows first.",
    )
    parser.add_argument(
        "--merge-from",
        nargs="+",
        default=None,
        help="Merge worker rows/report dirs into --output-dir. No model load, no GPU.",
    )
    parser.add_argument("--n-seen", type=int, default=20)
    parser.add_argument("--n-unseen", type=int, default=8)
    parser.add_argument("--n-distractor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected prompts only. No model load, no GPU.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.merge_from:
        return _merge_worker_dirs(args)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"ckpt not found: {ckpt}")

    episodes = _load_jsonl_tasks(Path(args.episodes_jsonl))
    task_map = json.loads(Path(args.task_map).read_text())
    instruction_dir = Path(args.instruction_dir)
    template_cache: Dict[str, Dict[str, Any]] = {}

    config = load_config(args.config)
    OmegaConf.set_struct(config, False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    windows: List[Dict[str, Any]] = []
    for shard in range(int(args.n_windows)):
        batch = create_diverse_eval_batch(
            config,
            "robotwin_3d",
            batch_size=1,
            shard_ordinal=shard,
            seed=int(args.seed),
            use_eval_pool=True,
        )
        eid = str(_meta_first(batch, "episode_ids"))
        ep_i = _episode_index(eid)
        task = _task_of(task_map, ep_i)
        if task not in template_cache:
            path = instruction_dir / f"{task}.json"
            template_cache[task] = json.loads(path.read_text()) if path.exists() else {}
        catalog_text = str((batch.get("t5_input_messages") or [""])[0]).strip()
        seen = episodes.get(ep_i) or ([catalog_text] if catalog_text else [])
        other_task = next(
            name
            for name in task_map["tasks"]
            if name != task
        )
        other_start = int(task_map["tasks"][other_task]["episode_start"])
        distractors = episodes.get(other_start) or []
        slots = _recover_slots(seen, template_cache[task].get("seen") or [])
        prompts = _pick_texts(
            catalog_text=catalog_text,
            seen=seen,
            seen_templates=template_cache[task].get("seen") or [],
            unseen_templates=template_cache[task].get("unseen") or [],
            distractors=distractors,
            n_seen=int(args.n_seen),
            n_unseen=int(args.n_unseen),
            n_distractor=int(args.n_distractor),
            rng=rng,
        )
        windows.append(
            {
                "shard": shard,
                "episode_id": eid,
                "episode_index": ep_i,
                "task_name": task,
                "catalog_text": catalog_text,
                "n_seen_available": len(seen),
                "recovered_slots": slots,
                "prompts": prompts,
            }
        )
        LOG.info(
            "window shard=%d %s task=%s prompts=%d (catalog/seen/unseen/distractor=%s)",
            shard,
            eid,
            task,
            len(prompts),
            {
                split: sum(1 for p in prompts if p["split"] == split)
                for split in ("catalog", "seen", "unseen", "distractor")
            },
        )

    if args.shards is not None:
        wanted = {int(s) for s in args.shards}
        windows = [spec for spec in windows if int(spec["shard"]) in wanted]
        if not windows:
            raise SystemExit(f"no windows left for --shards {sorted(wanted)}")
        LOG.info("keeping shards %s", sorted(int(spec["shard"]) for spec in windows))

    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(windows, ensure_ascii=False, indent=2) + "\n")
    LOG.info("wrote prompt plan %s", plan_path)
    if args.dry_run:
        return 0

    attacher = _build_depth_attacher(config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            float(args.memory_fraction), device.index
        )
    dtype = torch.bfloat16
    model = create_eval_model(config)
    model.to(device)
    model.eval()
    loaded = _load_model_weights(model, ckpt)
    LOG.info("loaded weights from %s keys=%s", ckpt, list(loaded) if isinstance(loaded, dict) else type(loaded))

    inf = config.model.inference
    infer_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(getattr(inf, "num_inference_timesteps", 10))
    )
    guidance = float(getattr(inf, "guidance_scale", 1.0))
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

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for spec in windows:
        batch = create_diverse_eval_batch(
            config,
            "robotwin_3d",
            batch_size=1,
            shard_ordinal=int(spec["shard"]),
            seed=int(args.seed),
            use_eval_pool=True,
        )
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        present = batch.get("depth_present")
        if present is None or not bool(torch.as_tensor(present).reshape(-1)[0]):
            raise RuntimeError(f"window {spec['episode_id']} has no depth")
        rows: List[Dict[str, Any]] = []
        window_seed = int(args.seed) * 10007 + int(spec["shard"])
        for prompt in spec["prompts"]:
            # Same window + same noise; only the instruction changes.
            # Unseeded randn would confound wording with denoising noise.
            torch.manual_seed(window_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(window_seed)
            t0 = time.time()
            mse = _eval_prompt(
                model,
                config,
                attacher,
                batch,
                prompt["text"],
                device,
                dtype,
                infer_steps,
                guidance,
                action_params,
            )
            row = {
                "episode_id": spec["episode_id"],
                "episode_index": spec["episode_index"],
                "task_name": spec["task_name"],
                "shard": spec["shard"],
                "split": prompt["split"],
                "text": prompt["text"],
                "action_mse": mse,
                "seconds": time.time() - t0,
            }
            rows.append(row)
            all_rows.append(row)
            LOG.info(
                "%s %s mse=%.5f (%.1fs) %s",
                spec["episode_id"],
                prompt["split"],
                float(mse),
                row["seconds"],
                prompt["text"][:80],
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summary = {
            "episode_id": spec["episode_id"],
            "task_name": spec["task_name"],
            "shard": spec["shard"],
            **_summarize_window(rows),
        }
        summaries.append(summary)

    by_split: Dict[str, List[float]] = {}
    for row in all_rows:
        by_split.setdefault(row["split"], []).append(float(row["action_mse"]))
    report = {
        "ckpt": str(ckpt),
        "config": str(args.config),
        "n_windows": len(windows),
        "n_prompts": len(all_rows),
        "overall": {split: _stats(vals) for split, vals in by_split.items()},
        "windows": summaries,
        "verdict_hint": _verdict_from_windows(summaries),
    }
    (out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows)
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    LOG.info("wrote %s", out_dir / "report.json")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(report["verdict_hint"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
