#!/usr/bin/env python3
"""Clean Agibot_beta Motus Manifest records with action/state magnitude guard.

This script never decodes video. It optionally loads only the manifest
state/action storage used by training, applies the same magnitude guard, and
writes a cleaned JSONL plus a bad-record JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thinker_wm.data_prepare import ensure_motus_data_importable

ensure_motus_data_importable()

from motus_data.datasets.action_safety import (  # noqa: E402
    ActionMagnitudeOutlier,
    tensor_abs_max,
    validate_action_state_magnitude,
)
from motus_data.datasets.manifest import MotusManifestDataset  # noqa: E402


DEFAULT_BAD_TASKS = (
    "Water the flowers",
    "Print documents with a printer.",
    "Pack the medicine",
    "Clap hands",
    "Wave goodbye",
)


def parse_task_values(values: Optional[Iterable[str]]) -> set[str]:
    tasks: set[str] = set()
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                tasks.add(part)
    return tasks


def load_task_file(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    tasks = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tasks.add(line)
    return tasks


def make_action_loader(args: argparse.Namespace) -> MotusManifestDataset:
    loader = MotusManifestDataset.__new__(MotusManifestDataset)
    loader.frames_clip_num = int(args.num_video_frames) + 1
    loader.num_video_frames = int(args.num_video_frames)
    loader.video_action_freq_ratio = int(args.video_action_freq_ratio)
    loader.default_action_dim = int(args.action_dim)
    loader.default_state_dim = int(args.state_dim)
    loader.default_source_dataset = "agibot_beta"
    loader.default_robot_name = "agibot-a2-d"
    loader.action_chunk_size = loader.frames_clip_num * loader.video_action_freq_ratio - 1
    return loader


def record_task(record: Dict[str, Any]) -> str:
    return str(record.get("task") or "")


def check_record(
    loader: MotusManifestDataset,
    record: Dict[str, Any],
    *,
    threshold: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    initial_state, action_sequence = loader._load_state_action(record)
    source_dataset = loader._source_dataset(record)
    try:
        validate_action_state_magnitude(
            source_dataset=source_dataset,
            action_sequence=action_sequence,
            initial_state=initial_state,
            manifest_id=record.get("id"),
            manifest_task=record.get("task"),
            threshold=threshold,
            sources={"agibot_beta"},
        )
    except ActionMagnitudeOutlier as exc:
        return False, "action_magnitude_outlier", dict(exc.details)
    details = {
        "action_abs_max": tensor_abs_max(action_sequence),
        "state_abs_max": tensor_abs_max(initial_state),
    }
    return True, "", details


def write_jsonl_line(handle, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Agibot_beta Motus Manifest JSONL.")
    parser.add_argument("--output", required=True, help="Cleaned output Manifest JSONL.")
    parser.add_argument("--bad-output", required=True, help="Rejected records JSONL with reason/details.")
    parser.add_argument("--report", required=True, help="Summary report JSON.")
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--guard-scan", action="store_true", help="Load action/state and apply magnitude guard.")
    parser.add_argument(
        "--bad-task",
        action="append",
        default=None,
        help="Task to reject before guard scan. Can be repeated or comma-separated.",
    )
    parser.add_argument("--bad-task-file", default=None, help="Optional newline-separated task blacklist.")
    parser.add_argument("--no-default-bad-tasks", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--video-action-freq-ratio", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument("--state-dim", type=int, default=26)
    parser.add_argument("--shard-index", type=int, default=0, help="0-based line shard index.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total line shards.")
    parser.add_argument(
        "--keep-load-errors",
        action="store_true",
        help="Keep records whose action/state load fails instead of rejecting them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    input_path = Path(args.input)
    output_path = Path(args.output)
    bad_output_path = Path(args.bad_output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bad_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    bad_tasks = set()
    if not args.no_default_bad_tasks:
        bad_tasks.update(DEFAULT_BAD_TASKS)
    bad_tasks.update(parse_task_values(args.bad_task))
    bad_tasks.update(load_task_file(args.bad_task_file))

    loader = make_action_loader(args)
    counters: Counter[str] = Counter()
    bad_by_task: Counter[str] = Counter()
    bad_by_reason: Counter[str] = Counter()
    started = time.time()

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as clean, bad_output_path.open(
        "w", encoding="utf-8"
    ) as bad:
        for line_no, line in enumerate(src, start=1):
            if args.limit is not None and counters["seen"] >= args.limit:
                break
            if (line_no - 1) % args.num_shards != args.shard_index:
                continue

            counters["seen"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                counters["bad"] += 1
                bad_by_reason["json_decode_error"] += 1
                write_jsonl_line(
                    bad,
                    {
                        "line_no": line_no,
                        "reason": "json_decode_error",
                        "error": str(exc),
                        "raw": line.rstrip("\n")[:2000],
                    },
                )
                continue

            task = record_task(record)
            if task:
                counters[f"task_seen:{task}"] += 1

            reject_reason = ""
            details: Dict[str, Any] = {}
            if task in bad_tasks:
                reject_reason = "bad_task_blacklist"
            elif args.guard_scan:
                try:
                    ok, reason, details = check_record(loader, record, threshold=args.threshold)
                    if not ok:
                        reject_reason = reason
                except Exception as exc:
                    details = {"error": repr(exc)}
                    reject_reason = "action_state_load_error"
                    if args.keep_load_errors:
                        reject_reason = ""

            if reject_reason:
                counters["bad"] += 1
                bad_by_task[task] += 1
                bad_by_reason[reject_reason] += 1
                write_jsonl_line(
                    bad,
                    {
                        "line_no": line_no,
                        "reason": reject_reason,
                        "id": record.get("id"),
                        "task": task,
                        "source_dataset": record.get("source_dataset"),
                        "details": details,
                        "record": record,
                    },
                )
            else:
                counters["kept"] += 1
                clean.write(line)

            if args.progress_every > 0 and counters["seen"] % args.progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                rate = counters["seen"] / elapsed
                print(
                    f"seen={counters['seen']} kept={counters['kept']} bad={counters['bad']} "
                    f"rate={rate:.1f}/s shard={args.shard_index}/{args.num_shards}",
                    flush=True,
                )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "bad_output": str(bad_output_path),
        "threshold": float(args.threshold),
        "guard_scan": bool(args.guard_scan),
        "bad_tasks": sorted(bad_tasks),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "elapsed_seconds": time.time() - started,
        "counters": dict(counters),
        "bad_by_task": dict(bad_by_task.most_common()),
        "bad_by_reason": dict(bad_by_reason.most_common()),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
