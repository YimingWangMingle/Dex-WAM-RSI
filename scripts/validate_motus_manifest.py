#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "id",
    "source_dataset",
    "split",
    "text",
    "video",
    "robot",
    "state",
    "action",
    "loss",
}


def error(errors: List[str], message: str):
    errors.append(message)


def validate_record(record: Dict[str, Any], check_paths: bool = False) -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL - set(record.keys()))
    if missing:
        error(errors, f"missing top-level keys: {missing}")
        return errors
    if record.get("schema_version") != "motus_manifest_v1":
        error(errors, "schema_version must be motus_manifest_v1")

    text = record.get("text") or {}
    if not str(text.get("t5") or "").strip() and not str(text.get("vlm") or "").strip():
        error(errors, "text.t5 or text.vlm must be non-empty")

    video = record.get("video") or {}
    video_uri = video.get("uri") or video.get("path")
    if not video_uri:
        error(errors, "video.uri is required")
    if check_paths and video_uri and not Path(video_uri).exists():
        error(errors, f"video.uri does not exist: {video_uri}")
    video_backend = str(video.get("storage_backend") or video.get("storage") or "file")
    if video_backend in {"tar", "tar_file", "file_tar"} and not video.get("member"):
        error(errors, "video.member is required for tar backend")
    frame_indices = video.get("frame_indices") or video.get("indices") or []
    has_frame_indices = isinstance(frame_indices, list) and len(frame_indices) > 0
    if has_frame_indices:
        try:
            [int(item) for item in frame_indices]
        except (TypeError, ValueError):
            error(errors, "video.frame_indices must be integers")
    else:
        try:
            start_sec = float(video.get("start_sec", 0.0))
            end_sec = float(video.get("end_sec"))
            if end_sec <= start_sec:
                error(errors, "video.end_sec must be greater than video.start_sec")
        except (TypeError, ValueError):
            error(errors, "video.start_sec/end_sec must be numeric when video.frame_indices is absent")

    robot = record.get("robot") or {}
    if not robot.get("name"):
        error(errors, "robot.name is required")
    for key in ("state_dim", "action_dim"):
        try:
            if int(robot.get(key)) <= 0:
                error(errors, f"robot.{key} must be positive")
        except (TypeError, ValueError):
            error(errors, f"robot.{key} must be an integer")
    if str(robot.get("action_policy") or "") not in {"real", "learned_prompt", "zero_masked"}:
        error(errors, "robot.action_policy must be real, learned_prompt, or zero_masked")

    loss = record.get("loss") or {}
    for key in ("video_weight", "action_weight"):
        try:
            value = float(loss.get(key))
            if value < 0:
                error(errors, f"loss.{key} must be non-negative")
        except (TypeError, ValueError):
            error(errors, f"loss.{key} must be numeric")

    for key in ("state", "action"):
        item = record.get(key) or {}
        backend = str(item.get("storage_backend") or item.get("storage") or "none")
        if backend not in {
            "none",
            "file",
            "hdf5",
            "agibot_proprio_hdf5",
            "agibot_proprio_hdf5_tar",
            "robotwin_qpos_pt",
            "npy",
            "pt",
            "json",
            "tar",
            "tar_file",
            "file_tar",
            "streaming",
        }:
            error(errors, f"{key}.storage_backend unsupported: {backend}")
        uri = item.get("uri")
        if check_paths and backend not in {"none", "null"} and uri and not Path(uri).exists():
            error(errors, f"{key}.uri does not exist: {uri}")
        if backend in {"agibot_proprio_hdf5_tar", "tar", "tar_file", "file_tar"} and not item.get("member"):
            error(errors, f"{key}.member is required for tar backend")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Motus Manifest v1 JSONL.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    counters = Counter()
    errors_seen = 0
    scanned = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if args.limit is not None and scanned >= args.limit:
                break
            scanned += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] line={line_no}: bad json: {exc}")
                errors_seen += 1
                if errors_seen >= args.max_errors:
                    break
                continue

            record_errors = validate_record(record, check_paths=args.check_paths)
            if record_errors:
                for item in record_errors:
                    print(f"[ERROR] line={line_no} id={record.get('id')}: {item}")
                    errors_seen += 1
                    if errors_seen >= args.max_errors:
                        break
            counters[f"source:{record.get('source_dataset')}"] += 1
            counters[f"split:{record.get('split')}"] += 1
            counters[f"task:{record.get('task')}"] += 1
            if errors_seen >= args.max_errors:
                break

    print(f"scanned={scanned} errors={errors_seen}")
    for key, value in sorted(counters.items()):
        print(f"{key}={value}")
    return 1 if errors_seen else 0


if __name__ == "__main__":
    raise SystemExit(main())
