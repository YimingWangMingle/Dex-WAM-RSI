#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def select_time_window(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    target = record.get("target") or {}
    media = record.get("media") or {}
    for start_key, end_key in (("video_start_sec", "video_end_sec"),):
        start = target.get(start_key)
        end = target.get(end_key)
        if start is not None and end is not None:
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError):
                continue
            if end_f > start_f:
                return start_f, end_f
    try:
        start_f = float(media.get("start_sec", 0.0))
        end_f = float(media.get("end_sec"))
    except (TypeError, ValueError):
        return None, None
    return start_f, end_f


def format_ego4d_text(record: Dict[str, Any]) -> str:
    task = clean_text(record.get("task")) or "unknown"
    subtask = clean_text(record.get("subtask")) or "unknown"
    text = clean_text(record.get("text"))
    target_text = clean_text(record.get("target_text"))
    parts = [f"Ego4D task: {task}/{subtask}."]
    if text:
        parts.append(f"Input: {text}")
    if target_text:
        parts.append(f"Target: {target_text}")
    return "\n".join(parts)


def convert_record(record: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    media = record.get("media") or {}
    video_path = media.get("video_path")
    start_sec, end_sec = select_time_window(record)
    text = format_ego4d_text(record)
    if not video_path or start_sec is None or end_sec is None or end_sec <= start_sec or not text:
        return None

    return {
        "schema_version": "motus_manifest_v1",
        "id": record.get("id") or f"ego4d:{record.get('split')}:{record.get('video_uid')}",
        "source_dataset": args.source_dataset,
        "split": record.get("split"),
        "task": record.get("task"),
        "subtask": record.get("subtask"),
        "text": {
            "t5": text,
            "vlm": text,
            "target": clean_text(record.get("target_text")) or clean_text(record.get("text")),
        },
        "video": {
            "storage_backend": "file",
            "uri": video_path,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "time_is_local_to_clip": True,
        },
        "robot": {
            "name": args.robot_name,
            "state_dim": int(args.state_dim),
            "action_dim": int(args.action_dim),
            "action_policy": "learned_prompt",
        },
        "state": {
            "storage_backend": "none",
            "uri": None,
            "key": None,
            "index": None,
        },
        "action": {
            "storage_backend": "none",
            "uri": None,
            "key": None,
            "indices": None,
        },
        "loss": {
            "video_weight": 1.0,
            "action_weight": 0.0,
        },
        "metadata": {
            "converter": "convert_ego4d_window_index_to_motus_manifest.py",
            "original_record_id": record.get("id"),
            "video_uid": record.get("video_uid"),
            "clip_uid": record.get("clip_uid"),
            "source_file": record.get("source_file"),
            "target": record.get("target"),
            "window_metadata": record.get("window_metadata") or (record.get("metadata") or {}).get("window_precut"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Ego4D window index JSONL to Motus Manifest v1 JSONL.")
    parser.add_argument("--index", required=True, help="Input Ego4D window index JSONL.")
    parser.add_argument("--output", required=True, help="Output Motus Manifest JSONL.")
    parser.add_argument("--source-dataset", default="ego4d")
    parser.add_argument("--robot-name", default="ego4d-null")
    parser.add_argument("--state-dim", type=int, default=26)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument("--limit", type=int, default=None, help="Maximum input lines to scan.")
    parser.add_argument("--max-records", type=int, default=None, help="Maximum converted records to write.")
    parser.add_argument("--split", default=None, help="Optional exact split filter, e.g. train or trainval.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = written = skipped = 0
    with index_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if args.limit is not None and scanned >= args.limit:
                break
            scanned += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if args.split is not None and record.get("split") != args.split:
                skipped += 1
                continue
            manifest_record = convert_record(record, args)
            if manifest_record is None:
                skipped += 1
                continue
            dst.write(json.dumps(manifest_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if args.max_records is not None and written >= args.max_records:
                break

    print(f"scanned={scanned} written={written} skipped={skipped} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
