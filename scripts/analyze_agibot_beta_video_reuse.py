#!/usr/bin/env python3
"""Analyze video-level reuse potential for Agibot_beta Motus manifests.

This is a read-only planning tool for the video-level producer-consumer
pipeline. It groups manifest records by physical video key and estimates:

* how many records share one video
* how much frame decoding can be reduced by union-frame decode
* how many conservative intra-record subwindows are possible

The script intentionally does not read mp4/h5 payloads. It only scans JSONL
metadata, so it is safe to run while training or conversion jobs are active.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def video_key(record: Dict[str, Any]) -> str:
    video = record.get("video") or {}
    uri = str(video.get("uri") or video.get("path") or "")
    member = video.get("member")
    if member:
        return f"{uri}!{member}"
    return uri


def frame_indices(record: Dict[str, Any]) -> List[int]:
    video = record.get("video") or {}
    indices = video.get("frame_indices") or video.get("indices") or []
    if isinstance(indices, str):
        return []
    try:
        return [int(index) for index in indices]
    except (TypeError, ValueError):
        return []


def action_segments(record: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    metadata = record.get("metadata") or {}
    segments = []
    for item in metadata.get("action_config") or []:
        try:
            start = int(item.get("start_frame"))
            end = int(item.get("end_frame"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("action_text") or "")
        if end > start:
            segments.append((start, end, text))
    return segments


def containing_action_segment(
    start_frame: int,
    end_frame: int,
    segments: List[Tuple[int, int, str]],
) -> Optional[Tuple[int, int, str]]:
    for segment_start, segment_end, text in segments:
        if start_frame >= segment_start and end_frame <= segment_end:
            return segment_start, segment_end, text
    return None


def count_subwindows(
    record: Dict[str, Any],
    frames_clip_num: int,
    stride_frames: int,
    max_subwindows_per_record: Optional[int],
    require_single_action_segment: bool,
) -> Tuple[int, int]:
    """Return conservative subwindow count and rejected crossing-boundary count."""
    indices = frame_indices(record)
    if not indices:
        return 0, 0
    first = min(indices)
    last = max(indices)
    if last - first + 1 < frames_clip_num:
        return 0, 0

    segments = action_segments(record)
    count = 0
    rejected_crossing = 0
    start = first
    while start + frames_clip_num - 1 <= last:
        end = start + frames_clip_num - 1
        if require_single_action_segment and segments:
            if containing_action_segment(start, end, segments) is None:
                rejected_crossing += 1
                start += stride_frames
                continue
        count += 1
        if max_subwindows_per_record is not None and count >= max_subwindows_per_record:
            break
        start += stride_frames
    return count, rejected_crossing


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def iter_manifest(path: Path, limit: Optional[int]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_top_groups(path: Path, groups: Dict[str, List[Tuple[int, List[int]]]], top_k: int) -> None:
    ranked = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[:top_k]
    with path.open("w", encoding="utf-8") as f:
        f.write("video_key\trecords\trequested_frames\tunique_frames\toverlap_reduction\n")
        for key, items in ranked:
            requested = sum(len(indices) for _, indices in items)
            unique = len({index for _, indices in items for index in indices})
            reduction = 1.0 - (unique / requested) if requested else 0.0
            f.write(f"{key}\t{len(items)}\t{requested}\t{unique}\t{reduction:.6f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Agibot_beta video reuse potential.")
    parser.add_argument("--manifest", required=True, help="Agibot_beta Motus manifest JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Scan at most this many manifest lines.")
    parser.add_argument("--frames-clip-num", type=int, default=49, help="Frames per training sample including first frame.")
    parser.add_argument("--subwindow-stride-frames", type=int, default=49, help="Stride for intra-record subwindow estimate.")
    parser.add_argument("--max-subwindows-per-record", type=int, default=None)
    parser.add_argument("--allow-cross-action-segment", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--progress-interval", type=int, default=100000)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[Tuple[int, List[int]]]] = defaultdict(list)
    source_counts = Counter()
    task_counts = Counter()
    frame_span_values = []
    requested_frame_values = []
    inter_requested_total = 0
    subwindow_total = 0
    subwindow_rejected_crossing = 0
    malformed = 0

    for scanned, (line_no, record) in enumerate(iter_manifest(manifest_path, args.limit), start=1):
        try:
            key = video_key(record)
            indices = frame_indices(record)
            if not key or not indices:
                malformed += 1
                continue
            groups[key].append((line_no, indices))
            inter_requested_total += len(indices)
            requested_frame_values.append(len(indices))
            frame_span_values.append(max(indices) - min(indices) + 1)
            source_counts[str(record.get("source_dataset") or "unknown")] += 1
            task_counts[str(record.get("task") or "unknown")] += 1
            sub_count, rejected = count_subwindows(
                record,
                frames_clip_num=int(args.frames_clip_num),
                stride_frames=int(args.subwindow_stride_frames),
                max_subwindows_per_record=args.max_subwindows_per_record,
                require_single_action_segment=not bool(args.allow_cross_action_segment),
            )
            subwindow_total += sub_count
            subwindow_rejected_crossing += rejected
        except Exception:
            malformed += 1
            continue

        if args.progress_interval > 0 and scanned % args.progress_interval == 0:
            print(
                f"[progress] scanned={scanned} video_keys={len(groups)} "
                f"subwindows={subwindow_total} malformed={malformed}",
                flush=True,
            )

    group_record_counts = sorted((len(items) for items in groups.values()))
    per_group_requested = []
    per_group_unique = []
    per_group_reduction = []
    groups_with_multiple_records = 0
    for items in groups.values():
        requested = sum(len(indices) for _, indices in items)
        unique = len({index for _, indices in items for index in indices})
        per_group_requested.append(requested)
        per_group_unique.append(unique)
        if len(items) > 1:
            groups_with_multiple_records += 1
        if requested:
            per_group_reduction.append(1.0 - (unique / requested))

    inter_unique_total = sum(per_group_unique)
    frame_span_values.sort()
    requested_frame_values.sort()
    per_group_reduction.sort()

    summary = {
        "manifest": str(manifest_path),
        "limit": args.limit,
        "records_scanned": sum(len(items) for items in groups.values()) + malformed,
        "records_valid": sum(len(items) for items in groups.values()),
        "records_malformed_or_missing_video_frames": malformed,
        "video_keys": len(groups),
        "groups_with_multiple_records": groups_with_multiple_records,
        "records_per_video_key": {
            "p50": percentile(group_record_counts, 0.50),
            "p90": percentile(group_record_counts, 0.90),
            "p95": percentile(group_record_counts, 0.95),
            "p99": percentile(group_record_counts, 0.99),
            "max": max(group_record_counts) if group_record_counts else 0,
        },
        "frame_span_per_record": {
            "p50": percentile(frame_span_values, 0.50),
            "p90": percentile(frame_span_values, 0.90),
            "p95": percentile(frame_span_values, 0.95),
            "p99": percentile(frame_span_values, 0.99),
            "max": max(frame_span_values) if frame_span_values else 0,
        },
        "requested_frames_per_record": {
            "p50": percentile(requested_frame_values, 0.50),
            "p90": percentile(requested_frame_values, 0.90),
            "p95": percentile(requested_frame_values, 0.95),
            "p99": percentile(requested_frame_values, 0.99),
            "max": max(requested_frame_values) if requested_frame_values else 0,
        },
        "inter_record_decode": {
            "requested_frames_total": inter_requested_total,
            "unique_frames_by_video_key_total": inter_unique_total,
            "estimated_frame_reduction": (
                1.0 - (inter_unique_total / inter_requested_total)
                if inter_requested_total
                else 0.0
            ),
            "per_group_reduction_p50": percentile(per_group_reduction, 0.50),
            "per_group_reduction_p90": percentile(per_group_reduction, 0.90),
            "per_group_reduction_p95": percentile(per_group_reduction, 0.95),
            "per_group_reduction_p99": percentile(per_group_reduction, 0.99),
        },
        "intra_record_subwindows": {
            "frames_clip_num": int(args.frames_clip_num),
            "stride_frames": int(args.subwindow_stride_frames),
            "max_subwindows_per_record": args.max_subwindows_per_record,
            "require_single_action_segment": not bool(args.allow_cross_action_segment),
            "estimated_subwindows": subwindow_total,
            "rejected_cross_action_segment": subwindow_rejected_crossing,
        },
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
    }

    summary_path = output_dir / "video_reuse_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    write_top_groups(output_dir / "top_video_groups.tsv", groups, int(args.top_k))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
