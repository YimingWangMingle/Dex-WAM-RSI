#!/usr/bin/env python3
"""Estimate Ego4D grouped batch locality for different temporal buckets."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    vals = sorted(values)

    def q(p: float) -> float:
        return vals[min(len(vals) - 1, int(round((len(vals) - 1) * p)))]

    return {
        "count": len(vals),
        "min": vals[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def frame_indices(start_sec: float, end_sec: float, fps: float, frames: int) -> List[int]:
    start_idx = max(0, int(round(start_sec * fps)))
    end_idx = max(start_idx, int(round(end_sec * fps)))
    if end_idx == start_idx:
        return [start_idx] * frames
    indices = np.linspace(start_idx, end_idx, num=frames)
    return np.rint(indices).astype(np.int64).tolist()


def group_key(video_uri: str, start_sec: float, bucket_seconds: float | None) -> str:
    if bucket_seconds is None or bucket_seconds <= 0:
        return video_uri
    bucket = int(math.floor(max(start_sec, 0.0) / bucket_seconds))
    return f"{video_uri}::bucket:{bucket}"


def iter_batches(indices: List[int], batch_size: int, drop_last: bool) -> Iterable[List[int]]:
    end = len(indices) - (len(indices) % batch_size if drop_last else 0)
    for start in range(0, end, batch_size):
        batch = indices[start : start + batch_size]
        if len(batch) == batch_size or (batch and not drop_last):
            yield batch


def virtual_windows(
    start_sec: float,
    end_sec: float,
    enabled: bool,
    window_seconds: float,
    stride_seconds: float,
    max_windows_per_record: int | None,
) -> Iterable[Tuple[float, float, int]]:
    if end_sec <= start_sec:
        return
    if not enabled:
        yield start_sec, end_sec, 0
        return
    duration = end_sec - start_sec
    if duration <= window_seconds:
        yield start_sec, end_sec, 0
        return

    count = 0
    current = start_sec
    epsilon = 1e-6
    while current + window_seconds <= end_sec + epsilon:
        yield current, min(current + window_seconds, end_sec), count
        count += 1
        if max_windows_per_record is not None and count >= max_windows_per_record:
            break
        current += stride_seconds


def analyze(
    records: List[dict],
    bucket_seconds: float | None,
    batch_size: int,
    frames: int,
    fps: float,
    virtual_enabled: bool,
    window_seconds: float,
    stride_seconds: float,
    max_windows_per_record: int | None,
):
    groups: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    videos = set()
    entries = []
    for index, record in enumerate(records):
        video = record["video"]
        uri = video["uri"]
        start = float(video["start_sec"])
        end = float(video["end_sec"])
        videos.add(uri)
        for window_start, window_end, window_index in virtual_windows(
            start,
            end,
            enabled=virtual_enabled,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            max_windows_per_record=max_windows_per_record,
        ):
            entry_index = len(entries)
            entries.append(
                {
                    "record_index": index,
                    "window_index": window_index,
                    "uri": uri,
                    "start_sec": window_start,
                    "end_sec": window_end,
                }
            )
            groups[group_key(uri, window_start, bucket_seconds)].append((window_start, window_end, entry_index))

    batches = []
    dropped_records = 0
    for items in groups.values():
        items.sort(key=lambda item: (item[0], item[1], item[2]))
        ordered = [index for _, _, index in items]
        dropped_records += len(ordered) % batch_size
        batches.extend(iter_batches(ordered, batch_size=batch_size, drop_last=True))

    spans = []
    reductions = []
    unique_frames = []
    for batch in batches:
        starts = [float(entries[index]["start_sec"]) for index in batch]
        ends = [float(entries[index]["end_sec"]) for index in batch]
        frame_lists = [
            frame_indices(start, end, fps=fps, frames=frames)
            for start, end in zip(starts, ends)
        ]
        requested = sum(len(indices) for indices in frame_lists)
        unique = len({frame for indices in frame_lists for frame in indices})
        spans.append(max(ends) - min(starts))
        unique_frames.append(unique)
        reductions.append(1.0 - unique / max(requested, 1))

    return {
        "bucket_seconds": bucket_seconds,
        "records": len(records),
        "virtual_windows": virtual_enabled,
        "virtual_window_seconds": window_seconds if virtual_enabled else None,
        "virtual_stride_seconds": stride_seconds if virtual_enabled else None,
        "max_windows_per_record": max_windows_per_record if virtual_enabled else None,
        "virtual_entries": len(entries),
        "video_keys": len(videos),
        "group_keys": len(groups),
        "batches": len(batches),
        "dropped_records_drop_last": dropped_records,
        "span_sec": stats(spans),
        "unique_frames": stats(unique_frames),
        "frame_reduction": stats(reductions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate temporal bucket grouping quality for Ego4D Manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bucket-seconds", type=float, nargs="+", default=[0, 30, 60, 120])
    parser.add_argument("--virtual-windows", action="store_true")
    parser.add_argument("--virtual-window-seconds", type=float, default=8.0)
    parser.add_argument("--virtual-stride-seconds", type=float, default=4.0)
    parser.add_argument("--max-virtual-windows-per-record", type=int, default=8)
    args = parser.parse_args()

    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            video = record.get("video") or {}
            if not video.get("uri") or video.get("start_sec") is None or video.get("end_sec") is None:
                continue
            records.append(record)
            if args.max_samples is not None and len(records) >= args.max_samples:
                break

    for bucket in args.bucket_seconds:
        bucket_value = None if bucket <= 0 else float(bucket)
        print(
            json.dumps(
                analyze(
                    records,
                    bucket_value,
                    args.batch_size,
                    args.frames,
                    args.fps,
                    virtual_enabled=args.virtual_windows,
                    window_seconds=args.virtual_window_seconds,
                    stride_seconds=args.virtual_stride_seconds,
                    max_windows_per_record=args.max_virtual_windows_per_record,
                ),
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
