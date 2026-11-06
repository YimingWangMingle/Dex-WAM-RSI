#!/usr/bin/env python3
"""Precut Ego4D full videos into bounded-duration training windows.

This script reads the original Ego4D unified JSONL index and cuts physical
short mp4 clips directly from ``full_scale`` source videos. It does not depend
on the earlier annotation-level clip output.

For every record, the script first selects the same training time window used
by ``Ego4DDataset``:

1. ``target.video_start_sec`` / ``target.video_end_sec`` when valid.
2. Otherwise ``media.start_sec`` / ``media.end_sec``.

If that window is longer than ``--max-window-seconds``, it is split into
multiple non-overlapping windows. Each output JSONL record points to one short
mp4 and rewrites media/target time fields to chunk-local seconds, so the
existing Ego4DDataset can read it without code changes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUPPORTED_TASKS = {"nlq", "moments", "goalstep", "av", "fho"}
TRAIN_SPLITS = {"train", "trainval", "all"}
VAL_SPLITS = {"val"}
DEFAULT_INDEX = "/code1/7TB/v2/annotations/processed/unified_index.filtered.jsonl"
DEFAULT_OUTPUT_ROOT = "/code1/7TB/v2/ego4d_clips_window_v1"
WINDOW_VERSION = "ego4d_clips_window_v1"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def select_time_window(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    """Match Ego4DDataset._select_time_window."""
    target = record.get("target") or {}
    media = record.get("media") or {}

    target_start = target.get("video_start_sec")
    target_end = target.get("video_end_sec")
    if target_start is not None and target_end is not None:
        try:
            target_start = float(target_start)
            target_end = float(target_end)
        except (TypeError, ValueError):
            target_start = target_end = None
        if target_start is not None and target_end is not None and target_end > target_start:
            return target_start, target_end, "target.video_start_sec"

    try:
        return float(media.get("start_sec")), float(media.get("end_sec")), "media.start_sec"
    except (TypeError, ValueError):
        return None, None, "invalid"


def split_allowed(record_split: Any, split_mode: str) -> bool:
    split = str(record_split)
    if split_mode == "all":
        return True
    if split_mode == "val":
        return split in VAL_SPLITS
    return split in TRAIN_SPLITS


def keep_record(record: Dict[str, Any], split_mode: str) -> Tuple[bool, str]:
    task = record.get("task")
    subtask = str(record.get("subtask") or "")
    target = record.get("target") or {}
    media = record.get("media") or {}

    if not split_allowed(record.get("split"), split_mode):
        return False, "split"
    if task not in SUPPORTED_TASKS:
        return False, "task"
    if "unannotated" in subtask or target.get("type") == "unannotated_clip":
        return False, "unannotated"
    if not media.get("video_path"):
        return False, "missing_video_path"

    start_sec, end_sec, _ = select_time_window(record)
    if start_sec is None or end_sec is None or end_sec <= start_sec:
        return False, "bad_time"

    if not clean_text(record.get("text")) and not clean_text(record.get("target_text")):
        return False, "empty_text"

    return True, ""


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def safe_task_name(record: Dict[str, Any]) -> str:
    task = clean_text(record.get("task")) or "unknown"
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in task)


def source_video_path(record: Dict[str, Any]) -> Path:
    media = record.get("media") or {}
    return Path(str(media.get("video_path") or ""))


def make_windows(
    segment_start: float,
    segment_end: float,
    max_window_seconds: float,
    min_window_seconds: float,
) -> List[Tuple[float, float]]:
    duration = segment_end - segment_start
    if duration <= 0:
        return []
    if duration <= max_window_seconds:
        return [(segment_start, segment_end)]

    windows: List[Tuple[float, float]] = []
    current = segment_start
    while current < segment_end:
        next_end = min(segment_end, current + max_window_seconds)
        windows.append((current, next_end))
        current = next_end

    if len(windows) >= 2:
        last_start, last_end = windows[-1]
        last_duration = last_end - last_start
        if last_duration < min_window_seconds:
            prev_start, _ = windows[-2]
            merged_duration = last_end - prev_start
            if merged_duration <= max_window_seconds:
                windows[-2] = (prev_start, last_end)
                windows.pop()
            else:
                windows.pop()

    # Keep short original annotations, but drop tiny tail windows from long
    # segments when merging would violate max_window_seconds.
    return [(s, e) for s, e in windows if e > s]


def window_output_path(output_root: Path, record: Dict[str, Any], window_index: int) -> Path:
    source_video = source_video_path(record)
    video_uid = clean_text(record.get("video_uid")) or source_video.stem or "unknown_video"
    sample_id = clean_text(record.get("id")) or json.dumps(record, sort_keys=True)
    base_hash = stable_hash(sample_id)
    name = f"{video_uid}_{base_hash}_w{window_index:04d}.mp4"
    return output_root / "clips" / safe_task_name(record) / name


def run_ffmpeg(
    source_video: Path,
    output_video: Path,
    window_start: float,
    window_end: float,
    ffmpeg_bin: str,
    crf: int,
    preset: str,
    timeout: Optional[float],
    overwrite: bool,
    dry_run: bool,
) -> Tuple[bool, Dict[str, Any]]:
    duration = max(0.001, window_end - window_start)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{window_start:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    info = {
        "cmd": " ".join(cmd),
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "window_duration_sec": duration,
    }
    if dry_run:
        return True, info

    output_video.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        info.update({"reason": "ffmpeg_timeout", "seconds": time.perf_counter() - start})
        info["stderr"] = str(exc)
        return False, info
    except FileNotFoundError as exc:
        info.update({"reason": "ffmpeg_not_found", "seconds": time.perf_counter() - start})
        info["stderr"] = str(exc)
        return False, info

    info.update(
        {
            "returncode": proc.returncode,
            "seconds": time.perf_counter() - start,
            "stderr": proc.stderr[-4000:],
        }
    )
    return proc.returncode == 0, info


def build_window_record(
    record: Dict[str, Any],
    output_video: Path,
    segment_start: float,
    segment_end: float,
    window_start: float,
    window_end: float,
    window_index: int,
    window_count: int,
    time_source: str,
    max_window_seconds: float,
) -> Dict[str, Any]:
    new_record = copy.deepcopy(record)
    media = dict(new_record.get("media") or {})
    target = dict(new_record.get("target") or {})
    metadata = dict(new_record.get("metadata") or {})

    source_media = record.get("media") or {}
    chunk_duration = window_end - window_start

    parent_id = clean_text(record.get("id")) or stable_hash(json.dumps(record, sort_keys=True))
    new_record["id"] = f"{parent_id}:window:{window_index:04d}"

    if "start_sec" in media:
        media["source_media_start_sec"] = media.get("start_sec")
    if "end_sec" in media:
        media["source_media_end_sec"] = media.get("end_sec")
    if "start_frame" in media:
        media["source_start_frame"] = media.get("start_frame")
        media.pop("start_frame", None)
    if "end_frame" in media:
        media["source_end_frame"] = media.get("end_frame")
        media.pop("end_frame", None)

    media["video_path"] = str(output_video)
    media["start_sec"] = 0.0
    media["end_sec"] = chunk_duration
    media["source_video_path"] = source_media.get("video_path")
    media["source_segment_start_sec"] = segment_start
    media["source_segment_end_sec"] = segment_end
    media["window_start_sec_original"] = window_start
    media["window_end_sec_original"] = window_end
    media["window_version"] = WINDOW_VERSION
    media["time_source"] = time_source

    if target:
        if "video_start_sec" in target:
            target["source_video_start_sec"] = target.get("video_start_sec")
        if "video_end_sec" in target:
            target["source_video_end_sec"] = target.get("video_end_sec")
        target["video_start_sec"] = 0.0
        target["video_end_sec"] = chunk_duration

    window_metadata = {
        "version": WINDOW_VERSION,
        "parent_id": parent_id,
        "window_index": window_index,
        "window_count": window_count,
        "original_video_path": source_media.get("video_path"),
        "original_segment_start_sec": segment_start,
        "original_segment_end_sec": segment_end,
        "window_start_sec_original": window_start,
        "window_end_sec_original": window_end,
        "chunk_duration_sec": chunk_duration,
        "max_window_seconds": max_window_seconds,
        "time_source": time_source,
    }

    metadata["window_precut"] = window_metadata
    new_record["media"] = media
    if target:
        new_record["target"] = target
    new_record["window_metadata"] = window_metadata
    new_record["metadata"] = metadata
    return new_record


def process_record(args: Tuple[int, Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    line_index, record, cfg = args
    started = time.perf_counter()
    ok, reason = keep_record(record, cfg["split"])
    if not ok:
        return {
            "status": "skipped",
            "reason": reason,
            "line_index": line_index,
            "id": record.get("id"),
        }

    source_video = source_video_path(record)
    if not cfg["dry_run"] and not source_video.exists():
        return {
            "status": "failed_record",
            "reason": "source_video_missing",
            "line_index": line_index,
            "id": record.get("id"),
            "source_video_path": str(source_video),
        }

    segment_start, segment_end, time_source = select_time_window(record)
    if segment_start is None or segment_end is None or segment_end <= segment_start:
        return {
            "status": "failed_record",
            "reason": "bad_time",
            "line_index": line_index,
            "id": record.get("id"),
            "source_video_path": str(source_video),
        }

    windows = make_windows(
        segment_start,
        segment_end,
        cfg["max_window_seconds"],
        cfg["min_window_seconds"],
    )
    if not windows:
        return {
            "status": "skipped",
            "reason": "window_too_short",
            "line_index": line_index,
            "id": record.get("id"),
            "source_video_path": str(source_video),
            "segment_start_sec": segment_start,
            "segment_end_sec": segment_end,
        }

    output_root = Path(cfg["output_root"])
    ok_records = []
    failures = []
    reused = 0
    cut = 0

    for window_index, (window_start, window_end) in enumerate(windows):
        output_video = window_output_path(output_root, record, window_index)
        window_duration = window_end - window_start

        if output_video.exists() and not cfg["overwrite"]:
            if output_video.stat().st_size >= cfg["min_bytes"]:
                reused += 1
                ok_records.append(
                    build_window_record(
                        record=record,
                        output_video=output_video,
                        segment_start=segment_start,
                        segment_end=segment_end,
                        window_start=window_start,
                        window_end=window_end,
                        window_index=window_index,
                        window_count=len(windows),
                        time_source=time_source,
                        max_window_seconds=cfg["max_window_seconds"],
                    )
                )
                continue
            failures.append(
                {
                    "status": "failed_window",
                    "reason": "existing_output_too_small",
                    "line_index": line_index,
                    "id": record.get("id"),
                    "window_index": window_index,
                    "output_video_path": str(output_video),
                    "output_size": output_video.stat().st_size,
                }
            )
            continue

        success, ffmpeg_info = run_ffmpeg(
            source_video=source_video,
            output_video=output_video,
            window_start=window_start,
            window_end=window_end,
            ffmpeg_bin=cfg["ffmpeg_bin"],
            crf=cfg["crf"],
            preset=cfg["preset"],
            timeout=cfg["timeout"],
            overwrite=cfg["overwrite"],
            dry_run=cfg["dry_run"],
        )
        if not success:
            failures.append(
                {
                    "status": "failed_window",
                    "reason": ffmpeg_info.get("reason", "ffmpeg_failed"),
                    "line_index": line_index,
                    "id": record.get("id"),
                    "window_index": window_index,
                    "source_video_path": str(source_video),
                    "output_video_path": str(output_video),
                    "window_start_sec_original": window_start,
                    "window_end_sec_original": window_end,
                    "ffmpeg": ffmpeg_info,
                }
            )
            continue

        if not cfg["dry_run"]:
            if not output_video.exists():
                failures.append(
                    {
                        "status": "failed_window",
                        "reason": "output_missing",
                        "line_index": line_index,
                        "id": record.get("id"),
                        "window_index": window_index,
                        "source_video_path": str(source_video),
                        "output_video_path": str(output_video),
                        "ffmpeg": ffmpeg_info,
                    }
                )
                continue
            if output_video.stat().st_size < cfg["min_bytes"]:
                failures.append(
                    {
                        "status": "failed_window",
                        "reason": "output_too_small",
                        "line_index": line_index,
                        "id": record.get("id"),
                        "window_index": window_index,
                        "source_video_path": str(source_video),
                        "output_video_path": str(output_video),
                        "output_size": output_video.stat().st_size,
                        "ffmpeg": ffmpeg_info,
                    }
                )
                continue

        if not cfg["dry_run"]:
            cut += 1
        ok_records.append(
            build_window_record(
                record=record,
                output_video=output_video,
                segment_start=segment_start,
                segment_end=segment_end,
                window_start=window_start,
                window_end=window_end,
                window_index=window_index,
                window_count=len(windows),
                time_source=time_source,
                max_window_seconds=cfg["max_window_seconds"],
            )
        )

    status = "ok" if ok_records and not failures else "partial" if ok_records else "failed_record"
    return {
        "status": status,
        "line_index": line_index,
        "id": record.get("id"),
        "source_video_path": str(source_video),
        "input_segment_duration_sec": segment_end - segment_start,
        "window_count_planned": len(windows),
        "window_count_ok": len(ok_records),
        "window_count_failed": len(failures),
        "reused": reused,
        "cut": cut,
        "records": ok_records,
        "failures": failures,
        "seconds": time.perf_counter() - started,
        "task": record.get("task"),
        "split": record.get("split"),
        "time_source": time_source,
        "window_duration_bins": [round(e - s, 6) for s, e in windows],
        "expanded": len(windows) > 1,
    }


def iter_index(index_path: Path, limit: Optional[int], start_line: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    yielded = 0
    with index_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index < start_line:
                continue
            if limit is not None and yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yielded += 1
            yield line_index, json.loads(line)


def count_source_records(index_path: Path, limit: Optional[int], start_line: int) -> int:
    total = 0
    with index_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index < start_line:
                continue
            if limit is not None and total >= limit:
                break
            if line.strip():
                total += 1
    return total


def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def unordered_process_records(
    records_iter: Iterable[Tuple[int, Dict[str, Any], Dict[str, Any]]],
    num_workers: int,
) -> Iterable[Dict[str, Any]]:
    """Yield worker results as soon as each source record finishes.

    ProcessPoolExecutor.map preserves input order. That is bad for this job:
    one slow source record can block the main process from writing many later
    completed window records, while workers continue creating mp4 files. A
    killed run then leaves valid clips on disk without matching index lines.
    """
    max_in_flight = max(num_workers * 2, 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        pending = set()
        source_iter = iter(records_iter)

        def submit_until_full() -> None:
            while len(pending) < max_in_flight:
                try:
                    item = next(source_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(process_record, item))

        submit_until_full()
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                yield future.result()
            submit_until_full()


def bump_counter(counter: Dict[str, int], key: Any, amount: int = 1) -> None:
    key = str(key)
    counter[key] = counter.get(key, 0) + int(amount)


def duration_bin(seconds: float) -> str:
    bins = [0, 2, 4, 8, 16, 32, 60, 90, 120, 180, 240, 10**9]
    eps = 1e-6
    for low, high in zip(bins, bins[1:]):
        if low <= seconds < high or abs(seconds - high) <= eps:
            return f"{low}-{high}"
    return "unknown"


def format_duration(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def output_paths(output_root: Path, max_window_seconds: float, limit: Optional[int], start_line: int) -> Tuple[Path, Path, Path]:
    out_index_dir = output_root / "index"
    out_index_dir.mkdir(parents=True, exist_ok=True)
    window_label = f"window{max_window_seconds:g}"
    suffix = f".{window_label}"
    if limit is not None:
        suffix += f".limit{limit}"
    if start_line:
        suffix += f".from{start_line}"
    return (
        out_index_dir / f"unified_index.filtered{suffix}.jsonl",
        out_index_dir / f"unified_index.filtered{suffix}.failed.jsonl",
        out_index_dir / f"unified_index.filtered{suffix}.stats.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precut Ego4D full videos into bounded-duration training windows.")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Source Ego4D unified JSONL index.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root for windowed clips and index.")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"], help="Split filtering mode.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many JSONL lines after start-line.")
    parser.add_argument("--start-line", type=int, default=0, help="Skip source JSONL lines before this index.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel ffmpeg workers.")
    parser.add_argument("--max-window-seconds", type=float, default=16.0, help="Maximum duration for each output training window.")
    parser.add_argument("--min-window-seconds", type=float, default=2.0, help="Minimum output window duration; tiny tail windows are merged or skipped.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable.")
    parser.add_argument("--crf", type=int, default=23, help="libx264 CRF for clip re-encoding.")
    parser.add_argument("--preset", default="veryfast", help="libx264 preset.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-window ffmpeg timeout in seconds. <=0 disables it.")
    parser.add_argument("--min-bytes", type=int, default=10 * 1024, help="Minimum acceptable output mp4 size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing window clip files.")
    parser.add_argument("--dry-run", action="store_true", help="Do not run ffmpeg; only generate planned records.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any record/window failed.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N processed source records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index)
    output_root = Path(args.output_root)
    clip_index_path, failed_path, stats_path = output_paths(
        output_root,
        max_window_seconds=args.max_window_seconds,
        limit=args.limit,
        start_line=args.start_line,
    )

    if not index_path.exists():
        raise FileNotFoundError(f"Source index does not exist: {index_path}")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.max_window_seconds <= 0:
        raise ValueError("--max-window-seconds must be positive")
    if args.min_window_seconds <= 0:
        raise ValueError("--min-window-seconds must be positive")
    if args.min_window_seconds > args.max_window_seconds:
        raise ValueError("--min-window-seconds must be <= --max-window-seconds")
    if not args.dry_run and shutil.which(args.ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"ffmpeg executable not found: {args.ffmpeg_bin}. "
            "Pass --ffmpeg-bin /absolute/path/to/ffmpeg on the cluster."
        )

    timeout = None if args.timeout is not None and args.timeout <= 0 else args.timeout
    cfg = {
        "output_root": str(output_root),
        "split": args.split,
        "max_window_seconds": float(args.max_window_seconds),
        "min_window_seconds": float(args.min_window_seconds),
        "ffmpeg_bin": args.ffmpeg_bin,
        "crf": args.crf,
        "preset": args.preset,
        "timeout": timeout,
        "min_bytes": args.min_bytes,
        "overwrite": args.overwrite,
        "dry_run": args.dry_run,
        "fail_on_error": args.fail_on_error,
    }

    print("Ego4D windowed precut clips")
    print(f"  source index       : {index_path}")
    print(f"  output root        : {output_root}")
    print(f"  window index       : {clip_index_path}")
    print(f"  failed jsonl       : {failed_path}")
    print(f"  stats json         : {stats_path}")
    print(f"  split={args.split} limit={args.limit} start_line={args.start_line} workers={args.num_workers}")
    print(f"  max_window={args.max_window_seconds}s min_window={args.min_window_seconds}s")
    print(f"  crf={args.crf} preset={args.preset} overwrite={args.overwrite} dry_run={args.dry_run}")

    print("Counting source records for progress...", flush=True)
    total_source_records = count_source_records(index_path, args.limit, args.start_line)
    print(f"  source records     : {total_source_records}", flush=True)

    stats: Dict[str, Any] = {
        "version": WINDOW_VERSION,
        "source_index": str(index_path),
        "output_root": str(output_root),
        "clip_index": str(clip_index_path),
        "failed_jsonl": str(failed_path),
        "split": args.split,
        "limit": args.limit,
        "start_line": args.start_line,
        "num_workers": args.num_workers,
        "max_window_seconds": args.max_window_seconds,
        "min_window_seconds": args.min_window_seconds,
        "crf": args.crf,
        "preset": args.preset,
        "overwrite": args.overwrite,
        "dry_run": args.dry_run,
        "total_source_records": total_source_records,
        "processed_records": 0,
        "ok_records": 0,
        "partial_records": 0,
        "failed_records": 0,
        "skipped_records": 0,
        "output_windows": 0,
        "cut_windows": 0,
        "reused_windows": 0,
        "failed_windows": 0,
        "expanded_records": 0,
        "reasons": {},
        "task_counts": {},
        "split_counts": {},
        "input_duration_bins": {},
        "output_window_duration_bins": {},
        "windows_per_record_bins": {},
    }
    start = time.perf_counter()

    records_iter = ((idx, rec, cfg) for idx, rec in iter_index(index_path, args.limit, args.start_line))
    with clip_index_path.open("w", encoding="utf-8") as ok_fp, failed_path.open("w", encoding="utf-8") as fail_fp:
        if args.num_workers == 1:
            result_iter = map(process_record, records_iter)
        else:
            result_iter = unordered_process_records(records_iter, args.num_workers)

        for result in result_iter:
            stats["processed_records"] += 1
            status = result.get("status")

            if status == "ok":
                stats["ok_records"] += 1
            elif status == "partial":
                stats["partial_records"] += 1
            elif status == "skipped":
                stats["skipped_records"] += 1
                bump_counter(stats["reasons"], result.get("reason", "unknown"))
                write_jsonl_line(fail_fp, result)
                fail_fp.flush()
                continue
            else:
                stats["failed_records"] += 1
                bump_counter(stats["reasons"], result.get("reason", "unknown"))
                write_jsonl_line(fail_fp, result)
                fail_fp.flush()
                continue

            records = result.get("records") or []
            failures = result.get("failures") or []
            for record in records:
                write_jsonl_line(ok_fp, record)
            for failure in failures:
                bump_counter(stats["reasons"], failure.get("reason", "unknown"))
                write_jsonl_line(fail_fp, failure)
            ok_fp.flush()
            fail_fp.flush()

            output_count = len(records)
            failed_count = len(failures)
            stats["output_windows"] += output_count
            stats["failed_windows"] += failed_count
            stats["reused_windows"] += int(result.get("reused", 0))
            stats["cut_windows"] += int(result.get("cut", 0))
            if result.get("expanded"):
                stats["expanded_records"] += 1

            bump_counter(stats["task_counts"], result.get("task", "unknown"))
            bump_counter(stats["split_counts"], result.get("split", "unknown"))
            input_duration = float(result.get("input_segment_duration_sec") or 0.0)
            bump_counter(stats["input_duration_bins"], duration_bin(input_duration))
            bump_counter(stats["windows_per_record_bins"], output_count + failed_count)
            for window_duration in result.get("window_duration_bins") or []:
                bump_counter(stats["output_window_duration_bins"], duration_bin(float(window_duration)))

            if stats["processed_records"] % max(1, args.progress_interval) == 0:
                elapsed = time.perf_counter() - start
                rate = stats["processed_records"] / max(elapsed, 1e-6)
                remaining = max(total_source_records - stats["processed_records"], 0)
                percent = 100.0 * stats["processed_records"] / max(total_source_records, 1)
                eta = remaining / max(rate, 1e-6)
                print(
                    f"[progress] processed={stats['processed_records']}/{total_source_records} "
                    f"remaining={remaining} pct={percent:.2f}% "
                    f"ok_records={stats['ok_records']} partial={stats['partial_records']} "
                    f"failed_records={stats['failed_records']} skipped={stats['skipped_records']} "
                    f"windows={stats['output_windows']} failed_windows={stats['failed_windows']} "
                    f"reused={stats['reused_windows']} cut={stats['cut_windows']} "
                    f"rate={rate:.2f}/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                    flush=True,
                )

    stats["seconds"] = time.perf_counter() - start
    stats["rate_records_per_second"] = stats["processed_records"] / max(stats["seconds"], 1e-6)
    stats["rate_windows_per_second"] = stats["output_windows"] / max(stats["seconds"], 1e-6)
    if stats["ok_records"] + stats["partial_records"] > 0:
        stats["expansion_factor"] = stats["output_windows"] / max(
            stats["ok_records"] + stats["partial_records"], 1
        )
    else:
        stats["expansion_factor"] = 0.0

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Done.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.fail_on_error and (stats["failed_records"] > 0 or stats["failed_windows"] > 0):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
