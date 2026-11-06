#!/usr/bin/env python3
"""Mark a fixed record-level holdout split inside a Motus Manifest JSONL.

The script keeps a single manifest file as the source for both training and
evaluation. It rewrites selected records from ``split=train`` to ``split=test``
and leaves all other records unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _maybe_report_progress(
    *,
    label: str,
    path: Path,
    bytes_done: int,
    total_bytes: int,
    last_percent: Optional[int],
    every_percent: int,
) -> int:
    if total_bytes <= 0:
        return last_percent if last_percent is not None else -1
    percent = min(100, int(bytes_done * 100 / total_bytes))
    if last_percent is None or percent >= last_percent + every_percent or percent == 100:
        print(f"[{label}] {path.name}: {percent}% ({bytes_done}/{total_bytes} bytes)", flush=True)
        return percent
    return last_percent


def _read_jsonl_record_ids(path: Path, source_split: str, progress_every_percent: int) -> Dict[str, Any]:
    ids: List[str] = []
    duplicate_ids: Set[str] = set()
    seen: Set[str] = set()
    split_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    total_bytes = path.stat().st_size
    last_percent: Optional[int] = None

    bytes_done = 0
    with path.open("rb") as f:
        for line_no, line in enumerate(f, start=1):
            bytes_done += len(line)
            last_percent = _maybe_report_progress(
                label="scan",
                path=path,
                bytes_done=bytes_done,
                total_bytes=total_bytes,
                last_percent=last_percent,
                every_percent=progress_every_percent,
            )
            record = json.loads(line)
            schema_counts[str(record.get("schema_version"))] += 1
            split = str(record.get("split", ""))
            split_counts[split] += 1
            if split != source_split:
                continue
            record_id = record.get("id")
            if not record_id:
                raise ValueError(f"Missing id in {path} line {line_no}")
            record_id = str(record_id)
            if record_id in seen:
                duplicate_ids.add(record_id)
            seen.add(record_id)
            ids.append(record_id)

    return {
        "ids": ids,
        "duplicate_ids": sorted(duplicate_ids),
        "split_counts": dict(split_counts),
        "schema_counts": dict(schema_counts),
    }


def _resolve_num_holdout(
    *,
    eligible_count: int,
    num_holdout: Optional[int],
    holdout_fraction: Optional[float],
    min_holdout: int,
    max_holdout: Optional[int],
) -> int:
    if eligible_count <= 0:
        return 0
    if holdout_fraction is not None:
        if holdout_fraction <= 0 or holdout_fraction >= 1:
            raise ValueError(f"--holdout-fraction must be in (0, 1), got {holdout_fraction}")
        count = max(int(min_holdout), int(round(eligible_count * holdout_fraction)))
    elif num_holdout is not None:
        count = int(num_holdout)
    else:
        raise ValueError("Either --num-holdout or --holdout-fraction must be set")
    if max_holdout is not None:
        count = min(count, int(max_holdout))
    return max(0, min(count, eligible_count))


def _select_holdout_ids(ids: List[str], num_holdout: int, seed: int, allow_fewer: bool) -> List[str]:
    unique_ids = sorted(set(ids))
    if len(unique_ids) < num_holdout and not allow_fewer:
        raise ValueError(
            f"Not enough unique eligible records for holdout: "
            f"eligible={len(unique_ids)} requested={num_holdout}"
        )
    count = min(len(unique_ids), int(num_holdout))
    rng = random.Random(int(seed))
    return sorted(rng.sample(unique_ids, k=count))


def _rewrite_manifest(
    input_path: Path,
    output_path: Path,
    holdout_ids: Set[str],
    source_split: str,
    holdout_split: str,
    progress_every_percent: int,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    before_splits: Counter[str] = Counter()
    after_splits: Counter[str] = Counter()
    selected_seen: Set[str] = set()
    total_bytes = input_path.stat().st_size
    last_percent: Optional[int] = None

    bytes_done = 0
    with input_path.open("rb") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            bytes_done += len(line)
            last_percent = _maybe_report_progress(
                label="rewrite",
                path=input_path,
                bytes_done=bytes_done,
                total_bytes=total_bytes,
                last_percent=last_percent,
                every_percent=progress_every_percent,
            )
            record = json.loads(line)
            before_split = str(record.get("split", ""))
            before_splits[before_split] += 1
            record_id = str(record.get("id", ""))
            if before_split == source_split and record_id in holdout_ids:
                record["split"] = holdout_split
                selected_seen.add(record_id)
                counts["marked_holdout"] += 1
            else:
                counts["unchanged"] += 1
            after_splits[str(record.get("split", ""))] += 1
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            dst.write("\n")

    missing_selected = sorted(holdout_ids - selected_seen)
    if missing_selected:
        raise RuntimeError(
            f"{len(missing_selected)} selected holdout ids were not rewritten; "
            f"first missing ids: {missing_selected[:10]}"
        )
    return {
        "rewrite_counts": dict(counts),
        "before_split_counts": dict(before_splits),
        "after_split_counts": dict(after_splits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Motus Manifest JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL with train/test split marks.")
    parser.add_argument("--summary", required=True, help="Output summary JSON path.")
    parser.add_argument("--num-holdout", type=int, default=None, help="Exact number of records to mark as holdout.")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.01,
        help="Fraction of eligible records to mark as holdout, e.g. 0.01 for 1%%.",
    )
    parser.add_argument("--min-holdout", type=int, default=1, help="Minimum holdout count when using fraction.")
    parser.add_argument("--max-holdout", type=int, default=None, help="Optional maximum holdout count.")
    parser.add_argument("--seed", type=int, default=20260713, help="Deterministic random seed.")
    parser.add_argument("--source-split", default="train", help="Only records with this split are eligible.")
    parser.add_argument("--holdout-split", default="test", help="Split value assigned to selected records.")
    parser.add_argument("--name", default=None, help="Optional child dataset name for summary metadata.")
    parser.add_argument("--allow-fewer", action="store_true", help="Allow fewer holdout records if input is small.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output files.")
    parser.add_argument(
        "--progress-every-percent",
        type=int,
        default=1,
        help="Print scan/rewrite progress at this percent interval. Set <=0 to disable.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    if not input_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_path}")
    for path in (output_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")

    progress_every_percent = max(101, int(args.progress_every_percent)) if int(args.progress_every_percent) <= 0 else max(1, int(args.progress_every_percent))
    scan = _read_jsonl_record_ids(
        input_path,
        source_split=str(args.source_split),
        progress_every_percent=progress_every_percent,
    )
    resolved_num_holdout = _resolve_num_holdout(
        eligible_count=len(set(scan["ids"])),
        num_holdout=args.num_holdout,
        holdout_fraction=args.holdout_fraction,
        min_holdout=int(args.min_holdout),
        max_holdout=args.max_holdout,
    )
    holdout_ids = _select_holdout_ids(
        ids=scan["ids"],
        num_holdout=int(resolved_num_holdout),
        seed=int(args.seed),
        allow_fewer=bool(args.allow_fewer),
    )
    rewrite = _rewrite_manifest(
        input_path=input_path,
        output_path=output_path,
        holdout_ids=set(holdout_ids),
        source_split=str(args.source_split),
        holdout_split=str(args.holdout_split),
        progress_every_percent=progress_every_percent,
    )

    summary = {
        "name": args.name,
        "input": str(input_path),
        "output": str(output_path),
        "seed": int(args.seed),
        "source_split": str(args.source_split),
        "holdout_split": str(args.holdout_split),
        "requested_holdout": None if args.num_holdout is None else int(args.num_holdout),
        "holdout_fraction": args.holdout_fraction,
        "min_holdout": int(args.min_holdout),
        "max_holdout": args.max_holdout,
        "resolved_holdout": int(resolved_num_holdout),
        "selected_holdout": len(holdout_ids),
        "eligible_records": len(scan["ids"]),
        "eligible_unique_records": len(set(scan["ids"])),
        "actual_holdout_fraction": (len(holdout_ids) / len(set(scan["ids"]))) if scan["ids"] else 0.0,
        "duplicate_id_count": len(scan["duplicate_ids"]),
        "duplicate_ids_sample": scan["duplicate_ids"][:20],
        "schema_counts": scan["schema_counts"],
        **rewrite,
        "holdout_ids": holdout_ids,
        "holdout_ids_sample": holdout_ids[:20],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: summary[k] for k in (
        "name",
        "input",
        "output",
        "seed",
        "selected_holdout",
        "eligible_records",
        "before_split_counts",
        "after_split_counts",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
