#!/usr/bin/env python3
"""Audit an EGOSCALE delivery without modifying raw data or building a Catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data.catalog.egoscale import (
    EGOSCALE_PARQUET_COLUMNS,
    EGOSCALE_REQUIRED_PATHS,
    _episode_record,
    egoscale_source_group_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def _schema_signature(schema: Any, *, ordered: bool) -> str:
    fields = [
        (field.name, str(field.type), bool(field.nullable)) for field in schema
    ]
    if not ordered:
        fields.sort()
    payload = json.dumps(fields, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sample_error_key(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _audit_episode(episode: Path, published_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"episode": episode.name}
    try:
        result["source_group_key"] = egoscale_source_group_key(episode.name)
        record, statistics = _episode_record(
            episode, published_root=published_root
        )
        parquet_columns = tuple(record["meta"]["raw_parquet_columns"])
        result.update(
            {
                "status": "ok",
                "missing_paths": [],
                "empty_paths": [],
                "column_count": len(parquet_columns),
                "columns_exact_order": bool(
                    statistics["parquet_column_order_matches_canonical"]
                ),
                "columns_same_set": set(parquet_columns)
                == set(EGOSCALE_PARQUET_COLUMNS),
                "parquet_column_order_fingerprint": str(
                    statistics["parquet_column_order_fingerprint"]
                ),
                "frames": int(statistics["frames"]),
                "texts": int(statistics["texts"]),
                "fine_actions": int(statistics["fine_actions"]),
                "task_usable": bool(statistics["task_usable"]),
                "annotation_uses_source_group_id": bool(
                    statistics["annotation_uses_source_group_id"]
                ),
                "null_value_columns": list(
                    statistics["null_value_columns"]
                ),
                "source_data_type": str(
                    record.get("meta", {}).get("source_data_type", "")
                ),
            }
        )
    except Exception as error:  # report every fail-closed builder rejection
        result["status"] = "error"
        result["error"] = _sample_error_key(error)
        missing = []
        empty = []
        for relative in EGOSCALE_REQUIRED_PATHS:
            path = episode / relative
            if not path.is_file():
                missing.append(relative)
            elif path.stat().st_size <= 0:
                empty.append(relative)
        result["missing_paths"] = missing
        result["empty_paths"] = empty
        parquet_path = episode / "data/chunk-000/episode_000000.parquet"
        if parquet_path.is_file() and parquet_path.stat().st_size > 0:
            try:
                schema = pq.ParquetFile(parquet_path).schema_arrow
                names = tuple(schema.names)
                result.update(
                    {
                        "column_count": len(names),
                        "columns_exact_order": names
                        == EGOSCALE_PARQUET_COLUMNS,
                        "columns_same_set": set(names)
                        == set(EGOSCALE_PARQUET_COLUMNS),
                        "ordered_schema_signature": _schema_signature(
                            schema, ordered=True
                        ),
                        "set_schema_signature": _schema_signature(
                            schema, ordered=False
                        ),
                        "column_names": list(names),
                    }
                )
            except Exception as schema_error:
                result["schema_probe_error"] = _sample_error_key(schema_error)
    return result


def _chunks(values: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _append_example(
    examples: dict[str, list[str]], key: str, episode: str, limit: int
) -> None:
    if len(examples[key]) < limit:
        examples[key].append(episode)


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("workers, batch-size and progress-every must be positive")
    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output == root or root in output.parents:
        raise ValueError("audit output must not be inside the read-only data root")
    output.parent.mkdir(parents=True, exist_ok=True)

    episodes = sorted(
        (
            Path(entry.path)
            for entry in os.scandir(root)
            if entry.is_dir(follow_symlinks=False)
        ),
        key=lambda path: path.name,
    )
    total_inventory = len(episodes)
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("max-episodes must be positive")
        episodes = episodes[: args.max_episodes]

    started = time.monotonic()
    counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    error_counts: Counter[str] = Counter()
    error_examples: dict[str, list[str]] = defaultdict(list)
    path_missing_counts: Counter[str] = Counter()
    path_empty_counts: Counter[str] = Counter()
    frames = 0
    texts = 0
    fine_actions = 0
    source_groups = set()
    processed = 0

    worker = lambda path: _audit_episode(path, root)  # noqa: E731
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch in _chunks(episodes, args.batch_size):
            for result in executor.map(worker, batch):
                processed += 1
                status = str(result["status"])
                counts[status] += 1
                group = result.get("source_group_key")
                if group:
                    source_groups.add(str(group))
                for path in result.get("missing_paths", []):
                    path_missing_counts[str(path)] += 1
                for path in result.get("empty_paths", []):
                    path_empty_counts[str(path)] += 1
                for field in (
                    "column_count",
                    "columns_exact_order",
                    "columns_same_set",
                    "parquet_column_order_fingerprint",
                    "ordered_schema_signature",
                    "set_schema_signature",
                    "source_data_type",
                    "task_usable",
                    "annotation_uses_source_group_id",
                ):
                    if field in result:
                        value = json.dumps(result[field], sort_keys=True)
                        value_counts[field][value] += 1
                        _append_example(
                            examples,
                            f"{field}={value}",
                            str(result["episode"]),
                            args.max_examples,
                        )
                for column in result.get("null_value_columns", []):
                    value_counts["null_value_columns"][str(column)] += 1
                if status == "ok":
                    frames += int(result["frames"])
                    texts += int(result["texts"])
                    fine_actions += int(result["fine_actions"])
                else:
                    error = str(result.get("error", "unknown error"))
                    error_counts[error] += 1
                    _append_example(
                        error_examples,
                        error,
                        str(result["episode"]),
                        args.max_examples,
                    )
                if processed % args.progress_every == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"processed={processed}/{len(episodes)} "
                        f"percent={processed / len(episodes) * 100:.2f} "
                        f"ok={counts['ok']} error={counts['error']} "
                        f"rate={processed / max(elapsed, 1e-9):.1f}/s",
                        flush=True,
                    )

    elapsed = time.monotonic() - started
    report: Mapping[str, Any] = {
        "status": "complete",
        "data_root": str(root),
        "inventory_episode_dirs": total_inventory,
        "audited_episode_dirs": len(episodes),
        "source_groups": len(source_groups),
        "counts": dict(sorted(counts.items())),
        "frames_in_accepted_episodes": frames,
        "texts_in_accepted_episodes": texts,
        "fine_actions_in_accepted_episodes": fine_actions,
        "required_paths": list(EGOSCALE_REQUIRED_PATHS),
        "expected_parquet_columns": list(EGOSCALE_PARQUET_COLUMNS),
        "value_counts": {
            key: dict(sorted(values.items()))
            for key, values in sorted(value_counts.items())
        },
        "examples": dict(sorted(examples.items())),
        "missing_path_counts": dict(sorted(path_missing_counts.items())),
        "empty_path_counts": dict(sorted(path_empty_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "error_examples": dict(sorted(error_examples.items())),
        "elapsed_seconds": elapsed,
        "rate_episodes_per_second": processed / max(elapsed, 1e-9),
    }
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        f"DONE audited={len(episodes)} ok={counts['ok']} "
        f"error={counts['error']} elapsed={elapsed:.1f}s output={output}",
        flush=True,
    )
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
