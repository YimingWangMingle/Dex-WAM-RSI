#!/usr/bin/env python3
"""Build and index one immutable EGOSCALE Episode Catalog shard."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data.catalog.egoscale import (  # noqa: E402
    build_egoscale_episode_catalog,
)
from motus_data.catalog.sqlite_index import (  # noqa: E402
    CatalogOffsetIndex,
    build_catalog_offset_index,
    catalog_metadata_path,
    default_catalog_index_path,
)
from motus_data.index_contracts import (  # noqa: E402
    validate_catalog_metadata_v2,
)


DEFAULT_DATA_VERSION = "20260816_shards_v1"
PARTITION_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ProgressPrinter:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = float(interval_seconds)
        self.last_print = 0.0

    def __call__(self, state: Mapping[str, Any]) -> None:
        now = time.monotonic()
        if not state.get("done") and now - self.last_print < self.interval_seconds:
            return
        self.last_print = now
        if "total_bytes" in state:
            completed = int(state.get("bytes", 0))
            total = int(state.get("total_bytes", 0))
            percent = 100.0 * completed / max(total, 1)
            detail = f"records={state.get('records', 0)}"
            phase = "index"
        else:
            percent = float(state.get("percent", 0.0))
            processed = int(state.get("processed", 0))
            total = int(state.get("total", 0))
            elapsed = float(state.get("elapsed_seconds", 0.0))
            rate = processed / elapsed if elapsed > 0.0 else 0.0
            remaining = max(total - processed, 0)
            eta = remaining / rate if rate > 0.0 else float("inf")
            detail = (
                f"episodes={processed}/{total} frames={state.get('frames', 0)} "
                f"rate={rate:.2f}/s "
                f"eta_seconds={eta:.0f}"
            )
            if state.get("resumed"):
                detail += f" resumed_from={state.get('resumed_from', 0)}"
            phase = str(state.get("phase", "catalog"))
        print(
            f"phase={phase} progress={percent:.2f}% {detail}",
            flush=True,
        )


def _parse_shard_indices(value: str) -> tuple[int, ...]:
    indices: list[int] = []
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError("empty shard index component")
        if "-" in part:
            pieces = part.split("-", 1)
            try:
                start, end = (int(piece) for piece in pieces)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid shard range={part!r}"
                ) from error
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"shard range must be ascending: {part!r}"
                )
            indices.extend(range(start, end + 1))
        else:
            try:
                indices.append(int(part))
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid shard index={part!r}"
                ) from error
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("shard indices must not contain duplicates")
    return tuple(sorted(indices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--dataset-name",
        default="egoscale_first500h",
        help="stable child/file prefix; defaults to the existing 500H release",
    )
    parser.add_argument("--data-version", default=DEFAULT_DATA_VERSION)
    parser.add_argument("--shard-count", type=int, default=10)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shard-index", type=int)
    selection.add_argument(
        "--shard-indices",
        type=_parse_shard_indices,
        help="comma-separated indices and/or ranges, for example 1-9 or 1,3-5",
    )
    parser.add_argument(
        "--partition-name",
        help="stable artifact label; required for a multi-bucket release",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching durable partial Catalog checkpoint",
    )
    parser.add_argument(
        "--checkpoint-interval-episodes",
        type=int,
        default=500,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            payload = handle.read(8 * 1024 * 1024)
            if not payload:
                return digest.hexdigest()
            digest.update(payload)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    dataset_name = str(args.dataset_name).strip().lower()
    if PARTITION_NAME_PATTERN.fullmatch(dataset_name) is None:
        raise ValueError("dataset-name must match ^[a-z0-9][a-z0-9_-]*$")
    selected_indices = (
        (args.shard_index,)
        if args.shard_indices is None
        else tuple(args.shard_indices)
    )
    if any(not 0 <= index < args.shard_count for index in selected_indices):
        raise ValueError("shard indices must satisfy 0 <= index < shard-count")
    if len(selected_indices) > 1 and not args.partition_name:
        raise ValueError("--partition-name is required for a multi-bucket release")
    if args.partition_name:
        partition_name = str(args.partition_name).strip().lower()
        if PARTITION_NAME_PATTERN.fullmatch(partition_name) is None:
            raise ValueError(
                "partition-name must match ^[a-z0-9][a-z0-9_-]*$"
            )
    else:
        partition_name = f"shard{selected_indices[0]:02d}_of{args.shard_count:02d}"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_id = f"{dataset_name}_{partition_name}_{args.data_version}"
    catalog = output_root / (
        f"{dataset_name}.{partition_name}.{args.data_version}"
        ".episode_catalog.jsonl"
    )
    metadata_path = catalog_metadata_path(catalog)
    index_path = default_catalog_index_path(catalog)
    progress = ProgressPrinter(args.progress_interval_seconds)
    lock_path = output_root / f".{partition_name}.build.lock"

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another EGOSCALE build holds {lock_path}") from error

        if args.force:
            catalog.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)

        status = "reused"
        if catalog.exists() or metadata_path.exists():
            if not catalog.is_file() or not metadata_path.is_file():
                raise RuntimeError(f"incomplete EGOSCALE Catalog artifacts: {catalog}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validate_catalog_metadata_v2(metadata)
            if metadata["catalog_id"] != catalog_id:
                raise RuntimeError("EGOSCALE catalog_id mismatch")
            if metadata["source"] != "egoscale":
                raise RuntimeError("EGOSCALE Catalog source mismatch")
            if _sha256(catalog) != metadata["sha256"]:
                raise RuntimeError("EGOSCALE Catalog SHA256 mismatch")
        else:
            metadata = build_egoscale_episode_catalog(
                data_root=args.data_root,
                storage_root=args.storage_root,
                output_path=catalog,
                catalog_id=catalog_id,
                data_version=args.data_version,
                child=dataset_name,
                shard_count=args.shard_count,
                shard_index=(
                    selected_indices[0] if args.shard_indices is None else None
                ),
                shard_indices=(
                    None if args.shard_indices is None else selected_indices
                ),
                max_workers=args.workers,
                progress_callback=progress,
                resume=args.resume,
                checkpoint_interval_episodes=args.checkpoint_interval_episodes,
                force=args.force,
            )
            status = "built"

        sharding = metadata["meta"]["sharding"]
        actual_indices = tuple(
            sharding.get("selected_shard_indices")
            or [sharding.get("shard_index")]
        )
        if actual_indices != selected_indices:
            raise RuntimeError(
                "EGOSCALE Catalog shard selection mismatch: "
                f"expected={selected_indices} actual={actual_indices}"
            )

        if not index_path.exists():
            build_catalog_offset_index(
                catalog,
                progress_callback=progress,
            )
        index = CatalogOffsetIndex(
            index_path,
            catalog_path=catalog,
            catalog_metadata=metadata,
        )
        try:
            if len(index) != int(metadata["episodes"]):
                raise RuntimeError("EGOSCALE offset index count mismatch")
        finally:
            index.close()

    report = {
        "status": status,
        "catalog": str(catalog),
        "metadata": str(metadata_path),
        "index": str(index_path),
        "catalog_id": metadata["catalog_id"],
        "episodes": metadata["episodes"],
        "frames": metadata["meta"]["build"]["frames"],
        "sha256": metadata["sha256"],
        "sharding": metadata["meta"]["sharding"],
    }
    report_path = output_root / f"{dataset_name}.{partition_name}.report.json"
    _atomic_json(report_path, report)
    print("EGOSCALE_RELEASE_DONE " + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
