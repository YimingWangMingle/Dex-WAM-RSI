#!/usr/bin/env python3
"""Build and index the four split-neutral LIBERO suite Catalogs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data.catalog.libero import (
    LIBERO_CHILDREN,
    build_libero_episode_catalog,
)
from motus_data.catalog.sqlite_index import (
    CatalogOffsetIndex,
    build_catalog_offset_index,
    catalog_metadata_path,
    default_catalog_index_path,
)
from motus_data.index_contracts import validate_catalog_metadata_v2


DEFAULT_DATA_VERSION = "libero_parquet_v1"


class ProgressPrinter:
    def __init__(self, child: str, interval_seconds: float) -> None:
        self.child = child
        self.interval_seconds = float(interval_seconds)
        self.last_print = 0.0

    def __call__(self, state: Mapping[str, Any]) -> None:
        now = time.monotonic()
        if (
            not state.get("done")
            and now - self.last_print < self.interval_seconds
        ):
            return
        self.last_print = now
        if "total_bytes" in state:
            completed = int(state.get("bytes", 0))
            total = int(state.get("total_bytes", 0))
            percent = 100.0 * completed / max(total, 1)
            details = f"records={state.get('records', 0)}"
            phase = "index"
        else:
            percent = float(state.get("percent", 0.0))
            details = (
                f"episodes={state.get('processed', 0)}/"
                f"{state.get('total', 0)} frames={state.get('frames', 0)}"
            )
            phase = str(state.get("phase", "catalog"))
        print(
            f"child={self.child} phase={phase} "
            f"progress={percent:.2f}% {details}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-version", default=DEFAULT_DATA_VERSION)
    parser.add_argument(
        "--children",
        nargs="+",
        choices=tuple(LIBERO_CHILDREN),
        default=list(LIBERO_CHILDREN),
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--validate-numeric-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
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


def _catalog_path(root: Path, child: str, data_version: str) -> Path:
    return root / f"{child}.{data_version}.episode_catalog.jsonl"


def _load_completed(
    catalog: Path,
    *,
    child: str,
    data_version: str,
) -> dict[str, Any] | None:
    metadata_path = catalog_metadata_path(catalog)
    if not catalog.exists() and not metadata_path.exists():
        return None
    if not catalog.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            f"incomplete LIBERO Catalog artifacts: {catalog}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_catalog_metadata_v2(metadata)
    expected_id = f"{child}_{data_version}"
    if metadata["catalog_id"] != expected_id:
        raise RuntimeError(
            f"LIBERO catalog_id mismatch: {metadata['catalog_id']!r}"
        )
    if metadata["source"] != "libero" or metadata["child"] != child:
        raise RuntimeError("LIBERO Catalog source/child mismatch")
    if metadata["meta"].get("data_version") != data_version:
        raise RuntimeError("LIBERO Catalog data_version mismatch")
    actual_sha256 = _sha256(catalog)
    if actual_sha256 != metadata["sha256"]:
        raise RuntimeError(
            f"LIBERO Catalog SHA256 mismatch: {catalog}"
        )
    return metadata


def _verify_or_build_index(
    catalog: Path,
    metadata: Mapping[str, Any],
    *,
    progress: ProgressPrinter,
    force: bool,
) -> dict[str, Any]:
    index_path = default_catalog_index_path(catalog)
    if force and index_path.exists():
        index_path.unlink()
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
            raise RuntimeError("LIBERO offset index Episode count mismatch")
        return {
            "path": str(index_path),
            "episodes": len(index),
            "size_bytes": index_path.stat().st_size,
        }
    finally:
        index.close()


def _atomic_report(path: Path, value: Mapping[str, Any]) -> None:
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
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".libero_release_build.lock"
    completed: dict[str, Any] = {}
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another LIBERO release build holds {lock_path}"
            ) from error
        for child in args.children:
            catalog = _catalog_path(
                output_root,
                child,
                args.data_version,
            )
            progress = ProgressPrinter(
                child,
                args.progress_interval_seconds,
            )
            if args.force:
                catalog.unlink(missing_ok=True)
                catalog_metadata_path(catalog).unlink(missing_ok=True)
            metadata = _load_completed(
                catalog,
                child=child,
                data_version=args.data_version,
            )
            status = "reused"
            if metadata is None:
                metadata = build_libero_episode_catalog(
                    data_root=args.data_root,
                    child=child,
                    output_path=catalog,
                    catalog_id=f"{child}_{args.data_version}",
                    data_version=args.data_version,
                    storage_root=args.storage_root,
                    validate_numeric_finite=args.validate_numeric_finite,
                    max_episodes=args.max_episodes,
                    progress_callback=progress,
                )
                status = "built"
            index = _verify_or_build_index(
                catalog,
                metadata,
                progress=progress,
                force=args.force,
            )
            completed[child] = {
                "status": status,
                "catalog": str(catalog),
                "metadata": str(catalog_metadata_path(catalog)),
                "catalog_id": metadata["catalog_id"],
                "episodes": metadata["episodes"],
                "frames": metadata["meta"]["build"]["frames"],
                "sha256": metadata["sha256"],
                "index": index,
            }
            print(
                "LIBERO_CHILD_DONE "
                + json.dumps(completed[child], sort_keys=True),
                flush=True,
            )
    report = {
        "data_version": args.data_version,
        "data_root": str(args.data_root),
        "storage_root": str(args.storage_root),
        "children": completed,
    }
    report_path = output_root / f"libero.{args.data_version}.report.json"
    _atomic_report(report_path, report)
    print(f"LIBERO_RELEASE_DONE report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
