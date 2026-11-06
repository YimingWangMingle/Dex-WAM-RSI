#!/usr/bin/env python3
"""Build the first split-neutral EgoVerse release, one child at a time."""

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

from motus_data.catalog.egoverse import build_egoverse_episode_catalog
from motus_data.catalog.sqlite_index import (
    CatalogOffsetIndex,
    build_catalog_offset_index,
    catalog_metadata_path,
    default_catalog_index_path,
)
from motus_data.index_contracts import validate_catalog_metadata_v2


DEFAULT_DATA_VERSION = "20260730_first6_v1"
PROFILE_ROOTS = {
    "aria_bimanual": "processed_v3/aria",
    "aria_left_arm": "processed_v3/aria",
    "aria_right_arm": "processed_v3/aria",
    "eva_bimanual": "processed_v3/eva",
    "scale_bimanual": "processed_v3/scale",
    "mecka_freeform_bimanual": "processed_v3/mecka/freeform",
}


class ProgressPrinter:
    def __init__(self, profile: str, interval_seconds: float = 5.0):
        self.profile = profile
        self.interval_seconds = float(interval_seconds)
        self.last_phase: str | None = None
        self.last_print = 0.0

    def __call__(self, state: Mapping[str, Any]) -> None:
        now = time.monotonic()
        phase = str(
            state.get("phase")
            or ("index" if "total_bytes" in state else "build")
        )
        done = bool(state.get("done"))
        if (
            not done
            and phase == self.last_phase
            and now - self.last_print < self.interval_seconds
        ):
            return
        self.last_phase = phase
        self.last_print = now

        if "total_bytes" in state:
            total = int(state.get("total_bytes", 0))
            completed = int(state.get("bytes", 0))
            percent = 100.0 * completed / max(1, total)
        else:
            percent = float(state.get("percent", 0.0))
        fields = [
            f"profile={self.profile}",
            f"phase={phase}",
            f"progress={percent:.2f}%",
        ]
        for key in (
            "processed",
            "total",
            "discovered_episodes",
            "scan_errors",
            "written",
            "skipped",
            "records",
            "elapsed_seconds",
        ):
            if key not in state:
                continue
            value = state[key]
            if key == "elapsed_seconds":
                fields.append(f"{key}={float(value):.1f}")
            else:
                fields.append(f"{key}={value}")
        print(" ".join(fields), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--egoverse-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-version", default=DEFAULT_DATA_VERSION)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_ROOTS),
        default=list(PROFILE_ROOTS),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--quarantine-episode-errors",
        action="store_true",
        help=(
            "publish only fully validated Episodes and write every rejected "
            "Episode to a bound quarantine sidecar"
        ),
    )
    return parser.parse_args()


def _catalog_path(
    output_root: Path,
    profile: str,
    data_version: str,
) -> Path:
    return (
        output_root
        / f"egoverse_{profile}.{data_version}.episode_catalog.jsonl"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_quarantine(
    catalog: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    quarantine = metadata["meta"]["build"].get("quarantine")
    if not isinstance(quarantine, Mapping):
        raise RuntimeError("EgoVerse Catalog metadata lacks quarantine binding")
    path = catalog.parent / str(quarantine["filename"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_digest = _sha256(path)
    if actual_digest != quarantine["sha256"]:
        raise RuntimeError(
            f"quarantine SHA256 mismatch: {actual_digest} "
            f"!= {quarantine['sha256']}"
        )
    if path.stat().st_size != int(quarantine["size_bytes"]):
        raise RuntimeError("quarantine size differs from metadata")
    with path.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != int(quarantine["episodes"]):
        raise RuntimeError(
            f"quarantine rows={rows} != metadata={quarantine['episodes']}"
        )
    return {
        "quarantine_path": str(path),
        "quarantined_episodes": rows,
        "quarantine_sha256": actual_digest,
    }


def _load_completed_metadata(
    catalog: Path,
    *,
    profile: str,
    data_version: str,
    quarantine_episode_errors: bool,
) -> dict[str, Any] | None:
    metadata_path = catalog_metadata_path(catalog)
    index_path = default_catalog_index_path(catalog)
    present = {
        "catalog": catalog.is_file(),
        "metadata": metadata_path.is_file(),
        "index": index_path.is_file(),
    }
    if not present["catalog"] and not present["metadata"]:
        if present["index"]:
            raise RuntimeError(
                f"orphan Catalog index must be removed manually: {index_path}"
            )
        return None
    if not (present["catalog"] and present["metadata"]):
        raise RuntimeError(
            f"incomplete published Catalog artifacts for profile={profile}: "
            f"{present}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_catalog_metadata_v2(metadata)
    expected_id = f"egoverse_{profile}_{data_version}"
    expected_child = f"egoverse_{profile}"
    if metadata["catalog_id"] != expected_id:
        raise RuntimeError(
            f"Catalog ID mismatch: {metadata['catalog_id']!r} != {expected_id!r}"
        )
    if metadata["source"] != "egoverse":
        raise RuntimeError("completed Catalog source is not EgoVerse")
    if metadata["child"] != expected_child:
        raise RuntimeError(
            f"Catalog child mismatch: {metadata['child']!r} "
            f"!= {expected_child!r}"
        )
    if metadata["meta"]["data_version"] != data_version:
        raise RuntimeError("completed Catalog data_version mismatch")
    expected_fail_closed = not quarantine_episode_errors
    actual_fail_closed = bool(
        metadata["meta"]["build"].get("fail_on_episode_error", True)
    )
    if actual_fail_closed != expected_fail_closed:
        raise RuntimeError(
            "completed Catalog episode-error policy differs from this run"
        )
    actual_digest = _sha256(catalog)
    if actual_digest != metadata["sha256"]:
        raise RuntimeError(
            f"Catalog SHA256 mismatch for profile={profile}: "
            f"{actual_digest} != {metadata['sha256']}"
        )
    return metadata


def _verify_index(
    catalog: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    index_path = default_catalog_index_path(catalog)
    index = CatalogOffsetIndex(
        index_path,
        catalog_path=catalog,
        catalog_metadata=metadata,
    )
    try:
        return {
            "index_path": str(index_path),
            "episodes": len(index),
            "index_size_bytes": index_path.stat().st_size,
        }
    finally:
        index.close()


def _write_release_report(
    output_root: Path,
    data_version: str,
    completed: Mapping[str, Any],
) -> Path:
    report_path = output_root / f"egoverse_first6.{data_version}.report.json"
    temporary = report_path.with_name(
        f".{report_path.name}.tmp.{os.getpid()}"
    )
    payload = {
        "format": "motus_egoverse_release_build_report_v1",
        "data_version": data_version,
        "profiles": dict(completed),
        "completed_profiles": len(completed),
        "expected_profiles": len(PROFILE_ROOTS),
        "done": len(completed) == len(PROFILE_ROOTS),
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report_path)
    return report_path


def main() -> int:
    args = parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    root = Path(args.egoverse_root)
    output_root = Path(args.output_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".egoverse_first6_release.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(
            f"another EgoVerse release builder holds {lock_path}"
        ) from error
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()}\n")
    lock_handle.flush()
    completed: dict[str, Any] = {}

    for profile in args.profiles:
        source_root = root / PROFILE_ROOTS[profile]
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        catalog = _catalog_path(output_root, profile, args.data_version)
        metadata = _load_completed_metadata(
            catalog,
            profile=profile,
            data_version=args.data_version,
            quarantine_episode_errors=args.quarantine_episode_errors,
        )
        progress = ProgressPrinter(
            profile,
            interval_seconds=args.progress_interval_seconds,
        )
        print(
            f"EGOVERSE_RELEASE_CHILD_START profile={profile} "
            f"source_root={source_root} catalog={catalog}",
            flush=True,
        )
        if metadata is None:
            metadata = build_egoverse_episode_catalog(
                data_root=source_root,
                profile=profile,
                output_path=catalog,
                catalog_id=f"egoverse_{profile}_{args.data_version}",
                data_version=args.data_version,
                source_child=f"egoverse_{profile}",
                validate_numeric_finite=True,
                validate_jpeg=True,
                fail_on_episode_error=(
                    not args.quarantine_episode_errors
                ),
                fail_on_store_error=True,
                resume=True,
                checkpoint_interval=args.checkpoint_interval,
                progress_callback=progress,
            )
        else:
            print(
                f"EGOVERSE_RELEASE_CATALOG_VERIFIED profile={profile} "
                f"episodes={metadata['episodes']}",
                flush=True,
            )

        index_path = default_catalog_index_path(catalog)
        if not index_path.is_file():
            build_catalog_offset_index(
                catalog,
                progress_callback=progress,
            )
        index_result = _verify_index(catalog, metadata)
        quarantine_result = _verify_quarantine(catalog, metadata)
        build = metadata["meta"]["build"]
        completed[profile] = {
            "catalog": str(catalog),
            "metadata": str(catalog_metadata_path(catalog)),
            "index": str(index_path),
            "episodes": int(metadata["episodes"]),
            "catalog_sha256": str(metadata["sha256"]),
            "scan_error_count": int(build["scan_error_count"]),
            "skipped_episodes": int(build["skipped_episodes"]),
            "action_source_signal": build["profile_contract"][
                "action_source_signal"
            ],
            "action_is_control_command": bool(
                build["profile_contract"]["action_is_control_command"]
            ),
            **index_result,
            **quarantine_result,
        }
        report_path = _write_release_report(
            output_root,
            args.data_version,
            completed,
        )
        print(
            f"EGOVERSE_RELEASE_CHILD_DONE profile={profile} "
            f"episodes={metadata['episodes']} report={report_path}",
            flush=True,
        )

    report_path = _write_release_report(
        output_root,
        args.data_version,
        completed,
    )
    print(
        f"EGOVERSE_FIRST6_RELEASE_DONE profiles={len(completed)} "
        f"report={report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
