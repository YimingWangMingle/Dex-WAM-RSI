#!/usr/bin/env python3
"""Read-only audit of Agibot Beta effector schemas from an existing Catalog."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from motus_data.agibot_beta_profiles import (
    expected_agibot_beta_profile_for_task,
    inspect_agibot_beta_proprio,
)
from motus_data.index_contracts import (
    validate_catalog_metadata,
    validate_episode_catalog_binding,
)
from motus_data.storage import BoundedFileReader


def _metadata_path(catalog: Path) -> Path:
    if catalog.suffix == ".jsonl":
        return catalog.with_suffix(".meta.json")
    return Path(str(catalog) + ".meta.json")


def _parse_task_ids(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank audit row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid audit JSON at line {line_number}; do not resume "
                    "until the partial final row is removed"
                ) from error
            episode_id = str(row.get("episode_id") or "")
            if not episode_id or episode_id in completed:
                raise ValueError(
                    f"invalid or duplicate episode_id at audit line {line_number}"
                )
            completed.add(episode_id)
    return completed


def _state_storage(record: Mapping[str, Any]) -> Mapping[str, Any]:
    storage = record["state"]["storage"]
    if storage.get("type") != "tar":
        raise ValueError(
            f"Agibot Beta state storage must be tar, got {storage.get('type')!r}"
        )
    required = ("path", "member", "offset", "size")
    missing = [key for key in required if key not in storage]
    if missing:
        raise ValueError(f"Agibot Beta state storage missing fields: {missing}")
    return storage


def _summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    task_profiles: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    mismatch_examples = []
    mismatch_count = 0
    error_examples = []
    total = 0
    for row in rows:
        total += 1
        status = str(row.get("status"))
        statuses[status] += 1
        task_id = str(row.get("task_id"))
        tasks[task_id] += 1
        if status == "ok":
            profile = str(row["profile"])
            profiles[profile] += 1
            task_profiles[f"{task_id}:{profile}"] += 1
            shape = row["shapes"]["state/effector/position"]
            shapes[str(tuple(int(value) for value in shape))] += 1
            if not bool(row["task_profile_matches_shape"]):
                mismatch_count += 1
                if len(mismatch_examples) < 30:
                    mismatch_examples.append(dict(row))
        elif len(error_examples) < 30:
            error_examples.append(dict(row))
    return {
        "audited_episodes": total,
        "status_counts": dict(sorted(statuses.items())),
        "profile_counts": dict(sorted(profiles.items())),
        "effector_shape_counts": dict(sorted(shapes.items())),
        "task_counts": dict(sorted(tasks.items(), key=lambda item: int(item[0]))),
        "task_profile_counts": dict(
            sorted(
                task_profiles.items(),
                key=lambda item: (int(item[0].split(":", 1)[0]), item[0]),
            )
        ),
        "task_profile_mismatch_count": mismatch_count,
        "task_profile_mismatch_examples": mismatch_examples,
        "error_examples": error_examples,
    }


def _read_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--task-ids",
        help="Optional comma-separated task IDs. Omit for the full Catalog.",
    )
    parser.add_argument("--max-per-task", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--numeric-mode",
        choices=("none", "effector", "all"),
        default="none",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fsync-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = Path(args.catalog).resolve()
    metadata_path = _metadata_path(catalog)
    output = Path(args.output_jsonl).resolve()
    summary_path = Path(args.summary_json).resolve()
    if not catalog.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"audit requires Catalog and metadata: {catalog}, {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_catalog_metadata(metadata)
    if str(metadata.get("source")) != "agibot_beta":
        raise ValueError("audit input must be an Agibot Beta Catalog")
    if int(args.fsync_every) <= 0 or int(args.progress_every) <= 0:
        raise ValueError("fsync-every and progress-every must be positive")
    if args.max_per_task is not None and int(args.max_per_task) <= 0:
        raise ValueError("max-per-task must be positive")
    if args.max_episodes is not None and int(args.max_episodes) <= 0:
        raise ValueError("max-episodes must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"audit output exists; use --resume: {output}")
    completed = _load_completed(output) if args.resume else set()
    task_filter = _parse_task_ids(args.task_ids)
    selected_per_task: Counter[int] = Counter()
    catalog_size = catalog.stat().st_size
    started = time.monotonic()
    scanned = 0
    selected = 0
    written = 0
    errors = 0

    mode = "a" if args.resume else "x"
    with output.open(mode, encoding="utf-8") as output_handle:
        with catalog.open("rb") as catalog_handle:
            for line_number, payload in enumerate(catalog_handle, start=1):
                scanned += 1
                try:
                    record = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid input Catalog JSON at line {line_number}"
                    ) from error
                validate_episode_catalog_binding(record, metadata)
                task_id = int(record["meta"]["task_id"])
                if task_filter is not None and task_id not in task_filter:
                    continue
                if (
                    args.max_per_task is not None
                    and selected_per_task[task_id] >= int(args.max_per_task)
                ):
                    continue
                selected_per_task[task_id] += 1
                selected += 1
                episode_id = str(record["episode_id"])
                if episode_id in completed:
                    continue
                row: dict[str, Any] = {
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "raw_episode_id": int(record["meta"]["raw_episode_id"]),
                }
                try:
                    storage = _state_storage(record)
                    with BoundedFileReader(
                        storage["path"], storage["offset"], storage["size"]
                    ) as member:
                        inspection = inspect_agibot_beta_proprio(
                            member, numeric_mode=args.numeric_mode
                        )
                    expected = expected_agibot_beta_profile_for_task(task_id)
                    row.update(inspection)
                    row.update(
                        {
                            "status": "ok",
                            "expected_profile_from_task": expected,
                            "task_profile_matches_shape": (
                                str(inspection["profile"]) == expected
                            ),
                            "storage": {
                                "path": str(storage["path"]),
                                "member": str(storage["member"]),
                                "offset": int(storage["offset"]),
                                "size": int(storage["size"]),
                            },
                        }
                    )
                except Exception as error:
                    errors += 1
                    row.update(
                        {
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    if args.fail_fast:
                        raise
                output_handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                written += 1
                if written % int(args.fsync_every) == 0:
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                if written % int(args.progress_every) == 0:
                    elapsed = time.monotonic() - started
                    byte_percent = (
                        100.0 * catalog_handle.tell() / max(1, catalog_size)
                    )
                    print(
                        "AUDIT_PROGRESS "
                        f"scanned={scanned} selected={selected} written={written} "
                        f"errors={errors} catalog_bytes={byte_percent:.2f}% "
                        f"rate={written / max(elapsed, 1e-9):.2f}/s "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                if (
                    args.max_episodes is not None
                    and selected >= int(args.max_episodes)
                ):
                    break
        output_handle.flush()
        os.fsync(output_handle.fileno())

    summary = _summary(_read_rows(output))
    summary.update(
        {
            "format": "agibot_beta_effector_profile_audit_v1",
            "input_catalog": str(catalog),
            "input_catalog_id": str(metadata["catalog_id"]),
            "input_catalog_sha256": str(
                metadata.get("sha256")
                or (metadata.get("statistics") or {}).get("catalog_sha256")
            ),
            "task_filter": sorted(task_filter) if task_filter is not None else None,
            "numeric_mode": str(args.numeric_mode),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    temporary_summary = summary_path.with_name(
        f".{summary_path.name}.tmp.{os.getpid()}"
    )
    with temporary_summary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_summary, summary_path)
    print(
        "AUDIT_DONE "
        + json.dumps(
            {
                "output": str(output),
                "summary": str(summary_path),
                "audited_episodes": summary["audited_episodes"],
                "profiles": summary["profile_counts"],
                "errors": summary["status_counts"].get("error", 0),
                "task_profile_mismatches": summary[
                    "task_profile_mismatch_count"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 2 if summary["status_counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
