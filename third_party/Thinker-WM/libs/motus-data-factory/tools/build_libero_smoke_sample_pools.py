#!/usr/bin/env python3
"""Build deterministic LIBERO SamplePools for data-path validation only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data.catalog.libero import LIBERO_CHILDREN
from motus_data.catalog.sqlite_index import default_catalog_index_path
from motus_data.repository import EpisodeRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-version", default="libero_parquet_v1")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--windows-per-episode", type=int, default=4)
    parser.add_argument("--video-frames", type=int, default=9)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _catalog_path(root: Path, child: str, version: str) -> Path:
    return root / f"{child}.{version}.episode_catalog.jsonl"


def _window_starts(
    count: int,
    *,
    video_frames: int,
    windows: int,
) -> tuple[int, ...]:
    maximum = int(count) - int(video_frames)
    if maximum < 0:
        raise ValueError(
            f"Episode count={count} is shorter than video_frames={video_frames}"
        )
    if windows == 1:
        return (maximum // 2,)
    starts = tuple(
        round(position * maximum / (windows - 1))
        for position in range(windows)
    )
    if len(set(starts)) != windows:
        raise ValueError(
            f"Episode count={count} cannot provide {windows} distinct windows"
        )
    return starts


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if args.windows_per_episode <= 0:
        raise ValueError("windows_per_episode must be positive")
    if args.video_frames < 2:
        raise ValueError("video_frames must be at least 2")
    catalog_root = Path(args.catalog_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    completed: dict[str, Any] = {}

    for child, spec in LIBERO_CHILDREN.items():
        catalog = _catalog_path(catalog_root, child, args.data_version)
        index = default_catalog_index_path(catalog)
        repository = EpisodeRepository(
            catalog,
            index_path=index,
            verify_sha256=False,
        )
        try:
            selected_by_task: dict[int, list[str]] = defaultdict(list)
            for episode_id in repository.iter_episode_ids():
                episode = repository.get_episode(episode_id)
                task_index = int(episode.info.metadata["task_index"])
                if len(selected_by_task[task_index]) < args.episodes_per_task:
                    selected_by_task[task_index].append(episode_id)
                expected_tasks = set(range(spec.task_start, spec.task_stop))
                if (
                    set(selected_by_task) == expected_tasks
                    and all(
                        len(selected_by_task[index]) >= args.episodes_per_task
                        for index in expected_tasks
                    )
                ):
                    break
            expected_tasks = set(range(spec.task_start, spec.task_stop))
            if set(selected_by_task) != expected_tasks:
                raise RuntimeError(
                    f"{child} smoke selection does not cover every task"
                )

            selections = []
            episode_task_map: dict[str, int] = {}
            for task_index in sorted(selected_by_task):
                for episode_id in selected_by_task[task_index]:
                    episode = repository.get_episode(episode_id)
                    count = min(
                        episode.info.video_count,
                        episode.info.state_count,
                        episode.info.action_count,
                    )
                    starts = _window_starts(
                        count,
                        video_frames=args.video_frames,
                        windows=args.windows_per_episode,
                    )
                    episode_task_map[episode_id] = task_index
                    for start in starts:
                        selections.append(
                            {
                                "episode_id": episode_id,
                                "video_frame_indices": list(
                                    range(start, start + args.video_frames)
                                ),
                                "state_indices": [start],
                                "action_indices": list(
                                    range(
                                        start,
                                        start + args.video_frames - 1,
                                    )
                                ),
                                "text_indices": [0],
                            }
                        )

            pool_id = f"{child}_{args.data_version}_smoke_v1"
            rows: list[Mapping[str, Any]] = [
                {
                    "format": "motus_sample_pool_v2",
                    "pool_id": pool_id,
                    "catalog_id": repository.catalog_id,
                    "samples": len(selections),
                    "meta": {
                        "purpose": "data_factory_validation_only",
                        "model_production_pool": False,
                        "episodes_per_task": args.episodes_per_task,
                        "windows_per_episode": args.windows_per_episode,
                        "video_frames": args.video_frames,
                        "state_policy": "initial_observation_only",
                        "action_policy": "transition_commands",
                    },
                },
                *selections,
            ]
            output = output_root / f"{child}.{args.data_version}.smoke_pool.jsonl"
            if output.exists() and not args.force:
                raise FileExistsError(output)
            _atomic_jsonl(output, rows)
            completed[child] = {
                "pool": str(output),
                "pool_id": pool_id,
                "catalog_id": repository.catalog_id,
                "tasks": len(selected_by_task),
                "episodes": len(episode_task_map),
                "samples": len(selections),
                "episode_task_map": episode_task_map,
            }
            print(
                "LIBERO_SMOKE_POOL_DONE "
                + json.dumps(completed[child], sort_keys=True),
                flush=True,
            )
        finally:
            repository.close()

    report = {
        "data_version": args.data_version,
        "purpose": "data_factory_validation_only",
        "children": completed,
    }
    report_path = output_root / f"libero.{args.data_version}.smoke_pools.report.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(report_path)
    _atomic_json(report_path, report)
    print(f"LIBERO_SMOKE_POOLS_READY report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
