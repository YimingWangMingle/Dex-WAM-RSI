#!/usr/bin/env python3
"""Build deterministic engineering-only SamplePools for all ten releases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from motus_data import open_dataset
from motus_data.index_contracts import (
    MOTUS_SAMPLE_POOL_HEADER_V2,
    validate_sample_pool_header_v2,
    validate_sample_selection_v1,
)


DATASETS = (
    "agibot",
    "agibot_beta",
    "robotwin_aloha",
    "robotwin_ur5",
    "robotwin_arx",
    "robotwin_piper",
    "robotmind_tianyi",
    "robotmind_arkmobile",
    "robotmind_ark",
    "robotmind_tienkung",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        default="/data/jiyuan.wang/EpisodeCatalogs",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "/data/jiyuan.wang/SamplePools/_20260727/"
            "ten_source_smoke_v1"
        ),
    )
    parser.add_argument("--samples-per-child", type=int, default=8)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _ratio(dataset: str, episode: Any) -> int:
    if dataset == "agibot":
        ratio = int(episode.info.metadata["action_video_fps_ratio"])
        if ratio != 6:
            raise RuntimeError(
                f"unexpected Agibot action/video ratio={ratio}"
            )
        return ratio
    return 1


def _text_indices(episode: Any, start: int) -> list[int]:
    items = episode.text.items
    if episode.info.source_dataset in {"agibot", "agibot_beta"}:
        context = [
            item.index for item in items if item.kind == "episode_context"
        ]
        segments = [
            item.index
            for item in items
            if item.kind == "action_segment"
            and item.frame_range is not None
            and item.frame_range[0] <= start < item.frame_range[1]
        ]
        if len(context) != 1 or len(segments) != 1:
            return []
        return [context[0], segments[0]]

    english = [item.index for item in items if item.kind == "instruction_en"]
    if english:
        return [english[0]]
    return [items[0].index] if items else []


def _start_bounds(
    dataset: str,
    episode: Any,
    *,
    num_video_frames: int,
    ratio: int,
) -> tuple[int, int]:
    minimum = 0
    if dataset == "agibot":
        effective = episode.info.metadata["effective_video_frames"]
        minimum = int(effective[0])
        maximum = int(effective[1]) - num_video_frames
    else:
        maximum = episode.info.video_count - num_video_frames - 1

    maximum = min(
        maximum,
        (episode.info.action_count - 1) // ratio - num_video_frames,
        (episode.info.state_count - 1) // ratio,
    )
    return minimum, maximum


def _selection(
    episode: Any,
    *,
    start: int,
    num_video_frames: int,
    ratio: int,
) -> dict[str, Any] | None:
    text_indices = _text_indices(episode, start)
    if not text_indices:
        return None
    video_indices = list(range(start, start + num_video_frames + 1))
    state_start = start * ratio
    action_start = state_start + 1
    action_stop = video_indices[-1] * ratio + 1
    selection = {
        "episode_id": episode.episode_id,
        "video_frame_indices": video_indices,
        "state_indices": [state_start],
        "action_indices": list(range(action_start, action_stop)),
        "text_indices": text_indices,
    }
    validate_sample_selection_v1(selection)
    episode._validated_selection_indices(selection)
    return selection


def _selections(
    repository: Any,
    dataset: str,
    *,
    samples: int,
    num_video_frames: int,
) -> tuple[list[dict[str, Any]], int]:
    result = []
    selected_ratio = None
    for episode_id in repository.iter_episode_ids():
        episode = repository.get_episode(episode_id)
        ratio = _ratio(dataset, episode)
        minimum, maximum = _start_bounds(
            dataset,
            episode,
            num_video_frames=num_video_frames,
            ratio=ratio,
        )
        for start in range(minimum, maximum + 1):
            selection = _selection(
                episode,
                start=start,
                num_video_frames=num_video_frames,
                ratio=ratio,
            )
            if selection is None:
                continue
            result.append(selection)
            selected_ratio = ratio
            if len(result) == samples:
                return result, int(selected_ratio)
    raise RuntimeError(
        f"{dataset} produced only {len(result)}/{samples} valid smoke selections"
    )


def _write_pool(
    output: Path,
    *,
    dataset: str,
    catalog_id: str,
    catalog_episodes: int,
    selections: list[dict[str, Any]],
    ratio: int,
    num_video_frames: int,
    force: bool,
) -> None:
    header = {
        "format": MOTUS_SAMPLE_POOL_HEADER_V2,
        "pool_id": f"{dataset}_engineering_smoke_20260727_v1",
        "catalog_id": catalog_id,
        "samples": len(selections),
        "meta": {
            "purpose": "engineering_smoke_only",
            "production_sampling_policy": False,
            "catalog_episodes": int(catalog_episodes),
            "num_video_frames": int(num_video_frames),
            "action_video_fps_ratio": int(ratio),
        },
    }
    validate_sample_pool_header_v2(header)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"output exists; use --force: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in (header, *selections):
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            output.unlink(missing_ok=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    if int(args.samples_per_child) <= 0:
        raise ValueError("--samples-per-child must be positive")
    if int(args.num_video_frames) <= 0:
        raise ValueError("--num-video-frames must be positive")

    output_root = Path(args.output_root)
    for dataset in DATASETS:
        repository = open_dataset(
            dataset,
            catalog_root=args.catalog_root,
            verify_sha256=False,
        )
        try:
            selections, ratio = _selections(
                repository,
                dataset,
                samples=int(args.samples_per_child),
                num_video_frames=int(args.num_video_frames),
            )
            output = output_root / f"{dataset}.engineering_smoke.sample_pool.jsonl"
            _write_pool(
                output,
                dataset=dataset,
                catalog_id=repository.catalog_id,
                catalog_episodes=len(repository),
                selections=selections,
                ratio=ratio,
                num_video_frames=int(args.num_video_frames),
                force=bool(args.force),
            )
            print(
                f"WROTE dataset={dataset} catalog_id={repository.catalog_id} "
                f"catalog_episodes={len(repository)} samples={len(selections)} "
                f"ratio={ratio} output={output}"
            )
        finally:
            repository.close()
    print("TEN_SOURCE_SMOKE_SAMPLE_POOLS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
