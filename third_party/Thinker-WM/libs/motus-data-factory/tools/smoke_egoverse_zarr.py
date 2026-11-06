#!/usr/bin/env python3
"""Run a bounded real-payload EgoVerse Catalog, reader, and queue smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from motus_data import (
    MOTUS_MODEL_BATCH_V1,
    EpisodeDatasetRelease,
    EpisodeRepository,
    open_training_data,
)
from motus_data.catalog.egoverse import build_egoverse_episode_catalog
from motus_data.catalog.sqlite_index import build_catalog_offset_index
from motus_data.egoverse import get_egoverse_profile
from motus_data.index_contracts import MOTUS_SAMPLE_POOL_HEADER_V2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--video-height", type=int, default=64)
    parser.add_argument("--video-width", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_pool(
    path: Path,
    *,
    catalog_id: str,
    episode: Any,
    num_video_frames: int,
    text_index: int,
    force: bool,
) -> None:
    starts = (0, 1)
    selections = []
    for start in starts:
        selections.append(
            {
                "episode_id": episode.episode_id,
                "video_frame_indices": list(
                    range(start, start + num_video_frames + 1)
                ),
                "state_indices": [start],
                "action_indices": list(
                    range(start + 1, start + num_video_frames + 1)
                ),
                "text_indices": [text_index],
            }
        )
    header = {
        "format": MOTUS_SAMPLE_POOL_HEADER_V2,
        "pool_id": f"{episode.info.source_child}_real_payload_smoke",
        "catalog_id": catalog_id,
        "samples": len(selections),
        "meta": {
            "purpose": "engineering_smoke_only",
            "production_sampling_policy": False,
        },
    }
    if path.exists() and not force:
        raise FileExistsError(f"SamplePool exists; use --force: {path}")
    rows = [header, *selections]
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _usable_episode(
    repository: EpisodeRepository,
    *,
    required_frames: int,
) -> tuple[Any, int] | None:
    for episode_id in repository.iter_episode_ids():
        episode = repository.get_episode(episode_id)
        if (
            episode.info.video_count < required_frames
            or episode.info.state_count < required_frames
            or episode.info.action_count < required_frames
        ):
            continue
        for item in episode.text.items:
            if item.text.strip():
                return episode, item.index
    return None


def main() -> int:
    args = _parse_args()
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if args.num_video_frames <= 0:
        raise ValueError("--num-video-frames must be positive")
    if args.video_height <= 0 or args.video_width <= 0:
        raise ValueError("video dimensions must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = output_dir / f"{args.profile}.smoke.episode_catalog.jsonl"
    catalog_id = f"egoverse_{args.profile}_real_payload_smoke"
    metadata = build_egoverse_episode_catalog(
        data_root=args.data_root,
        profile=args.profile,
        output_path=catalog,
        catalog_id=catalog_id,
        data_version="real_payload_smoke",
        max_episodes=args.max_episodes,
        force=args.force,
    )
    profile = get_egoverse_profile(args.profile)
    build = metadata["meta"]["build"]
    contract = build["profile_contract"]
    if contract["action_source_signal"] != profile.action_source_signal:
        raise RuntimeError("Catalog action source signal does not match profile")
    if (
        bool(contract["action_is_control_command"])
        != profile.action_is_control_command
    ):
        raise RuntimeError("Catalog action command flag does not match profile")
    if int(build["quality"]["episodes"]) != int(metadata["episodes"]):
        raise RuntimeError("Catalog quality Episode count mismatch")
    index_result = build_catalog_offset_index(
        catalog,
        force=args.force,
    )
    index = Path(index_result["index_path"])

    repository = EpisodeRepository(
        catalog,
        index_path=index,
        verify_sha256=False,
    )
    try:
        first_id = next(repository.iter_episode_ids())
        first = repository.get_episode(first_id)
        action_schema = first.info.modality_schemas["action"]
        if (
            action_schema.metadata.get("source_signal")
            != profile.action_source_signal
        ):
            raise RuntimeError("reader action source signal mismatch")
        if (
            bool(action_schema.metadata.get("is_control_command"))
            != profile.action_is_control_command
        ):
            raise RuntimeError("reader action command flag mismatch")
        last = first.info.video_count - 1
        payload = first.read_selection(
            {
                "episode_id": first.episode_id,
                "video_frame_indices": [0, last // 2, last],
                "state_indices": [0, last],
                "action_indices": [last // 3, last],
                "text_indices": list(range(first.info.text_count)),
            },
            video_output_format="rgb_float32_chw",
            video_size=(args.video_height, args.video_width),
        )
        if not torch.isfinite(payload.video).all():
            raise RuntimeError("decoded EgoVerse video is non-finite")
        if not torch.isfinite(payload.state).all():
            raise RuntimeError("decoded EgoVerse state is non-finite")
        if not torch.isfinite(payload.action).all():
            raise RuntimeError("decoded EgoVerse action is non-finite")
        print(
            "EGOVERSE_READER_OK "
            f"profile={args.profile} episode={first.episode_id} "
            f"valid_frames={first.info.video_count} "
            f"video={tuple(payload.video.shape)} "
            f"state={tuple(payload.state.shape)} "
            f"action={tuple(payload.action.shape)} "
            f"text_items={first.info.text_count} "
            f"action_source={profile.action_source_signal} "
            f"action_is_control_command={profile.action_is_control_command}",
            flush=True,
        )

        selected = _usable_episode(
            repository,
            required_frames=args.num_video_frames + 2,
        )
        if selected is None:
            print(
                "EGOVERSE_QUEUE_SKIPPED_NO_USABLE_TEXT "
                f"profile={args.profile} episodes={len(repository)}",
                flush=True,
            )
            return 0
        episode, text_index = selected
    finally:
        repository.close()

    pool = output_dir / f"{args.profile}.smoke.sample_pool.jsonl"
    _write_pool(
        pool,
        catalog_id=catalog_id,
        episode=episode,
        num_video_frames=args.num_video_frames,
        text_index=text_index,
        force=args.force,
    )
    release = EpisodeDatasetRelease(
        dataset=f"egoverse_{args.profile}",
        release="real_payload_smoke",
        catalog_path=str(catalog),
        index_path=str(index),
        default_video_size=(args.video_height, args.video_width),
        default_queue_size=1,
        action_magnitude_guard_threshold=None,
    )
    pipeline = open_training_data(
        release.dataset,
        release=release,
        catalog_path=catalog,
        index_path=index,
        sample_pool=pool,
        batch_size=2,
        output_adapter=MOTUS_MODEL_BATCH_V1,
        adapter_options={
            "num_video_frames": args.num_video_frames,
            "action_video_fps_ratio": 1,
            "video_loss_weight": 1.0,
            "action_loss_weight": 0.0,
            "state_action_policy": "real",
        },
        video_size=(args.video_height, args.video_width),
        num_workers=0,
        pin_memory=False,
        queue_enabled=True,
        queue_size=1,
        queue_clone_tensors=False,
        shuffle=False,
        drop_last=True,
    )
    try:
        with pipeline:
            batch, stats = pipeline.next()
        print(
            "EGOVERSE_QUEUE_OK "
            f"profile={args.profile} "
            f"batch={len(batch['episode_ids'])} "
            f"video={tuple(batch['video_frames'].shape)} "
            f"state={tuple(batch['initial_state'].shape)} "
            f"action={tuple(batch['action_sequence'].shape)} "
            f"produced={stats['ready_queue_produced_count']} "
            f"consumed={stats['ready_queue_consumed_count']}",
            flush=True,
        )
    finally:
        pipeline.close()
    print(
        "EGOVERSE_REAL_PAYLOAD_SMOKE_OK "
        f"profile={args.profile} episodes={metadata['episodes']} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
