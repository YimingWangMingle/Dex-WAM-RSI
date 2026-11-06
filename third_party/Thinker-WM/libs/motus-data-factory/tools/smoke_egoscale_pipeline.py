#!/usr/bin/env python3
"""Exercise real EGOSCALE Episode, SamplePool, batch, and queue reads."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data.contracts import (  # noqa: E402
    validate_motus_native_multimodal_batch_v1,
)
from motus_data.indexed_pipeline import open_training_data  # noqa: E402
from motus_data.releases import EpisodeDatasetRelease  # noqa: E402
from motus_data.repository import EpisodeRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--video-height", type=int, default=120)
    parser.add_argument("--video-width", type=int, default=160)
    parser.add_argument("--probe-episode-id")
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument(
        "--camera-mode",
        choices=("base_0", "base_1", "stereo_horizontal"),
        default="base_0",
    )
    parser.add_argument(
        "--pose-profile",
        choices=("camera_pose_with_validity", "world_pose_with_validity"),
        default="camera_pose_with_validity",
    )
    return parser.parse_args()


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


def _catalog_rows(path: Path, count: int) -> list[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= count:
                break
    if len(rows) < count:
        raise RuntimeError(
            f"Catalog has only {len(rows)} rows; smoke requires {count}"
        )
    return rows


def _build_pool(
    catalog: Path,
    output: Path,
    *,
    catalog_id: str,
    samples: int,
    samples_per_episode: int,
    num_video_frames: int,
) -> list[Mapping[str, Any]]:
    episode_rows = _catalog_rows(
        catalog,
        max(1, (samples + samples_per_episode - 1) // samples_per_episode),
    )
    selections = []
    video_count = int(num_video_frames) + 1
    for sample_index in range(samples):
        episode = episode_rows[
            (sample_index // samples_per_episode) % len(episode_rows)
        ]
        count = int(episode["video"]["count"])
        if count < video_count:
            raise RuntimeError(
                f"Episode {episode['episode_id']} is too short for smoke"
            )
        max_start = count - video_count
        start = min((sample_index % samples_per_episode) * 5, max_start)
        text_index = next(
            (
                int(item["index"])
                for item in episode["texts"]
                if item["kind"] == "task_description_en"
            ),
            0,
        )
        selections.append(
            {
                "episode_id": episode["episode_id"],
                "video_frame_indices": list(
                    range(start, start + video_count)
                ),
                "state_indices": [start],
                "action_indices": list(
                    range(start, start + int(num_video_frames))
                ),
                "text_indices": [text_index],
            }
        )
    header = {
        "format": "motus_sample_pool_v2",
        "pool_id": "egoscale_real_smoke_pool",
        "catalog_id": str(catalog_id),
        "samples": len(selections),
    }
    with output.open("w", encoding="utf-8") as handle:
        for row in [header, *selections]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return selections


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        raise ValueError("batch-size and batches must be positive")
    catalog = Path(args.catalog).resolve()
    index = Path(args.index).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = catalog.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pool = None
    selections = []
    if not args.direct_only:
        sample_count = args.batch_size * max(args.batches, 2)
        pool = output_dir / "egoscale_real_smoke.sample_pool.jsonl"
        selections = _build_pool(
            catalog,
            pool,
            catalog_id=metadata["catalog_id"],
            samples=sample_count,
            samples_per_episode=args.batch_size,
            num_video_frames=args.num_video_frames,
        )

    direct_repository = EpisodeRepository(
        catalog,
        index_path=index,
        verify_sha256=False,
        reader_options={
            "egoscale": {
                "camera_mode": args.camera_mode,
                "pose_profile": args.pose_profile,
                "resize_mode": "pad",
            }
        },
    )
    try:
        probe_episode_id = args.probe_episode_id
        if probe_episode_id is None:
            probe_episode_id = (
                selections[0]["episode_id"]
                if selections
                else _catalog_rows(catalog, 1)[0]["episode_id"]
            )
        episode = direct_repository.get_episode(probe_episode_id)
        probe_indices = [0, episode.info.video_count // 2, episode.info.video_count - 1]
        video = episode.read_video(
            probe_indices,
            output_format="rgb_float32_chw",
            video_size=(args.video_height, args.video_width),
        )
        state = episode.read_state(probe_indices)
        action = episode.read_action(probe_indices)
        raw = episode.read_source_modalities(
            {
                "head/valid": probe_indices,
                "left_hand/mano/valid": probe_indices,
                "right_hand/mano/valid": probe_indices,
                "left_hand/tracks": probe_indices,
            }
        )
        direct = {
            "episode_id": episode.episode_id,
            "video_shape": list(video.shape),
            "state_shape": list(state.shape),
            "action_shape": list(action.shape),
            "state_finite": bool(torch.isfinite(state).all()),
            "action_finite": bool(torch.isfinite(action).all()),
            "raw_shapes": {name: list(value.shape) for name, value in raw.items()},
            "validity": {
                name: value.tolist()
                for name, value in raw.items()
                if name.endswith("/valid") or name == "head/valid"
            },
        }
    finally:
        direct_repository.close()

    batches = []
    if not args.direct_only:
        release = EpisodeDatasetRelease(
            dataset="egoscale_first500h_smoke",
            release="real_smoke_v1",
            catalog_path=str(catalog),
            index_path=str(index),
            default_video_size=(args.video_height, args.video_width),
            default_num_workers=0,
            default_pin_memory=False,
            default_queue_size=2,
            action_magnitude_guard_threshold=None,
            reader_options={
                "egoscale": {
                    "camera_mode": args.camera_mode,
                    "pose_profile": args.pose_profile,
                    "resize_mode": "pad",
                }
            },
        )
        pipeline = open_training_data(
            release.dataset,
            release=release,
            sample_pool=pool,
            batch_size=args.batch_size,
            output_adapter="motus_native_multimodal_batch_v1",
            adapter_options={
                "num_video_frames": args.num_video_frames,
                "action_video_fps_ratio": 1,
            },
            queue_enabled=True,
            queue_size=2,
            queue_timeout_seconds=300.0,
            shuffle=False,
            drop_last=True,
        )
        try:
            for _ in range(args.batches):
                batch, stats = pipeline.next()
                validate_motus_native_multimodal_batch_v1(
                    batch,
                    expected_batch_size=args.batch_size,
                    num_video_frames=args.num_video_frames,
                    video_action_freq_ratio=1,
                    expected_state_dimension=24,
                    expected_action_dimension=24,
                )
                batches.append(
                    {
                        "first_frame_shape": list(batch["first_frame"].shape),
                        "video_frames_shape": list(batch["video_frames"].shape),
                        "state_shape": list(batch["initial_state"].shape),
                        "action_shape": list(batch["action_sequence"].shape),
                        "queue_stats": dict(stats),
                    }
                )
        finally:
            pipeline.close()

    report = {
        "catalog_id": metadata["catalog_id"],
        "catalog_episodes": metadata["episodes"],
        "sample_pool": None if pool is None else str(pool),
        "camera_mode": args.camera_mode,
        "pose_profile": args.pose_profile,
        "direct_episode_access": direct,
        "queue_batches": batches,
        "passed": True,
    }
    report_path = output_dir / "egoscale_real_smoke.report.json"
    _atomic_json(report_path, report)
    print("EGOSCALE_PIPELINE_SMOKE_DONE " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
