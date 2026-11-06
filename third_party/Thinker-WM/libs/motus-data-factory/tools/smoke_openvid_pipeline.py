#!/usr/bin/env python3
"""Validate OpenVid Episode access through DataLoader and ready queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from motus_data import (
    EpisodeDatasetRelease,
    get_episode_dataset_release,
    open_dataset,
    open_training_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog")
    parser.add_argument("--index")
    parser.add_argument("--registered-release", action="store_true")
    parser.add_argument(
        "--catalog-root",
        default="/data/jiyuan.wang/EpisodeCatalogs",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-video-frames", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _selection(
    episode_id: str,
    *,
    start: int,
    num_video_frames: int,
    frame_stride: int,
) -> dict:
    return {
        "episode_id": episode_id,
        "video_frame_indices": [
            start + frame_stride * position
            for position in range(num_video_frames + 1)
        ],
        "state_indices": [],
        "action_indices": [],
        "text_indices": [0],
    }


def main() -> int:
    args = parse_args()
    for name in (
        "batch_size",
        "num_video_frames",
        "frame_stride",
        "height",
        "width",
        "queue_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be >= 0")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.registered_release:
        if args.catalog or args.index:
            raise ValueError(
                "--registered-release cannot be combined with "
                "--catalog/--index"
            )
        release = get_episode_dataset_release("openvid")
        catalog, index = release.resolve_paths(args.catalog_root)
        release_arg = None
        catalog_root_arg = args.catalog_root
    else:
        if not (args.catalog and args.index):
            raise ValueError(
                "--catalog and --index are required unless "
                "--registered-release is used"
            )
        catalog = Path(args.catalog).resolve()
        index = Path(args.index).resolve()
        release = EpisodeDatasetRelease(
            dataset="openvid",
            release="pipeline_smoke",
            catalog_path=str(catalog),
            index_path=str(index),
            default_video_size=(int(args.height), int(args.width)),
            default_num_workers=int(args.num_workers),
            default_queue_size=int(args.queue_size),
            action_magnitude_guard_sources=None,
        )
        release_arg = release
        catalog_root_arg = None

    repository = open_dataset(
        "openvid",
        release=release_arg,
        catalog_root=catalog_root_arg,
        verify_sha256=False,
    )
    try:
        episode = None
        candidates = (
            [str(args.episode_id)]
            if args.episode_id
            else repository.iter_episode_ids()
        )
        required_span = (
            int(args.batch_size)
            * int(args.num_video_frames)
            * int(args.frame_stride)
            + 1
        )
        for episode_id in candidates:
            candidate = repository.get_episode(episode_id)
            if candidate.info.video_count < required_span:
                continue
            text = candidate.read_text([0])
            if text and text[0].text.strip():
                episode = candidate
                break
        if episode is None:
            raise RuntimeError(
                "no Episode has enough frames and non-empty text for smoke"
            )
        episode_summary = {
            "episode_id": episode.episode_id,
            "video_count": episode.info.video_count,
            "state_count": episode.info.state_count,
            "action_count": episode.info.action_count,
            "text_count": episode.info.text_count,
            "schemas": {
                name: {
                    "count": schema.count,
                    "dtype": schema.dtype,
                    "item_shape": schema.item_shape,
                    "index_semantics": schema.index_semantics,
                }
                for name, schema in episode.info.modality_schemas.items()
            },
        }
    finally:
        repository.close()

    selections = [
        _selection(
            episode_summary["episode_id"],
            start=(
                sample_index
                * int(args.num_video_frames)
                * int(args.frame_stride)
            ),
            num_video_frames=int(args.num_video_frames),
            frame_stride=int(args.frame_stride),
        )
        for sample_index in range(int(args.batch_size))
    ]
    pool_path = output_dir / "openvid.pipeline_smoke.sample_pool.jsonl"
    rows = [
        {
            "format": "motus_sample_pool_v2",
            "pool_id": "openvid_pipeline_smoke",
            "catalog_id": "openvid_1m_full_20260730",
            "samples": len(selections),
        },
        *selections,
    ]
    # Bind to the actual immutable Catalog ID, including subset smoke builds.
    metadata = json.loads(
        catalog.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    rows[0]["catalog_id"] = str(metadata["catalog_id"])
    pool_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    pipeline = open_training_data(
        "openvid",
        release=release_arg,
        catalog_root=catalog_root_arg,
        sample_pool=pool_path,
        batch_size=int(args.batch_size),
        output_adapter="motus_model_batch_v1",
        adapter_options={
            "num_video_frames": int(args.num_video_frames),
            "action_video_fps_ratio": 1,
            "video_loss_weight": 1.0,
            "action_loss_weight": 0.0,
            "state_action_policy": "absent",
            "allow_missing_state_action": True,
            "placeholder_state_action_dim": 26,
        },
        video_size=(int(args.height), int(args.width)),
        queue_enabled=True,
        num_workers=int(args.num_workers),
        queue_size=int(args.queue_size),
        shuffle=False,
        drop_last=True,
        verify_sha256=False,
    )
    try:
        batch, queue_stats = pipeline.next()
        if not torch.equal(
            batch["action_loss_weights"],
            torch.zeros(int(args.batch_size)),
        ):
            raise RuntimeError("OpenVid action loss weights are not zero")
        if not torch.equal(
            batch["initial_state"],
            torch.zeros_like(batch["initial_state"]),
        ):
            raise RuntimeError("OpenVid placeholder state is not zero")
        if not torch.equal(
            batch["action_sequence"],
            torch.zeros_like(batch["action_sequence"]),
        ):
            raise RuntimeError("OpenVid placeholder action is not zero")
        if not all(
            item["synthetic_state_action"]
            for item in batch["metadata"]
        ):
            raise RuntimeError("OpenVid synthetic state/action audit flag is absent")
        result = {
            "episode": episode_summary,
            "sample_pool": str(pool_path),
            "batch": {
                "first_frame": list(batch["first_frame"].shape),
                "video_frames": list(batch["video_frames"].shape),
                "initial_state": list(batch["initial_state"].shape),
                "action_sequence": list(batch["action_sequence"].shape),
                "source_datasets": batch["source_datasets"],
                "action_loss_weights": (
                    batch["action_loss_weights"].tolist()
                ),
                "video_loss_weights": (
                    batch["video_loss_weights"].tolist()
                ),
                "state_action_policies": batch["state_action_policies"],
                "text_prefixes": [
                    text[:80] for text in batch["t5_input_messages"]
                ],
            },
            "queue": queue_stats,
        }
        print(
            "OPENVID_PIPELINE_SMOKE_OK "
            + json.dumps(result, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
