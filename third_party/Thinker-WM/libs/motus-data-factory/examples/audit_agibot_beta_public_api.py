#!/usr/bin/env python3
"""Acceptance for the minimal Agibot Beta public training-data API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from motus_data import (
    MotusModelBatchV1Adapter,
    open_training_data,
    validate_motus_model_batch_v1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-pool", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--catalog-root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--action-video-fps-ratio", type=int, default=1)
    return parser.parse_args()


def _write_report(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"report already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    for name in (
        "batch_size",
        "batches",
        "queue_size",
        "video_height",
        "video_width",
        "num_video_frames",
        "action_video_fps_ratio",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")

    adapter = MotusModelBatchV1Adapter(
        num_video_frames=int(args.num_video_frames),
        action_video_fps_ratio=int(args.action_video_fps_ratio),
        action_loss_weight=0.0,
    )
    pipeline = open_training_data(
        "agibot_beta",
        sample_pool=args.sample_pool,
        batch_size=int(args.batch_size),
        output_adapter=adapter,
        video_size=(int(args.video_height), int(args.video_width)),
        catalog_root=args.catalog_root,
        queue_size=int(args.queue_size),
        shuffle=False,
    )
    expected_batches = list(iter(pipeline.batch_sampler))
    if int(args.batches) > len(expected_batches):
        pipeline.close()
        raise ValueError(
            f"requested batches={args.batches}, available={len(expected_batches)}"
        )

    started = time.monotonic()
    consumed = []
    try:
        print(
            "AGIBOT_BETA_PUBLIC_API_READY "
            + json.dumps(
                {
                    "release": pipeline.release.release_id,
                    "repository_indexed": pipeline.repository.indexed,
                    "catalog_id": pipeline.repository.catalog_id,
                    "pool_id": pipeline.sample_pool.pool_id,
                    "pool_selections": len(pipeline.sample_pool),
                    "batch_size": int(args.batch_size),
                    "rank": pipeline.rank,
                    "world_size": pipeline.world_size,
                    "queue_enabled": pipeline.queue_enabled,
                }
            ),
            flush=True,
        )
        pipeline.start()
        for position in range(int(args.batches)):
            batch, stats = pipeline.next()
            validate_motus_model_batch_v1(
                batch,
                expected_batch_size=int(args.batch_size),
                num_video_frames=int(args.num_video_frames),
                video_action_freq_ratio=int(args.action_video_fps_ratio),
            )
            pool_indices = expected_batches[position]
            selections = [
                pipeline.sample_pool.selections[index]
                for index in pool_indices
            ]
            expected_episode_ids = [
                str(selection["episode_id"]) for selection in selections
            ]
            expected_frame_indices = [
                list(selection["video_frame_indices"])
                for selection in selections
            ]
            if list(batch["episode_ids"]) != expected_episode_ids:
                raise RuntimeError("public API batch changed Episode IDs")
            if list(batch["frame_indices"]) != expected_frame_indices:
                raise RuntimeError("public API batch changed frame indices")
            for key in (
                "first_frame",
                "video_frames",
                "initial_state",
                "action_sequence",
            ):
                if batch[key].device.type != "cpu":
                    raise RuntimeError(f"{key} is not a CPU tensor")
                if not torch.isfinite(batch[key]).all():
                    raise RuntimeError(f"{key} contains non-finite values")
            details = {
                "position": position,
                "pool_indices": list(pool_indices),
                "episode_id": expected_episode_ids[0],
                "first_frame_shape": list(batch["first_frame"].shape),
                "video_frames_shape": list(batch["video_frames"].shape),
                "initial_state_shape": list(batch["initial_state"].shape),
                "action_sequence_shape": list(
                    batch["action_sequence"].shape
                ),
                "queue_produced": stats["ready_queue_produced_count"],
                "queue_consumed": stats["ready_queue_consumed_count"],
                "queue_wait_seconds": stats[
                    "ready_queue_get_wait_seconds"
                ],
            }
            consumed.append(details)
            print(
                "AGIBOT_BETA_PUBLIC_API_BATCH_OK "
                + json.dumps(details),
                flush=True,
            )
        queue_snapshot = pipeline.snapshot()
    finally:
        pipeline.close()

    report = {
        "status": "ok",
        "release": pipeline.release.release_id,
        "catalog_id": pipeline.repository.catalog_id,
        "repository_indexed": pipeline.repository.indexed,
        "pool_id": pipeline.sample_pool.pool_id,
        "batches_consumed": len(consumed),
        "consumed": consumed,
        "queue_snapshot": queue_snapshot,
        "wall_seconds": time.monotonic() - started,
    }
    _write_report(args.report, report)
    print(
        "AGIBOT_BETA_PUBLIC_API_OK "
        + json.dumps(
            {
                "batches_consumed": len(consumed),
                "wall_seconds": report["wall_seconds"],
                "report": str(Path(args.report)),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
