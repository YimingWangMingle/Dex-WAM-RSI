#!/usr/bin/env python3
"""Real Agibot Beta Catalog/SamplePool/DataLoader/queue acceptance."""

from __future__ import annotations

import argparse
import json
import os
import resource
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from motus_data import (
    AgibotBetaEpisodeReader,
    EpisodeRepository,
    IndexedBatchCollator,
    SamplePool,
    SamplePoolDataset,
    build_indexed_batch_adapter,
    validate_motus_model_batch_v1,
)
from motus_data.loading import ReadyBatchQueue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--index-path")
    parser.add_argument(
        "--trust-index-binding",
        action="store_true",
        help="Skip the repeated full Catalog SHA scan for an audited sidecar.",
    )
    parser.add_argument("--sample-pool", required=True)
    parser.add_argument("--report")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--queue-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--action-video-fps-ratio", type=int, default=1)
    parser.add_argument("--video-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _write_report(path: str | Path, value: dict[str, Any]) -> None:
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


def _require_positive(name: str, value: int | float) -> None:
    if isinstance(value, bool) or float(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_batch(
    batch: dict[str, Any],
    *,
    expected_indices: list[int],
    pool: SamplePool,
    batch_size: int,
    num_video_frames: int,
    action_video_fps_ratio: int,
    video_height: int,
    video_width: int,
    action_loss_weight: float,
) -> dict[str, Any]:
    validate_motus_model_batch_v1(
        batch,
        expected_batch_size=batch_size,
        num_video_frames=num_video_frames,
        video_action_freq_ratio=action_video_fps_ratio,
    )
    expected_selections = [
        pool.selections[index] for index in expected_indices
    ]
    expected_episode_ids = [
        str(selection["episode_id"]) for selection in expected_selections
    ]
    expected_frame_indices = [
        list(selection["video_frame_indices"])
        for selection in expected_selections
    ]
    if list(batch["episode_ids"]) != expected_episode_ids:
        raise RuntimeError(
            "queue batch Episode IDs differ from SamplePool batch indices"
        )
    if list(batch["frame_indices"]) != expected_frame_indices:
        raise RuntimeError(
            "queue batch video indices differ from SamplePool selections"
        )
    if len(set(expected_episode_ids)) != 1:
        raise RuntimeError("expected grouped batch contains multiple Episodes")
    if tuple(batch["first_frame"].shape) != (
        batch_size,
        3,
        video_height,
        video_width,
    ):
        raise RuntimeError(
            f"unexpected first_frame shape={tuple(batch['first_frame'].shape)}"
        )
    if tuple(batch["video_frames"].shape) != (
        batch_size,
        num_video_frames,
        3,
        video_height,
        video_width,
    ):
        raise RuntimeError(
            f"unexpected video_frames shape={tuple(batch['video_frames'].shape)}"
        )
    expected_action_count = num_video_frames * action_video_fps_ratio
    if tuple(batch["initial_state"].shape) != (batch_size, 26):
        raise RuntimeError(
            f"unexpected initial_state shape={tuple(batch['initial_state'].shape)}"
        )
    if tuple(batch["action_sequence"].shape) != (
        batch_size,
        expected_action_count,
        26,
    ):
        raise RuntimeError(
            "unexpected action_sequence shape="
            f"{tuple(batch['action_sequence'].shape)}"
        )
    tensor_keys = (
        "first_frame",
        "video_frames",
        "initial_state",
        "action_sequence",
        "video_loss_weights",
        "action_loss_weights",
    )
    for key in tensor_keys:
        value = batch[key]
        if value.device.type != "cpu":
            raise RuntimeError(f"queue batch {key} is not a CPU tensor")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"queue batch {key} contains non-finite values")
    expected_action_weights = torch.full(
        (batch_size,),
        float(action_loss_weight),
        dtype=torch.float32,
    )
    if not torch.equal(batch["action_loss_weights"], expected_action_weights):
        raise RuntimeError("action loss weights differ from adapter policy")
    if (
        batch["t5_input_messages"] is None
        or any(not str(value).strip() for value in batch["t5_input_messages"])
    ):
        raise RuntimeError("queue batch contains empty T5 text")
    if (
        batch["vlm_input_messages"] is None
        or any(not str(value).strip() for value in batch["vlm_input_messages"])
    ):
        raise RuntimeError("queue batch contains empty VLM text")
    if batch["vlm_inputs"] is not None or batch["language_embedding"] is not None:
        raise RuntimeError("model-owned preprocess placeholders must remain None")
    if any(value != "agibot_beta" for value in batch["source_datasets"]):
        raise RuntimeError("queue batch source is not agibot_beta")
    if len(batch["first_frame_pils"]) != batch_size:
        raise RuntimeError("first_frame_pils length differs from batch size")
    return {
        "episode_id": expected_episode_ids[0],
        "pool_indices": list(expected_indices),
        "video_union_count": len(
            {
                index
                for selection in expected_selections
                for index in selection["video_frame_indices"]
            }
        ),
        "first_frame_shape": list(batch["first_frame"].shape),
        "video_frames_shape": list(batch["video_frames"].shape),
        "initial_state_shape": list(batch["initial_state"].shape),
        "action_sequence_shape": list(batch["action_sequence"].shape),
        "t5_text_preview": str(batch["t5_input_messages"][0])[:160],
    }


def main() -> int:
    args = parse_args()
    for name in (
        "batch_size",
        "batches",
        "queue_size",
        "queue_timeout_seconds",
        "video_height",
        "video_width",
        "num_video_frames",
        "action_video_fps_ratio",
        "world_size",
    ):
        _require_positive(name, getattr(args, name))
    if not 0 <= int(args.rank) < int(args.world_size):
        raise ValueError("rank must be within [0, world_size)")
    if args.trust_index_binding and not args.index_path:
        raise ValueError("--trust-index-binding requires --index-path")

    event_lock = threading.Lock()
    events: list[dict[str, Any]] = []

    def observe(event: str, details: dict[str, Any]) -> None:
        with event_lock:
            events.append({"event": str(event), **dict(details)})

    started = time.monotonic()
    repository = EpisodeRepository(
        args.catalog,
        readers={
            "agibot_beta": AgibotBetaEpisodeReader(event_callback=observe)
        },
        index_path=args.index_path,
        verify_sha256=not args.trust_index_binding,
    )
    queue = None
    try:
        pool = SamplePool(args.sample_pool, repository)
        dataset = SamplePoolDataset(
            repository,
            pool,
            video_output_format="rgb_float32_chw",
            video_size=(int(args.video_height), int(args.video_width)),
            action_magnitude_guard_threshold=100.0,
            action_magnitude_guard_sources=("agibot_beta",),
        )
        sampler = dataset.create_batch_sampler(
            batch_size=int(args.batch_size),
            drop_last=True,
            shuffle=False,
            num_replicas=int(args.world_size),
            rank=int(args.rank),
            seed=int(args.seed),
        )
        expected_batches = list(iter(sampler))
        if int(args.batches) > len(expected_batches):
            raise ValueError(
                f"requested batches={args.batches}, but rank sampler has "
                f"only {len(expected_batches)} batches before restart"
            )
        adapter = build_indexed_batch_adapter(
            "motus_model_batch_v1",
            num_video_frames=int(args.num_video_frames),
            action_video_fps_ratio=int(args.action_video_fps_ratio),
            video_loss_weight=float(args.video_loss_weight),
            action_loss_weight=float(args.action_loss_weight),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=IndexedBatchCollator(adapter),
            num_workers=0,
            pin_memory=False,
        )
        queue = ReadyBatchQueue(
            loader,
            max_size=int(args.queue_size),
            get_timeout_seconds=float(args.queue_timeout_seconds),
            low_watermark=0,
            name=f"agibot_beta.indexed.rank{int(args.rank)}",
            clone_tensors=False,
        )
        print(
            "REAL_INDEXED_QUEUE_READY "
            + json.dumps(
                {
                    "catalog_id": repository.catalog_id,
                    "repository_indexed": repository.indexed,
                    "catalog_sha256_verified": not args.trust_index_binding,
                    "pool_id": pool.pool_id,
                    "pool_selections": len(pool),
                    "episode_groups": len(pool.episode_groups),
                    "rank_batches": len(expected_batches),
                    "batch_size": int(args.batch_size),
                    "queue_size": int(args.queue_size),
                    "rss_mb": _rss_mb(),
                }
            ),
            flush=True,
        )

        queue.start()
        consumed = []
        for batch_position in range(int(args.batches)):
            batch, queue_stats = queue.next()
            details = _validate_batch(
                batch,
                expected_indices=expected_batches[batch_position],
                pool=pool,
                batch_size=int(args.batch_size),
                num_video_frames=int(args.num_video_frames),
                action_video_fps_ratio=int(args.action_video_fps_ratio),
                video_height=int(args.video_height),
                video_width=int(args.video_width),
                action_loss_weight=float(args.action_loss_weight),
            )
            with event_lock:
                event_snapshot = list(events)
            event_groups = {
                name: [
                    item for item in event_snapshot if item["event"] == name
                ]
                for name in (
                    "canonical_timeline_load",
                    "video_member_extract",
                    "video_decode",
                )
            }
            for name, values in event_groups.items():
                if len(values) < batch_position + 1:
                    raise RuntimeError(
                        f"consumed batch {batch_position} has no completed {name}"
                    )
            decode_event = event_groups["video_decode"][batch_position]
            if int(decode_event["unique_count"]) != details["video_union_count"]:
                raise RuntimeError(
                    "union decode count differs from SamplePool selections: "
                    f"event={decode_event['unique_count']} "
                    f"expected={details['video_union_count']}"
                )
            details["queue"] = {
                key: value
                for key, value in queue_stats.items()
                if key.startswith("ready_queue_")
            }
            consumed.append(details)
            print(
                "REAL_INDEXED_QUEUE_BATCH_OK "
                + json.dumps(
                    {
                        "batch_position": batch_position,
                        **details,
                        "rss_mb": _rss_mb(),
                    }
                ),
                flush=True,
            )

        queue_snapshot = queue.snapshot()
        queue.close(timeout=float(args.queue_timeout_seconds))
        queue = None
        with event_lock:
            final_events = list(events)
        event_counts = Counter(item["event"] for item in final_events)
        for required in (
            "canonical_timeline_load",
            "video_member_extract",
            "video_decode",
        ):
            if event_counts[required] < int(args.batches):
                raise RuntimeError(
                    f"expected at least {args.batches} {required} events, "
                    f"observed={event_counts[required]}"
                )
        report = {
            "status": "ok",
            "catalog_id": repository.catalog_id,
            "catalog_path": str(Path(args.catalog)),
            "index_path": (
                None if args.index_path is None else str(Path(args.index_path))
            ),
            "repository_indexed": repository.indexed,
            "catalog_sha256_verified": not args.trust_index_binding,
            "sample_pool_path": str(Path(args.sample_pool)),
            "pool_id": pool.pool_id,
            "pool_selections": len(pool),
            "batch_size": int(args.batch_size),
            "batches_consumed": len(consumed),
            "consumed_batches": consumed,
            "event_counts": dict(event_counts),
            "queue_snapshot": queue_snapshot,
            "wall_seconds": time.monotonic() - started,
            "max_rss_mb": _rss_mb(),
        }
        if args.report:
            _write_report(args.report, report)
        print(
            "REAL_AGIBOT_BETA_INDEXED_QUEUE_OK "
            + json.dumps(
                {
                    "batches_consumed": len(consumed),
                    "event_counts": dict(event_counts),
                    "wall_seconds": report["wall_seconds"],
                    "max_rss_mb": report["max_rss_mb"],
                    "report": args.report,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if queue is not None:
            queue.close(timeout=float(args.queue_timeout_seconds))
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
