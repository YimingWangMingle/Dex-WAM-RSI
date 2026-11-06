#!/usr/bin/env python3
"""Audit ten-source Catalog/Pool binding and real indexed queue consumption."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from motus_data import (
    MOTUS_MODEL_BATCH_V1,
    IndexedTrainingSource,
    open_mixed_training_data,
    open_training_data,
    validate_motus_model_batch_v1,
)


DATASETS = (
    ("agibot", "agibot", 6),
    ("agibot_beta", "agibot_beta", 1),
    ("robotwin_aloha", "robotwin", 1),
    ("robotwin_ur5", "robotwin", 1),
    ("robotwin_arx", "robotwin", 1),
    ("robotwin_piper", "robotwin", 1),
    ("robotmind_tianyi", "robotmind", 1),
    ("robotmind_arkmobile", "robotmind", 1),
    ("robotmind_ark", "robotmind", 1),
    ("robotmind_tienkung", "robotmind", 1),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        default="/data/jiyuan.wang/EpisodeCatalogs",
    )
    parser.add_argument(
        "--sample-pool-root",
        default=(
            "/data/jiyuan.wang/SamplePools/_20260727/"
            "ten_source_smoke_v1"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--weighted-batches", type=int, default=4)
    parser.add_argument("--quota-batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    return parser.parse_args()


def _distributed_values(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.rank
    world_size = args.world_size
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return int(rank), int(world_size)


def _pool_path(root: Path, dataset: str) -> Path:
    return root / f"{dataset}.engineering_smoke.sample_pool.jsonl"


def _pool_header(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _sources(
    catalog_root: Path,
    pool_root: Path,
) -> list[IndexedTrainingSource]:
    sources = []
    for dataset, family, ratio in DATASETS:
        pool = _pool_path(pool_root, dataset)
        header = _pool_header(pool)
        weight = float(header["meta"]["catalog_episodes"])
        sources.append(
            IndexedTrainingSource(
                name=dataset,
                dataset=dataset,
                family=family,
                sample_pool=pool,
                catalog_root=catalog_root,
                weight=weight,
                output_adapter=MOTUS_MODEL_BATCH_V1,
                adapter_options={
                    "num_video_frames": 48,
                    "action_video_fps_ratio": ratio,
                    "video_loss_weight": 1.0,
                    "action_loss_weight": 0.0,
                    "state_action_policy": "real",
                },
            )
        )
    return sources


def _validate_batch(
    batch: dict[str, Any],
    *,
    dataset: str,
    family: str,
    ratio: int,
    batch_size: int,
    size: tuple[int, int],
) -> None:
    validate_motus_model_batch_v1(
        batch,
        expected_batch_size=batch_size,
        num_video_frames=48,
        video_action_freq_ratio=ratio,
    )
    expected_first = (batch_size, 3, *size)
    expected_video = (batch_size, 48, 3, *size)
    if tuple(batch["first_frame"].shape) != expected_first:
        raise RuntimeError(
            f"{dataset} first_frame={tuple(batch['first_frame'].shape)} "
            f"expected={expected_first}"
        )
    if tuple(batch["video_frames"].shape) != expected_video:
        raise RuntimeError(
            f"{dataset} video_frames={tuple(batch['video_frames'].shape)} "
            f"expected={expected_video}"
        )
    if set(batch["source_datasets"]) != {family}:
        raise RuntimeError(
            f"{dataset} source_datasets={batch['source_datasets']}"
        )
    if set(batch["source_children"]) != {dataset}:
        raise RuntimeError(
            f"{dataset} source_children={batch['source_children']}"
        )
    if not torch.equal(
        batch["action_loss_weights"],
        torch.zeros(batch_size, dtype=torch.float32),
    ):
        raise RuntimeError(f"{dataset} action loss is not disabled")
    if not torch.equal(
        batch["video_loss_weights"],
        torch.ones(batch_size, dtype=torch.float32),
    ):
        raise RuntimeError(f"{dataset} video loss is not enabled")


def _metadata_gate(
    sources: list[IndexedTrainingSource],
    *,
    batch_size: int,
    quota_batches: int,
    seed: int,
    rank: int,
    world_size: int,
) -> None:
    pipeline = open_mixed_training_data(
        sources,
        batch_size=batch_size,
        batches_per_epoch=quota_batches,
        video_size=(64, 64),
        seed=seed,
        rank=rank,
        world_size=world_size,
        queue_enabled=False,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    try:
        quotas = dict(
            zip(
                (source.name for source in pipeline.sources),
                pipeline.stream.source_quotas(),
            )
        )
        if sum(quotas.values()) != quota_batches:
            raise RuntimeError("weighted quotas do not sum to the epoch size")
        if any(value <= 0 for value in quotas.values()):
            raise RuntimeError(
                f"quota epoch is too short to cover all children: {quotas}"
            )
        print(
            "WEIGHTED_QUOTAS "
            + json.dumps(quotas, sort_keys=True, separators=(",", ":"))
        )
        print(
            "NORMALIZED_WEIGHTS "
            + json.dumps(
                pipeline.snapshot()["mix_normalized_weights"],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    finally:
        pipeline.close()
    print("TEN_SOURCE_METADATA_BINDING_OK")


def _child_coverage_gate(
    *,
    catalog_root: Path,
    pool_root: Path,
    batch_size: int,
    size: tuple[int, int],
    queue_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> None:
    for dataset, family, ratio in DATASETS:
        pipeline = open_training_data(
            dataset,
            sample_pool=_pool_path(pool_root, dataset),
            batch_size=batch_size,
            output_adapter=MOTUS_MODEL_BATCH_V1,
            adapter_options={
                "num_video_frames": 48,
                "action_video_fps_ratio": ratio,
                "video_loss_weight": 1.0,
                "action_loss_weight": 0.0,
                "state_action_policy": "real",
            },
            video_size=size,
            seed=seed,
            rank=rank,
            world_size=world_size,
            queue_enabled=True,
            catalog_root=catalog_root,
            num_workers=0,
            pin_memory=False,
            queue_size=queue_size,
            queue_clone_tensors=False,
            shuffle=True,
            drop_last=True,
        )
        try:
            with pipeline:
                batch, stats = pipeline.next()
            _validate_batch(
                batch,
                dataset=dataset,
                family=family,
                ratio=ratio,
                batch_size=batch_size,
                size=size,
            )
            message = str(batch["t5_input_messages"][0]).replace("\n", " ")
            print(
                f"CHILD_QUEUE_OK dataset={dataset} family={family} ratio={ratio} "
                f"episode={batch['episode_ids'][0]} "
                f"video={tuple(batch['video_frames'].shape)} "
                f"action={tuple(batch['action_sequence'].shape)} "
                f"produced={stats['ready_queue_produced_count']} "
                f"consumed={stats['ready_queue_consumed_count']} "
                f"text={message[:100]!r}"
            )
        finally:
            pipeline.close()


def _weighted_queue_gate(
    sources: list[IndexedTrainingSource],
    *,
    batch_size: int,
    size: tuple[int, int],
    queue_size: int,
    weighted_batches: int,
    quota_batches: int,
    seed: int,
    rank: int,
    world_size: int,
) -> None:
    expected = {name: (family, ratio) for name, family, ratio in DATASETS}
    pipeline = open_mixed_training_data(
        sources,
        batch_size=batch_size,
        batches_per_epoch=quota_batches,
        video_size=size,
        seed=seed,
        rank=rank,
        world_size=world_size,
        queue_enabled=True,
        num_workers=0,
        pin_memory=False,
        queue_size=queue_size,
        queue_clone_tensors=False,
        queue_timeout_seconds=600.0,
        shuffle=True,
        drop_last=True,
    )
    observed = []
    try:
        with pipeline:
            for step in range(weighted_batches):
                batch, stats = pipeline.next()
                dataset = str(stats["mix_source_name"])
                family, ratio = expected[dataset]
                _validate_batch(
                    batch,
                    dataset=dataset,
                    family=family,
                    ratio=ratio,
                    batch_size=batch_size,
                    size=size,
                )
                observed.append(dataset)
                print(
                    f"MIX_QUEUE_BATCH_OK step={step} dataset={dataset} "
                    f"ratio={ratio} episode={batch['episode_ids'][0]}"
                )
        snapshot = pipeline.snapshot()
    finally:
        pipeline.close()
    print(
        "MIX_OBSERVED "
        + json.dumps(observed, separators=(",", ":"))
    )
    print(
        "MIX_QUEUE_STATS "
        + json.dumps(
            {
                "produced": snapshot["ready_queue_produced_count"],
                "consumed": snapshot["ready_queue_consumed_count"],
                "current_size": snapshot["ready_queue_current_size"],
                "max_size": snapshot["ready_queue_max_size"],
                "consumed_by_source": snapshot["mix_consumed_by_source"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    print("TEN_SOURCE_WEIGHTED_QUEUE_OK")


def main() -> int:
    args = _parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.weighted_batches) <= 0:
        raise ValueError("--weighted-batches must be positive")
    rank, world_size = _distributed_values(args)
    catalog_root = Path(args.catalog_root)
    pool_root = Path(args.sample_pool_root)
    size = (int(args.height), int(args.width))
    sources = _sources(catalog_root, pool_root)

    _metadata_gate(
        sources,
        batch_size=int(args.batch_size),
        quota_batches=int(args.quota_batches),
        seed=int(args.seed),
        rank=rank,
        world_size=world_size,
    )
    if args.metadata_only:
        return 0
    _child_coverage_gate(
        catalog_root=catalog_root,
        pool_root=pool_root,
        batch_size=int(args.batch_size),
        size=size,
        queue_size=int(args.queue_size),
        seed=int(args.seed),
        rank=rank,
        world_size=world_size,
    )
    _weighted_queue_gate(
        sources,
        batch_size=int(args.batch_size),
        size=size,
        queue_size=int(args.queue_size),
        weighted_batches=int(args.weighted_batches),
        quota_batches=int(args.quota_batches),
        seed=int(args.seed),
        rank=rank,
        world_size=world_size,
    )
    print("TEN_SOURCE_INDEXED_MIX_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
