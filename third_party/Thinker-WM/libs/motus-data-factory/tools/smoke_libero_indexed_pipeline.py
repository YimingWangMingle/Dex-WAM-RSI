#!/usr/bin/env python3
"""Validate four-child LIBERO Dataset/DataLoader/queue consumption."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data import (
    EpisodeDatasetRelease,
    IndexedTrainingSource,
    MOTUS_NATIVE_MULTIMODAL_BATCH_V1,
    open_mixed_training_data,
    validate_motus_native_multimodal_batch_v1,
)
from motus_data.catalog.libero import LIBERO_CHILDREN


def _resource_snapshot() -> dict[str, float]:
    result = {
        "process_max_rss_mib": (
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        )
    }
    shm = Path("/dev/shm")
    if shm.is_dir():
        usage = shutil.disk_usage(shm)
        result.update(
            {
                "shm_used_mib": float(usage.used) / (1024.0**2),
                "shm_total_mib": float(usage.total) / (1024.0**2),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--data-version", default="libero_parquet_v1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches-per-child", type=int, default=2)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--camera-mode",
        choices=("main", "wrist", "main_wrist_horizontal"),
        default="main",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("raw", "zscore"),
        default="zscore",
    )
    parser.add_argument("--statistics-path")
    return parser.parse_args()


def _source(
    root: Path,
    pool_root: Path,
    child: str,
    version: str,
    *,
    camera_mode: str,
    normalization_mode: str,
    statistics_path: str | None,
    weight: float,
) -> IndexedTrainingSource:
    catalog = root / f"{child}.{version}.episode_catalog.jsonl"
    index = root / f"{child}.{version}.episode_catalog.index.sqlite"
    pool = pool_root / f"{child}.{version}.smoke_pool.jsonl"
    release = EpisodeDatasetRelease(
        dataset=child,
        release=f"{version}_smoke",
        catalog_path=str(catalog),
        index_path=str(index),
        default_num_workers=0,
        default_queue_size=2,
        action_magnitude_guard_threshold=None,
        reader_options={
            "libero": {
                "camera_mode": camera_mode,
                "resize_mode": "pad",
                "normalization_mode": normalization_mode,
                **(
                    {"statistics_path": statistics_path}
                    if statistics_path is not None
                    else {}
                ),
            }
        },
    )
    return IndexedTrainingSource(
        name=child,
        dataset=child,
        family="libero",
        sample_pool=pool,
        release=release,
        weight=float(weight),
        verify_sha256=False,
    )


def _validate_batch(
    batch: dict[str, object],
    *,
    child: str,
    batch_size: int,
    num_video_frames: int,
    size: tuple[int, int],
) -> dict[str, object]:
    validate_motus_native_multimodal_batch_v1(
        batch,
        expected_batch_size=batch_size,
        num_video_frames=num_video_frames,
        video_action_freq_ratio=1,
        expected_state_dimension=8,
        expected_action_dimension=7,
    )
    if set(batch["source_datasets"]) != {"libero"}:
        raise RuntimeError("LIBERO source family changed")
    if set(batch["source_children"]) != {child}:
        raise RuntimeError("LIBERO source child changed")
    episode_ids = set(batch["episode_ids"])
    if len(episode_ids) != 1:
        raise RuntimeError("LIBERO batch crossed Episode boundaries")
    if tuple(batch["first_frame"].shape) != (batch_size, 3, *size):
        raise RuntimeError("LIBERO first-frame shape changed")
    if tuple(batch["video_frames"].shape) != (
        batch_size,
        num_video_frames,
        3,
        *size,
    ):
        raise RuntimeError("LIBERO future-video shape changed")
    if tuple(batch["initial_state"].shape) != (batch_size, 8):
        raise RuntimeError("LIBERO state shape changed")
    if tuple(batch["action_sequence"].shape) != (
        batch_size,
        num_video_frames,
        7,
    ):
        raise RuntimeError("LIBERO action shape changed")
    if not all(str(value).strip() for value in batch["t5_input_messages"]):
        raise RuntimeError("LIBERO batch text is empty")
    return {
        "child": child,
        "episode_id": next(iter(episode_ids)),
        "batch_size": batch_size,
        "video_shape": list(batch["video_frames"].shape),
        "state_shape": list(batch["initial_state"].shape),
        "action_shape": list(batch["action_sequence"].shape),
        "normalization_modes": batch["normalization_modes"],
        "text": batch["t5_input_messages"][0],
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.batches_per_child <= 0:
        raise ValueError("batch sizes must be positive")
    size = (int(args.height), int(args.width))
    catalog_root = Path(args.catalog_root)
    pool_root = Path(args.pool_root)

    real_weights = {
        child: spec.episode_count for child, spec in LIBERO_CHILDREN.items()
    }
    weighted_sources = [
        _source(
            catalog_root,
            pool_root,
            child,
            args.data_version,
            camera_mode=args.camera_mode,
            normalization_mode=args.normalization_mode,
            statistics_path=args.statistics_path,
            weight=real_weights[child],
        )
        for child in LIBERO_CHILDREN
    ]
    metadata_pipeline = open_mixed_training_data(
        weighted_sources,
        batch_size=args.batch_size,
        batches_per_epoch=64,
        output_adapter=MOTUS_NATIVE_MULTIMODAL_BATCH_V1,
        adapter_options={
            "num_video_frames": args.num_video_frames,
            "action_video_fps_ratio": 1,
        },
        video_size=size,
        seed=args.seed,
        rank=0,
        world_size=1,
        queue_enabled=False,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    try:
        metadata_snapshot = metadata_pipeline.snapshot()
        quotas = metadata_snapshot["mix_epoch_quotas"]
        if set(quotas) != set(LIBERO_CHILDREN):
            raise RuntimeError("LIBERO weighted quotas lack a child")
        if any(int(value) <= 0 for value in quotas.values()):
            raise RuntimeError("LIBERO weighted quotas omit a child")
        print(
            "LIBERO_WEIGHTED_QUOTAS "
            + json.dumps(quotas, sort_keys=True),
            flush=True,
        )
        print(
            "LIBERO_NORMALIZED_WEIGHTS "
            + json.dumps(
                metadata_snapshot["mix_normalized_weights"],
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        metadata_pipeline.close()

    balanced_sources = [
        _source(
            catalog_root,
            pool_root,
            child,
            args.data_version,
            camera_mode=args.camera_mode,
            normalization_mode=args.normalization_mode,
            statistics_path=args.statistics_path,
            weight=1.0,
        )
        for child in LIBERO_CHILDREN
    ]
    total_batches = args.batches_per_child * len(balanced_sources)
    pipeline = open_mixed_training_data(
        balanced_sources,
        batch_size=args.batch_size,
        batches_per_epoch=total_batches,
        output_adapter=MOTUS_NATIVE_MULTIMODAL_BATCH_V1,
        adapter_options={
            "num_video_frames": args.num_video_frames,
            "action_video_fps_ratio": 1,
        },
        video_size=size,
        seed=args.seed,
        rank=0,
        world_size=1,
        queue_enabled=True,
        num_workers=0,
        pin_memory=False,
        queue_size=args.queue_size,
        queue_timeout_seconds=600.0,
        queue_clone_tensors=False,
        shuffle=True,
        drop_last=True,
    )
    observed: Counter[str] = Counter()
    started = time.monotonic()
    try:
        pipeline.set_epoch(3)
        with pipeline:
            for step in range(total_batches):
                batch, stats = pipeline.next()
                child = str(stats["mix_source_name"])
                result = _validate_batch(
                    batch,
                    child=child,
                    batch_size=args.batch_size,
                    num_video_frames=args.num_video_frames,
                    size=size,
                )
                result["resources"] = _resource_snapshot()
                if not stats["ready_queue_enabled"]:
                    raise RuntimeError("LIBERO ReadyBatchQueue is not enabled")
                observed[child] += 1
                print(
                    f"LIBERO_QUEUE_BATCH step={step + 1}/{total_batches} "
                    + json.dumps(result, ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
    finally:
        pipeline.close()
    expected = {
        child: args.batches_per_child for child in LIBERO_CHILDREN
    }
    if dict(observed) != expected:
        raise RuntimeError(
            f"LIBERO balanced mix coverage changed: {dict(observed)}"
        )
    print(
        "LIBERO_INDEXED_PIPELINE_PASS "
        + json.dumps(
            {
                "camera_mode": args.camera_mode,
                "consumed": dict(observed),
                "elapsed_seconds": time.monotonic() - started,
                "normalization_mode": args.normalization_mode,
                "num_workers": 0,
                "queue_size": args.queue_size,
                "resources": _resource_snapshot(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
