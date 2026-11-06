#!/usr/bin/env python3
"""Audit OpenVid plus Agibot Beta weighted mix and rank sharding."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from motus_data import (
    MOTUS_MODEL_BATCH_V1,
    EpisodeDatasetRelease,
    IndexedTrainingSource,
    open_dataset,
    open_mixed_training_data,
    open_training_data,
    validate_motus_model_batch_v1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        default="/data/jiyuan.wang/EpisodeCatalogs",
    )
    parser.add_argument(
        "--openvid-catalog",
        default=(
            "/data/jiyuan.wang/EpisodeCatalogs/_20260730/openvid/"
            "openvid_1m_full_20260730.episode_catalog.jsonl"
        ),
    )
    parser.add_argument(
        "--openvid-index",
        default=(
            "/data/jiyuan.wang/EpisodeCatalogs/_20260730/openvid/"
            "openvid_1m_full_20260730.episode_catalog.index.sqlite"
        ),
    )
    parser.add_argument(
        "--agibot-beta-pool",
        default=(
            "/data/jiyuan.wang/SamplePools/_20260727/"
            "ten_source_smoke_v1/"
            "agibot_beta.engineering_smoke.sample_pool.jsonl"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--batches-per-epoch", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".meta.json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _openvid_release(
    catalog: Path,
    index: Path,
    *,
    height: int,
    width: int,
) -> EpisodeDatasetRelease:
    return EpisodeDatasetRelease(
        dataset="openvid",
        release="20260730_full_v1_gate",
        catalog_path=str(catalog),
        index_path=str(index),
        default_video_size=(height, width),
        default_num_workers=0,
        default_queue_size=2,
        action_magnitude_guard_sources=None,
    )


def _write_openvid_pool(
    output_path: Path,
    *,
    release: EpisodeDatasetRelease,
    catalog_id: str,
    catalog_episodes: int,
    batch_size: int,
    num_video_frames: int,
) -> tuple[str, int]:
    repository = open_dataset(
        "openvid",
        release=release,
        verify_sha256=False,
    )
    try:
        sample_count = max(8, 2 * batch_size)
        required_frames = num_video_frames + sample_count
        selected_episode = None
        for episode_id in repository.iter_episode_ids():
            episode = repository.get_episode(episode_id)
            if episode.info.video_count < required_frames:
                continue
            texts = episode.read_text([0])
            if texts and texts[0].text.strip():
                selected_episode = episode
                break
        if selected_episode is None:
            raise RuntimeError(
                "OpenVid has no Episode suitable for the engineering pool"
            )
        selections = [
            {
                "episode_id": selected_episode.episode_id,
                "video_frame_indices": list(
                    range(start, start + num_video_frames + 1)
                ),
                "state_indices": [],
                "action_indices": [],
                "text_indices": [0],
            }
            for start in range(sample_count)
        ]
        rows = [
            {
                "format": "motus_sample_pool_v2",
                "pool_id": "openvid_agibot_beta_mix_gate_20260730",
                "catalog_id": catalog_id,
                "samples": len(selections),
                "meta": {
                    "purpose": "engineering_smoke_only",
                    "production_sampling_policy": False,
                    "catalog_episodes": catalog_episodes,
                    "num_video_frames": num_video_frames,
                    "action_video_fps_ratio": 1,
                },
            },
            *selections,
        ]
    finally:
        repository.close()

    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return selected_episode.episode_id, len(selections)


def _pool_catalog_episodes(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    return int(header["meta"]["catalog_episodes"])


def _sources(
    *,
    catalog_root: Path,
    openvid_release: EpisodeDatasetRelease,
    openvid_pool: Path,
    openvid_episodes: int,
    agibot_beta_pool: Path,
    agibot_beta_episodes: int,
    num_video_frames: int,
) -> list[IndexedTrainingSource]:
    common = {
        "num_video_frames": num_video_frames,
        "action_video_fps_ratio": 1,
        "video_loss_weight": 1.0,
        "action_loss_weight": 0.0,
    }
    return [
        IndexedTrainingSource(
            name="openvid",
            dataset="openvid",
            family="openvid",
            sample_pool=openvid_pool,
            weight=float(openvid_episodes),
            release=openvid_release,
            verify_sha256=False,
            output_adapter=MOTUS_MODEL_BATCH_V1,
            adapter_options={
                **common,
                "state_action_policy": "absent",
                "allow_missing_state_action": True,
                "placeholder_state_action_dim": 26,
            },
        ),
        IndexedTrainingSource(
            name="agibot_beta",
            dataset="agibot_beta",
            family="agibot_beta",
            sample_pool=agibot_beta_pool,
            weight=float(agibot_beta_episodes),
            catalog_root=catalog_root,
            output_adapter=MOTUS_MODEL_BATCH_V1,
            adapter_options={
                **common,
                "state_action_policy": "real",
            },
        ),
    ]


def _child_pipeline(
    source: IndexedTrainingSource,
    *,
    batch_size: int,
    video_size: tuple[int, int],
    seed: int,
    rank: int,
    world_size: int,
):
    return open_training_data(
        source.dataset,
        release=source.release,
        sample_pool=source.sample_pool,
        batch_size=batch_size,
        video_size=video_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        queue_enabled=False,
        catalog_root=source.catalog_root,
        catalog_path=source.catalog_path,
        index_path=source.index_path,
        use_index=source.use_index,
        verify_sha256=source.verify_sha256,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )


def _rank_sharding_gate(
    sources: list[IndexedTrainingSource],
    *,
    batch_size: int,
    video_size: tuple[int, int],
    batches_per_epoch: int,
    seed: int,
) -> None:
    for source in sources:
        by_epoch = {}
        for epoch in (0, 1):
            rank_batches = []
            pipelines = []
            try:
                for rank in (0, 1):
                    pipeline = _child_pipeline(
                        source,
                        batch_size=batch_size,
                        video_size=video_size,
                        seed=seed,
                        rank=rank,
                        world_size=2,
                    )
                    pipelines.append(pipeline)
                    pipeline.set_epoch(epoch)
                    batches = tuple(
                        tuple(batch)
                        for batch in pipeline.batch_sampler
                    )
                    rank_batches.append(batches)
                left = {
                    index
                    for batch in rank_batches[0]
                    for index in batch
                }
                right = {
                    index
                    for batch in rank_batches[1]
                    for index in batch
                }
                if left & right:
                    raise RuntimeError(
                        f"{source.name} rank SamplePool rows overlap"
                    )
                if not left or not right:
                    raise RuntimeError(
                        f"{source.name} rank sharding produced an empty rank"
                    )
                by_epoch[epoch] = {
                    "rank0": rank_batches[0],
                    "rank1": rank_batches[1],
                }
            finally:
                for pipeline in pipelines:
                    pipeline.close()
        print(
            f"RANK_CHILD_SHARD_OK source={source.name} "
            f"epoch0={by_epoch[0]} epoch1={by_epoch[1]}",
            flush=True,
        )

    mixed = []
    try:
        for rank in (0, 1):
            mixed.append(
                open_mixed_training_data(
                    sources,
                    batch_size=batch_size,
                    batches_per_epoch=batches_per_epoch,
                    video_size=video_size,
                    seed=seed,
                    rank=rank,
                    world_size=2,
                    queue_enabled=False,
                    num_workers=0,
                    shuffle=True,
                    drop_last=True,
                )
            )
        schedule0 = mixed[0].stream.source_schedule()
        schedule1 = mixed[1].stream.source_schedule()
        if schedule0 != schedule1:
            raise RuntimeError("source schedules differ across ranks")
        mixed[0].set_epoch(1)
        mixed[1].set_epoch(1)
        epoch1_schedule0 = mixed[0].stream.source_schedule()
        epoch1_schedule1 = mixed[1].stream.source_schedule()
        if epoch1_schedule0 != epoch1_schedule1:
            raise RuntimeError("epoch-1 source schedules differ across ranks")
        print(
            "MIX_RANK_SCHEDULE_OK "
            + json.dumps(
                {
                    "epoch0": schedule0,
                    "epoch1": epoch1_schedule0,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    finally:
        for pipeline in mixed:
            pipeline.close()


def _validate_batch(
    batch: dict[str, Any],
    *,
    source_name: str,
    batch_size: int,
    num_video_frames: int,
    video_size: tuple[int, int],
) -> None:
    validate_motus_model_batch_v1(
        batch,
        expected_batch_size=batch_size,
        num_video_frames=num_video_frames,
        video_action_freq_ratio=1,
    )
    if tuple(batch["first_frame"].shape) != (
        batch_size,
        3,
        *video_size,
    ):
        raise RuntimeError(
            f"{source_name} first-frame shape is invalid"
        )
    if tuple(batch["video_frames"].shape) != (
        batch_size,
        num_video_frames,
        3,
        *video_size,
    ):
        raise RuntimeError(f"{source_name} video shape is invalid")
    if set(batch["source_datasets"]) != {source_name}:
        raise RuntimeError(
            f"{source_name} batch mixed source datasets"
        )
    if set(batch["source_children"]) != {source_name}:
        raise RuntimeError(
            f"{source_name} batch mixed source children"
        )
    if not torch.equal(
        batch["action_loss_weights"],
        torch.zeros(batch_size),
    ):
        raise RuntimeError(f"{source_name} Stage1 action loss is not zero")
    if not torch.equal(
        batch["video_loss_weights"],
        torch.ones(batch_size),
    ):
        raise RuntimeError(f"{source_name} video loss is not one")
    if source_name == "openvid":
        if set(batch["state_action_policies"]) != {"absent"}:
            raise RuntimeError("OpenVid state/action policy changed")
        if torch.count_nonzero(batch["initial_state"]).item() != 0:
            raise RuntimeError("OpenVid placeholder state is not zero")
        if torch.count_nonzero(batch["action_sequence"]).item() != 0:
            raise RuntimeError("OpenVid placeholder action is not zero")
        if not all(
            item["synthetic_state_action"]
            for item in batch["metadata"]
        ):
            raise RuntimeError("OpenVid synthetic audit flag is missing")
    else:
        if set(batch["state_action_policies"]) != {"real"}:
            raise RuntimeError("Agibot Beta real state/action policy changed")
        if not torch.isfinite(batch["initial_state"]).all():
            raise RuntimeError("Agibot Beta state is non-finite")
        if not torch.isfinite(batch["action_sequence"]).all():
            raise RuntimeError("Agibot Beta action is non-finite")
        if "metadata" in batch:
            raise RuntimeError(
                "Agibot Beta batch gained OpenVid-only metadata"
            )


def _weighted_queue_gate(
    sources: list[IndexedTrainingSource],
    *,
    batch_size: int,
    num_video_frames: int,
    video_size: tuple[int, int],
    queue_size: int,
    batches_per_epoch: int,
    seed: int,
) -> None:
    pipeline = open_mixed_training_data(
        sources,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        video_size=video_size,
        seed=seed,
        rank=0,
        world_size=1,
        queue_enabled=True,
        num_workers=0,
        pin_memory=False,
        queue_size=queue_size,
        queue_clone_tensors=False,
        queue_timeout_seconds=600.0,
        shuffle=True,
        drop_last=True,
    )
    quotas = {
        source.name: quota
        for source, quota in zip(
            pipeline.sources,
            pipeline.stream.source_quotas(),
        )
    }
    if sum(quotas.values()) != batches_per_epoch:
        raise RuntimeError("weighted quotas do not fill the epoch")
    if any(quota <= 0 for quota in quotas.values()):
        raise RuntimeError(
            f"weighted epoch does not cover every source: {quotas}"
        )

    observed = Counter()
    try:
        with pipeline:
            for step in range(batches_per_epoch):
                batch, stats = pipeline.next()
                source_name = str(stats["mix_source_name"])
                _validate_batch(
                    batch,
                    source_name=source_name,
                    batch_size=batch_size,
                    num_video_frames=num_video_frames,
                    video_size=video_size,
                )
                observed[source_name] += 1
                print(
                    f"MIX_BATCH_OK step={step} source={source_name} "
                    f"episode={batch['episode_ids'][0]} "
                    f"video={tuple(batch['video_frames'].shape)} "
                    f"state={tuple(batch['initial_state'].shape)} "
                    f"action={tuple(batch['action_sequence'].shape)}",
                    flush=True,
                )
        snapshot = pipeline.snapshot()
    finally:
        pipeline.close()
    if dict(observed) != quotas:
        raise RuntimeError(
            f"observed source counts={dict(observed)} quotas={quotas}"
        )
    if snapshot["ready_queue_producer_exception_type"] is not None:
        raise RuntimeError(
            "mixed queue producer failed: "
            f"{snapshot['ready_queue_producer_exception_message']}"
        )
    print(
        "OPENVID_AGIBOT_BETA_WEIGHTED_QUEUE_OK "
        + json.dumps(
            {
                "weights": snapshot["mix_normalized_weights"],
                "quotas": quotas,
                "observed": dict(observed),
                "queue": {
                    "produced": snapshot["ready_queue_produced_count"],
                    "consumed": snapshot["ready_queue_consumed_count"],
                    "current_size": snapshot["ready_queue_current_size"],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    for name in (
        "batch_size",
        "num_video_frames",
        "height",
        "width",
        "queue_size",
        "batches_per_epoch",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_root = Path(args.catalog_root).resolve()
    openvid_catalog = Path(args.openvid_catalog).resolve()
    openvid_index = Path(args.openvid_index).resolve()
    agibot_beta_pool = Path(args.agibot_beta_pool).resolve()
    for path in (
        openvid_catalog,
        openvid_index,
        agibot_beta_pool,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    video_size = (int(args.height), int(args.width))
    openvid_metadata = _metadata(openvid_catalog)
    openvid_episodes = int(openvid_metadata["episodes"])
    release = _openvid_release(
        openvid_catalog,
        openvid_index,
        height=video_size[0],
        width=video_size[1],
    )
    openvid_pool = output_dir / "openvid.mix_gate.sample_pool.jsonl"
    episode_id, sample_count = _write_openvid_pool(
        openvid_pool,
        release=release,
        catalog_id=str(openvid_metadata["catalog_id"]),
        catalog_episodes=openvid_episodes,
        batch_size=int(args.batch_size),
        num_video_frames=int(args.num_video_frames),
    )
    agibot_beta_episodes = _pool_catalog_episodes(agibot_beta_pool)
    sources = _sources(
        catalog_root=catalog_root,
        openvid_release=release,
        openvid_pool=openvid_pool,
        openvid_episodes=openvid_episodes,
        agibot_beta_pool=agibot_beta_pool,
        agibot_beta_episodes=agibot_beta_episodes,
        num_video_frames=int(args.num_video_frames),
    )
    print(
        "MIX_GATE_INPUT "
        + json.dumps(
            {
                "openvid_episode_id": episode_id,
                "openvid_pool_samples": sample_count,
                "catalog_episodes": {
                    "openvid": openvid_episodes,
                    "agibot_beta": agibot_beta_episodes,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    _rank_sharding_gate(
        sources,
        batch_size=int(args.batch_size),
        video_size=video_size,
        batches_per_epoch=int(args.batches_per_epoch),
        seed=int(args.seed),
    )
    if args.metadata_only:
        print(
            "OPENVID_AGIBOT_BETA_MIX_METADATA_OK",
            flush=True,
        )
        return 0
    _weighted_queue_gate(
        sources,
        batch_size=int(args.batch_size),
        num_video_frames=int(args.num_video_frames),
        video_size=video_size,
        queue_size=int(args.queue_size),
        batches_per_epoch=int(args.batches_per_epoch),
        seed=int(args.seed),
    )
    print("OPENVID_AGIBOT_BETA_MIX_AUDIT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
