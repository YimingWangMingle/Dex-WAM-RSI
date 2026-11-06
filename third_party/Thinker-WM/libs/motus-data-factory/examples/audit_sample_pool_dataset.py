#!/usr/bin/env python3
"""Real-data acceptance smoke for SamplePool grouped union decoding."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from motus_data import (
    AgibotBetaEpisodeReader,
    EpisodeRepository,
    IndexedEpisodeBatch,
    SamplePool,
    SamplePoolDataset,
)


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--index-path")
    parser.add_argument(
        "--trust-index-binding",
        action="store_true",
        help="Skip a full Catalog SHA256 scan after loading a verified offset index.",
    )
    parser.add_argument("--sample-pool", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trust_index_binding and not args.index_path:
        raise ValueError("--trust-index-binding requires --index-path")
    events = Counter()
    event_details = []

    def observe(event, details):
        events[str(event)] += 1
        event_details.append({"event": str(event), **dict(details)})

    started = time.monotonic()
    repository = EpisodeRepository(
        Path(args.catalog),
        readers={
            "agibot_beta": AgibotBetaEpisodeReader(event_callback=observe)
        },
        index_path=args.index_path,
        verify_sha256=not args.trust_index_binding,
    )
    pool = SamplePool(Path(args.sample_pool), repository)
    dataset = SamplePoolDataset(repository, pool)
    sampler = dataset.create_batch_sampler(
        batch_size=int(args.batch_size),
        drop_last=True,
        shuffle=False,
        num_replicas=1,
        rank=0,
        seed=20260723,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    print(
        "SAMPLE_POOL_READY "
        + json.dumps(
            {
                "catalog_id": repository.catalog_id,
                "sample_pool_id": pool.pool_id,
                "selections": len(pool),
                "episode_groups": len(pool.episode_groups),
                "group_sizes": {
                    key: len(value) for key, value in pool.episode_groups.items()
                },
                "batches": len(sampler),
                "rss_mb": _rss_mb(),
                "repository_indexed": repository.indexed,
                "catalog_sha256_verified": not args.trust_index_binding,
            }
        ),
        flush=True,
    )

    batch = next(iter(loader))
    if not isinstance(batch, IndexedEpisodeBatch):
        raise RuntimeError(f"unexpected batch type={type(batch)!r}")
    if batch.batch_size != int(args.batch_size):
        raise RuntimeError(
            f"batch size mismatch: {batch.batch_size} vs {int(args.batch_size)}"
        )
    if len({sample.info.episode_id for sample in batch.samples}) != 1:
        raise RuntimeError("grouped batch contains multiple Episodes")
    expected_video_union = sorted(
        {
            index
            for sample in batch.samples
            for index in sample.video_frame_indices
        }
    )
    required_events = {
        "video_member_extract": 1,
        "video_decode": 1,
        "canonical_timeline_load": 1,
    }
    for event, expected in required_events.items():
        if events[event] != expected:
            raise RuntimeError(
                f"expected {event}={expected}, observed {events[event]}"
            )
    decode_event = next(
        detail for detail in event_details if detail["event"] == "video_decode"
    )
    if int(decode_event["unique_count"]) != len(expected_video_union):
        raise RuntimeError(
            "union video decode count mismatch: "
            f"event={decode_event['unique_count']} expected={len(expected_video_union)}"
        )
    for sample, selection in zip(
        batch.samples,
        [pool.selections[index] for index in next(iter(sampler))],
    ):
        if list(sample.video_frame_indices) != selection["video_frame_indices"]:
            raise RuntimeError("video indices were not reconstructed exactly")
        if list(sample.state_indices) != selection["state_indices"]:
            raise RuntimeError("state indices were not reconstructed exactly")
        if list(sample.action_indices) != selection["action_indices"]:
            raise RuntimeError("action indices were not reconstructed exactly")
        if list(sample.text_indices) != selection["text_indices"]:
            raise RuntimeError("text indices were not reconstructed exactly")
    print(
        "GROUPED_UNION_READ_OK "
        + json.dumps(
            {
                "episode_id": batch.episode_id,
                "batch_size": batch.batch_size,
                "sample_video_lengths": [
                    len(sample.video_frame_indices) for sample in batch.samples
                ],
                "sample_state_lengths": [
                    len(sample.state_indices) for sample in batch.samples
                ],
                "sample_action_lengths": [
                    len(sample.action_indices) for sample in batch.samples
                ],
                "sample_text_lengths": [
                    len(sample.text_indices) for sample in batch.samples
                ],
                "union_video_frames": len(expected_video_union),
                "events": dict(events),
                "rss_mb": _rss_mb(),
            }
        ),
        flush=True,
    )

    if len(pool) >= 2:
        rank0 = dataset.create_batch_sampler(
            batch_size=1,
            drop_last=True,
            shuffle=True,
            num_replicas=2,
            rank=0,
            seed=20260723,
        )
        rank1 = dataset.create_batch_sampler(
            batch_size=1,
            drop_last=True,
            shuffle=True,
            num_replicas=2,
            rank=1,
            seed=20260723,
        )
        rank0_indices = {index for values in rank0 for index in values}
        rank1_indices = {index for values in rank1 for index in values}
        if rank0_indices & rank1_indices:
            raise RuntimeError("distributed sampler ranks overlap")
        if rank0_indices | rank1_indices != set(range(len(pool))):
            raise RuntimeError("distributed sampler did not cover the complete Pool")
        print(
            "DISTRIBUTED_GROUPED_SAMPLER_OK "
            + json.dumps(
                {
                    "rank0": sorted(rank0_indices),
                    "rank1": sorted(rank1_indices),
                }
            ),
            flush=True,
        )
    print(
        "SAMPLE_POOL_DATASET_OK "
        + json.dumps(
            {
                "wall_seconds": time.monotonic() - started,
                "rss_mb": _rss_mb(),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
