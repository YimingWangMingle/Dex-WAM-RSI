#!/usr/bin/env python3
"""Run real SamplePool -> grouped batch -> mixed ReadyBatchQueue smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from motus_data import open_dataset, open_mixed_training_data


PROFILES = (
    ("agibot_beta_parallel_gripper", 14),
    ("agibot_beta_dexterous_hand", 24),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=2)
    return parser.parse_args()


def _pool_rows(catalog_id: str, episode, *, pool_id: str):
    sample_count = 4
    if (
        episode.info.video_count < sample_count + 2
        or episode.info.state_count < sample_count
        or episode.info.action_count < sample_count + 1
        or episode.info.text_count < 1
    ):
        raise ValueError(f"Episode is too short for smoke: {episode.episode_id}")
    selections = []
    for start in range(sample_count):
        selections.append(
            {
                "episode_id": episode.episode_id,
                "video_frame_indices": [start, start + 1, start + 2],
                "state_indices": [start],
                "action_indices": [start, start + 1],
                "text_indices": [0],
            }
        )
    return [
        {
            "format": "motus_sample_pool_v2",
            "pool_id": pool_id,
            "catalog_id": catalog_id,
            "samples": len(selections),
        },
        *selections,
    ]


def _write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if int(args.batch_size) != 2:
        raise ValueError("this frozen smoke expects batch-size=2")
    pools = {}
    expected_dimensions = {}
    selected_episodes = {}
    for dataset, dimension in PROFILES:
        repository = open_dataset(dataset, verify_sha256=False)
        try:
            episode_id = next(repository.iter_episode_ids())
            episode = repository.get_episode(episode_id)
            pool_path = output / f"{dataset}.smoke.sample_pool.jsonl"
            _write_jsonl(
                pool_path,
                _pool_rows(
                    repository.catalog_id,
                    episode,
                    pool_id=f"{dataset}_real_queue_smoke_v1",
                ),
            )
            pools[dataset] = pool_path
            expected_dimensions[dataset] = dimension
            selected_episodes[dataset] = episode_id
        finally:
            repository.close()

    sources = [
        {
            "name": dataset,
            "dataset": dataset,
            "sample_pool": pool_path,
            "weight": 1.0,
            "family": "agibot_beta",
        }
        for dataset, pool_path in pools.items()
    ]
    consumed = []
    pipeline = open_mixed_training_data(
        sources,
        batch_size=2,
        batches_per_epoch=4,
        output_adapter=None,
        video_size=(96, 128),
        queue_enabled=True,
        num_workers=0,
        pin_memory=False,
        queue_size=int(args.queue_size),
        shuffle=False,
        drop_last=True,
    )
    try:
        with pipeline:
            for _ in range(4):
                batch, stats = pipeline.next()
                source = str(stats["mix_source_name"])
                dimension = expected_dimensions[source]
                if batch.batch_size != 2:
                    raise ValueError("mixed queue returned wrong batch size")
                if batch.info.source_child != source:
                    raise ValueError("mixed batch source child mismatch")
                sample_rows = []
                for sample in batch.samples:
                    if tuple(sample.state.shape) != (1, dimension):
                        raise ValueError(
                            f"state shape mismatch: {sample.state.shape}"
                        )
                    if tuple(sample.action.shape) != (2, dimension):
                        raise ValueError(
                            f"action shape mismatch: {sample.action.shape}"
                        )
                    if (
                        sample.video.ndim != 4
                        or tuple(sample.video.shape[1:]) != (3, 96, 128)
                    ):
                        raise ValueError(
                            f"video decode shape mismatch: {sample.video.shape}"
                        )
                    if (
                        not torch.isfinite(sample.state).all()
                        or not torch.isfinite(sample.action).all()
                        or not torch.isfinite(sample.video).all()
                    ):
                        raise ValueError("mixed queue sample contains non-finite data")
                    sample_rows.append(
                        {
                            "video_shape": list(sample.video.shape),
                            "state_shape": list(sample.state.shape),
                            "action_shape": list(sample.action.shape),
                            "maximum_absolute_state_action": max(
                                float(sample.state.abs().max()),
                                float(sample.action.abs().max()),
                            ),
                        }
                    )
                consumed.append(
                    {
                        "source": source,
                        "episode_id": batch.episode_id,
                        "queue_enabled": bool(stats["ready_queue_enabled"]),
                        "samples": sample_rows,
                    }
                )
    finally:
        pipeline.close()
    source_counts = {
        dataset: sum(row["source"] == dataset for row in consumed)
        for dataset, _ in PROFILES
    }
    if source_counts != {
        "agibot_beta_parallel_gripper": 2,
        "agibot_beta_dexterous_hand": 2,
    }:
        raise ValueError(f"unexpected source counts: {source_counts}")
    report = {
        "format": "agibot_beta_profile_mixed_queue_smoke_v1",
        "selected_episodes": selected_episodes,
        "source_counts": source_counts,
        "batches": consumed,
        "pipeline_snapshot": pipeline.snapshot(),
    }
    report_path = output / "report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print("PROFILE_MIXED_QUEUE_SMOKE_DONE " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
