#!/usr/bin/env python3
"""Validate the model-facing batch contract for both Agibot Beta profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from motus_data import open_mixed_training_data


PROFILES = (
    ("agibot_beta_parallel_gripper", 14, "agibot-beta-parallel-gripper"),
    ("agibot_beta_dexterous_hand", 24, "agibot-beta-dexterous-hand"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queue-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool_directory = Path(args.pool_directory).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = []
    expected = {}
    for dataset, dimension, robot_name in PROFILES:
        pool = pool_directory / f"{dataset}.smoke.sample_pool.jsonl"
        if not pool.is_file():
            raise FileNotFoundError(pool)
        sources.append(
            {
                "name": dataset,
                "dataset": dataset,
                "sample_pool": pool,
                "weight": 1.0,
                "family": "agibot_beta",
            }
        )
        expected[dataset] = (dimension, robot_name)

    rows = []
    pipeline = open_mixed_training_data(
        sources,
        batch_size=2,
        batches_per_epoch=4,
        output_adapter="motus_model_batch_v1",
        adapter_options={
            "num_video_frames": 2,
            "action_video_fps_ratio": 1,
            "video_loss_weight": 1.0,
            "action_loss_weight": 1.0,
            "state_action_policy": "real",
        },
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
                dimension, robot_name = expected[source]
                expected_shapes = {
                    "first_frame": (2, 3, 96, 128),
                    "video_frames": (2, 2, 3, 96, 128),
                    "initial_state": (2, dimension),
                    "action_sequence": (2, 2, dimension),
                }
                for key, shape in expected_shapes.items():
                    value = batch.get(key)
                    if not torch.is_tensor(value) or tuple(value.shape) != shape:
                        raise ValueError(
                            f"model batch {key} mismatch for {source}: "
                            f"{getattr(value, 'shape', None)}"
                        )
                    if not torch.isfinite(value).all():
                        raise ValueError(f"model batch {key} is non-finite")
                if batch["robot_names"] != [robot_name, robot_name]:
                    raise ValueError("model batch robot_names mismatch")
                if batch["source_children"] != [source, source]:
                    raise ValueError("model batch source_children mismatch")
                if batch["source_datasets"] != ["agibot_beta", "agibot_beta"]:
                    raise ValueError("model batch source_datasets mismatch")
                rows.append(
                    {
                        "source": source,
                        "robot_names": batch["robot_names"],
                        "first_frame_shape": list(batch["first_frame"].shape),
                        "video_frames_shape": list(batch["video_frames"].shape),
                        "initial_state_shape": list(batch["initial_state"].shape),
                        "action_sequence_shape": list(
                            batch["action_sequence"].shape
                        ),
                        "maximum_absolute_state_action": max(
                            float(batch["initial_state"].abs().max()),
                            float(batch["action_sequence"].abs().max()),
                        ),
                        "queue_enabled": bool(stats["ready_queue_enabled"]),
                    }
                )
    finally:
        pipeline.close()
    counts = {
        dataset: sum(row["source"] == dataset for row in rows)
        for dataset, _, _ in PROFILES
    }
    if any(value != 2 for value in counts.values()):
        raise ValueError(f"model batch source counts mismatch: {counts}")
    report = {
        "format": "agibot_beta_profile_model_batch_smoke_v1",
        "source_counts": counts,
        "batches": rows,
        "pipeline_snapshot": pipeline.snapshot(),
    }
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print("PROFILE_MODEL_BATCH_SMOKE_DONE " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
