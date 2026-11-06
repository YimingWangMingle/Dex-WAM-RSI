#!/usr/bin/env python3
"""Validate direct load_window parity and the replaceable Motus batch adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from motus_data import (
    EpisodeRepository,
    IndexedBatchCollator,
    SamplePool,
    SamplePoolDataset,
    build_dataset,
    build_indexed_batch_adapter,
    validate_motus_model_batch_v1,
)
from motus_data.baseline import batch_signature, sample_signature, tensor_signature
from motus_data.collate import collate_samples
from motus_data.datasets.agibot_beta import AgibotBetaDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--sample-pool", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--config",
        help="Direct Agibot Beta config; defaults to baseline.config_path.",
    )
    parser.add_argument("--samples", type=int, default=2)
    return parser.parse_args()


def _assert_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(
            f"{name} shape mismatch: new={tuple(actual.shape)} "
            f"direct={tuple(expected.shape)}"
        )
    difference = (actual.float() - expected.float()).abs()
    raise RuntimeError(
        f"{name} differs: max_abs_diff="
        f"{float(difference.max()) if difference.numel() else 0.0}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_samples(
    dataset: AgibotBetaDataset, selections
) -> list[dict]:
    positions = {
        str(episode["episode_uid"]): index
        for index, episode in enumerate(dataset.episodes)
    }
    result = []
    for selection in selections:
        episode_id = str(selection["episode_id"])
        if episode_id not in positions:
            raise KeyError(f"direct Dataset is missing episode={episode_id}")
        video = [int(value) for value in selection["video_frame_indices"]]
        sample = dataset.load_window(positions[episode_id], video[0])
        if sample["frame_indices"] != video:
            raise RuntimeError(
                f"direct frame replay mismatch for episode={episode_id}"
            )
        result.append(sample)
    return result


def main() -> int:
    args = parse_args()
    if int(args.samples) <= 0:
        raise ValueError("--samples must be positive")
    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("format") != "agibot_beta_direct_baseline_v1":
        raise ValueError("unsupported baseline format")
    config_path = Path(args.config or baseline["config_path"])
    expected_config_sha256 = str(baseline["config_sha256"])
    actual_config_sha256 = _file_sha256(config_path)
    if actual_config_sha256 != expected_config_sha256:
        raise RuntimeError(
            f"direct config differs from frozen baseline: "
            f"actual={actual_config_sha256} expected={expected_config_sha256}"
        )
    config = OmegaConf.load(config_path)

    print(f"PHASE build_direct_dataset config={config_path}", flush=True)
    direct_dataset = build_dataset(
        config,
        split="train",
        rank=0,
        seed=int(baseline["seed"]),
    )
    if not isinstance(direct_dataset, AgibotBetaDataset):
        raise TypeError(
            "phase-6 parity requires one direct AgibotBetaDataset, got "
            f"{type(direct_dataset).__name__}"
        )

    repository = EpisodeRepository(args.catalog)
    pool = SamplePool(args.sample_pool, repository)
    count = min(int(args.samples), len(pool))
    if count != int(args.samples):
        raise ValueError(
            f"SamplePool has {len(pool)} selections, requested {args.samples}"
        )
    pool_indices = list(range(count))
    selected = [pool.selections[index] for index in pool_indices]
    if len({str(row["episode_id"]) for row in selected}) != 1:
        raise ValueError("frozen phase-6 gate requires one same-Episode batch")

    frozen_selections = baseline.get("selections") or []
    for position, selection in enumerate(selected):
        frozen = frozen_selections[position]
        if (
            str(selection["episode_id"]) != str(frozen["episode_id"])
            or list(selection["video_frame_indices"])
            != [int(value) for value in frozen["video_frame_indices"]]
        ):
            raise RuntimeError(
                f"parity Pool selection {position} differs from frozen baseline"
            )
    print(
        "PARITY_INPUT_READY "
        + json.dumps(
            {
                "catalog_id": repository.catalog_id,
                "sample_pool_id": pool.pool_id,
                "samples": count,
                "episode_id": selected[0]["episode_id"],
                "video_size": list(direct_dataset.video_size),
            }
        ),
        flush=True,
    )

    direct_samples = _direct_samples(direct_dataset, selected)
    for position, direct_sample in enumerate(direct_samples):
        if sample_signature(direct_sample) != baseline["samples"][position]:
            raise RuntimeError(
                f"direct replay no longer matches frozen baseline sample={position}"
            )
    print("DIRECT_BASELINE_REPLAY_OK", flush=True)

    indexed_dataset = SamplePoolDataset(
        repository,
        pool,
        video_output_format="rgb_float32_chw",
        video_size=tuple(int(value) for value in direct_dataset.video_size),
    )
    sampler = indexed_dataset.create_batch_sampler(
        batch_size=count,
        drop_last=True,
        shuffle=False,
        num_replicas=1,
        rank=0,
        seed=int(baseline["seed"]),
    )
    adapter = build_indexed_batch_adapter(
        "motus_model_batch_v1",
        num_video_frames=int(direct_dataset.num_video_frames),
        action_video_fps_ratio=int(direct_dataset.action_video_fps_ratio),
        video_loss_weight=1.0,
        action_loss_weight=1.0,
    )
    loader = DataLoader(
        indexed_dataset,
        batch_sampler=sampler,
        collate_fn=IndexedBatchCollator(adapter),
        num_workers=0,
        pin_memory=False,
    )
    indexed_batch = next(iter(loader))
    validate_motus_model_batch_v1(
        indexed_batch,
        expected_batch_size=count,
        num_video_frames=int(direct_dataset.num_video_frames),
        video_action_freq_ratio=int(direct_dataset.action_video_fps_ratio),
    )

    tensor_keys = (
        "first_frame",
        "video_frames",
        "initial_state",
        "action_sequence",
    )
    scalar_list_keys = (
        "t5_input_message",
        "vlm_input_message",
        "robot_name",
        "source_dataset",
        "source_child",
        "episode_id",
        "frame_indices",
        "action_video_fps_ratio",
    )
    batch_list_keys = {
        "t5_input_message": "t5_input_messages",
        "vlm_input_message": "vlm_input_messages",
        "robot_name": "robot_names",
        "source_dataset": "source_datasets",
        "source_child": "source_children",
        "episode_id": "episode_ids",
        "frame_indices": "frame_indices",
        "action_video_fps_ratio": "action_video_fps_ratios",
    }
    for position, direct_sample in enumerate(direct_samples):
        for key in tensor_keys:
            _assert_tensor(
                f"sample[{position}].{key}",
                indexed_batch[key][position],
                direct_sample[key],
            )
            expected_signature = baseline["samples"][position]["tensors"][key]
            if tensor_signature(indexed_batch[key][position]) != expected_signature:
                raise RuntimeError(
                    f"indexed tensor signature differs from baseline "
                    f"sample={position} key={key}"
                )
        for key in scalar_list_keys:
            actual = indexed_batch[batch_list_keys[key]][position]
            expected = direct_sample[key]
            if actual != expected:
                raise RuntimeError(
                    f"sample[{position}].{key} mismatch: "
                    f"new={actual!r} direct={expected!r}"
                )
        direct_pil = np.asarray(direct_sample["first_frame_pil"])
        indexed_pil = np.asarray(indexed_batch["first_frame_pils"][position])
        if not np.array_equal(indexed_pil, direct_pil):
            raise RuntimeError(f"sample[{position}].first_frame_pil differs")
    print(
        "INDEXED_SAMPLE_PARITY_OK "
        + json.dumps({"samples": count, "tensor_keys": list(tensor_keys)}),
        flush=True,
    )

    direct_batch = collate_samples(direct_samples)
    if direct_batch is None:
        raise RuntimeError("direct collate returned no batch")
    direct_signature = batch_signature(direct_batch)
    indexed_signature = batch_signature(indexed_batch)
    if indexed_signature["tensors"] != direct_signature["tensors"]:
        raise RuntimeError("adapted and direct batch tensor signatures differ")
    comparable_lists = (
        "episode_ids",
        "frame_indices",
        "source_datasets",
        "source_children",
        "robot_names",
        "t5_input_messages",
        "vlm_input_messages",
        "state_action_policies",
        "action_video_fps_ratios",
    )
    for key in comparable_lists:
        if indexed_signature["lists"][key] != direct_signature["lists"][key]:
            raise RuntimeError(f"adapted and direct batch list differs: {key}")
    if indexed_signature["tensors"] != baseline["batch"]["tensors"]:
        raise RuntimeError("adapted batch tensor signatures differ from frozen baseline")
    print(
        "MOTUS_MODEL_BATCH_V1_PARITY_OK "
        + json.dumps(
            {
                "batch_size": count,
                "first_frame_shape": list(indexed_batch["first_frame"].shape),
                "video_frames_shape": list(indexed_batch["video_frames"].shape),
                "initial_state_shape": list(indexed_batch["initial_state"].shape),
                "action_sequence_shape": list(
                    indexed_batch["action_sequence"].shape
                ),
                "manifest_ids_compared": False,
                "manifest_id_note": (
                    "new IDs identify SamplePool selections; old IDs identify "
                    "direct-Dataset windows"
                ),
            }
        ),
        flush=True,
    )
    print("PHASE6_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
