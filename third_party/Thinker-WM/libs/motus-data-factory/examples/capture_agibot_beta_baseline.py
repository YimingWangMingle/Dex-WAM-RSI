#!/usr/bin/env python3
"""Capture deterministic real-data Agibot Beta sample and batch signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from motus_data import build_dataset, validate_motus_model_batch_v1
from motus_data.baseline import (
    AGIBOT_BETA_BASELINE_V1,
    batch_signature,
    sample_signature,
    selection_records,
    write_json_atomic,
)
from motus_data.collate import collate_samples
from motus_data.datasets.agibot_beta import AgibotBetaDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--selection-file",
        help="Optional prior baseline JSON; replay its selections instead of auto-selecting.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_replay_samples(
    dataset: AgibotBetaDataset, selection_file: str, limit: int
):
    report = json.loads(Path(selection_file).read_text(encoding="utf-8"))
    selections = report.get("selections") or []
    if len(selections) < limit:
        raise ValueError(
            f"selection file contains {len(selections)} selections, requested {limit}"
        )
    episode_positions = {
        str(episode["episode_uid"]): position
        for position, episode in enumerate(dataset.episodes)
    }
    samples = []
    for selection in selections[:limit]:
        episode_id = str(selection["episode_id"])
        if episode_id not in episode_positions:
            raise KeyError(f"baseline episode is absent from current split: {episode_id}")
        sample = dataset.load_window(
            episode_positions[episode_id], int(selection["start_frame"])
        )
        expected = [int(value) for value in selection["video_frame_indices"]]
        if sample["frame_indices"] != expected:
            raise RuntimeError(
                f"replayed frame indices differ for {episode_id}: "
                f"got={sample['frame_indices']} expected={expected}"
            )
        samples.append(sample)
    return samples


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    config_path = Path(args.config).resolve()
    print(f"PHASE load_config path={config_path}", flush=True)
    config = OmegaConf.load(config_path)
    print("PHASE build_current_direct_dataset", flush=True)
    dataset = build_dataset(config, split="train", rank=0, seed=args.seed)
    if not isinstance(dataset, AgibotBetaDataset):
        raise TypeError(
            "baseline capture requires a single type=agibot_beta dataset, got "
            f"{type(dataset).__name__}"
        )
    if args.samples > dataset.virtual_samples_per_episode:
        raise ValueError(
            f"--samples={args.samples} exceeds virtual_samples_per_episode="
            f"{dataset.virtual_samples_per_episode}"
        )

    print(
        f"DATASET_READY split={dataset.split} episodes={len(dataset.episodes)} "
        f"total_usable={dataset.total_episode_count} scan_stats={dataset.scan_stats}",
        flush=True,
    )
    if args.selection_file:
        print(f"PHASE replay_selections path={args.selection_file}", flush=True)
        samples = load_replay_samples(dataset, args.selection_file, args.samples)
        selection_mode = "replay"
    else:
        print(
            f"PHASE deterministic_auto_selection samples={args.samples} seed={args.seed}",
            flush=True,
        )
        samples = dataset.get_grouped_samples(list(range(args.samples)))
        selection_mode = "auto"

    for position, sample in enumerate(samples):
        print(
            f"SAMPLE_READY index={position} episode_id={sample['episode_id']} "
            f"start_frame={sample['metadata']['start_frame']} "
            f"frames={len(sample['frame_indices'])}",
            flush=True,
        )

    print("PHASE collate_and_validate", flush=True)
    batch = collate_samples(samples)
    if batch is None:
        raise RuntimeError("Agibot Beta baseline collate returned no batch")
    validate_motus_model_batch_v1(
        batch,
        expected_batch_size=len(samples),
        num_video_frames=dataset.num_video_frames,
        video_action_freq_ratio=dataset.action_video_fps_ratio,
    )

    print("PHASE hash_full_tensor_contents", flush=True)
    report = {
        "format": AGIBOT_BETA_BASELINE_V1,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "seed": int(args.seed),
        "selection_mode": selection_mode,
        "dataset": {
            "class": type(dataset).__name__,
            "split": dataset.split,
            "episodes": len(dataset.episodes),
            "total_usable_episodes": dataset.total_episode_count,
            "scan_stats": dataset.scan_stats,
            "num_video_frames": dataset.num_video_frames,
            "action_video_fps_ratio": dataset.action_video_fps_ratio,
            "video_size": list(dataset.video_size),
        },
        "selections": selection_records(samples),
        "samples": [sample_signature(sample) for sample in samples],
        "batch": batch_signature(batch),
        "contract": {
            "name": "motus_model_batch_v1",
            "passed": True,
        },
    }
    write_json_atomic(args.output, report)
    print(
        f"BASELINE_OK output={Path(args.output).resolve()} "
        f"samples={len(samples)} batch_size={batch['first_frame'].shape[0]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
