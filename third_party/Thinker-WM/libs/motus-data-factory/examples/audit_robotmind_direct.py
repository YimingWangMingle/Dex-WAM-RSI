#!/usr/bin/env python3
"""Audit a direct RobotMind source before enabling it in mixed training."""

from __future__ import annotations

import argparse
from collections import Counter

from omegaconf import OmegaConf

from motus_data import build_dataset, validate_sample
from motus_data.datasets.robotmind import (
    RobotMindDataset,
    scan_robotmind_episodes,
    split_robotmind_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--inspect-all", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OmegaConf.load(args.config)
    dataset = build_dataset(config)
    if not isinstance(dataset, RobotMindDataset):
        raise TypeError("audit_robotmind_direct.py requires one dataset.type=robotmind child")

    node = config.dataset
    all_episodes = scan_robotmind_episodes(node.data_root, node.embodiment)
    test_fraction = float(node.get("test_fraction", 0.01))
    split_seed = int(node.get("split_seed", 20260720))
    train = split_robotmind_episodes(all_episodes, "train", test_fraction, split_seed)
    test = split_robotmind_episodes(all_episodes, "test", test_fraction, split_seed)
    train_ids = {episode.episode_uid for episode in train}
    test_ids = {episode.episode_uid for episode in test}
    print(
        f"embodiment={dataset.schema.name} root={dataset.data_root} total={len(all_episodes)} "
        f"train={len(train)} test={len(test)} overlap={len(train_ids & test_ids)} "
        f"schema_dim={dataset.schema.state_action_dim} keys="
        f"{[name for name, _ in dataset.schema.state_action_keys]}",
        flush=True,
    )
    if len(train) + len(test) != len(all_episodes) or train_ids & test_ids:
        raise RuntimeError("RobotMind deterministic train/test split is not exhaustive and disjoint")

    inspect_episodes = all_episodes if args.inspect_all else dataset.episodes[: args.samples]
    failures = Counter()
    valid = 0
    min_frames = None
    max_frames = 0
    for position, episode in enumerate(inspect_episodes, start=1):
        try:
            info = dataset._inspect_episode(episode)
            valid += 1
            min_frames = info.total_frames if min_frames is None else min(min_frames, info.total_frames)
            max_frames = max(max_frames, info.total_frames)
        except Exception as error:
            failures[f"{type(error).__name__}:{error}"] += 1
        if position % max(1, args.progress_every) == 0 or position == len(inspect_episodes):
            print(
                f"inspect={position}/{len(inspect_episodes)} "
                f"percent={100.0 * position / max(1, len(inspect_episodes)):.2f}% "
                f"valid={valid} failed={sum(failures.values())}",
                flush=True,
            )
    print(
        f"inspection_done valid={valid} failed={sum(failures.values())} "
        f"min_frames={min_frames} max_frames={max_frames}",
        flush=True,
    )
    for reason, count in failures.most_common(20):
        print(f"failure count={count} reason={reason}", flush=True)
    if failures:
        raise RuntimeError("RobotMind metadata audit found invalid episodes")

    sample_count = min(int(args.samples), len(dataset))
    seen_episodes = set()
    for index in range(sample_count):
        sample = dataset[index]
        validate_sample(sample)
        seen_episodes.add(sample["episode_id"])
        print(
            f"sample={index + 1}/{sample_count} episode={sample['episode_id']} "
            f"frames={tuple(sample['video_frames'].shape)} state={tuple(sample['initial_state'].shape)} "
            f"action={tuple(sample['action_sequence'].shape)} indices="
            f"{sample['frame_indices'][0]}..{sample['frame_indices'][-1]} "
            f"lang={dataset.instruction_lang} text={sample['t5_input_message'].strip()!r}",
            flush=True,
        )
    print(f"PASS sampled={sample_count} unique_episodes={len(seen_episodes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
