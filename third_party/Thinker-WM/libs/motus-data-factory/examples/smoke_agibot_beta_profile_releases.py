#!/usr/bin/env python3
"""Smoke native Agibot Beta profile releases against real storage."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from motus_data import open_dataset
from motus_data.repository import EpisodeRepositoryError


PROFILES = (
    ("agibot_beta_parallel_gripper", 14),
    ("agibot_beta_dexterous_hand", 24),
)
EXCLUDED_DEXTEROUS_EPISODE = "agibot_beta:554:913085"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes-per-profile", type=int, default=3)
    return parser.parse_args()


def _positions(length: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("episodes-per-profile must be positive")
    if length <= 0:
        raise ValueError("profile Catalog is empty")
    if count == 1:
        return [0]
    return sorted(
        {
            round(index * (length - 1) / (count - 1))
            for index in range(count)
        }
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"format": "agibot_beta_profile_release_smoke_v1", "profiles": {}}
    for dataset, expected_dimension in PROFILES:
        repository = open_dataset(dataset, verify_sha256=False)
        try:
            episode_ids = list(repository.iter_episode_ids())
            selected_ids = [
                episode_ids[position]
                for position in _positions(
                    len(episode_ids), int(args.episodes_per_profile)
                )
            ]
            rows = []
            for episode_id in selected_ids:
                episode = repository.get_episode(episode_id)
                indices = sorted(
                    {
                        0,
                        episode.info.state_count // 2,
                        episode.info.state_count - 1,
                    }
                )
                state = episode.read_state(indices)
                action = episode.read_action(indices)
                expected_shape = (len(indices), expected_dimension)
                if tuple(state.shape) != expected_shape:
                    raise ValueError(
                        f"state shape mismatch for {episode_id}: {state.shape}"
                    )
                if tuple(action.shape) != expected_shape:
                    raise ValueError(
                        f"action shape mismatch for {episode_id}: {action.shape}"
                    )
                if not torch.isfinite(state).all() or not torch.isfinite(action).all():
                    raise ValueError(f"non-finite profile output for {episode_id}")
                maximum = max(
                    float(state.abs().max()), float(action.abs().max())
                )
                if not math.isfinite(maximum) or maximum > 20.0:
                    raise ValueError(
                        f"profile output exceeds guard for {episode_id}: {maximum}"
                    )
                rows.append(
                    {
                        "episode_id": episode_id,
                        "indices": indices,
                        "state_shape": list(state.shape),
                        "action_shape": list(action.shape),
                        "maximum_absolute_value": maximum,
                        "robot_name": episode.info.robot_name,
                        "source_child": episode.info.source_child,
                    }
                )
            profile_report = {
                "catalog_id": repository.catalog_id,
                "episodes": len(repository),
                "expected_dimension": expected_dimension,
                "samples": rows,
            }
            if dataset == "agibot_beta_dexterous_hand":
                try:
                    repository.get_episode(
                        EXCLUDED_DEXTEROUS_EPISODE
                    ).read_state([0])
                except EpisodeRepositoryError as error:
                    profile_report["quality_exclusion"] = {
                        "episode_id": EXCLUDED_DEXTEROUS_EPISODE,
                        "rejected": True,
                        "error": str(error),
                    }
                else:
                    raise ValueError("known numeric anomaly was not rejected")
            report["profiles"][dataset] = profile_report
        finally:
            repository.close()
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print("PROFILE_RELEASE_SMOKE_DONE " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
