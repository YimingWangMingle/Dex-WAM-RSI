#!/usr/bin/env python3
"""Decode one exact multimodal selection from each source-family smoke Catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from motus_data.catalog.sqlite_index import default_catalog_index_path
from motus_data.repository import EpisodeRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=(
            "/data/jiyuan.wang/EpisodeCatalogs/_20260727/"
            "ten_source_smoke"
        ),
    )
    return parser.parse_args()


def main() -> int:
    root = Path(parse_args().root)
    catalogs = [
        root / "agibot.smoke1.episode_catalog.jsonl",
        root / "robotwin_aloha.smoke1.episode_catalog.jsonl",
        root / "robotmind_tianyi.smoke1.episode_catalog.jsonl",
    ]
    results = []
    for catalog in catalogs:
        repository = EpisodeRepository(
            catalog,
            index_path=default_catalog_index_path(catalog),
            verify_sha256=False,
        )
        try:
            episode_id = next(repository.iter_episode_ids())
            episode = repository.get_episode(episode_id)
            sample = episode.read_selection(
                {
                    "episode_id": episode_id,
                    "video_frame_indices": [0, 1],
                    "state_indices": [0],
                    "action_indices": [0, 1],
                    "text_indices": [0],
                },
                video_output_format="rgb_float32_chw",
                video_size=(64, 64),
            )
            tensors = (sample.video, sample.state, sample.action)
            if not all(torch.isfinite(value).all().item() for value in tensors):
                raise RuntimeError(f"non-finite tensor for {episode_id}")
            result = {
                "catalog": str(catalog),
                "episode_id": episode_id,
                "source_dataset": sample.info.source_dataset,
                "source_child": sample.info.source_child,
                "robot_name": sample.info.robot_name,
                "video_shape": list(sample.video.shape),
                "state_shape": list(sample.state.shape),
                "action_shape": list(sample.action.shape),
                "text": [item.text for item in sample.text],
            }
            results.append(result)
            print(
                "SOURCE_FAMILY_DECODE_OK "
                + json.dumps(result, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        finally:
            repository.close()
    if {result["source_dataset"] for result in results} != {
        "agibot",
        "robotwin",
        "robotmind",
    }:
        raise RuntimeError("source-family smoke did not cover all three families")
    print("THREE_FAMILY_CATALOG_SMOKE_AUDIT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
