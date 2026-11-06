#!/usr/bin/env python3
"""Decode one exact multimodal selection from each requested Catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from motus_data.catalog.sqlite_index import default_catalog_index_path
from motus_data.repository import EpisodeRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", action="append", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--video-height", type=int, default=64)
    parser.add_argument("--video-width", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalogs = [Path(value) for value in args.catalog]
    if args.expected_count is not None and len(catalogs) != args.expected_count:
        raise ValueError(
            f"catalog count={len(catalogs)} expected={args.expected_count}"
        )
    seen_children = set()
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
                video_size=(args.video_height, args.video_width),
            )
            tensors = (sample.video, sample.state, sample.action)
            if not all(torch.isfinite(value).all().item() for value in tensors):
                raise RuntimeError(f"non-finite tensor for {episode_id}")
            child = sample.info.source_child
            if child in seen_children:
                raise RuntimeError(f"duplicate source_child={child!r}")
            seen_children.add(child)
            result = {
                "catalog": str(catalog),
                "episode_id": episode_id,
                "source_dataset": sample.info.source_dataset,
                "source_child": child,
                "robot_name": sample.info.robot_name,
                "video_shape": list(sample.video.shape),
                "state_shape": list(sample.state.shape),
                "action_shape": list(sample.action.shape),
                "text_kind": sample.text[0].kind,
                "text": sample.text[0].text,
            }
            results.append(result)
            print(
                "CATALOG_DECODE_OK "
                + json.dumps(result, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        finally:
            repository.close()
    print(
        "CATALOG_SET_AUDIT_OK "
        + json.dumps(
            {
                "catalog_count": len(results),
                "source_children": sorted(seen_children),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
