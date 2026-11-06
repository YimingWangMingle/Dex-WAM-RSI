#!/usr/bin/env python3
"""Read exact multimodal selections from one LIBERO Catalog Episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data.libero import LIBERO_CAMERA_MODES, LiberoParquetEpisodeReader
from motus_data.repository import EpisodeRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--index")
    parser.add_argument("--episode-id")
    parser.add_argument(
        "--camera-mode",
        choices=LIBERO_CAMERA_MODES,
        default="main",
    )
    parser.add_argument("--video-height", type=int)
    parser.add_argument("--video-width", type=int)
    return parser.parse_args()


def _tensor_summary(value: torch.Tensor) -> dict[str, object]:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    result: dict[str, object] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if value.numel() and value.is_floating_point():
        result.update(
            {
                "finite": bool(torch.isfinite(value).all()),
                "min": float(value.min()),
                "max": float(value.max()),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    if (args.video_height is None) != (args.video_width is None):
        raise ValueError(
            "--video-height and --video-width must be provided together"
        )
    video_size = (
        None
        if args.video_height is None
        else (int(args.video_height), int(args.video_width))
    )
    repository = EpisodeRepository(
        args.catalog,
        index_path=args.index,
        readers={
            "libero": LiberoParquetEpisodeReader(
                camera_mode=args.camera_mode
            )
        },
    )
    try:
        episode_id = args.episode_id or next(repository.iter_episode_ids())
        episode = repository.get_episode(episode_id)
        video_count = episode.info.video_count
        numeric_count = min(
            episode.info.state_count,
            episode.info.action_count,
        )
        video_indices = sorted({0, video_count // 2, video_count - 1})
        numeric_indices = sorted({0, numeric_count // 2, numeric_count - 1})
        selection = episode.read_selection(
            {
                "episode_id": episode_id,
                "video_frame_indices": video_indices,
                "state_indices": numeric_indices,
                "action_indices": numeric_indices,
                "text_indices": [0],
            },
            video_output_format="rgb_float32_chw",
            video_size=video_size,
        )
        report = {
            "status": "PASS",
            "catalog_id": repository.catalog_id,
            "episode_id": episode_id,
            "source": episode.info.source_dataset,
            "child": episode.info.source_child,
            "camera_mode": args.camera_mode,
            "video_indices": video_indices,
            "state_indices": numeric_indices,
            "action_indices": numeric_indices,
            "video": _tensor_summary(selection.video),
            "state": _tensor_summary(selection.state),
            "action": _tensor_summary(selection.action),
            "texts": [item.text for item in selection.text],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
