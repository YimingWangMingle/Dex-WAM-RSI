#!/usr/bin/env python3
"""Real-data acceptance smoke for the Catalog-backed EpisodeRepository."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from motus_data import EpisodeMemoryLimitError, EpisodeRepository


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--video-frame-a", type=int, default=0)
    parser.add_argument("--video-frame-b", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    repository = EpisodeRepository(Path(args.catalog))
    load_seconds = time.monotonic() - started
    infos = list(repository.iter_episode_infos(split=args.split))
    if not infos:
        raise RuntimeError(f"no Episode matches split={args.split!r}")
    episode_id = args.episode_id or infos[0].episode_id
    episode = repository.get_episode(episode_id)
    print(
        "REPOSITORY_READY "
        + json.dumps(
            {
                "catalog_id": repository.catalog_id,
                "protocol_version": repository.protocol_version,
                "catalog_records": len(repository),
                "matching_infos": len(infos),
                "catalog_sha256": repository.catalog_sha256,
                "load_seconds": load_seconds,
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    frame_a = int(args.video_frame_a)
    frame_b = int(args.video_frame_b)
    if frame_a == frame_b:
        raise ValueError("video-frame-a and video-frame-b must differ")
    if not 0 <= frame_a < episode.info.video_count:
        raise ValueError(f"video-frame-a={frame_a} is out of bounds")
    if not 0 <= frame_b < episode.info.video_count:
        raise ValueError(f"video-frame-b={frame_b} is out of bounds")
    last_state = episode.info.state_count - 1
    last_action = episode.info.action_count - 1
    last_text = episode.info.text_count - 1
    selection = {
        "episode_id": episode.episode_id,
        "video_frame_indices": [frame_a, frame_b, frame_a],
        "state_indices": [0, last_state, 0],
        "action_indices": [0, last_action, 0],
        "text_indices": [0, last_text, 0],
    }
    selected = episode.read_selection(selection)
    if not torch.equal(selected.video[0], selected.video[2]):
        raise RuntimeError("duplicate video index did not preserve equal frames")
    if not torch.equal(selected.state[0], selected.state[2]):
        raise RuntimeError("duplicate state index did not preserve equal rows")
    if not torch.equal(selected.action[0], selected.action[2]):
        raise RuntimeError("duplicate action index did not preserve equal rows")
    if [item.index for item in selected.text] != [0, last_text, 0]:
        raise RuntimeError("text index order/repetition was not preserved")
    if not torch.isfinite(selected.state).all() or not torch.isfinite(
        selected.action
    ).all():
        raise RuntimeError("state/action selection contains non-finite values")
    print(
        "INDEXED_READ_OK "
        + json.dumps(
            {
                "episode_id": episode.episode_id,
                "episode_split": episode.info.split,
                "video_indices": list(selected.video_frame_indices),
                "video_shape": list(selected.video.shape),
                "video_dtype": str(selected.video.dtype),
                "state_indices": list(selected.state_indices),
                "state_shape": list(selected.state.shape),
                "action_indices": list(selected.action_indices),
                "action_shape": list(selected.action.shape),
                "text_indices": list(selected.text_indices),
                "selected_text_kinds": [item.kind for item in selected.text],
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    complete = episode.materialize(modalities=("state", "action", "text"))
    if tuple(complete.state.shape) != (episode.info.state_count, 26):
        raise RuntimeError(f"unexpected complete state shape={complete.state.shape}")
    if tuple(complete.action.shape) != (episode.info.action_count, 26):
        raise RuntimeError(f"unexpected complete action shape={complete.action.shape}")
    if len(complete.text) != episode.info.text_count:
        raise RuntimeError("complete text count mismatch")
    if not torch.equal(
        complete.state[torch.tensor(selection["state_indices"])], selected.state
    ):
        raise RuntimeError("complete state and indexed state differ")
    if not torch.equal(
        complete.action[torch.tensor(selection["action_indices"])], selected.action
    ):
        raise RuntimeError("complete action and indexed action differ")
    print(
        "COMPLETE_NONVIDEO_OK "
        + json.dumps(
            {
                "state_shape": list(complete.state.shape),
                "action_shape": list(complete.action.shape),
                "text_count": len(complete.text),
                "state_dtype": str(complete.state.dtype),
                "action_dtype": str(complete.action.dtype),
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    try:
        episode.materialize_video(max_bytes=1)
    except EpisodeMemoryLimitError as error:
        memory_guard = str(error)
    else:
        raise RuntimeError("complete-video memory guard did not reject max_bytes=1")
    print(
        "VIDEO_MEMORY_GUARD_OK "
        + json.dumps({"error": memory_guard, "rss_mb": _rss_mb()}),
        flush=True,
    )
    print(
        "EPISODE_REPOSITORY_OK "
        + json.dumps(
            {
                "episode_id": episode.episode_id,
                "video_count": episode.info.video_count,
                "state_count": episode.info.state_count,
                "action_count": episode.info.action_count,
                "text_count": episode.info.text_count,
                "wall_seconds": time.monotonic() - started,
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
