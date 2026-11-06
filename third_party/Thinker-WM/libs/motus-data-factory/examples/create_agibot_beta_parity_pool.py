#!/usr/bin/env python3
"""Convert the frozen direct-Dataset baseline into a parity-only SamplePool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from motus_data import EpisodeRepository
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V1,
    MOTUS_SAMPLE_POOL_HEADER_V2,
    MOTUS_SAMPLE_POOL_HEADER_V1,
    MOTUS_SAMPLE_SELECTION_V1,
    validate_sample_pool_header,
    validate_sample_selection_v1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-pool-id", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--action-video-fps-ratio", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_indices_for_legacy_start(episode, start: int) -> list[int]:
    context = [item.index for item in episode.text.items if item.kind == "episode_context"]
    segments = [
        item.index
        for item in episode.text.items
        if item.kind == "action_segment"
        and item.frame_range is not None
        and item.frame_range[0] <= int(start) < item.frame_range[1]
    ]
    if len(context) != 1 or len(segments) != 1:
        raise RuntimeError(
            f"legacy parity requires one context and one action segment at "
            f"episode={episode.episode_id} start={start}; "
            f"context={context} segments={segments}"
        )
    return [context[0], segments[0]]


def main() -> int:
    args = parse_args()
    if int(args.samples) <= 0:
        raise ValueError("--samples must be positive")
    ratio = int(args.action_video_fps_ratio)
    if ratio <= 0:
        raise ValueError("--action-video-fps-ratio must be positive")

    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("format") != "agibot_beta_direct_baseline_v1":
        raise ValueError("unsupported Agibot Beta baseline format")
    frozen = baseline.get("selections") or []
    if len(frozen) < int(args.samples):
        raise ValueError(
            f"baseline has {len(frozen)} selections, requested {args.samples}"
        )

    repository = EpisodeRepository(args.catalog)
    selections = []
    episode_infos = []
    for position, frozen_selection in enumerate(frozen[: int(args.samples)]):
        episode = repository.get_episode(str(frozen_selection["episode_id"]))
        video = [
            int(value) for value in frozen_selection["video_frame_indices"]
        ]
        if not video or video != list(range(video[0], video[0] + len(video))):
            raise ValueError(
                f"baseline selection {position} is not a contiguous legacy window"
            )
        start = video[0]
        state_start = start * ratio
        action_start = state_start + 1
        action_stop = video[-1] * ratio + 1
        selection = {
            "episode_id": episode.episode_id,
            "video_frame_indices": video,
            "state_indices": [state_start],
            "action_indices": list(range(action_start, action_stop)),
            "text_indices": _text_indices_for_legacy_start(episode, start),
        }
        validate_sample_selection_v1(selection)
        episode._validated_selection_indices(selection)
        selections.append(selection)
        episode_infos.append(episode.info)

    identities = {
        (info.source_dataset, info.source_child) for info in episode_infos
    }
    if len(identities) != 1:
        raise ValueError(
            "one parity Pool must be homogeneous by source and child"
        )
    source_dataset, source_child = next(iter(identities))
    is_v1 = (
        repository.protocol_version == MOTUS_EPISODE_CATALOG_METADATA_V1
    )
    generation_meta = {
        "legacy_parity_fixture_only": True,
        "warning": "not a production model-owned SamplePool",
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": _sha256(baseline_path),
        "action_video_fps_ratio": ratio,
    }
    if is_v1:
        splits = {info.split for info in episode_infos}
        if len(splits) != 1:
            raise ValueError("one v1 parity Pool must be split-homogeneous")
        header = {
            "schema_version": MOTUS_SAMPLE_POOL_HEADER_V1,
            "record_type": "sample_pool_header",
            "record_schema_version": MOTUS_SAMPLE_SELECTION_V1,
            "sample_pool_id": str(args.sample_pool_id),
            "catalog_id": repository.catalog_id,
            "source_dataset": source_dataset,
            "source_child": source_child,
            "record_count": len(selections),
            "generation_metadata": generation_meta,
            "split": next(iter(splits)),
        }
    else:
        header = {
            "format": MOTUS_SAMPLE_POOL_HEADER_V2,
            "pool_id": str(args.sample_pool_id),
            "catalog_id": repository.catalog_id,
            "samples": len(selections),
            "meta": generation_meta,
        }
    validate_sample_pool_header(header)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        raise FileExistsError(f"output exists; use --force to replace: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in [header, *selections]:
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
        if args.force:
            output.unlink(missing_ok=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        "PARITY_POOL_OK "
        + json.dumps(
            {
                "output": str(output.resolve()),
                "catalog_id": repository.catalog_id,
                "sample_pool_id": (
                    header.get("sample_pool_id") or header.get("pool_id")
                ),
                "record_count": len(selections),
                "selections": selections,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
