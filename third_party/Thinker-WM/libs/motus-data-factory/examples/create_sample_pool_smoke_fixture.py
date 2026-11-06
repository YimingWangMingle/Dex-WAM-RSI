#!/usr/bin/env python3
"""Create an explicit smoke-only SamplePool for one known Catalog Episode."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--index-path")
    parser.add_argument(
        "--trust-index-binding",
        action="store_true",
        help="Skip a full Catalog SHA256 scan after loading a verified offset index.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-pool-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--starts", default="0,16,32,48")
    parser.add_argument("--video-count", type=int, default=49)
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--action-start-offset", type=int, default=1)
    parser.add_argument("--action-count", type=int, default=48)
    parser.add_argument("--text-indices", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _integers(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated integer")
    return values


def main() -> int:
    args = parse_args()
    if args.trust_index_binding and not args.index_path:
        raise ValueError("--trust-index-binding requires --index-path")
    repository = EpisodeRepository(
        args.catalog,
        index_path=args.index_path,
        verify_sha256=not args.trust_index_binding,
    )
    episode = repository.get_episode(args.episode_id)
    starts = _integers(args.starts)
    text_indices = _integers(args.text_indices)
    selections = []
    for start in starts:
        selection = {
            "episode_id": episode.episode_id,
            "video_frame_indices": list(
                range(int(start), int(start) + int(args.video_count))
            ),
            "state_indices": [int(start) + int(args.state_offset)],
            "action_indices": list(
                range(
                    int(start) + int(args.action_start_offset),
                    int(start)
                    + int(args.action_start_offset)
                    + int(args.action_count),
                )
            ),
            "text_indices": list(text_indices),
        }
        validate_sample_selection_v1(selection)
        episode._validated_selection_indices(selection)
        selections.append(selection)

    is_v1 = (
        repository.protocol_version == MOTUS_EPISODE_CATALOG_METADATA_V1
    )
    generation_meta = {
        "smoke_fixture_only": True,
        "warning": "not a production model-owned SamplePool",
        "starts": starts,
        "video_count": int(args.video_count),
        "state_offset": int(args.state_offset),
        "action_start_offset": int(args.action_start_offset),
        "action_count": int(args.action_count),
        "text_indices": text_indices,
    }
    if is_v1:
        header = {
            "schema_version": MOTUS_SAMPLE_POOL_HEADER_V1,
            "record_type": "sample_pool_header",
            "record_schema_version": MOTUS_SAMPLE_SELECTION_V1,
            "sample_pool_id": str(args.sample_pool_id),
            "catalog_id": repository.catalog_id,
            "source_dataset": episode.info.source_dataset,
            "source_child": episode.info.source_child,
            "split": episode.info.split,
            "record_count": len(selections),
            "generation_metadata": generation_meta,
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
        "SMOKE_SAMPLE_POOL_OK "
        + json.dumps(
            {
                "output": str(output),
                "catalog_id": repository.catalog_id,
                "episode_id": episode.episode_id,
                "record_count": len(selections),
                "starts": starts,
                "repository_indexed": repository.indexed,
                "catalog_sha256_verified": not args.trust_index_binding,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
