#!/usr/bin/env python3
"""Read-only modality diagnostics for one EgoVerse Zarr Episode."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motus_data.egoverse import (  # noqa: E402
    decode_egoverse_annotation,
    get_egoverse_profile,
    open_egoverse_zarr_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    return parser.parse_args()


def _encoded_bytes(value: Any) -> bytes:
    if isinstance(value, np.void):
        value = value.item()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise ValueError(f"expected encoded bytes, found {type(value)!r}")
    return value


def main() -> int:
    args = parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    profile = get_egoverse_profile(args.profile)
    path = Path(args.zarr_path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    group = open_egoverse_zarr_group(str(path))
    attributes = dict(group.attrs)
    total_frames = int(attributes.get("total_frames", 0))
    print(
        "EGOVERSE_DIAG_START "
        f"path={path} profile={profile.name} total_frames={total_frames}",
        flush=True,
    )
    errors: list[dict[str, Any]] = []

    for key in sorted(set(profile.state_keys + profile.action_keys)):
        try:
            array = group[key]
            for start in range(0, total_frames, args.chunk_rows):
                end = min(start + args.chunk_rows, total_frames)
                try:
                    values = np.asarray(array[start:end])
                except Exception as error:
                    raise ValueError(
                        f"cannot decode rows=[{start},{end}): {error}"
                    ) from error
                if not np.isfinite(values).all():
                    raise ValueError(f"NaN/Inf rows=[{start},{end})")
            print(
                f"EGOVERSE_DIAG_NUMERIC_OK key={key} "
                f"shape={tuple(array.shape)} dtype={array.dtype}",
                flush=True,
            )
        except Exception as error:
            item = {
                "modality": "numeric",
                "key": key,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            errors.append(item)
            print(
                "EGOVERSE_DIAG_ERROR "
                + json.dumps(item, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

    try:
        video = group[profile.video_key]
        shapes = []
        for index in sorted(set((0, total_frames - 1))):
            values = np.asarray(
                video[index : index + 1],
                dtype=object,
            ).reshape(-1)
            with Image.open(io.BytesIO(_encoded_bytes(values[0]))) as image:
                shapes.append((index, image.convert("RGB").size))
        print(
            f"EGOVERSE_DIAG_VIDEO_OK key={profile.video_key} "
            f"shape={tuple(video.shape)} decoded={shapes}",
            flush=True,
        )
    except Exception as error:
        item = {
            "modality": "video",
            "key": profile.video_key,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        errors.append(item)
        print(
            "EGOVERSE_DIAG_ERROR "
            + json.dumps(item, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    if "annotations" in group:
        try:
            annotations = np.asarray(
                group["annotations"][:],
                dtype=object,
            ).reshape(-1)
            for value in annotations:
                decode_egoverse_annotation(value)
            print(
                f"EGOVERSE_DIAG_TEXT_OK annotations={len(annotations)}",
                flush=True,
            )
        except Exception as error:
            item = {
                "modality": "text",
                "key": "annotations",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            errors.append(item)
            print(
                "EGOVERSE_DIAG_ERROR "
                + json.dumps(item, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

    print(
        f"EGOVERSE_DIAG_DONE errors={len(errors)}",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
