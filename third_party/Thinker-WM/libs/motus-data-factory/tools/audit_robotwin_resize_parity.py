#!/usr/bin/env python3
"""Compare RoboTwin Reader resizing with the frozen legacy pad algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from motus_data.repository import decode_video_frame_map_ffmpeg
from motus_data.source_readers import RobotwinEpisodeReader


def _legacy_resize_with_padding(
    native_frames: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    frames = native_frames.permute(0, 3, 1, 2).float().div(255.0)
    frames_np = (
        frames.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    target_h, target_w = size
    resized = []
    for frame in frames_np:
        height, width = frame.shape[:2]
        scale = min(target_h / height, target_w / width)
        new_h = max(1, int(height * scale))
        new_w = max(1, int(width * scale))
        resized_frame = cv2.resize(frame, (new_w, new_h))
        padded = np.zeros(
            (target_h, target_w, frame.shape[2]),
            dtype=frame.dtype,
        )
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[
            y_offset : y_offset + new_h,
            x_offset : x_offset + new_w,
        ] = resized_frame
        resized.append(padded)
    return torch.from_numpy(np.stack(resized, axis=0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="RoboTwin MP4 path; repeat for multiple robot embodiments.",
    )
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    indices = tuple(int(value) for value in args.indices)
    target_size = (int(args.height), int(args.width))
    failed = False

    for raw_path in args.video:
        path = Path(raw_path).expanduser().resolve()
        decoded = decode_video_frame_map_ffmpeg(str(path), indices)
        native = torch.from_numpy(
            np.stack([np.asarray(decoded[index]) for index in indices])
        ).to(torch.uint8)

        def cached_decoder(_path: str, requested: list[int]):
            return {index: decoded[index] for index in requested}

        reader = RobotwinEpisodeReader(video_decoder=cached_decoder)
        actual = reader.read_video(
            {
                "source_dataset": "robotwin",
                "video": {
                    "locator": {
                        "storage_backend": "file",
                        "uri": str(path),
                    }
                },
            },
            indices,
            output_format="rgb_uint8_hwc",
            video_size=target_size,
        )
        expected = _legacy_resize_with_padding(native, target_size)
        difference = (
            actual.to(torch.int16) - expected.to(torch.int16)
        ).abs()
        mismatch_pixels = int(torch.count_nonzero(difference).item())
        max_abs_diff = int(difference.max().item())
        print(
            f"video={path} native_shape={tuple(native.shape)} "
            f"output_shape={tuple(actual.shape)} "
            f"mismatch_values={mismatch_pixels} max_abs_diff={max_abs_diff}"
        )
        failed = failed or mismatch_pixels != 0

    if failed:
        print("ROBOTWIN_RESIZE_PARITY_FAILED")
        return 1
    print("ROBOTWIN_RESIZE_PARITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
