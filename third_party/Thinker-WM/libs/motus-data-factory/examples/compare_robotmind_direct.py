#!/usr/bin/env python3
"""Compare direct RobotMind windows with a few legacy Manifest records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
from omegaconf import OmegaConf

from motus_data import build_dataset
from motus_data.datasets.manifest import MotusManifestDataset
from motus_data.datasets.robotmind import RobotMindDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-scan", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.dataset.split = "train"
    config.dataset.test_fraction = 0.0
    dataset = build_dataset(config)
    if not isinstance(dataset, RobotMindDataset):
        raise TypeError("compare_robotmind_direct.py requires one dataset.type=robotmind child")

    records = []
    seen_paths = set()
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        for scanned, line in enumerate(handle, start=1):
            if scanned > int(args.max_scan):
                break
            record = json.loads(line)
            if str((record.get("robot") or {}).get("name")) != dataset.robot_name:
                continue
            video = record.get("video") or {}
            path = str(video.get("uri") or video.get("path") or "")
            indices = [int(value) for value in video.get("frame_indices") or []]
            if path not in dataset._path_to_index or len(indices) != dataset.num_video_frames + 1:
                continue
            if any(
                right - left != dataset.video_frame_step
                for left, right in zip(indices, indices[1:])
            ):
                continue
            if path in seen_paths:
                continue
            seen_paths.add(path)
            records.append(record)
            if len(records) >= int(args.samples):
                break
    if len(records) < int(args.samples):
        raise RuntimeError(
            f"found only {len(records)} comparable records after scanning up to {args.max_scan}"
        )

    for position, record in enumerate(records, start=1):
        video = record["video"]
        path = str(video.get("uri") or video.get("path"))
        start = int(video["frame_indices"][0])
        direct = dataset.load_window(dataset.episode_index_for_path(path), start)
        with tempfile.TemporaryDirectory(prefix="motus_robotmind_parity_") as temp_dir:
            manifest_path = Path(temp_dir) / "one.jsonl"
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            legacy = MotusManifestDataset(
                manifest_path=str(manifest_path),
                split=str(record.get("split") or "train"),
                video_size=dataset.video_size,
                num_video_frames=dataset.num_video_frames,
                video_action_freq_ratio=dataset.action_video_fps_ratio,
                action_dim=dataset.schema.state_action_dim,
                state_dim=dataset.schema.state_action_dim,
                source_dataset="robotmind",
                robot_name=dataset.robot_name,
                instruction_lang=dataset.instruction_lang,
                max_decode_retries=1,
            )
            fixed = legacy[0]
            legacy.close()

        for key in ("first_frame", "video_frames", "initial_state", "action_sequence"):
            if not torch.equal(direct[key], fixed[key]):
                difference = float((direct[key] - fixed[key]).abs().max())
                raise AssertionError(
                    f"parity failed record={record.get('id')} key={key} max_abs_diff={difference}"
                )
        for key in ("t5_input_message", "vlm_input_message", "robot_name"):
            if direct[key] != fixed[key]:
                raise AssertionError(
                    f"parity failed record={record.get('id')} key={key}: "
                    f"direct={direct[key]!r} legacy={fixed[key]!r}"
                )
        print(
            f"PASS sample={position}/{len(records)} id={record.get('id')} "
            f"episode={direct['episode_id']} start={start} "
            f"video={tuple(direct['video_frames'].shape)} state={tuple(direct['initial_state'].shape)} "
            f"action={tuple(direct['action_sequence'].shape)} max_abs_diff=0",
            flush=True,
        )
    print(f"PASS robotmind_direct_parity samples={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
