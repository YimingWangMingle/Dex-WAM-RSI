#!/usr/bin/env python3
"""Compare a direct Agibot Dataset window with the fixed Manifest adapter."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from motus_data.datasets.agibot_native import AgibotDataset
from motus_data.datasets.agibot_common import read_agibot_eef_actions
from motus_data.datasets.manifest import MotusManifestDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-folders", nargs="+", required=True)
    parser.add_argument("--data-root-folders", nargs="+", required=True)
    parser.add_argument("--dataset-names", nargs="+", default=["agibot-world"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    direct = AgibotDataset(
        args.annotation_folders,
        args.data_root_folders,
        args.dataset_names,
        split=args.split,
        video_size=(args.video_height, args.video_width),
        num_video_frames=args.num_video_frames,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_decode_retries=1,
    )
    episode = direct.episodes[args.episode_offset]
    eef_actions = read_agibot_eef_actions(episode["proprio_path"])
    frame_numbers = int(eef_actions.shape[0] // int(episode["action_video_fps_ratio"]))
    effective_start, effective_end = episode["effective_frames"]
    max_start = max(
        min(effective_end, frame_numbers - (args.num_video_frames + 1)), effective_start
    )
    start = args.start_frame
    if start is None:
        start = effective_start + (max_start - effective_start) // 2
    sample = direct.load_window(args.episode_offset, start)
    frame_indices = list(sample["frame_indices"])
    frame_mask = list(sample["frame_mask"])
    record = {
        "schema_version": "motus_manifest_v1",
        "id": sample["manifest_id"],
        "source_dataset": "agibot",
        "split": args.split,
        "task": episode.get("task_name", ""),
        "text": {
            "t5": sample["t5_input_message"],
            "vlm": sample["vlm_input_message"],
        },
        "video": {
            "storage_backend": "file",
            "uri": episode["video_path"],
            "frame_indices": frame_indices,
        },
        "robot": {
            "name": "agibot-a2-d",
            "state_dim": 26,
            "action_dim": 26,
            "action_policy": "real",
        },
        "state": {
            "storage_backend": "agibot_proprio_hdf5",
            "uri": episode["proprio_path"],
            "index": start,
        },
        "action": {
            "storage_backend": "agibot_proprio_hdf5",
            "uri": episode["proprio_path"],
            "frame_indices": frame_indices,
            "frame_mask": frame_mask,
            "action_video_fps_ratio": episode["action_video_fps_ratio"],
        },
        "loss": {"video_weight": 1.0, "action_weight": 1.0},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "window.jsonl"
        manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        fixed = MotusManifestDataset(
            str(manifest_path),
            split=args.split,
            video_size=(args.video_height, args.video_width),
            num_video_frames=args.num_video_frames,
            action_dim=26,
            state_dim=26,
            source_dataset="agibot",
        )
        fixed_sample = fixed[0]
        comparisons = {}
        for key in ("first_frame", "video_frames", "initial_state", "action_sequence"):
            left = sample[key]
            right = fixed_sample[key]
            comparisons[key] = {
                "shape": list(left.shape),
                "exact": bool(torch.equal(left, right)),
                "max_abs_diff": float((left - right).abs().max()),
            }
        text_equal = (
            sample["t5_input_message"] == fixed_sample["t5_input_message"]
            and sample["vlm_input_message"] == fixed_sample["vlm_input_message"]
        )
        passed = text_equal and all(
            item["max_abs_diff"] <= 1e-6 for item in comparisons.values()
        )
        print(
            json.dumps(
                {
                    "passed": passed,
                    "episode_uid": episode["episode_uid"],
                    "start_frame": start,
                    "frame_indices": frame_indices,
                    "text_equal": text_equal,
                    "tensors": comparisons,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        fixed.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
