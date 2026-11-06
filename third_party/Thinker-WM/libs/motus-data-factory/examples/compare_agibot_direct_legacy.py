#!/usr/bin/env python3
"""Compare the direct data-factory Agibot adapter with the pre-Manifest code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from motus_data.datasets.agibot_common import read_agibot_eef_actions
from motus_data.datasets.agibot_native import AgibotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--annotation-folders", nargs="+", required=True)
    parser.add_argument("--data-root-folders", nargs="+", required=True)
    parser.add_argument("--dataset-names", nargs="+", default=["agibot-world"])
    parser.add_argument("--episode-offsets", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    sys.path.insert(0, str(legacy_root))
    from data.agibot_dr_su import agibot_dataset as legacy_agibot

    direct = AgibotDataset(
        args.annotation_folders,
        args.data_root_folders,
        args.dataset_names,
        split="train",
        video_size=(args.video_height, args.video_width),
        num_video_frames=args.num_video_frames,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_decode_retries=1,
    )
    reports = []
    for episode_offset in args.episode_offsets:
        episode = direct.episodes[int(episode_offset)]
        ratio = int(episode["action_video_fps_ratio"])
        eef_actions = read_agibot_eef_actions(episode["proprio_path"])
        frame_numbers = int(eef_actions.shape[0] // ratio)
        effective_start, effective_end = episode["effective_frames"]
        max_start = max(
            min(effective_end, frame_numbers - (args.num_video_frames + 1)),
            effective_start,
        )
        start_frame = effective_start + (max_start - effective_start) // 2
        denominator = max_start + 1 - effective_start
        random_value = (
            0.0
            if denominator <= 1
            else (start_frame - effective_start + 0.25) / denominator
        )

        legacy_item = {
            "video_paths": {"agibot_head_color": episode["video_path"]},
            "proprio_path": episode["proprio_path"],
            "task_id": episode["task_id"],
            "task_name": episode["task_name"],
            "init_scene_text": episode["init_scene_text"],
            "episode_id": episode["episode_id"],
            "action_video_fps_ratio": ratio,
            "action_label": episode["action_labels"],
            "effective_frames": episode["effective_frames"],
        }
        with mock.patch.object(
            legacy_agibot.np.random,
            "random_sample",
            return_value=random_value,
        ):
            legacy_prepared = legacy_agibot.agibot_prepare(
                legacy_item,
                (args.video_height, args.video_width),
                args.num_video_frames + 1,
            )
        legacy_sample = legacy_agibot.su_motus_bridge(
            legacy_prepared, robot_name="agibot-a2-d"
        )
        direct_sample = direct.load_window(episode_offset, start_frame)

        tensor_results = {}
        for key in ("first_frame", "video_frames", "initial_state", "action_sequence"):
            old_value = legacy_sample[key]
            new_value = direct_sample[key]
            difference = old_value.detach().double() - new_value.detach().double()
            tensor_results[key] = {
                "old_shape": list(old_value.shape),
                "new_shape": list(new_value.shape),
                "old_dtype": str(old_value.dtype),
                "new_dtype": str(new_value.dtype),
                "max_abs_diff": float(difference.abs().max()),
                "allclose": bool(
                    torch.allclose(
                        old_value.detach().double(),
                        new_value.detach().double(),
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                ),
            }
        old_pil = np.asarray(legacy_sample["first_frame_pil"])
        new_pil = np.asarray(direct_sample["first_frame_pil"])
        pil_equal = bool(np.array_equal(old_pil, new_pil))
        text_equal_after_boundary_normalization = (
            legacy_sample["t5_input_message"].strip()
            == direct_sample["t5_input_message"]
            and legacy_sample["vlm_input_message"].strip()
            == direct_sample["vlm_input_message"]
        )
        passed = (
            all(result["allclose"] for result in tensor_results.values())
            and pil_equal
            and text_equal_after_boundary_normalization
            and direct_sample["frame_indices"][0] == start_frame
        )
        reports.append(
            {
                "passed": passed,
                "episode_offset": episode_offset,
                "episode_uid": episode["episode_uid"],
                "start_frame": start_frame,
                "action_video_fps_ratio": ratio,
                "text_equal_after_boundary_normalization": text_equal_after_boundary_normalization,
                "pil_equal": pil_equal,
                "tensors": tensor_results,
            }
        )

    result = {
        "passed": all(report["passed"] for report in reports),
        "comparison": "pre_manifest_agibot_vs_direct_data_factory",
        "reports": reports,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
