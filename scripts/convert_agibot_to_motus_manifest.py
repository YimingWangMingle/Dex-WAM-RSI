#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py


TASK_PROMPT = "Scene Description: {scene}\nTask: {task}.\n"
ACTION_PROMPT = "Action: {action}\n"
VLM_PROMPT = "Based on the following situation, please predict what the robot will do: \n"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_video_frame_idx(start_point: int, total_frames: int, frame_length: int) -> Tuple[List[int], List[bool]]:
    frame_mask = [True] * frame_length
    pad_num = min(total_frames - (start_point + frame_length), 0)
    if pad_num < 0:
        for i in range(frame_length + pad_num, frame_length):
            frame_mask[i] = False
    indices = list(range(start_point, min(total_frames, start_point + frame_length)))
    indices += [total_frames - 1] * (-pad_num)
    return indices, frame_mask


def iter_agibot_episodes(
    annotation_folders: List[str],
    data_root_folders: List[str],
    dataset_names: List[str],
) -> Iterable[Dict[str, Any]]:
    if not (len(annotation_folders) == len(data_root_folders) == len(dataset_names)):
        raise ValueError("annotation_folders, data_root_folders and dataset_names must have the same length")

    for annotation_folder, data_root_folder, dataset_name in zip(
        annotation_folders, data_root_folders, dataset_names
    ):
        if "agibot-world" not in dataset_name.lower():
            raise NotImplementedError(f"Unsupported Agibot dataset_name: {dataset_name}")
        annotation_root = Path(annotation_folder)
        for task_info_file in sorted(os.listdir(annotation_root)):
            if not task_info_file.endswith(".json"):
                continue
            task_info_path = annotation_root / task_info_file
            with task_info_path.open("r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                observation_path = (
                    Path(data_root_folder)
                    / "observations"
                    / str(item["task_id"])
                    / str(item["episode_id"])
                    / "videos_5fps"
                )
                yield {
                    "dataset_name": "agibot",
                    "dataset_source_name": dataset_name,
                    "task_info_file": str(task_info_path),
                    "video_path": str(observation_path / "head_color.mp4"),
                    "proprio_path": str(
                        Path(data_root_folder)
                        / "proprio_stats"
                        / str(item["task_id"])
                        / str(item["episode_id"])
                        / "proprio_stats.h5"
                    ),
                    "task_id": item["task_id"],
                    "task_name": item["task_name"],
                    "init_scene_text": item["init_scene_text"],
                    "episode_id": item["episode_id"],
                    "action_video_fps_ratio": item["action_video_fps_ratio"],
                    "action_label": item["label_info"]["flatten_action_config"],
                    "effective_frames": item["effective_frames"],
                }


def split_episodes(episodes: List[Dict[str, Any]], split: str, val_num: int) -> List[Dict[str, Any]]:
    val_list = episodes[:val_num]
    train_list = episodes[val_num:]
    if split == "val":
        selected = val_list
    elif split == "train":
        selected = train_list
    elif split == "all":
        selected = episodes
    else:
        raise ValueError(f"Unsupported split: {split}")
    return sorted(selected, key=lambda item: (int(item["task_id"]), int(item["episode_id"])))


def action_text_for_start(action_label: Dict[str, Any], start_point: int) -> str:
    action_info = action_label.get(str(start_point))
    text = ""
    if action_info is not None:
        if clean_text(action_info.get("action_text")):
            text = clean_text(action_info.get("action_text"))
        elif clean_text(action_info.get("skill")):
            text = clean_text(action_info.get("skill")) + "."
    return ACTION_PROMPT.format(action=text)


def h5_timesteps(proprio_path: str) -> int:
    with h5py.File(proprio_path, "r") as f:
        return int(f["state/end/position"].shape[0])


def convert_episode(
    episode: Dict[str, Any],
    split: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    video_path = episode["video_path"]
    proprio_path = episode["proprio_path"]
    if args.check_files and not Path(video_path).exists():
        return [], "missing_video"
    if args.check_files and not Path(proprio_path).exists():
        return [], "missing_proprio"

    try:
        timesteps = h5_timesteps(proprio_path)
    except Exception:
        return [], "bad_proprio"

    ratio = int(episode["action_video_fps_ratio"])
    if ratio <= 0:
        return [], "bad_ratio"
    frame_numbers = int(timesteps // ratio)
    frame_length = int(args.num_video_frames) + 1
    if frame_numbers <= 0:
        return [], "no_frames"

    effective_frames = episode["effective_frames"]
    sampling_start = int(effective_frames[0])
    sampling_end = max(min(int(effective_frames[1]), frame_numbers - frame_length), sampling_start)
    if sampling_end < sampling_start:
        return [], "bad_effective_frames"

    starts = range(sampling_start, sampling_end + 1, int(args.window_stride))
    records = []
    task_text = TASK_PROMPT.format(
        scene=clean_text(episode["init_scene_text"]),
        task=clean_text(episode["task_name"]),
    )
    windows_written = 0
    for start_point in starts:
        if args.max_windows_per_episode is not None and windows_written >= args.max_windows_per_episode:
            break
        frame_indices, frame_mask = get_video_frame_idx(start_point, frame_numbers, frame_length)
        action_text = action_text_for_start(episode["action_label"], start_point)
        t5_text = f"{task_text}{action_text}"
        vlm_text = f"{VLM_PROMPT}{task_text}{action_text}"
        record_id = (
            f"{args.source_dataset}:{split}:{episode['task_id']}:{episode['episode_id']}:"
            f"window:{int(start_point):06d}"
        )
        records.append(
            {
                "schema_version": "motus_manifest_v1",
                "id": record_id,
                "source_dataset": args.source_dataset,
                "split": split,
                "task": clean_text(episode["task_name"]),
                "subtask": "robot_action_prediction",
                "text": {
                    "t5": t5_text,
                    "vlm": vlm_text,
                    "target": clean_text(action_text),
                },
                "video": {
                    "storage_backend": "file",
                    "uri": video_path,
                    "frame_indices": frame_indices,
                    "view": "agibot_head_color",
                },
                "robot": {
                    "name": args.robot_name,
                    "state_dim": int(args.state_dim),
                    "action_dim": int(args.action_dim),
                    "action_policy": "real",
                },
                "state": {
                    "storage_backend": "agibot_proprio_hdf5",
                    "uri": proprio_path,
                    "key": "state/end+state/effector",
                    "index": int(start_point),
                },
                "action": {
                    "storage_backend": "agibot_proprio_hdf5",
                    "uri": proprio_path,
                    "key": "state/end+state/effector",
                    "frame_indices": frame_indices,
                    "frame_mask": frame_mask,
                    "action_video_fps_ratio": ratio,
                    "start_point": int(start_point),
                    "normalization": "agibot_v1",
                },
                "loss": {
                    "video_weight": 1.0,
                    "action_weight": 1.0,
                },
                "metadata": {
                    "converter": "convert_agibot_to_motus_manifest.py",
                    "task_info_file": episode["task_info_file"],
                    "dataset_source_name": episode["dataset_source_name"],
                    "task_id": episode["task_id"],
                    "episode_id": episode["episode_id"],
                    "effective_frames": effective_frames,
                    "timesteps": timesteps,
                    "frame_numbers": frame_numbers,
                    "window_stride": int(args.window_stride),
                    "num_video_frames": int(args.num_video_frames),
                },
            }
        )
        windows_written += 1
    return records, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Agibot annotations to Motus Manifest v1 JSONL.")
    parser.add_argument("--annotation-folders", nargs="+", required=True)
    parser.add_argument("--data-root-folders", nargs="+", required=True)
    parser.add_argument("--dataset-names", nargs="+", default=["agibot-world"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--val-num", type=int, default=128)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-windows-per-episode", type=int, default=None)
    parser.add_argument("--source-dataset", default="agibot")
    parser.add_argument("--robot-name", default="agibot-a2-d")
    parser.add_argument("--state-dim", type=int, default=26)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument("--check-files", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.window_stride <= 0:
        raise ValueError("--window-stride must be positive")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_episodes = list(
        iter_agibot_episodes(args.annotation_folders, args.data_root_folders, args.dataset_names)
    )
    episodes = split_episodes(all_episodes, args.split, args.val_num)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    scanned = written = 0
    skipped_reasons: Dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as dst:
        for episode in episodes:
            scanned += 1
            records, skip_reason = convert_episode(episode, args.split, args)
            if skip_reason:
                skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                continue
            for record in records:
                dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                if args.max_records is not None and written >= args.max_records:
                    break
            if args.max_records is not None and written >= args.max_records:
                break

    print(
        f"episodes_total={len(all_episodes)} episodes_scanned={scanned} written={written} "
        f"skipped={sum(skipped_reasons.values())} output={output_path}"
    )
    for reason, count in sorted(skipped_reasons.items()):
        print(f"skip:{reason}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
