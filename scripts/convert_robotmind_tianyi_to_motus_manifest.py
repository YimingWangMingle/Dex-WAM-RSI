#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import h5py
except ImportError:
    h5py = None


DEFAULT_RGB_KEY = "camera_observations/color_images/camera_top"
DEFAULT_LEFT_ARM_KEY = "puppet/arm_left_position_align/data"
DEFAULT_RIGHT_ARM_KEY = "puppet/arm_right_position_align/data"
DEFAULT_LEFT_GRIPPER_KEY = "puppet/end_effector_left_position_align/data"
DEFAULT_RIGHT_GRIPPER_KEY = "puppet/end_effector_right_position_align/data"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def text_for_instruction(instruction: str) -> Tuple[str, str, str]:
    target = clean_text(instruction)
    t5_text = f"Task: {target}.\n"
    vlm_text = "Based on the following situation, please predict what the robot will do: \n" + target
    return t5_text, vlm_text, target


def iter_episodes(data_root: str, check_files: bool = False) -> Iterable[Dict[str, Any]]:
    root = Path(data_root)
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        success_dir = task_dir / "success_episodes"
        if not success_dir.exists():
            continue
        zh_instruction = read_text(task_dir / "zh_description.txt")
        for episode_dir in sorted(path for path in success_dir.iterdir() if path.is_dir()):
            hdf5_path = episode_dir / "data" / "trajectory.hdf5"
            if check_files and not hdf5_path.exists():
                continue
            yield {
                "data_root": str(root),
                "task_name": task_dir.name,
                "episode_name": episode_dir.name,
                "hdf5_path": str(hdf5_path),
                "zh_instruction": zh_instruction,
            }


def inspect_episode(episode: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if h5py is None:
        raise ImportError("convert_robotmind_tianyi_to_motus_manifest.py requires h5py")
    path = episode["hdf5_path"]
    try:
        with h5py.File(path, "r") as f:
            required_keys = [
                DEFAULT_RGB_KEY,
                DEFAULT_LEFT_ARM_KEY,
                DEFAULT_RIGHT_ARM_KEY,
                DEFAULT_LEFT_GRIPPER_KEY,
                DEFAULT_RIGHT_GRIPPER_KEY,
            ]
            missing = [key for key in required_keys if key not in f]
            if missing:
                return None, "missing_key:" + ",".join(missing)
            lengths = [int(f[key].shape[0]) for key in required_keys]
            trajectory_length = int(f["metadata"].attrs.get("trajectory_length", lengths[0])) if "metadata" in f else lengths[0]
            total_frames = min(lengths + [trajectory_length])
            if total_frames <= 1:
                return None, "too_short"
            color_res = tuple(int(x) for x in f["camera_color_resolution/camera_top"][()]) if "camera_color_resolution/camera_top" in f else None
            en_instruction = ""
            if "metadata" in f:
                en_instruction = clean_text(f["metadata"].attrs.get("language_instruction", ""))
        return {
            "total_frames": total_frames,
            "raw_lengths": lengths,
            "trajectory_length": trajectory_length,
            "color_resolution": color_res,
            "en_instruction": en_instruction,
        }, None
    except Exception as exc:
        return None, f"bad_hdf5:{type(exc).__name__}"


def sampling_indices(
    condition_frame_idx: int,
    total_frames: int,
    action_chunk_size: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
    frames_clip_num: int,
) -> Tuple[List[int], List[int]]:
    action_indices = []
    for i in range(action_chunk_size):
        action_idx = condition_frame_idx + (i + 1) * global_downsample_rate
        action_indices.append(min(action_idx, total_frames - 1))

    video_indices = []
    for i in range(frames_clip_num):
        action_step = (i + 1) * video_action_freq_ratio - 1
        if action_step < len(action_indices):
            video_indices.append(action_indices[action_step])
        else:
            video_indices.append(action_indices[-1])
    return video_indices, action_indices


def choose_instruction(episode: Dict[str, Any], info: Dict[str, Any], language: str) -> str:
    zh = clean_text(episode.get("zh_instruction"))
    en = clean_text(info.get("en_instruction"))
    if language == "zh":
        return zh or en
    if language == "both":
        if zh and en:
            return f"{zh} / {en}"
        return zh or en
    return en or zh


def convert_episode(
    episode: Dict[str, Any],
    split: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    info, skip_reason = inspect_episode(episode)
    if skip_reason:
        return [], skip_reason
    assert info is not None

    instruction = choose_instruction(episode, info, args.instruction_lang)
    if not instruction:
        return [], "missing_text"
    t5_text, vlm_text, target_text = text_for_instruction(instruction)

    total_frames = int(info["total_frames"])
    frames_clip_num = int(args.num_video_frames) + 1
    action_chunk_size = frames_clip_num * int(args.video_action_freq_ratio)
    physical_chunk_size = action_chunk_size * int(args.global_downsample_rate)
    max_condition_idx = total_frames - physical_chunk_size - 1
    if max_condition_idx < 0:
        starts = [0] if args.keep_short else []
    else:
        starts = list(range(0, max_condition_idx + 1, int(args.window_stride)))
        if args.include_last and starts and starts[-1] != max_condition_idx:
            starts.append(max_condition_idx)
    if not starts:
        return [], "no_windows"

    records = []
    for window_idx, start_idx in enumerate(starts):
        if args.max_windows_per_episode is not None and len(records) >= args.max_windows_per_episode:
            break
        video_indices, action_indices = sampling_indices(
            condition_frame_idx=int(start_idx),
            total_frames=total_frames,
            action_chunk_size=action_chunk_size,
            global_downsample_rate=int(args.global_downsample_rate),
            video_action_freq_ratio=int(args.video_action_freq_ratio),
            frames_clip_num=frames_clip_num,
        )
        record_id = (
            f"{args.source_dataset}:{split}:{args.robot_name}:"
            f"{episode['task_name']}:{episode['episode_name']}:window:{int(start_idx):06d}"
        )
        records.append(
            {
                "schema_version": "motus_manifest_v1",
                "id": record_id,
                "source_dataset": args.source_dataset,
                "split": split,
                "task": clean_text(episode["task_name"]),
                "subtask": "robot_video_prediction",
                "text": {
                    "t5": t5_text,
                    "vlm": vlm_text,
                    "target": target_text,
                    "zh": clean_text(episode.get("zh_instruction")),
                    "en": clean_text(info.get("en_instruction")),
                },
                "video": {
                    "storage_backend": "robotmind_hdf5_jpeg",
                    "uri": episode["hdf5_path"],
                    "key": DEFAULT_RGB_KEY,
                    "frame_indices": video_indices,
                    "resize_mode": args.resize_mode,
                    "view": "camera_top",
                },
                "robot": {
                    "name": args.robot_name,
                    "state_dim": 16,
                    "action_dim": 16,
                    "action_policy": "real",
                },
                "state": {
                    "storage_backend": "robotmind_tianyi_hdf5",
                    "uri": episode["hdf5_path"],
                    "keys": {
                        "left_arm": DEFAULT_LEFT_ARM_KEY,
                        "right_arm": DEFAULT_RIGHT_ARM_KEY,
                        "left_gripper": DEFAULT_LEFT_GRIPPER_KEY,
                        "right_gripper": DEFAULT_RIGHT_GRIPPER_KEY,
                    },
                    "index": int(action_indices[0]),
                    "normalization": "none",
                },
                "action": {
                    "storage_backend": "robotmind_tianyi_hdf5",
                    "uri": episode["hdf5_path"],
                    "keys": {
                        "left_arm": DEFAULT_LEFT_ARM_KEY,
                        "right_arm": DEFAULT_RIGHT_ARM_KEY,
                        "left_gripper": DEFAULT_LEFT_GRIPPER_KEY,
                        "right_gripper": DEFAULT_RIGHT_GRIPPER_KEY,
                    },
                    "frame_indices": action_indices,
                    "global_downsample_rate": int(args.global_downsample_rate),
                    "video_action_freq_ratio": int(args.video_action_freq_ratio),
                    "normalization": "none",
                },
                "loss": {
                    "video_weight": 1.0,
                    "action_weight": 1.0,
                },
                "metadata": {
                    "converter": "convert_robotmind_tianyi_to_motus_manifest.py",
                    "data_root": episode["data_root"],
                    "episode_name": episode["episode_name"],
                    "hdf5_path": episode["hdf5_path"],
                    "condition_frame_idx": int(start_idx),
                    "window_index": int(window_idx),
                    "total_frames": int(total_frames),
                    "trajectory_length": int(info["trajectory_length"]),
                    "raw_lengths": [int(x) for x in info["raw_lengths"]],
                    "color_resolution": list(info["color_resolution"]) if info.get("color_resolution") else None,
                    "window_stride": int(args.window_stride),
                    "num_video_frames": int(args.num_video_frames),
                    "sampling": "frame_index",
                    "instruction_lang": args.instruction_lang,
                },
            }
        )
        if args.max_records is not None and len(records) >= args.max_records:
            break
    return records, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RobotMind TianYi HDF5 data to Motus Manifest v1 JSONL.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--source-dataset", default="robotmind")
    parser.add_argument("--robot-name", default="robotmind-tianyi")
    parser.add_argument("--instruction-lang", choices=["en", "zh", "both"], default="zh")
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--global-downsample-rate", type=int, default=1)
    parser.add_argument("--video-action-freq-ratio", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=24)
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--keep-short", action="store_true")
    parser.add_argument("--resize-mode", choices=["interpolate", "bilinear", "pad"], default="interpolate")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-windows-per-episode", type=int, default=None)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.window_stride <= 0:
        raise ValueError("--window-stride must be positive")
    if args.global_downsample_rate <= 0:
        raise ValueError("--global-downsample-rate must be positive")
    if args.video_action_freq_ratio <= 0:
        raise ValueError("--video-action-freq-ratio must be positive")

    output_path = Path(args.output)
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = written = 0
    skipped = Counter()
    task_counts = Counter()

    dst = None if args.dry_run else output_path.open("w", encoding="utf-8")
    try:
        for episode in iter_episodes(args.data_root, check_files=args.check_files):
            scanned += 1
            records, skip_reason = convert_episode(episode, args.split, args)
            if skip_reason:
                skipped[skip_reason] += 1
            else:
                for record in records:
                    if args.max_records is not None and written >= args.max_records:
                        break
                    task_counts[record["task"]] += 1
                    if dst is not None:
                        dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
            if args.max_records is not None and written >= args.max_records:
                break
            if args.max_episodes is not None and scanned >= args.max_episodes:
                break
    finally:
        if dst is not None:
            dst.close()

    print(
        f"episodes_scanned={scanned} written={written} skipped={sum(skipped.values())} "
        f"output={output_path} dry_run={args.dry_run}"
    )
    for reason, count in sorted(skipped.items()):
        print(f"skip:{reason}={count}")
    for task, count in sorted(task_counts.items()):
        print(f"task:{task}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
