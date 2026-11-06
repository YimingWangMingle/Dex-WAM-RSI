#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    import decord
except ImportError:
    decord = None

try:
    import cv2
except ImportError:
    cv2 = None


ROBOT_ALIASES = {
    "aloha-agilex": ["aloha-agilex", "aloha", "agilex"],
    "franka-panda": ["franka-panda", "franka", "panda"],
    "ARX-X5": ["arx-x5", "arx", "x5"],
    "piper": ["piper"],
    "ur5-wsg": ["ur5-wsg", "ur5", "wsg"],
}

ROBOT_ACTION_DIMS = {
    "aloha-agilex": 14,
    "franka-panda": 7,
    "ARX-X5": 14,
    "piper": 14,
    "ur5-wsg": 14,
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def infer_robot_name(path: Path, default: Optional[str] = None) -> Optional[str]:
    search_text = " ".join(str(part).lower() for part in path.parts)
    alias_items = []
    for name, aliases in ROBOT_ALIASES.items():
        for alias in aliases + [name]:
            alias_items.append((alias.lower(), name))
    for alias, name in sorted(alias_items, key=lambda item: len(item[0]), reverse=True):
        if alias in search_text:
            return name
    return default


def video_frame_count(path: str) -> int:
    if decord is not None:
        return int(len(decord.VideoReader(path, num_threads=1)))
    if cv2 is not None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"failed to open video: {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    raise ImportError("convert_robotwin_to_motus_manifest.py requires decord or cv2")


def qpos_shape(path: str) -> Tuple[int, int]:
    tensor = torch.load(path, map_location="cpu")
    if tensor.ndim != 2:
        raise ValueError(f"qpos tensor must be rank-2, got shape={tuple(tensor.shape)} path={path}")
    return int(tensor.shape[0]), int(tensor.shape[1])


def load_instructions(meta_path: Path) -> List[str]:
    if not meta_path.exists():
        return []
    with meta_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]


def iter_episodes(
    dataset_dirs: List[str],
    data_mode: str,
    task_mode: str,
    task_name: Optional[str],
    robot_name: Optional[str],
    check_files: bool,
) -> Iterable[Dict[str, Any]]:
    data_splits = ["clean", "randomized"] if data_mode == "both" else [data_mode]
    for dataset_dir_str in dataset_dirs:
        dataset_dir = Path(dataset_dir_str)
        inferred_robot = infer_robot_name(dataset_dir, default=robot_name)
        for data_split in data_splits:
            split_dir = dataset_dir / data_split
            if not split_dir.exists():
                continue
            task_dirs = [split_dir / task_name] if task_mode == "single" else sorted(
                path for path in split_dir.iterdir() if path.is_dir()
            )
            for task_dir in task_dirs:
                if not task_dir.exists():
                    continue
                qpos_dir = task_dir / "qpos"
                videos_dir = task_dir / "videos"
                metas_dir = task_dir / "metas"
                if not qpos_dir.exists() or not videos_dir.exists():
                    continue
                for qpos_path in sorted(qpos_dir.glob("*.pt")):
                    episode_name = qpos_path.stem
                    video_path = videos_dir / f"{episode_name}.mp4"
                    meta_path = metas_dir / f"{episode_name}.txt"
                    if check_files and not video_path.exists():
                        continue
                    if check_files and not meta_path.exists():
                        continue
                    yield {
                        "dataset_root": str(dataset_dir),
                        "data_split": data_split,
                        "task_name": task_dir.name,
                        "episode_name": episode_name,
                        "robot_name": infer_robot_name(task_dir, default=inferred_robot),
                        "qpos_path": str(qpos_path),
                        "video_path": str(video_path),
                        "meta_path": str(meta_path),
                    }


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


def text_for_instruction(raw_instruction: str) -> Tuple[str, str, str]:
    text_instruction = clean_text(raw_instruction).split(":")[-1].strip()
    t5_text = f"Task: {text_instruction}.\n"
    vlm_text = "Based on the following situation, please predict what the robot will do: \n" + text_instruction
    return t5_text, vlm_text, text_instruction


def convert_episode(
    episode: Dict[str, Any],
    split: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, int]]:
    robot_name = episode.get("robot_name") or args.robot_name
    if not robot_name:
        return [], "missing_robot_name", {}
    configured_dim = args.action_dim or ROBOT_ACTION_DIMS.get(robot_name)
    if configured_dim is None:
        return [], "missing_action_dim", {}

    try:
        qpos_len, qpos_dim = qpos_shape(episode["qpos_path"])
    except Exception:
        return [], "bad_qpos", {}
    if int(qpos_dim) != int(configured_dim):
        return [], f"qpos_dim_{qpos_dim}_expected_{configured_dim}", {"qpos_dim": qpos_dim}

    try:
        video_len = video_frame_count(episode["video_path"])
    except Exception:
        return [], "bad_video", {"qpos_dim": qpos_dim}

    total_frames = min(qpos_len, video_len)
    if total_frames < 2:
        return [], "too_short", {"qpos_dim": qpos_dim}

    instructions = load_instructions(Path(episode["meta_path"]))
    if not instructions:
        return [], "missing_text", {"qpos_dim": qpos_dim}
    if args.instruction_mode == "first":
        instruction_items = [(0, instructions[0])]
    else:
        instruction_items = list(enumerate(instructions))

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
        return [], "no_windows", {"qpos_dim": qpos_dim}

    records = []
    for start_idx in starts:
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
        for instruction_idx, instruction in instruction_items:
            t5_text, vlm_text, target_text = text_for_instruction(instruction)
            record_id = (
                f"{args.source_dataset}:{split}:{robot_name}:{episode['data_split']}:"
                f"{episode['task_name']}:{episode['episode_name']}:"
                f"window:{int(start_idx):06d}:instruction:{int(instruction_idx):03d}"
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
                        "target": target_text,
                    },
                    "video": {
                        "storage_backend": "file",
                        "uri": episode["video_path"],
                        "frame_indices": video_indices,
                        "resize_mode": "pad",
                        "view": "robotwin_main",
                    },
                    "robot": {
                        "name": robot_name,
                        "state_dim": int(qpos_dim),
                        "action_dim": int(qpos_dim),
                        "action_policy": "real",
                    },
                    "state": {
                        "storage_backend": "robotwin_qpos_pt",
                        "uri": episode["qpos_path"],
                        "key": "qpos",
                        "index": int(action_indices[0]),
                    },
                    "action": {
                        "storage_backend": "robotwin_qpos_pt",
                        "uri": episode["qpos_path"],
                        "key": "qpos",
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
                        "converter": "convert_robotwin_to_motus_manifest.py",
                        "dataset_root": episode["dataset_root"],
                        "data_split": episode["data_split"],
                        "episode_name": episode["episode_name"],
                        "meta_path": episode["meta_path"],
                        "instruction_idx": int(instruction_idx),
                        "condition_frame_idx": int(start_idx),
                        "qpos_frames": int(qpos_len),
                        "video_frames": int(video_len),
                        "effective_total_frames": int(total_frames),
                        "window_stride": int(args.window_stride),
                        "num_video_frames": int(args.num_video_frames),
                    },
                }
            )
            if args.max_records is not None and len(records) >= args.max_records:
                break
        if args.max_records is not None and len(records) >= args.max_records:
            break
    return records, None, {"qpos_dim": qpos_dim}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RoboTwin data to Motus Manifest v1 JSONL.")
    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-mode", choices=["clean", "randomized", "both"], default="both")
    parser.add_argument("--task-mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--source-dataset", default="robotwin")
    parser.add_argument("--robot-name", default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--global-downsample-rate", type=int, default=1)
    parser.add_argument("--video-action-freq-ratio", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=24)
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--keep-short", action="store_true")
    parser.add_argument("--instruction-mode", choices=["first", "all"], default="first")
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

    episodes_iter = iter_episodes(
        dataset_dirs=args.dataset_dirs,
        data_mode=args.data_mode,
        task_mode=args.task_mode,
        task_name=args.task_name,
        robot_name=args.robot_name,
        check_files=args.check_files,
    )

    output_path = Path(args.output)
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = written = 0
    skipped = Counter()
    qpos_dims = Counter()
    task_counts = Counter()

    dst = None if args.dry_run else output_path.open("w", encoding="utf-8")
    try:
        for episode in episodes_iter:
            scanned += 1
            records, skip_reason, stats = convert_episode(episode, args.split, args)
            if "qpos_dim" in stats:
                qpos_dims[int(stats["qpos_dim"])] += 1
            if skip_reason:
                skipped[skip_reason] += 1
            else:
                for record in records:
                    if args.max_records is not None and written >= args.max_records:
                        break
                    task_counts[episode["task_name"]] += 1
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
    for dim, count in sorted(qpos_dims.items()):
        print(f"qpos_dim:{dim}={count}")
    for reason, count in sorted(skipped.items()):
        print(f"skip:{reason}={count}")
    for task, count in sorted(task_counts.items()):
        print(f"task:{task}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
