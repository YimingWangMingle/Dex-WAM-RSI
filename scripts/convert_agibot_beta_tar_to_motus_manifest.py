#!/usr/bin/env python3
import argparse
import json
import re
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TASK_PROMPT = "Scene Description: {scene}\nTask: {task}.\n"
ACTION_PROMPT = "Action: {action}\n"
VLM_PROMPT = "Based on the following situation, please predict what the robot will do: \n"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def task_id_from_path(path: Path) -> int:
    match = re.search(r"task_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot infer task_id from task info path: {path}")
    return int(match.group(1))


def get_video_frame_idx(start_point: int, total_frames: int, frame_length: int) -> Tuple[List[int], List[bool]]:
    frame_mask = [True] * frame_length
    pad_num = min(total_frames - (start_point + frame_length), 0)
    if pad_num < 0:
        for i in range(frame_length + pad_num, frame_length):
            frame_mask[i] = False
    indices = list(range(start_point, min(total_frames, start_point + frame_length)))
    indices += [total_frames - 1] * (-pad_num)
    return indices, frame_mask


def action_text_for_start(action_config: List[Dict[str, Any]], start_point: int) -> str:
    for action in action_config:
        start = int(action.get("start_frame", 0))
        end = int(action.get("end_frame", start))
        if start <= start_point < end:
            text = clean_text(action.get("action_text")) or clean_text(action.get("skill"))
            return ACTION_PROMPT.format(action=text)
    return ""


def iter_task_info_files(task_info_dir: Path, task_ids: Optional[Set[int]]) -> Iterable[Path]:
    for path in sorted(task_info_dir.glob("task_*.json")):
        task_id = task_id_from_path(path)
        if task_ids is not None and task_id not in task_ids:
            continue
        yield path


def load_episodes(task_info_path: Path) -> List[Dict[str, Any]]:
    task_id = task_id_from_path(task_info_path)
    with task_info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Task info must contain a list: {task_info_path}")
    episodes = []
    for item in data:
        item = dict(item)
        item["task_id"] = int(item.get("task_id") or task_id)
        episodes.append(item)
    return episodes


def episode_span(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    actions = (item.get("label_info") or {}).get("action_config") or []
    starts = []
    ends = []
    for action in actions:
        try:
            starts.append(int(action.get("start_frame")))
            ends.append(int(action.get("end_frame")))
        except (TypeError, ValueError):
            continue
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def scan_tar_members(
    tar_paths: Iterable[Path],
    wanted_members: Set[str],
    label: str = "tar",
    progress_seconds: float = 30.0,
) -> Dict[str, Dict[str, Any]]:
    remaining = set(wanted_members)
    found: Dict[str, Dict[str, Any]] = {}
    if not remaining:
        return found

    tar_paths = list(tar_paths)
    start_time = time.monotonic()
    last_progress = start_time
    print(
        f"[index-scan-start] label={label} tars={len(tar_paths)} wanted={len(remaining)}",
        flush=True,
    )

    for tar_idx, tar_path in enumerate(tar_paths, start=1):
        if not remaining:
            break
        print(
            f"[index-scan-tar] label={label} tar={tar_idx}/{len(tar_paths)} "
            f"found={len(found)} remaining={len(remaining)} path={tar_path}",
            flush=True,
        )
        with tarfile.open(tar_path, "r") as tf:
            members_scanned = 0
            for member in tf:
                members_scanned += 1
                if member.name not in remaining:
                    current_time = time.monotonic()
                    if progress_seconds > 0 and current_time - last_progress >= progress_seconds:
                        print(
                            f"[index-scan-progress] label={label} tar={tar_idx}/{len(tar_paths)} "
                            f"members_scanned={members_scanned} found={len(found)} "
                            f"remaining={len(remaining)} elapsed_sec={current_time - start_time:.1f}",
                            flush=True,
                        )
                        last_progress = current_time
                    continue
                found[member.name] = {
                    "uri": str(tar_path),
                    "member": member.name,
                    "tar_offset_data": int(member.offset_data),
                    "tar_size": int(member.size),
                }
                remaining.remove(member.name)
                if not remaining:
                    break
        current_time = time.monotonic()
        print(
            f"[index-scan-done-tar] label={label} tar={tar_idx}/{len(tar_paths)} "
            f"found={len(found)} remaining={len(remaining)} elapsed_sec={current_time - start_time:.1f}",
            flush=True,
        )
        last_progress = current_time
    print(
        f"[index-scan-finish] label={label} found={len(found)} "
        f"missing={len(remaining)} elapsed_sec={time.monotonic() - start_time:.1f}",
        flush=True,
    )
    return found


def build_member_indices(
    data_root: Path,
    episodes: List[Dict[str, Any]],
    check_members: bool,
    progress_seconds: float = 30.0,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]], Dict[str, int]]:
    stats = {"missing_video_member": 0, "missing_proprio_member": 0}
    if not check_members:
        video_index = {}
        proprio_index = {}
        for item in episodes:
            task_id = int(item["task_id"])
            episode_id = int(item["episode_id"])
            video_index[(task_id, episode_id)] = {
                "uri": None,
                "member": f"{episode_id}/videos/head_color.mp4",
            }
            proprio_index[(task_id, episode_id)] = {
                "uri": None,
                "member": f"{task_id}/{episode_id}/proprio_stats.h5",
            }
        return video_index, proprio_index, stats

    video_index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    proprio_index: Dict[Tuple[int, int], Dict[str, Any]] = {}

    episodes_by_task: Dict[int, List[int]] = {}
    for item in episodes:
        episodes_by_task.setdefault(int(item["task_id"]), []).append(int(item["episode_id"]))

    for task_id, episode_ids in sorted(episodes_by_task.items()):
        wanted = {f"{episode_id}/videos/head_color.mp4" for episode_id in episode_ids}
        tar_paths = sorted((data_root / "observations" / str(task_id)).glob("*.tar"))
        found = scan_tar_members(
            tar_paths,
            wanted,
            label=f"video_task_{task_id}",
            progress_seconds=progress_seconds,
        )
        for episode_id in episode_ids:
            member = f"{episode_id}/videos/head_color.mp4"
            info = found.get(member)
            if info is None:
                stats["missing_video_member"] += 1
                continue
            video_index[(task_id, episode_id)] = info

    wanted_proprio = {
        f"{int(item['task_id'])}/{int(item['episode_id'])}/proprio_stats.h5"
        for item in episodes
    }
    proprio_tars = sorted((data_root / "proprio_stats").glob("*.tar"))
    found_proprio = scan_tar_members(
        proprio_tars,
        wanted_proprio,
        label="proprio",
        progress_seconds=progress_seconds,
    )
    for item in episodes:
        task_id = int(item["task_id"])
        episode_id = int(item["episode_id"])
        member = f"{task_id}/{episode_id}/proprio_stats.h5"
        info = found_proprio.get(member)
        if info is None:
            stats["missing_proprio_member"] += 1
            continue
        proprio_index[(task_id, episode_id)] = info

    return video_index, proprio_index, stats


def make_record(
    args: argparse.Namespace,
    item: Dict[str, Any],
    start_point: int,
    window_index: int,
    frame_indices: List[int],
    frame_mask: List[bool],
    video_ref: Dict[str, Any],
    proprio_ref: Dict[str, Any],
    split: str,
) -> Dict[str, Any]:
    task_id = int(item["task_id"])
    episode_id = int(item["episode_id"])
    action_config = (item.get("label_info") or {}).get("action_config") or []
    task_text = TASK_PROMPT.format(
        scene=clean_text(item.get("init_scene_text")),
        task=clean_text(item.get("task_name")),
    )
    action_text = action_text_for_start(action_config, start_point)
    t5_text = f"{task_text}{action_text}"
    vlm_text = f"{VLM_PROMPT}{task_text}{action_text}"
    record_id = f"{args.source_dataset}:{split}:{task_id}:{episode_id}:window:{int(start_point):06d}"

    return {
        "schema_version": "motus_manifest_v1",
        "id": record_id,
        "source_dataset": args.source_dataset,
        "split": split,
        "task": clean_text(item.get("task_name")),
        "subtask": "robot_action_prediction",
        "text": {
            "t5": t5_text,
            "vlm": vlm_text,
            "target": clean_text(action_text),
        },
        "video": {
            "storage_backend": "file_tar",
            "uri": video_ref["uri"],
            "member": video_ref["member"],
            "tar_offset_data": video_ref.get("tar_offset_data"),
            "tar_size": video_ref.get("tar_size"),
            "frame_indices": frame_indices,
            "view": "agibot_head_color",
            "resize_mode": "interpolate",
            "decode_backend": "ffmpeg",
        },
        "robot": {
            "name": args.robot_name,
            "state_dim": int(args.state_dim),
            "action_dim": int(args.action_dim),
            "action_policy": "real",
        },
        "state": {
            "storage_backend": "agibot_proprio_hdf5_tar",
            "uri": proprio_ref["uri"],
            "member": proprio_ref["member"],
            "tar_offset_data": proprio_ref.get("tar_offset_data"),
            "tar_size": proprio_ref.get("tar_size"),
            "key": "state/end+state/effector",
            "index": int(start_point),
        },
        "action": {
            "storage_backend": "agibot_proprio_hdf5_tar",
            "uri": proprio_ref["uri"],
            "member": proprio_ref["member"],
            "tar_offset_data": proprio_ref.get("tar_offset_data"),
            "tar_size": proprio_ref.get("tar_size"),
            "key": "state/end+state/effector",
            "frame_indices": frame_indices,
            "frame_mask": frame_mask,
            "action_video_fps_ratio": int(args.action_video_fps_ratio),
            "start_point": int(start_point),
            "normalization": "agibot_v1",
        },
        "loss": {
            "video_weight": 1.0,
            "action_weight": 1.0,
        },
        "metadata": {
            "converter": "convert_agibot_beta_tar_to_motus_manifest.py",
            "task_info_file": str(item.get("task_info_file")),
            "task_id": task_id,
            "episode_id": episode_id,
            "window_index": int(window_index),
            "window_stride": int(args.window_stride),
            "num_video_frames": int(args.num_video_frames),
            "action_config": action_config,
            "frame_source": "task_info.action_config",
        },
    }


def convert(args: argparse.Namespace) -> int:
    start_time = time.monotonic()
    data_root = Path(args.data_root)
    task_ids = None if not args.task_ids else {int(task_id) for task_id in args.task_ids}
    task_info_files = list(iter_task_info_files(data_root / "task_info", task_ids))
    print(
        f"[progress] phase=load_task_info files={len(task_info_files)} "
        f"task_ids={sorted(task_ids) if task_ids is not None else 'all'}",
        flush=True,
    )
    episodes: List[Dict[str, Any]] = []
    for task_info_idx, task_info_file in enumerate(task_info_files, start=1):
        for item in load_episodes(task_info_file):
            item["task_info_file"] = str(task_info_file)
            episodes.append(item)
        print(
            f"[progress] phase=load_task_info file={task_info_idx}/{len(task_info_files)} "
            f"episodes={len(episodes)} path={task_info_file}",
            flush=True,
        )

    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]
    print(
        f"[progress] phase=loaded episodes={len(episodes)} max_episodes={args.max_episodes}",
        flush=True,
    )

    video_index, proprio_index, index_stats = build_member_indices(
        data_root,
        episodes,
        check_members=not args.no_check_members,
        progress_seconds=float(args.progress_seconds),
    )
    print(
        f"[progress] phase=index_ready video_index={len(video_index)} proprio_index={len(proprio_index)} "
        f"elapsed_sec={time.monotonic() - start_time:.1f}",
        flush=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scanned = 0
    written = 0
    skipped = 0
    reasons: Dict[str, int] = {}
    frame_length = int(args.num_video_frames) + 1
    last_progress = time.monotonic()

    with output_path.open("w", encoding="utf-8") as dst:
        for item in episodes:
            scanned += 1
            task_id = int(item["task_id"])
            episode_id = int(item["episode_id"])
            current_time = time.monotonic()
            should_print = (
                args.progress_interval > 0 and scanned % int(args.progress_interval) == 0
            ) or (
                float(args.progress_seconds) > 0 and current_time - last_progress >= float(args.progress_seconds)
            )
            if should_print:
                print(
                    f"[progress] phase=write episodes_scanned={scanned}/{len(episodes)} "
                    f"written={written} skipped={skipped} current_task={task_id} "
                    f"current_episode={episode_id} elapsed_sec={current_time - start_time:.1f}",
                    flush=True,
                )
                last_progress = current_time
            key = (task_id, episode_id)
            video_ref = video_index.get(key)
            proprio_ref = proprio_index.get(key)
            if video_ref is None:
                skipped += 1
                reasons["missing_video_member"] = reasons.get("missing_video_member", 0) + 1
                continue
            if proprio_ref is None:
                skipped += 1
                reasons["missing_proprio_member"] = reasons.get("missing_proprio_member", 0) + 1
                continue

            span = episode_span(item)
            if span is None:
                skipped += 1
                reasons["missing_action_config"] = reasons.get("missing_action_config", 0) + 1
                continue
            start_min, end_max = span
            if end_max <= start_min or end_max <= frame_length:
                skipped += 1
                reasons["bad_action_span"] = reasons.get("bad_action_span", 0) + 1
                continue
            sampling_start = max(0, int(start_min))
            sampling_end = max(min(int(end_max) - frame_length, int(end_max) - 1), sampling_start)
            if sampling_end < sampling_start:
                skipped += 1
                reasons["short_episode"] = reasons.get("short_episode", 0) + 1
                continue

            window_index = 0
            for start_point in range(sampling_start, sampling_end + 1, int(args.window_stride)):
                frame_indices, frame_mask = get_video_frame_idx(start_point, int(end_max), frame_length)
                record = make_record(
                    args,
                    item,
                    start_point=start_point,
                    window_index=window_index,
                    frame_indices=frame_indices,
                    frame_mask=frame_mask,
                    video_ref=video_ref,
                    proprio_ref=proprio_ref,
                    split=args.split,
                )
                dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                window_index += 1
                if args.max_records is not None and written >= int(args.max_records):
                    print_summary(scanned, written, skipped, output_path, index_stats, reasons)
                    return 0

    print_summary(scanned, written, skipped, output_path, index_stats, reasons)
    return 0


def print_summary(
    scanned: int,
    written: int,
    skipped: int,
    output_path: Path,
    index_stats: Dict[str, int],
    reasons: Dict[str, int],
) -> None:
    print(f"episodes_scanned={scanned} written={written} skipped={skipped} output={output_path}")
    for key, value in sorted(index_stats.items()):
        print(f"index_{key}={value}")
    for key, value in sorted(reasons.items()):
        print(f"skip_{key}={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AgiBot World tar-pack data to Motus Manifest v1 JSONL.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-ids", nargs="*", type=int, default=None)
    parser.add_argument("--source-dataset", default="agibot_beta")
    parser.add_argument("--robot-name", default="agibot-a2-d")
    parser.add_argument("--state-dim", type=int, default=26)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--action-video-fps-ratio", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-check-members", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print manifest-writing progress every N scanned episodes; set <=0 to disable count-based progress.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help="Print progress at least every N seconds during long tar scans/writes; set <=0 to disable time-based progress.",
    )
    return parser.parse_args()


def main() -> int:
    return convert(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
