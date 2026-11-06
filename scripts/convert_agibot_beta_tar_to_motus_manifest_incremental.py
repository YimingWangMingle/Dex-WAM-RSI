#!/usr/bin/env python3
import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def load_base_module():
    script_path = Path(__file__).with_name("convert_agibot_beta_tar_to_motus_manifest.py")
    spec = importlib.util.spec_from_file_location("agibot_beta_base_converter", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base converter: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


Index = Dict[Tuple[int, int], Dict[str, Any]]


def parse_task_ids(values: Optional[List[int]]) -> Optional[Set[int]]:
    if not values:
        return None
    return {int(value) for value in values}


def load_all_episodes(data_root: Path, task_ids: Optional[Set[int]]) -> Dict[int, List[Dict[str, Any]]]:
    task_to_episodes: Dict[int, List[Dict[str, Any]]] = {}
    task_info_files = list(base.iter_task_info_files(data_root / "task_info", task_ids))
    print(
        f"[progress] phase=load_task_info files={len(task_info_files)} "
        f"task_ids={sorted(task_ids) if task_ids is not None else 'all'}",
        flush=True,
    )
    total = 0
    for idx, task_info_file in enumerate(task_info_files, start=1):
        task_id = int(base.task_id_from_path(task_info_file))
        episodes = []
        for item in base.load_episodes(task_info_file):
            item["task_info_file"] = str(task_info_file)
            episodes.append(item)
        task_to_episodes[task_id] = episodes
        total += len(episodes)
        print(
            f"[progress] phase=load_task_info file={idx}/{len(task_info_files)} "
            f"task_id={task_id} task_episodes={len(episodes)} episodes={total} path={task_info_file}",
            flush=True,
        )
    print(f"[progress] phase=loaded tasks={len(task_to_episodes)} episodes={total}", flush=True)
    return task_to_episodes


def index_cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.jsonl"


def done_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".done")


def load_index_cache(path: Path) -> Index:
    index: Index = {}
    if not path.exists() or not done_path(path).exists():
        return index
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (int(item["task_id"]), int(item["episode_id"]))
            index[key] = {
                "uri": item["uri"],
                "member": item["member"],
                "tar_offset_data": int(item["tar_offset_data"]),
                "tar_size": int(item["tar_size"]),
            }
    print(f"[cache-hit] path={path} entries={len(index)}", flush=True)
    return index


def save_index_cache(path: Path, index: Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for (task_id, episode_id), info in sorted(index.items()):
            row = {
                "task_id": int(task_id),
                "episode_id": int(episode_id),
                "uri": info["uri"],
                "member": info["member"],
                "tar_offset_data": int(info["tar_offset_data"]),
                "tar_size": int(info["tar_size"]),
            }
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)
    done_path(path).write_text(
        f"status=complete\nentries={len(index)}\ncreated_at={time.strftime('%Y-%m-%d %H:%M:%S %z')}\n",
        encoding="utf-8",
    )
    print(f"[cache-save] path={path} entries={len(index)}", flush=True)


def scan_proprio_index(
    data_root: Path,
    episodes_by_task: Dict[int, List[Dict[str, Any]]],
    cache_dir: Path,
    progress_seconds: float,
) -> Index:
    cache_path = index_cache_path(cache_dir, "proprio")
    cached = load_index_cache(cache_path)
    if cached:
        return cached

    wanted = {
        f"{int(item['task_id'])}/{int(item['episode_id'])}/proprio_stats.h5"
        for episodes in episodes_by_task.values()
        for item in episodes
    }
    found = base.scan_tar_members(
        sorted((data_root / "proprio_stats").glob("*.tar")),
        wanted,
        label="proprio",
        progress_seconds=progress_seconds,
    )
    index: Index = {}
    for episodes in episodes_by_task.values():
        for item in episodes:
            task_id = int(item["task_id"])
            episode_id = int(item["episode_id"])
            member = f"{task_id}/{episode_id}/proprio_stats.h5"
            info = found.get(member)
            if info is not None:
                index[(task_id, episode_id)] = info
    save_index_cache(cache_path, index)
    return index


def scan_video_index_for_task(
    data_root: Path,
    task_id: int,
    episodes: List[Dict[str, Any]],
    cache_dir: Path,
    progress_seconds: float,
) -> Index:
    cache_path = index_cache_path(cache_dir, f"video_task_{task_id}")
    cached = load_index_cache(cache_path)
    if cached:
        return cached

    episode_ids = [int(item["episode_id"]) for item in episodes]
    wanted = {f"{episode_id}/videos/head_color.mp4" for episode_id in episode_ids}
    found = base.scan_tar_members(
        sorted((data_root / "observations" / str(task_id)).glob("*.tar")),
        wanted,
        label=f"video_task_{task_id}",
        progress_seconds=progress_seconds,
    )
    index: Index = {}
    for episode_id in episode_ids:
        member = f"{episode_id}/videos/head_color.mp4"
        info = found.get(member)
        if info is not None:
            index[(task_id, episode_id)] = info
    save_index_cache(cache_path, index)
    return index


def write_task_shard(
    args: argparse.Namespace,
    task_id: int,
    episodes: List[Dict[str, Any]],
    video_index: Index,
    proprio_index: Index,
    shard_path: Path,
) -> Dict[str, int]:
    frame_length = int(args.num_video_frames) + 1
    scanned = 0
    written = 0
    skipped = 0
    reasons: Dict[str, int] = {}
    tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    with tmp.open("w", encoding="utf-8") as dst:
        for item in episodes:
            scanned += 1
            episode_id = int(item["episode_id"])
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

            span = base.episode_span(item)
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
                frame_indices, frame_mask = base.get_video_frame_idx(start_point, int(end_max), frame_length)
                record = base.make_record(
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
                    break
            if args.max_records is not None and written >= int(args.max_records):
                break

    tmp.replace(shard_path)
    done_path(shard_path).write_text(
        "\n".join(
            [
                "status=complete",
                f"task_id={task_id}",
                f"episodes_scanned={scanned}",
                f"written={written}",
                f"skipped={skipped}",
                f"created_at={time.strftime('%Y-%m-%d %H:%M:%S %z')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {"scanned": scanned, "written": written, "skipped": skipped}
    summary.update({f"skip_{key}": value for key, value in reasons.items()})
    return summary


def completed_shards(shard_dir: Path) -> List[Path]:
    shards = []
    for path in sorted(shard_dir.glob("task_*.motus_manifest.jsonl")):
        if done_path(path).exists():
            shards.append(path)
    return shards


def count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def write_snapshot(shard_dir: Path, snapshot_path: Path) -> int:
    shards = completed_shards(shard_dir)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    total = 0
    with tmp.open("w", encoding="utf-8") as dst:
        for shard in shards:
            with shard.open("r", encoding="utf-8") as src:
                for line in src:
                    dst.write(line)
                    total += 1
    tmp.replace(snapshot_path)
    done_path(snapshot_path).write_text(
        "\n".join(
            [
                "status=complete",
                f"shards={len(shards)}",
                f"records={total}",
                f"created_at={time.strftime('%Y-%m-%d %H:%M:%S %z')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[snapshot] output={snapshot_path} shards={len(shards)} records={total}",
        flush=True,
    )
    return total


def convert_incremental(args: argparse.Namespace) -> int:
    start_time = time.monotonic()
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    shard_dir = Path(args.shard_output_dir)
    snapshot_output = Path(args.snapshot_output)
    task_ids = parse_task_ids(args.task_ids)

    episodes_by_task = load_all_episodes(data_root, task_ids)
    if args.max_tasks is not None:
        keep = set(sorted(episodes_by_task)[: int(args.max_tasks)])
        episodes_by_task = {task_id: episodes_by_task[task_id] for task_id in sorted(keep)}
        print(f"[progress] phase=max_tasks tasks={len(episodes_by_task)}", flush=True)

    proprio_index = scan_proprio_index(
        data_root,
        episodes_by_task,
        cache_dir,
        progress_seconds=float(args.progress_seconds),
    )
    print(f"[progress] phase=proprio_ready entries={len(proprio_index)}", flush=True)

    completed_since_snapshot = 0
    total_written = 0
    for idx, task_id in enumerate(sorted(episodes_by_task), start=1):
        episodes = episodes_by_task[task_id]
        shard_path = shard_dir / f"task_{task_id}.motus_manifest.jsonl"
        if shard_path.exists() and done_path(shard_path).exists() and not args.force:
            lines = count_lines(shard_path)
            total_written += lines
            print(
                f"[shard-skip] task={idx}/{len(episodes_by_task)} task_id={task_id} "
                f"records={lines} path={shard_path}",
                flush=True,
            )
        else:
            print(
                f"[task-start] task={idx}/{len(episodes_by_task)} task_id={task_id} "
                f"episodes={len(episodes)} elapsed_sec={time.monotonic() - start_time:.1f}",
                flush=True,
            )
            video_index = scan_video_index_for_task(
                data_root,
                task_id,
                episodes,
                cache_dir,
                progress_seconds=float(args.progress_seconds),
            )
            summary = write_task_shard(args, task_id, episodes, video_index, proprio_index, shard_path)
            total_written += int(summary["written"])
            completed_since_snapshot += 1
            print(
                f"[task-done] task={idx}/{len(episodes_by_task)} task_id={task_id} "
                f"written={summary['written']} skipped={summary['skipped']} path={shard_path} "
                f"elapsed_sec={time.monotonic() - start_time:.1f}",
                flush=True,
            )

        if args.snapshot_every_tasks > 0 and completed_since_snapshot >= int(args.snapshot_every_tasks):
            write_snapshot(shard_dir, snapshot_output)
            completed_since_snapshot = 0

    write_snapshot(shard_dir, snapshot_output)
    print(
        f"[complete] tasks={len(episodes_by_task)} total_written_seen={total_written} "
        f"snapshot={snapshot_output} elapsed_sec={time.monotonic() - start_time:.1f}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally convert AgiBot beta tar-pack data to Motus Manifest shards with cache/resume."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--shard-output-dir", required=True)
    parser.add_argument("--snapshot-output", required=True)
    parser.add_argument("--cache-dir", required=True)
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
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--snapshot-every-tasks", type=int, default=5)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    return convert_incremental(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
