#!/usr/bin/env python3
"""Build a compact episode-level index for runtime Agibot window sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

import h5py


def stable_test_split(task_id: Any, episode_id: Any, fraction: float, seed: int) -> str:
    payload = f"{int(seed)}:{task_id}:{episode_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return "test" if value / float(1 << 64) < float(fraction) else "train"


def iter_annotations(annotation_roots: list[str]) -> Iterable[tuple[Path, Dict[str, Any]]]:
    for root_value in annotation_roots:
        root = Path(root_value)
        for annotation_path in sorted(root.glob("*.json")):
            with annotation_path.open("r", encoding="utf-8") as handle:
                items = json.load(handle)
            for item in items:
                yield annotation_path, item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-folders", nargs="+", required=True)
    parser.add_argument("--data-root-folders", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.annotation_folders) != len(args.data_root_folders):
        raise ValueError("annotation-folders and data-root-folders must have equal lengths")
    if not 0.0 <= args.holdout_fraction < 1.0:
        raise ValueError("holdout-fraction must be in [0, 1)")

    roots = {str(Path(annotation).resolve()): str(Path(data).resolve()) for annotation, data in zip(
        args.annotation_folders, args.data_root_folders
    )}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    with output.open("w", encoding="utf-8") as destination:
        for annotation_path, item in iter_annotations(args.annotation_folders):
            if args.max_episodes is not None and counters["seen"] >= args.max_episodes:
                break
            counters["seen"] += 1
            annotation_root = next(
                root for root in roots if annotation_path.parent.resolve() == Path(root)
            )
            data_root = Path(roots[annotation_root])
            task_id = item["task_id"]
            episode_id = item["episode_id"]
            video_path = data_root / "observations" / str(task_id) / str(episode_id) / "videos_5fps" / "head_color.mp4"
            proprio_path = data_root / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
            if args.check_files and not video_path.is_file():
                counters["missing_video"] += 1
                continue
            if args.check_files and not proprio_path.is_file():
                counters["missing_proprio"] += 1
                continue
            try:
                with h5py.File(proprio_path, "r") as handle:
                    timesteps = int(handle["state/end/position"].shape[0])
            except Exception:
                counters["bad_proprio"] += 1
                continue
            ratio = int(item["action_video_fps_ratio"])
            if ratio <= 0 or timesteps <= 0:
                counters["bad_timeline"] += 1
                continue
            frame_numbers = timesteps // ratio
            effective = [int(value) for value in item["effective_frames"]]
            min_start = effective[0]
            max_start = max(
                min(effective[1], frame_numbers - (int(args.num_video_frames) + 1)),
                min_start,
            )
            if min_start < 0 or min_start >= frame_numbers:
                counters["bad_effective_frames"] += 1
                continue
            split = stable_test_split(
                task_id, episode_id, args.holdout_fraction, args.split_seed
            )
            record = {
                "schema_version": "agibot_episode_index_v1",
                "episode_uid": f"agibot:{task_id}:{episode_id}",
                "split": split,
                "task_id": task_id,
                "episode_id": episode_id,
                "task_name": item.get("task_name", ""),
                "init_scene_text": item.get("init_scene_text", ""),
                "video_path": str(video_path),
                "proprio_path": str(proprio_path),
                "action_video_fps_ratio": ratio,
                "action_labels": (item.get("label_info") or {}).get("flatten_action_config") or {},
                "effective_frames": effective,
                "timesteps": timesteps,
                "frame_numbers": frame_numbers,
                "valid_start_frame": min_start,
                "valid_end_frame": max_start,
                "annotation_path": str(annotation_path),
            }
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counters[f"split:{split}"] += 1
            counters["written"] += 1
            if counters["seen"] % max(1, int(args.progress_every)) == 0:
                print(
                    f"seen={counters['seen']} written={counters['written']} "
                    f"train={counters['split:train']} test={counters['split:test']}",
                    flush=True,
                )
    print(json.dumps({"output": str(output), "counts": dict(counters)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
