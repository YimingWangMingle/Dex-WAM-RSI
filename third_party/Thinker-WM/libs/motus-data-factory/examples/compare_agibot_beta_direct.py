#!/usr/bin/env python3
"""Compare direct Agibot Beta samples with fixed records from an old Manifest."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from motus_data import build_dataset
from motus_data.datasets.manifest import MotusManifestDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-scan-lines", type=int, default=100000)
    return parser.parse_args()


def tensor_result(name: str, direct: torch.Tensor, legacy: torch.Tensor) -> dict:
    if tuple(direct.shape) != tuple(legacy.shape):
        return {
            "name": name,
            "direct_shape": list(direct.shape),
            "legacy_shape": list(legacy.shape),
            "shape_equal": False,
            "exact": False,
            "allclose": False,
            "max_abs_diff": None,
        }
    difference = (direct.float() - legacy.float()).abs()
    return {
        "name": name,
        "shape": list(direct.shape),
        "shape_equal": True,
        "exact": bool(torch.equal(direct, legacy)),
        "allclose": bool(torch.allclose(direct, legacy, atol=1e-6, rtol=1e-6)),
        "max_abs_diff": float(difference.max().item()) if difference.numel() else 0.0,
    }


def main() -> int:
    args = parse_args()
    print(f"PHASE load_config path={args.config}", flush=True)
    config = OmegaConf.load(args.config)
    print("PHASE build_direct_dataset", flush=True)
    direct = build_dataset(config, split="train", rank=0, seed=20260720)
    print(
        f"DIRECT_READY episodes={len(direct.episodes)} "
        f"total_usable={direct.total_episode_count} scan_stats={direct.scan_stats}",
        flush=True,
    )
    episode_positions = {
        (int(item["task_id"]), int(item["episode_id"])): index
        for index, item in enumerate(direct.episodes)
    }

    selected = []
    selected_episode_keys = set()
    scanned = 0
    print(f"PHASE scan_legacy_manifest path={args.legacy_manifest}", flush=True)
    with open(args.legacy_manifest, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            scanned = line_number
            if line_number > int(args.max_scan_lines):
                break
            if line_number % 10000 == 0:
                print(
                    f"MANIFEST_PROGRESS lines={line_number} selected={len(selected)}",
                    flush=True,
                )
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            try:
                key = (int(metadata["task_id"]), int(metadata["episode_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key not in episode_positions:
                continue
            if key in selected_episode_keys:
                continue
            frame_indices = (record.get("video") or {}).get("frame_indices") or []
            if len(frame_indices) != direct.frames_clip_num:
                continue
            record["split"] = "train"
            selected.append(record)
            selected_episode_keys.add(key)
            if len(selected) >= int(args.samples):
                break
    if len(selected) < int(args.samples):
        raise RuntimeError(
            f"found only {len(selected)} comparable records after scanning {scanned} lines"
        )

    common = config.get("common", {})
    failures = []
    print(
        f"SELECTED records={len(selected)} distinct_episodes={len(selected_episode_keys)} "
        f"scanned_lines={scanned}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="motus_beta_parity_") as temp_dir:
        manifest_path = Path(temp_dir) / "selected.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected),
            encoding="utf-8",
        )
        legacy = MotusManifestDataset(
            manifest_path=str(manifest_path),
            split="train",
            video_size=(
                int(common.get("video_height", 320)),
                int(common.get("video_width", 384)),
            ),
            num_video_frames=int(common.get("num_video_frames", 48)),
            video_action_freq_ratio=int(common.get("video_action_freq_ratio", 1)),
            action_dim=int(common.get("action_dim", 26)),
            state_dim=int(common.get("state_dim", 26)),
            source_dataset="agibot_beta",
            robot_name="agibot-a2-d",
            max_decode_retries=1,
            action_magnitude_guard_threshold=100,
            action_magnitude_guard_sources={"agibot_beta"},
        )

        for index, record in enumerate(selected):
            metadata = record["metadata"]
            key = (int(metadata["task_id"]), int(metadata["episode_id"]))
            start = int((record.get("action") or {}).get("start_point"))
            direct_sample = direct.load_window(episode_positions[key], start)
            legacy_sample = legacy[index]
            tensors = [
                tensor_result(name, direct_sample[name], legacy_sample[name])
                for name in (
                    "first_frame",
                    "video_frames",
                    "initial_state",
                    "action_sequence",
                )
            ]
            text_equal = (
                direct_sample["t5_input_message"].strip()
                == legacy_sample["t5_input_message"].strip()
                and direct_sample["vlm_input_message"].strip()
                == legacy_sample["vlm_input_message"].strip()
            )
            frame_indices_equal = direct_sample["frame_indices"] == list(
                record["video"]["frame_indices"]
            )
            passed = frame_indices_equal and text_equal and all(
                item["allclose"] for item in tensors
            )
            output = {
                "sample": index,
                "task_id": key[0],
                "episode_id": key[1],
                "start_frame": start,
                "frame_indices_equal": frame_indices_equal,
                "text_equal_after_strip": text_equal,
                "tensors": tensors,
                "passed": passed,
            }
            print(json.dumps(output, ensure_ascii=False), flush=True)
            if not passed:
                failures.append(output)
        legacy.close()
    summary = {
        "samples": len(selected),
        "passed": len(selected) - len(failures),
        "failed": len(failures),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
