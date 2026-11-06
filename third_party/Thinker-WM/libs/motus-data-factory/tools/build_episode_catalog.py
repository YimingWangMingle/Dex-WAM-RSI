#!/usr/bin/env python3
"""Build and index one split-neutral Episode Catalog."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from motus_data.catalog.agibot import build_agibot_episode_catalog
from motus_data.catalog.egoverse import build_egoverse_episode_catalog
from motus_data.catalog.openvid import build_openvid_episode_catalog
from motus_data.catalog.robotmind import build_robotmind_episode_catalog
from motus_data.catalog.source_families import (
    build_source_family_episode_catalog,
)
from motus_data.catalog.sqlite_index import build_catalog_offset_index


class ProgressPrinter:
    def __init__(self, interval_seconds: float = 5.0):
        self.interval_seconds = float(interval_seconds)
        self.last_print = 0.0

    def __call__(self, state: Mapping[str, Any]) -> None:
        now = time.monotonic()
        if not state.get("done") and now - self.last_print < self.interval_seconds:
            return
        self.last_print = now
        percent = float(state.get("percent", 0.0))
        fields = []
        if "phase" in state:
            fields.append(f"phase={state['phase']}")
        fields.append(f"progress={percent:.2f}%")
        for key in (
            "processed",
            "total",
            "discovered_episodes",
            "scan_errors",
            "input_rows",
            "written",
            "written_episodes",
            "skipped",
            "skipped_episodes",
            "elapsed_seconds",
        ):
            if key in state:
                value = state[key]
                fields.append(
                    f"{key}={float(value):.1f}"
                    if key == "elapsed_seconds"
                    else f"{key}={value}"
                )
        print(" ".join(fields), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        required=True,
        choices=("agibot", "robotwin", "robotmind", "egoverse", "openvid"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--source-child", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--csv-path")
    parser.add_argument("--data-root")
    parser.add_argument("--embodiment")
    parser.add_argument("--profile")
    parser.add_argument("--archive-numbers", nargs="+", type=int)
    parser.add_argument("--annotation-folders", nargs="+")
    parser.add_argument("--data-root-folders", nargs="+")
    parser.add_argument("--dataset-names", nargs="+")
    parser.add_argument("--robot-name")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--verify-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-episode-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-store-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="EgoVerse: fail instead of reporting incomplete Zarr stores",
    )
    parser.add_argument(
        "--validate-numeric-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-jpeg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-missing-hdf5",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--build-index",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="EgoVerse: resume from the last fsynced build checkpoint",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="EgoVerse: fsync and checkpoint after this many Episodes",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.family not in {"egoverse", "openvid"}:
        raise ValueError(
            "--resume is currently supported only for EgoVerse and OpenVid"
        )
    output = Path(args.output)
    progress = ProgressPrinter()
    if args.family == "agibot":
        if not (
            args.annotation_folders
            and args.data_root_folders
            and args.dataset_names
        ):
            raise ValueError(
                "Agibot requires --annotation-folders, --data-root-folders "
                "and --dataset-names"
            )
        if not (
            len(args.annotation_folders)
            == len(args.data_root_folders)
            == len(args.dataset_names)
        ):
            raise ValueError("Agibot source path lists must have equal lengths")
        metadata = build_agibot_episode_catalog(
            annotation_folders=args.annotation_folders,
            data_root_folders=args.data_root_folders,
            dataset_names=args.dataset_names,
            output_path=output,
            catalog_id=args.catalog_id,
            data_version=args.data_version,
            source_child=args.source_child,
            robot_name=args.robot_name or "agibot-a2-d",
            max_episodes=args.max_episodes,
            fail_on_episode_error=args.fail_on_episode_error,
            force=args.force,
            progress_callback=progress,
        )
    elif args.family == "egoverse":
        if not (args.data_root and args.profile):
            raise ValueError(
                "EgoVerse requires --data-root and --profile"
            )
        if args.manifest:
            raise ValueError(
                "EgoVerse Catalogs are built directly from Zarr Episodes; "
                "--manifest is not supported"
            )
        metadata = build_egoverse_episode_catalog(
            data_root=args.data_root,
            profile=args.profile,
            output_path=output,
            catalog_id=args.catalog_id,
            data_version=args.data_version,
            source_child=args.source_child,
            max_episodes=args.max_episodes,
            validate_numeric_finite=args.validate_numeric_finite,
            validate_jpeg=args.validate_jpeg,
            fail_on_episode_error=args.fail_on_episode_error,
            fail_on_store_error=args.fail_on_store_error,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            force=args.force,
            progress_callback=progress,
        )
    elif args.family == "openvid":
        if not (args.data_root and args.csv_path):
            raise ValueError(
                "OpenVid requires --data-root and --csv-path"
            )
        if args.manifest:
            raise ValueError(
                "OpenVid complete-video Catalogs are built from CSV+ZIP; "
                "--manifest is only an audit input"
            )
        metadata = build_openvid_episode_catalog(
            data_root=args.data_root,
            csv_path=args.csv_path,
            output_path=output,
            catalog_id=args.catalog_id,
            data_version=args.data_version,
            source_child=args.source_child,
            archive_numbers=args.archive_numbers,
            max_episodes=args.max_episodes,
            fail_on_row_error=args.fail_on_episode_error,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            force=args.force,
            progress_callback=progress,
        )
    elif args.family == "robotmind" and (args.data_root or args.embodiment):
        if not (args.data_root and args.embodiment):
            raise ValueError(
                "direct RobotMind construction requires --data-root and --embodiment"
            )
        if args.manifest:
            raise ValueError(
                "choose either direct RobotMind --data-root/--embodiment "
                "or legacy --manifest bootstrap"
            )
        metadata = build_robotmind_episode_catalog(
            data_root=args.data_root,
            embodiment=args.embodiment,
            output_path=output,
            catalog_id=args.catalog_id,
            data_version=args.data_version,
            source_child=args.source_child,
            robot_name=args.robot_name,
            max_episodes=args.max_episodes,
            validate_numeric_finite=args.validate_numeric_finite,
            validate_jpeg=args.validate_jpeg,
            fail_on_episode_error=args.fail_on_episode_error,
            fail_on_missing_hdf5=args.fail_on_missing_hdf5,
            force=args.force,
            progress_callback=progress,
        )
    else:
        if not args.manifest:
            raise ValueError(f"{args.family} requires --manifest")
        metadata = build_source_family_episode_catalog(
            manifest_path=args.manifest,
            output_path=output,
            family=args.family,
            source_child=args.source_child,
            catalog_id=args.catalog_id,
            data_version=args.data_version,
            verify_source=args.verify_source,
            max_episodes=args.max_episodes,
            fail_on_episode_error=args.fail_on_episode_error,
            force=args.force,
            progress_callback=progress,
        )

    result = {"catalog": str(output), "metadata": metadata}
    if args.build_index:
        result["index"] = build_catalog_offset_index(
            output,
            force=args.force,
            progress_callback=progress,
        )
    print(
        "CATALOG_BUILD_DONE "
        + json.dumps(result, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
