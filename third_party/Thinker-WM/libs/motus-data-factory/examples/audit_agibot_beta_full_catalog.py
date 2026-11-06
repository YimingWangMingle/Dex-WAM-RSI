#!/usr/bin/env python3
"""Final source and runtime audit for a full Agibot Beta Episode Catalog."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from motus_data.catalog.agibot_beta import _episode_record
from motus_data.datasets.agibot_beta import (
    TarMemberRef,
    scan_agibot_beta_episodes,
)
from motus_data.repository import AgibotBetaEpisodeReader, EpisodeRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tar-catalog-dir", required=True)
    parser.add_argument("--repository-samples", type=int, default=5)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _metadata_path(catalog: Path) -> Path:
    if catalog.suffix == ".jsonl":
        return catalog.with_suffix(".meta.json")
    return Path(str(catalog) + ".meta.json")


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _locator_details(reference: TarMemberRef) -> dict[str, Any]:
    path = Path(reference.uri)
    exists = path.is_file()
    file_size = path.stat().st_size if exists else None
    offset = int(reference.offset_data)
    size = int(reference.size)
    return {
        "path": str(path),
        "member": str(reference.member),
        "offset": offset,
        "size": size,
        "file_exists": exists,
        "file_size": file_size,
        "locator_positive": offset >= 0 and size > 0,
        "locator_within_file": (
            bool(exists)
            and offset >= 0
            and size > 0
            and offset + size <= int(file_size)
        ),
    }


def _audit_skipped_episodes(
    *,
    metadata: Mapping[str, Any],
    data_root: Path,
    tar_catalog_dir: Path,
) -> dict[str, Any]:
    build = metadata["meta"]["build"]
    expected_errors = list(build.get("error_examples") or [])
    skipped = int(build["skipped_episodes"])
    if skipped != len(expected_errors):
        raise RuntimeError(
            "error_examples do not enumerate every skipped Episode: "
            f"skipped={skipped} examples={len(expected_errors)}"
        )

    episodes, source_stats = scan_agibot_beta_episodes(
        data_root, tar_catalog_dir, ()
    )
    expected_source_stats = dict(build["source_scan"])
    if source_stats != expected_source_stats:
        raise RuntimeError(
            "current source scan differs from published Catalog metadata: "
            f"current={source_stats} published={expected_source_stats}"
        )

    by_id = {str(episode["episode_uid"]): episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise RuntimeError("source scan returned duplicate Episode IDs")

    results = []
    for expected in expected_errors:
        episode_id = str(expected["episode_id"])
        episode = by_id.get(episode_id)
        if episode is None:
            raise RuntimeError(
                f"published skipped Episode is absent from current source scan: {episode_id}"
            )
        try:
            _episode_record(episode, robot_name="agibot-a2-d")
        except Exception as error:
            actual_type = type(error).__name__
            actual_message = str(error)
        else:
            raise RuntimeError(
                f"published skipped Episode no longer reproduces its failure: {episode_id}"
            )

        expected_type = str(expected["error_type"])
        expected_message = str(expected["error"])
        if actual_type != expected_type or actual_message != expected_message:
            raise RuntimeError(
                f"skipped Episode failure changed for {episode_id}: "
                f"expected={expected_type}: {expected_message!r} "
                f"actual={actual_type}: {actual_message!r}"
            )

        action_ranges = []
        for source_index, segment in enumerate(episode["action_config"]):
            action_ranges.append(
                {
                    "source_index": source_index,
                    "start": segment.get("start_frame"),
                    "end": segment.get("end_frame"),
                }
            )
        result = {
            "episode_id": episode_id,
            "error_type": actual_type,
            "error": actual_message,
            "video": _locator_details(episode["video_ref"]),
            "proprio": _locator_details(episode["proprio_ref"]),
            "action_ranges": action_ranges,
        }
        results.append(result)
        print(
            "SKIPPED_EPISODE_REPRODUCED "
            + json.dumps(result, ensure_ascii=False),
            flush=True,
        )

    summary = {
        "source_stats": source_stats,
        "skipped_episodes": skipped,
        "reproduced": len(results),
        "episodes": results,
    }
    print(
        "SKIPPED_EPISODE_AUDIT_OK "
        + json.dumps(
            {
                "skipped_episodes": skipped,
                "reproduced": len(results),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def _sample_positions(total: int, samples: int) -> list[int]:
    if samples <= 0:
        raise ValueError("repository-samples must be positive")
    if total <= 0:
        raise ValueError("Catalog cannot be empty")
    if samples == 1:
        return [0]
    return sorted(
        {
            round((total - 1) * position / (samples - 1))
            for position in range(samples)
        }
    )


def _audit_repository(
    *,
    catalog: Path,
    bootstrap_path: Path,
    samples: int,
) -> dict[str, Any]:
    events: list[tuple[str, Mapping[str, Any]]] = []
    reader = AgibotBetaEpisodeReader(
        event_callback=lambda event, details: events.append((event, details))
    )
    started = time.monotonic()
    repository = EpisodeRepository(
        catalog,
        readers={"agibot_beta": reader},
    )
    load_seconds = time.monotonic() - started

    bootstrap_text = str(bootstrap_path)
    referenced_bootstrap_records = 0
    for record in repository._records.values():
        if bootstrap_text in json.dumps(record, ensure_ascii=False):
            referenced_bootstrap_records += 1
    if referenced_bootstrap_records:
        raise RuntimeError(
            "published Catalog still references bootstrap tar catalog: "
            f"records={referenced_bootstrap_records} path={bootstrap_text}"
        )

    episode_ids = list(repository._records)
    selected_ids = [
        episode_ids[position]
        for position in _sample_positions(len(episode_ids), int(samples))
    ]
    sample_results = []
    for episode_id in selected_ids:
        episode = repository.get_episode(episode_id)
        info = episode.info
        if min(
            info.video_count,
            info.state_count,
            info.action_count,
            info.text_count,
        ) <= 0:
            raise RuntimeError(f"Episode has an empty modality: {episode_id}")

        selection = {
            "episode_id": episode_id,
            "video_frame_indices": [0, info.video_count - 1, 0],
            "state_indices": [0, info.state_count - 1, 0],
            "action_indices": [0, info.action_count - 1, 0],
            "text_indices": [0, info.text_count - 1, 0],
        }
        event_start = len(events)
        selected = episode.read_selection(selection)
        sample_events = events[event_start:]
        event_counts = dict(Counter(event for event, _ in sample_events))
        expected_event_counts = {
            "canonical_timeline_load": 1,
            "video_member_extract": 1,
            "video_decode": 1,
        }
        if event_counts != expected_event_counts:
            raise RuntimeError(
                f"unexpected physical I/O counts for {episode_id}: "
                f"actual={event_counts} expected={expected_event_counts}"
            )
        if not torch.equal(selected.video[0], selected.video[2]):
            raise RuntimeError(f"video duplicate index mismatch: {episode_id}")
        if not torch.equal(selected.state[0], selected.state[2]):
            raise RuntimeError(f"state duplicate index mismatch: {episode_id}")
        if not torch.equal(selected.action[0], selected.action[2]):
            raise RuntimeError(f"action duplicate index mismatch: {episode_id}")
        if not torch.isfinite(selected.state).all():
            raise RuntimeError(f"state contains non-finite values: {episode_id}")
        if not torch.isfinite(selected.action).all():
            raise RuntimeError(f"action contains non-finite values: {episode_id}")
        if [item.index for item in selected.text] != [
            0,
            info.text_count - 1,
            0,
        ]:
            raise RuntimeError(f"text duplicate index mismatch: {episode_id}")

        result = {
            "episode_id": episode_id,
            "video_count": info.video_count,
            "state_count": info.state_count,
            "action_count": info.action_count,
            "text_count": info.text_count,
            "video_shape": list(selected.video.shape),
            "state_shape": list(selected.state.shape),
            "action_shape": list(selected.action.shape),
            "event_counts": event_counts,
        }
        sample_results.append(result)
        print(
            "FULL_CATALOG_REPOSITORY_SAMPLE_OK "
            + json.dumps(result, ensure_ascii=False),
            flush=True,
        )

    result = {
        "catalog_id": repository.catalog_id,
        "catalog_records": len(repository),
        "catalog_sha256": repository.catalog_sha256,
        "repository_load_seconds": load_seconds,
        "bootstrap_path": bootstrap_text,
        "bootstrap_references": referenced_bootstrap_records,
        "sample_episode_ids": selected_ids,
        "samples": sample_results,
        "rss_mb": _rss_mb(),
    }
    print(
        "FULL_CATALOG_REPOSITORY_OK "
        + json.dumps(
            {
                "catalog_records": len(repository),
                "samples": len(sample_results),
                "bootstrap_references": referenced_bootstrap_records,
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def main() -> int:
    args = parse_args()
    catalog = Path(args.catalog).resolve()
    metadata_path = _metadata_path(catalog)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    started = time.monotonic()

    exclusions = _audit_skipped_episodes(
        metadata=metadata,
        data_root=Path(args.data_root),
        tar_catalog_dir=Path(args.tar_catalog_dir),
    )
    repository = _audit_repository(
        catalog=catalog,
        bootstrap_path=Path(args.tar_catalog_dir),
        samples=int(args.repository_samples),
    )
    report = {
        "catalog": str(catalog),
        "metadata": str(metadata_path),
        "catalog_id": metadata["catalog_id"],
        "catalog_sha256": metadata["sha256"],
        "exclusions": exclusions,
        "repository": repository,
        "wall_seconds": time.monotonic() - started,
        "rss_mb": _rss_mb(),
    }
    report_path = Path(args.report).resolve()
    _write_report(report_path, report)
    print(
        "FULL_AGIBOT_BETA_CATALOG_ACCEPTANCE_OK "
        + json.dumps(
            {
                "catalog_id": metadata["catalog_id"],
                "catalog_records": metadata["episodes"],
                "skipped_reproduced": exclusions["reproduced"],
                "repository_samples": len(repository["samples"]),
                "report": str(report_path),
                "wall_seconds": report["wall_seconds"],
                "rss_mb": report["rss_mb"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
