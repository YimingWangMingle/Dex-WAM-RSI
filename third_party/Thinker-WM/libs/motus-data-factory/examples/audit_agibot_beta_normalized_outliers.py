#!/usr/bin/env python3
"""Locate normalized Agibot Beta outliers with Episode/timestep provenance."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from motus_data.agibot_beta_normalization import (
    AGIBOT_BETA_NORMALIZATION_FORMAT,
)
from motus_data.agibot_beta_profiles import read_agibot_beta_native_timeline
from motus_data.catalog.sqlite_index import catalog_metadata_path
from motus_data.index_contracts import (
    validate_catalog_metadata,
    validate_episode_catalog_binding,
)
from motus_data.storage import BoundedFileReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--absolute-threshold", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-indices-per-feature", type=int, default=20)
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _state_storage(record: Mapping[str, Any]) -> Mapping[str, Any]:
    storage = record["state"]["storage"]
    if storage.get("type") != "tar":
        raise ValueError("Agibot Beta outlier audit requires tar state storage")
    return storage


def main() -> int:
    args = parse_args()
    catalog = Path(args.catalog).resolve()
    metadata_path = catalog_metadata_path(catalog)
    normalization_path = Path(args.normalization).resolve()
    output = Path(args.output_jsonl).resolve()
    summary_path = Path(args.summary_json).resolve()
    if output.exists() or summary_path.exists():
        raise FileExistsError("outlier audit output already exists")
    if int(args.progress_every) <= 0 or int(args.max_indices_per_feature) <= 0:
        raise ValueError("progress and index limits must be positive")
    threshold = float(args.absolute_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("absolute-threshold must be finite and positive")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_catalog_metadata(metadata)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if normalization.get("format") != AGIBOT_BETA_NORMALIZATION_FORMAT:
        raise ValueError("unsupported normalization artifact format")
    expected_binding = {
        "catalog_id": str(metadata["catalog_id"]),
        "sha256": str(metadata["sha256"]),
        "source": str(metadata["source"]),
        "child": str(metadata["child"]),
        "episodes": int(metadata["episodes"]),
    }
    if normalization.get("catalog") != expected_binding:
        raise ValueError("normalization artifact is bound to another Catalog")
    dimension = int(normalization["dimension"])
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    scale = np.asarray(normalization["applied_scale"], dtype=np.float64)
    feature_names = tuple(str(value) for value in normalization["feature_names"])
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or len(feature_names) != dimension
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid normalization vectors")

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    started = time.monotonic()
    processed = 0
    violating_episodes = 0
    violating_values = 0
    feature_episode_counts: Counter[str] = Counter()
    feature_value_counts: Counter[str] = Counter()
    maximum: dict[str, Any] | None = None
    with temporary_output.open("x", encoding="utf-8") as output_handle:
        with catalog.open("rb") as catalog_handle:
            for payload in catalog_handle:
                record = json.loads(payload.decode("utf-8"))
                validate_episode_catalog_binding(record, metadata)
                storage = _state_storage(record)
                with BoundedFileReader(
                    storage["path"], storage["offset"], storage["size"]
                ) as member:
                    timeline, profile = read_agibot_beta_native_timeline(
                        member,
                        expected_profile=str(normalization["effector_profile"]),
                    )
                if timeline.shape[1] != dimension:
                    raise ValueError(
                        f"dimension mismatch for {record['episode_id']}"
                    )
                normalized = (timeline.astype(np.float64) - mean) / scale
                violations = np.abs(normalized) > threshold
                if np.any(violations):
                    violating_episodes += 1
                    episode_violations = []
                    for feature_index in np.flatnonzero(violations.any(axis=0)):
                        indices = np.flatnonzero(violations[:, feature_index])
                        name = feature_names[int(feature_index)]
                        feature_episode_counts[name] += 1
                        feature_value_counts[name] += int(indices.size)
                        violating_values += int(indices.size)
                        local = normalized[indices, feature_index]
                        raw = timeline[indices, feature_index]
                        local_position = int(np.argmax(np.abs(local)))
                        candidate = {
                            "episode_id": str(record["episode_id"]),
                            "task_id": int((record.get("meta") or {})["task_id"]),
                            "feature": name,
                            "feature_index": int(feature_index),
                            "timestep": int(indices[local_position]),
                            "raw_value": float(raw[local_position]),
                            "normalized_value": float(local[local_position]),
                        }
                        if maximum is None or abs(
                            candidate["normalized_value"]
                        ) > abs(maximum["normalized_value"]):
                            maximum = candidate
                        episode_violations.append(
                            {
                                "feature": name,
                                "feature_index": int(feature_index),
                                "count": int(indices.size),
                                "indices": [
                                    int(value)
                                    for value in indices[
                                        : int(args.max_indices_per_feature)
                                    ]
                                ],
                                "raw_min": float(raw.min()),
                                "raw_max": float(raw.max()),
                                "normalized_min": float(local.min()),
                                "normalized_max": float(local.max()),
                            }
                        )
                    row = {
                        "episode_id": str(record["episode_id"]),
                        "task_id": int((record.get("meta") or {})["task_id"]),
                        "profile": profile.name,
                        "violations": episode_violations,
                    }
                    output_handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                processed += 1
                if processed % int(args.progress_every) == 0:
                    elapsed = time.monotonic() - started
                    print(
                        "OUTLIER_AUDIT_PROGRESS "
                        f"processed={processed}/{metadata['episodes']} "
                        f"violating_episodes={violating_episodes} "
                        f"violating_values={violating_values} "
                        f"rate={processed / max(elapsed, 1e-9):.2f}/s "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if processed != int(metadata["episodes"]):
        raise ValueError("outlier audit did not cover the complete Catalog")
    os.replace(temporary_output, output)
    summary = {
        "format": "motus_agibot_beta_normalized_outlier_audit_v1",
        "catalog": expected_binding,
        "normalization": str(normalization_path),
        "normalization_profile": str(normalization["normalization_profile"]),
        "absolute_threshold": threshold,
        "processed_episodes": processed,
        "violating_episodes": violating_episodes,
        "violating_values": violating_values,
        "feature_episode_counts": dict(sorted(feature_episode_counts.items())),
        "feature_value_counts": dict(sorted(feature_value_counts.items())),
        "maximum_absolute_violation": maximum,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(summary_path, summary)
    print("OUTLIER_AUDIT_DONE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
