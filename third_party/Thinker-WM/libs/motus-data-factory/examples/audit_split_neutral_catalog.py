#!/usr/bin/env python3
"""Stream-audit a complete split-neutral Episode Catalog without source I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from motus_data import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    return parser.parse_args()


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _metadata_path(catalog: Path) -> Path:
    if catalog.suffix == ".jsonl":
        return catalog.with_suffix(".meta.json")
    return Path(str(catalog) + ".meta.json")


def main() -> int:
    args = parse_args()
    catalog = Path(args.catalog)
    metadata_path = _metadata_path(catalog)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_catalog_metadata_v2(metadata)
    if metadata["format"] != MOTUS_EPISODE_CATALOG_METADATA_V2:
        raise RuntimeError("Catalog is not concise split-neutral v2")
    if "split_policy" in metadata:
        raise RuntimeError("split-neutral metadata contains split_policy")

    started = time.monotonic()
    digest = hashlib.sha256()
    episode_ids: set[str] = set()
    first_episode_id = None
    rows = 0
    with catalog.open("rb") as handle:
        for line_number, payload in enumerate(handle, start=1):
            digest.update(payload)
            if not payload.strip():
                raise RuntimeError(f"blank Catalog row at line {line_number}")
            record = json.loads(payload.decode("utf-8"))
            validate_episode_catalog_binding_v2(record, metadata)
            if "split" in record:
                raise RuntimeError(
                    f"split-neutral Episode contains split at line {line_number}"
                )
            episode_id = str(record["episode_id"])
            if episode_id in episode_ids:
                raise RuntimeError(
                    f"duplicate episode_id={episode_id!r} at line {line_number}"
                )
            episode_ids.add(episode_id)
            if first_episode_id is None:
                first_episode_id = episode_id
            rows += 1

    expected_rows = int(metadata["episodes"])
    if rows != expected_rows:
        raise RuntimeError(
            f"Catalog rows={rows} differ from metadata={expected_rows}"
        )
    actual_sha256 = digest.hexdigest()
    expected_sha256 = str(metadata["sha256"])
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Catalog SHA256 mismatch: actual={actual_sha256} "
            f"expected={expected_sha256}"
        )
    print(
        "SPLIT_NEUTRAL_CATALOG_OK "
        + json.dumps(
            {
                "catalog": str(catalog.resolve()),
                "catalog_id": metadata["catalog_id"],
                "records": rows,
                "unique_episode_ids": len(episode_ids),
                "first_episode_id": first_episode_id,
                "catalog_size_bytes": catalog.stat().st_size,
                "catalog_sha256": actual_sha256,
                "split_policy_present": False,
                "episode_split_present": False,
                "scan_seconds": time.monotonic() - started,
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
