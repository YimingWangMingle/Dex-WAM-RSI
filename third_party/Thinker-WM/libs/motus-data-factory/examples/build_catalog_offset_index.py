#!/usr/bin/env python3
"""Build an immutable SQLite byte-offset sidecar for an Episode Catalog."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from motus_data.catalog.sqlite_index import build_catalog_offset_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output")
    parser.add_argument("--progress-percent", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = Path(args.catalog).resolve()
    total_bytes = catalog.stat().st_size
    next_percent = 0.0
    started = time.monotonic()

    def progress(state):
        nonlocal next_percent
        percent = 100.0 * int(state["bytes"]) / max(1, int(state["total_bytes"]))
        if percent + 1e-9 < next_percent and int(state["bytes"]) != total_bytes:
            return
        elapsed = time.monotonic() - started
        rate = int(state["bytes"]) / max(elapsed, 1e-9)
        eta = (total_bytes - int(state["bytes"])) / max(rate, 1e-9)
        print(
            f"CATALOG_INDEX_PROGRESS records={state['records']} "
            f"bytes={state['bytes']}/{total_bytes} percent={percent:.2f}% "
            f"rate_mb_s={rate / (1024 * 1024):.2f} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
        next_percent = percent + max(0.01, float(args.progress_percent))

    result = build_catalog_offset_index(
        catalog,
        output_path=args.output,
        force=bool(args.force),
        progress_callback=progress,
    )
    print(
        "CATALOG_OFFSET_INDEX_OK "
        + json.dumps(
            {
                **result,
                "wall_seconds": time.monotonic() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
