#!/usr/bin/env python3
"""Build or validate the SQLite offset index for one Episode Catalog."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Mapping

from motus_data.catalog.sqlite_index import build_catalog_offset_index


class ProgressPrinter:
    def __init__(self, interval_seconds: float = 5.0):
        self.interval_seconds = float(interval_seconds)
        self.last_print = 0.0

    def __call__(self, state: Mapping[str, Any]) -> None:
        now = time.monotonic()
        if now - self.last_print < self.interval_seconds:
            return
        self.last_print = now
        processed = int(state["bytes"])
        total = int(state["total_bytes"])
        percent = 100.0 * processed / total if total else 100.0
        print(
            f"index_progress={percent:.2f}% "
            f"records={int(state['records'])} "
            f"bytes={processed}/{total}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_catalog_offset_index(
        args.catalog,
        force=args.force,
        progress_callback=ProgressPrinter(),
    )
    print(
        "CATALOG_INDEX_DONE "
        + json.dumps(result, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
