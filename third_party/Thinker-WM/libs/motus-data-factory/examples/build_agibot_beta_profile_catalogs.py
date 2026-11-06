#!/usr/bin/env python3
"""Build two immutable Agibot Beta child Catalogs from a full shape audit."""

from __future__ import annotations

import argparse
import json
import time

from motus_data.catalog.agibot_beta_profiles import (
    build_agibot_beta_profile_catalogs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-catalog", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.progress_every) <= 0:
        raise ValueError("progress-every must be positive")
    last_reported = 0
    started = time.monotonic()

    def progress(state):
        nonlocal last_reported
        processed = int(state["processed"])
        total = int(state["total"])
        if processed != total and processed - last_reported < int(
            args.progress_every
        ):
            return
        last_reported = processed
        elapsed = float(state["elapsed_seconds"])
        percent = 100.0 * processed / max(1, total)
        print(
            "PROFILE_CATALOG_PROGRESS "
            f"processed={processed}/{total} percent={percent:.2f}% "
            f"counts={json.dumps(state['profile_counts'], sort_keys=True)} "
            f"rate={processed / max(elapsed, 1e-9):.2f}/s "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    report = build_agibot_beta_profile_catalogs(
        parent_catalog_path=args.parent_catalog,
        audit_jsonl_path=args.audit_jsonl,
        audit_summary_path=args.audit_summary,
        output_directory=args.output_directory,
        release=args.release,
        data_version=args.data_version,
        progress_callback=progress,
    )
    print(
        "PROFILE_CATALOG_DONE "
        + json.dumps(
            {
                "output_directory": args.output_directory,
                "children": report["children"],
                "union_episodes": report["union_episodes"],
                "elapsed_seconds": time.monotonic() - started,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
