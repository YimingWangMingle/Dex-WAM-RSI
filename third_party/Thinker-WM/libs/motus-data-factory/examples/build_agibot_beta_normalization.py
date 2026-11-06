#!/usr/bin/env python3
"""Build Catalog-bound native Agibot Beta normalization statistics."""

from __future__ import annotations

import argparse
import json

from motus_data.agibot_beta_normalization import (
    DEFAULT_CONSTANT_STD_THRESHOLD,
    build_agibot_beta_normalization_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--constant-std-threshold",
        type=float,
        default=DEFAULT_CONSTANT_STD_THRESHOLD,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.progress_every) <= 0:
        raise ValueError("progress-every must be positive")
    last_reported = 0

    def progress(state):
        nonlocal last_reported
        processed = int(state["processed_episodes"])
        total = int(state["total_episodes"])
        if processed != total and processed - last_reported < int(
            args.progress_every
        ):
            return
        last_reported = processed
        elapsed = float(state["elapsed_seconds"])
        print(
            "NORMALIZATION_PROGRESS "
            f"profile={state['profile']} processed={processed}/{total} "
            f"percent={100.0 * processed / max(1, total):.2f}% "
            f"timesteps={state['timesteps']} "
            f"rate={processed / max(elapsed, 1e-9):.2f}/s "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    report = build_agibot_beta_normalization_artifact(
        catalog_path=args.catalog,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        resume=bool(args.resume),
        checkpoint_every=int(args.checkpoint_every),
        constant_std_threshold=float(args.constant_std_threshold),
        progress_callback=progress,
    )
    print(
        "NORMALIZATION_DONE "
        + json.dumps(report, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
