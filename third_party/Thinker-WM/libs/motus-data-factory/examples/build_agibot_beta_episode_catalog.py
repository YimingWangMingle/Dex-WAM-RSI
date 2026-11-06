#!/usr/bin/env python3
"""Build one validated Agibot Beta Episode Catalog from source data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from omegaconf import OmegaConf

from motus_data.catalog.agibot_beta import build_agibot_beta_episode_catalog
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--progress-percent", type=float, default=1.0)
    parser.add_argument("--fail-on-episode-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = OmegaConf.load(config_path)
    node = config.dataset
    if str(node.get("type")) != "agibot_beta":
        raise ValueError("Catalog builder requires a single dataset.type=agibot_beta config")
    print(
        f"PHASE scan_source config={config_path} data_root={node.data_root} "
        f"tar_catalog_dir={node.tar_catalog_dir}",
        flush=True,
    )

    started = time.monotonic()
    next_percent = 0.0

    def progress(state):
        nonlocal next_percent
        processed = int(state["processed"])
        total = int(state["total"])
        percent = 100.0 * processed / max(1, total)
        if percent + 1e-9 < next_percent and processed != total:
            return
        elapsed = float(state["elapsed_seconds"])
        rate = processed / max(elapsed, 1e-9)
        eta = (total - processed) / max(rate, 1e-9)
        print(
            f"CATALOG_PROGRESS processed={processed}/{total} percent={percent:.2f}% "
            f"written={state['written']} skipped={state['skipped']} "
            f"rate={rate:.2f}/s elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
        next_percent = percent + max(0.01, float(args.progress_percent))

    metadata = build_agibot_beta_episode_catalog(
        data_root=str(node.data_root),
        tar_catalog_dir=str(node.tar_catalog_dir),
        output_path=args.output,
        catalog_id=args.catalog_id,
        data_version=args.data_version,
        source_dataset=str(node.get("source_dataset", "agibot_beta")),
        source_child=str(node.get("name", "agibot_beta")),
        robot_name=str(node.get("robot_name", "agibot-a2-d")),
        max_episodes=args.max_episodes,
        fail_on_episode_error=bool(args.fail_on_episode_error),
        force=bool(args.force),
        progress_callback=progress,
    )
    print(
        "CATALOG_OK "
        + json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "metadata": str(Path(args.output).resolve().with_suffix(".meta.json")),
                "episodes": metadata["episodes"],
                "sha256": metadata["sha256"],
                "build": metadata.get("meta", {}).get("build", {}),
                "wall_seconds": time.monotonic() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
