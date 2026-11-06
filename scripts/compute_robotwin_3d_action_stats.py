#!/usr/bin/env python3
"""Parquet-only action/state stats for robotwin_3d (no video / repository).

Same split + trimmed mean/std as data_prepare/compute_action_stats.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare.compute_action_stats import (  # noqa: E402
    DEFAULT_MAX_SAMPLES_PER_DIM,
    DEFAULT_Q_HIGH,
    DEFAULT_Q_LOW,
    _StatsAccumulator,
    _finalize_stats,
    _split_chunks,
)
from thinker_wm.data_prepare.robotwin_3d import QPOS_DIM  # noqa: E402
from thinker_wm.data_prepare.sampling_ranges import DEFAULT_SPLIT_SEED, _split_keeps  # noqa: E402


def _read_qpos(path: str, column: str) -> torch.Tensor:
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=[column])
    values = np.stack(
        [np.asarray(row, dtype=np.float32) for row in table.column(column).to_pylist()],
        axis=0,
    )
    if values.ndim != 2 or int(values.shape[1]) != QPOS_DIM:
        raise ValueError(f"{column} shape {values.shape} in {path}")
    return torch.from_numpy(values.copy())


def _scan_chunk(rows: list[tuple[str, str]], capacity: int, seed: int) -> dict:
    state_acc = _StatsAccumulator(QPOS_DIM, capacity, seed)
    action_acc = _StatsAccumulator(QPOS_DIM, capacity, seed + 1)
    skipped = 0
    for parquet_path, _eid in rows:
        try:
            state = _read_qpos(parquet_path, "observation.state")
            action = _read_qpos(parquet_path, "action")
            state_acc.update(state)
            action_acc.update(action)
        except Exception:
            skipped += 1
    return {
        "state": state_acc.partial(),
        "action": action_acc.partial(),
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default=str(
            _REPO / "tmp/catalogs/robotwin_3d/robotwin_3d.episode_catalog.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            _REPO / "tmp/catalogs/robotwin_3d/robotwin_3d.train.action_stats.json"
        ),
    )
    parser.add_argument("--split", default="train", choices=("all", "train", "holdout"))
    parser.add_argument("--holdout-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--robot", default="aloha")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--q-low", type=float, default=DEFAULT_Q_LOW)
    parser.add_argument("--q-high", type=float, default=DEFAULT_Q_HIGH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    catalog = Path(args.catalog)
    output = Path(args.output)
    if output.is_file() and not args.force:
        raise FileExistsError(output)

    rows: list[tuple[str, str]] = []
    catalog_id = ""
    meta_path = catalog.with_name(catalog.name.replace(".jsonl", ".meta.json"))
    if meta_path.is_file():
        catalog_id = str(json.loads(meta_path.read_text()).get("catalog_id") or "")
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            eid = str(rec["episode_id"])
            if not _split_keeps(
                args.split,
                eid,
                holdout_fraction=args.holdout_fraction,
                split_seed=int(args.split_seed),
            ):
                continue
            rows.append((str(rec["state"]["storage"]["path"]), eid))

    if not rows:
        raise RuntimeError("no episodes in split")

    capacity = DEFAULT_MAX_SAMPLES_PER_DIM
    if args.workers > 1:
        from multiprocessing import Pool

        chunks = _split_chunks(rows, int(args.workers))
        with Pool(min(int(args.workers), len(chunks))) as pool:
            partials = pool.starmap(
                _scan_chunk,
                [(chunk, capacity, int(args.seed) + i) for i, chunk in enumerate(chunks)],
            )
    else:
        partials = [_scan_chunk(rows, capacity, int(args.seed))]

    skipped = sum(int(p["skipped"]) for p in partials)
    state_stats = _finalize_stats([p["state"] for p in partials], args.q_low, args.q_high)
    action_stats = _finalize_stats([p["action"] for p in partials], args.q_low, args.q_high)

    def _shape(stats: dict) -> dict:
        return {
            "count": stats["count"],
            "n_seen": stats["n_seen"],
            "dim": stats["dim"],
            "mean": [d["mean"] for d in stats["per_dim"]],
            "std": [d["std"] for d in stats["per_dim"]],
            "mean_raw": [d["mean_raw"] for d in stats["per_dim"]],
            "std_raw": [d["std_raw"] for d in stats["per_dim"]],
            "q_low": [d["q_low"] for d in stats["per_dim"]],
            "q_high": [d["q_high"] for d in stats["per_dim"]],
            "median": [d["median"] for d in stats["per_dim"]],
            "min": [d["min"] for d in stats["per_dim"]],
            "max": [d["max"] for d in stats["per_dim"]],
            "inlier_count": [d["count"] for d in stats["per_dim"]],
        }

    payload = {
        "dataset": "robotwin_3d",
        "robot": args.robot,
        "catalog_id": catalog_id,
        "episodes": len(rows),
        "episodes_skipped": skipped,
        "split": args.split,
        "holdout_fraction": args.holdout_fraction,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "dim": QPOS_DIM,
        "state": _shape(state_stats),
        "action": _shape(action_stats),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(output)
    print(
        f"STATS_OK robot={args.robot} episodes={len(rows)} skipped={skipped} "
        f"state_rows={state_stats['count']} action_rows={action_stats['count']} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
