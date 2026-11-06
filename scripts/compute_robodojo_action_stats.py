#!/usr/bin/env python3
"""RoboDojo state/action statistics from the train split only (HDF5 rows, no video).

Same trimmed mean/std as data_prepare/compute_action_stats.py; measured state
and control action are accumulated separately. The output records the catalog
hash, split file and field-map digest so a stats file can be tied to its data.

  python scripts/compute_robodojo_action_stats.py --catalog tmp/catalogs/robodojo/robodojo.episode_catalog.jsonl \
      --split-file tmp/catalogs/robodojo/robodojo.split.json --field-map cfg.yaml --robot <name> \
      --output tmp/catalogs/robodojo/robodojo.train.action_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare.compute_action_stats import (  # noqa: E402
    DEFAULT_MAX_SAMPLES_PER_DIM, DEFAULT_Q_HIGH, DEFAULT_Q_LOW, _StatsAccumulator, _finalize_stats,
)
from thinker_wm.data_prepare.robodojo_core import FieldMap, RoboDojoHDF5  # noqa: E402


def _shape(stats: dict) -> dict:
    keys = ("mean", "std", "mean_raw", "std_raw", "q_low", "q_high", "median", "min", "max")
    out = {"count": stats["count"], "n_seen": stats["n_seen"], "dim": stats["dim"]}
    out.update({k: [d[k] for d in stats["per_dim"]] for k in keys})
    out["inlier_count"] = [d["count"] for d in stats["per_dim"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--field-map", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--q-low", type=float, default=DEFAULT_Q_LOW)
    ap.add_argument("--q-high", type=float, default=DEFAULT_Q_HIGH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists() and not a.force:
        raise FileExistsError(out)
    fm = FieldMap.load(a.field_map)
    io = RoboDojoHDF5(fm)
    split = json.loads(Path(a.split_file).read_text())
    train = set(split["train"])
    meta = json.loads(Path(a.catalog.replace(".jsonl", ".meta.json")).read_text())
    if meta.get("meta", {}).get("field_map_digest") not in (None, fm.digest):
        raise SystemExit("catalog was built with a different field map")
    state_acc = action_acc = None
    episodes = skipped = 0
    for line in open(a.catalog, encoding="utf-8"):
        rec = json.loads(line)
        if rec["episode_id"] not in train:
            continue
        path = rec["state"]["storage"]["path"]
        try:
            n = int(rec["state"]["count"])
            s = torch.from_numpy(io.read_rows(path, "state", range(n)))
            act = torch.from_numpy(io.read_rows(path, "action", range(n)))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if state_acc is None:
            state_acc = _StatsAccumulator(s.shape[1], DEFAULT_MAX_SAMPLES_PER_DIM, a.seed)
            action_acc = _StatsAccumulator(act.shape[1], DEFAULT_MAX_SAMPLES_PER_DIM, a.seed + 1)
        state_acc.update(s)
        action_acc.update(act)
        episodes += 1
    if not episodes:
        raise SystemExit("no train episodes readable")
    payload = {
        "dataset": "robodojo", "robot": a.robot, "split": "train",
        "catalog_id": meta.get("catalog_id"), "catalog_sha256": meta.get("sha256"),
        "split_meta": split.get("meta"), "field_map_digest": fm.digest,
        "action_semantics": fm.action_semantics,
        "episodes": episodes, "episodes_skipped": skipped, "q_low": a.q_low, "q_high": a.q_high,
        "state": _shape(_finalize_stats([state_acc.partial()], a.q_low, a.q_high)),
        "action": _shape(_finalize_stats([action_acc.partial()], a.q_low, a.q_high)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"STATS_OK episodes={episodes} skipped={skipped} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
