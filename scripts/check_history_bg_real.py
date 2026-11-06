#!/usr/bin/env python3
"""Real-data check: producer-side history == original foreground read, bitwise.

Builds the training pipeline from a launch config (CPU only), takes N queued
batches whose history was attached in the producer, re-attaches the same
batches with the original read path (one read per slot group, invalid rows
read as frame 0), and compares the nine history fields with torch.equal.
Also times original vs valid-row read on those batches.

Example: CUDA_VISIBLE_DEVICES= python scripts/check_history_bg_real.py \
  --config <config_at_launch.yaml> --batches 6 --output <result.json>
"""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from thinker_wm.data_prepare.factory import create_training_data_pipeline
from thinker_wm.data_prepare.history_attach import HISTORY_KEYS, HistoryAttacher, build_history_attacher
from thinker_wm.scripts.eval_video_ckpt import load_config


def _old_read(self, plans, hw):
    res = {}
    for i, (reader, ep, w) in plans.items():
        s, r = reader.rgb(ep, w.sparse_rows.tolist(), hw), reader.rgb(ep, w.recent_rows.tolist(), hw)
        res[i] = None if s is None or r is None else (s, r)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--rank", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.system.ready_batch_queue_size = 2
    world = 24
    pipe = create_training_data_pipeline(cfg, rank=args.rank, world_size=world,
                                         seed=int(cfg.training.seed))
    old_att, new_att = build_history_attacher(cfg), build_history_attacher(cfg)
    rows, result = [], dict(config=args.config, rank=args.rank, world_size=world)
    with pipe:
        for i in range(args.batches):
            batch, stats = pipe.next()
            if not all(k in batch for k in HISTORY_KEYS if k.endswith(("_rgb", "_valid"))):
                raise RuntimeError("producer did not attach history")
            bare = {k: v for k, v in batch.items() if k not in HISTORY_KEYS}
            t0 = time.perf_counter()
            with mock.patch.object(HistoryAttacher, "_read_rgb", _old_read):
                ref = old_att.attach(bare, step=i)
            t1 = time.perf_counter()
            again = new_att.attach(bare, step=i)
            t2 = time.perf_counter()
            mism = [k for k in HISTORY_KEYS if (k in ref) != (k in batch) or
                    (k in ref and not torch.equal(ref[k], batch[k]))]
            mism += [f"recompute:{k}" for k in HISTORY_KEYS if k in ref and not torch.equal(ref[k], again[k])]
            rec = dict(batch=i, episodes=list(batch.get("episode_ids") or []),
                       mismatched=mism, old_read_s=t1 - t0, new_read_s=t2 - t1,
                       producer_transform_s=stats.get("mix_batch_transform_seconds"),
                       sparse_valid=float(ref["history_sparse_valid"].float().sum(1).mean()),
                       recent_valid=float(ref["history_recent_valid"].float().sum(1).mean()),
                       metrics_equal=old_att.last_metrics == new_att.last_metrics)
            print(json.dumps(rec), flush=True)
            rows.append(rec)
    result.update(batches=rows, all_equal=all(not r["mismatched"] and r["metrics_equal"] for r in rows))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("ALL_EQUAL", result["all_equal"], flush=True)


if __name__ == "__main__":
    main()
