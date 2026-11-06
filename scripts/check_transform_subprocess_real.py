#!/usr/bin/env python3
"""Real-data check: producer transform in a subprocess == in the producer thread.

Builds two training pipelines from a launch config (CPU only, same rank and
seed), one with ``system.ready_batch_transform_subprocess`` off and one on,
and compares every batch field (tensors with torch.equal). Then measures how
much a pure-Python loop in the main thread is slowed while each pipeline's
producer works, which is the GIL contention the subprocess is meant to remove.

Example: CUDA_VISIBLE_DEVICES= python scripts/check_transform_subprocess_real.py \
  --config <config_at_launch.yaml> --batches 6 --output <result.json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from thinker_wm.data_prepare.factory import create_training_data_pipeline
from thinker_wm.scripts.eval_video_ckpt import load_config


def _diff(a, b, path=""):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ok = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
              and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
        return [] if ok else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}:keys"]
        return [d for k in a for d in _diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}:len"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def _loop_rate(seconds):
    n, end = 0, time.perf_counter() + seconds
    while time.perf_counter() < end:
        n += 1
    return n / seconds


def _pipeline(cfg, rank, subprocess):
    cfg.system.ready_batch_transform_subprocess = subprocess
    return create_training_data_pipeline(cfg, rank=rank, world_size=24, seed=int(cfg.training.seed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--gil-seconds", type=float, default=30.0)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.system.ready_batch_queue_size = 2
    result = dict(config=args.config, rank=args.rank, batches=[])

    thread_pipe, sub_pipe = _pipeline(cfg, args.rank, False), _pipeline(cfg, args.rank, True)
    with thread_pipe, sub_pipe:
        for i in range(args.batches):
            ref, ref_stats = thread_pipe.next()
            got, got_stats = sub_pipe.next()
            rec = dict(batch=i, episodes=list(ref.get("episode_ids") or []),
                       mismatched=_diff(ref, got),
                       thread_transform_s=ref_stats.get("mix_batch_transform_seconds"),
                       subprocess_transform_s=got_stats.get("mix_batch_transform_seconds"))
            print(json.dumps(rec), flush=True)
            result["batches"].append(rec)
    result["all_equal"] = all(not r["mismatched"] for r in result["batches"])
    print("ALL_EQUAL", result["all_equal"], flush=True)

    # Queue size 1 keeps the producer busy for most of the measurement window.
    cfg.system.ready_batch_queue_size = 1
    gil = dict(idle=_loop_rate(args.gil_seconds))
    for name, subprocess in (("thread", False), ("subprocess", True)):
        pipe = _pipeline(cfg, args.rank, subprocess)
        with pipe:
            pipe.next()
            gil[name] = _loop_rate(args.gil_seconds)
            pipe.next()
        print(name, json.dumps(gil), flush=True)
    gil["thread_slowdown"] = 1.0 - gil["thread"] / gil["idle"]
    gil["subprocess_slowdown"] = 1.0 - gil["subprocess"] / gil["idle"]
    result["main_thread_python_rate"] = gil
    print(json.dumps(gil), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
