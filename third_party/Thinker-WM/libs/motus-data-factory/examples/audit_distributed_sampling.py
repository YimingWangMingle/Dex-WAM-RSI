#!/usr/bin/env python3
"""Audit per-rank batch composition without launching distributed training."""

from __future__ import annotations

import argparse
import json

from omegaconf import OmegaConf

from motus_data import build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    dataset = build_dataset(config, split="train")
    if not hasattr(dataset, "create_batch_sampler"):
        raise ValueError("distributed audit currently requires a mixture/batch sampler")

    report = {}
    for rank in range(args.world_size):
        sampler = dataset.create_batch_sampler(
            batch_size=int(config.training.batch_size),
            drop_last=True,
            shuffle=True,
            num_replicas=args.world_size,
            rank=rank,
            seed=args.seed,
        )
        batches = []
        for step, indices in enumerate(sampler):
            if step >= args.batches:
                break
            child_ids = {
                max(i for i, offset in enumerate(dataset._offsets) if offset <= index)
                for index in indices
            }
            if len(child_ids) != 1:
                raise RuntimeError(f"rank {rank} step {step} mixed child datasets: {child_ids}")
            child_id = next(iter(child_ids))
            batches.append({
                "step": step,
                "child": dataset.names[child_id],
                "batch_size": len(indices),
                "index_head": list(indices[:3]),
            })
        report[str(rank)] = batches
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

