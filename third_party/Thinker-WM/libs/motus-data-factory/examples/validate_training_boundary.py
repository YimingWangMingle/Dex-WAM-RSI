#!/usr/bin/env python3
"""Validate real or synthetic data through queue and optimizer boundaries."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from motus_data import build_data_pipeline, validate_motus_model_batch_v1
from motus_data.verification import run_optimizer_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--contract-profile",
        choices=["generic", "motus_model_batch_v1"],
        default="generic",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.system.num_workers = int(args.num_workers)
        if args.num_workers == 0:
            config.system.persistent_workers = False
            config.system.prefetch_factor = None
    if args.queue_size is not None:
        config.system.ready_batch_queue_size = int(args.queue_size)

    pipeline = build_data_pipeline(
        config,
        split="train",
        rank=args.rank,
        world_size=args.world_size,
        seed=args.seed,
    )
    dataset = pipeline.dataset
    description = {
        "dataset_class": type(dataset).__name__,
        "dataset_length": len(dataset),
        "batch_size": int(config.training.batch_size),
        "num_workers": int(config.system.get("num_workers", 0)),
        "queue_size": int(config.system.get("ready_batch_queue_size", 1)),
        "rank": args.rank,
        "world_size": args.world_size,
    }
    if hasattr(dataset, "names"):
        description.update({
            "children": list(dataset.names),
            "child_lengths": list(dataset.lengths),
            "normalized_weights": list(dataset.weights),
        })
    print(json.dumps({"pipeline": description}, ensure_ascii=False, indent=2), flush=True)
    def progress(record):
        step = int(record["step"])
        if step != 1 and step % max(1, int(args.log_interval)) != 0 and step != args.steps:
            return
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record["pid"] = os.getpid()
        record["rss_max_mb"] = float(usage.ru_maxrss) / 1024.0
        if Path("/dev/shm").exists():
            shm = shutil.disk_usage("/dev/shm")
            record["shm_used_gb"] = float(shm.used) / (1024 ** 3)
            record["shm_free_gb"] = float(shm.free) / (1024 ** 3)
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            record["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            record["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
        print(json.dumps({"progress": record}, ensure_ascii=False), flush=True)

    result = run_optimizer_smoke(
        pipeline,
        steps=args.steps,
        device=args.device,
        progress_callback=progress,
        batch_validator=(
            lambda batch: validate_motus_model_batch_v1(
                batch,
                expected_batch_size=int(config.training.batch_size),
                num_video_frames=int(config.common.num_video_frames),
                video_action_freq_ratio=int(
                    config.common.get("video_action_freq_ratio", 1)
                ),
            )
            if args.contract_profile == "motus_model_batch_v1"
            else None
        ),
    )
    print(json.dumps({"result": result}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
