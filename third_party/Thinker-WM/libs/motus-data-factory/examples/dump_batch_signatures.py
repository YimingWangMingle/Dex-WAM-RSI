#!/usr/bin/env python3
"""Dump deterministic batch signatures from the new or legacy data pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def tensor_signature(value):
    if not torch.is_tensor(value):
        return None
    cpu = value.detach().float().cpu().contiguous()
    flat = cpu.reshape(-1)
    stride = max(1, flat.numel() // 4096)
    probe = flat[::stride][:4096].numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(cpu.mean()) if cpu.numel() else None,
        "std": float(cpu.std()) if cpu.numel() > 1 else 0.0,
        "probe_sha256": hashlib.sha256(probe).hexdigest(),
    }


def batch_signature(batch):
    return {
        "tensors": {
            key: tensor_signature(batch.get(key))
            for key in ("first_frame", "video_frames", "initial_state", "action_sequence")
        },
        "source_datasets": batch.get("source_datasets"),
        "robot_names": batch.get("robot_names"),
        "manifest_ids": batch.get("manifest_ids"),
        "manifest_tasks": batch.get("manifest_tasks"),
        "t5_input_messages": batch.get("t5_input_messages"),
        "vlm_input_messages": batch.get("vlm_input_messages"),
    }


def legacy_loader(config, legacy_root: Path, rank: int, world_size: int, seed: int):
    sys.path.insert(0, str(legacy_root))
    from data.dataset import collate_fn, create_dataset

    dataset = create_dataset(config, val=False)
    collate = getattr(dataset, "collate_fn", None)
    collate = collate if callable(collate) else collate_fn
    kwargs = {
        "num_workers": int(config.system.get("num_workers", 0)),
        "pin_memory": bool(config.system.get("pin_memory", False)),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(config.system.get("persistent_workers", False))
        if config.system.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(config.system.prefetch_factor)
    sampler = dataset.create_batch_sampler(
        batch_size=int(config.training.batch_size),
        drop_last=True,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, **kwargs)


def new_loader(config, rank: int, world_size: int, seed: int):
    from motus_data import build_dataloader
    return build_dataloader(config, split="train", rank=rank, world_size=world_size, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("new", "legacy"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--child",
        default=None,
        help="Restrict a mix config to one named child before dataset construction.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = OmegaConf.load(args.config)
    config.system.num_workers = 0
    config.system.persistent_workers = False
    config.system.prefetch_factor = None
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)

    if args.child is not None:
        children = [
            child for child in config.dataset.datasets
            if str(child.get("name")) == str(args.child)
        ]
        if len(children) != 1:
            available = [str(child.get("name")) for child in config.dataset.datasets]
            raise ValueError(
                f"expected one child named {args.child!r}, found {len(children)}; "
                f"available={available}"
            )
        config.dataset.datasets = OmegaConf.create([
            OmegaConf.to_container(children[0], resolve=True)
        ])
        config.dataset.batches_per_epoch = max(int(args.batches), 1)

    print(
        f"[{args.backend}] building dataset child={args.child or 'ALL'} "
        f"batch_size={config.training.batch_size}",
        flush=True,
    )

    if args.backend == "legacy":
        if not args.legacy_root:
            raise ValueError("--legacy-root is required for the legacy backend")
        loader = legacy_loader(
            config, Path(args.legacy_root).resolve(), args.rank, args.world_size, args.seed
        )
    else:
        loader = new_loader(config, args.rank, args.world_size, args.seed)

    dataset = loader.dataset
    print(
        f"[{args.backend}] dataset ready class={type(dataset).__name__} "
        f"length={len(dataset)}; reading {args.batches} batch(es)",
        flush=True,
    )

    signatures = []
    for index, batch in enumerate(loader):
        if index >= args.batches:
            break
        signatures.append(batch_signature(batch))
        sources = batch.get("source_datasets") or []
        ids = batch.get("manifest_ids") or []
        print(
            f"[{args.backend}] batch={index + 1}/{args.batches} "
            f"source={sources[:1]} manifest_id={ids[:1]}",
            flush=True,
        )
    output = {
        "backend": args.backend,
        "config": str(Path(args.config).resolve()),
        "rank": args.rank,
        "world_size": args.world_size,
        "seed": args.seed,
        "child": args.child,
        "batches": signatures,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.backend}] wrote {path} batches={len(signatures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
