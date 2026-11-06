#!/usr/bin/env python3
"""Audit the existing ten sources plus OpenVid through the public mix API."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from motus_data import (
    MOTUS_MODEL_BATCH_V1,
    EpisodeDatasetRelease,
    IndexedTrainingSource,
    open_mixed_training_data,
    validate_motus_model_batch_v1,
)


EXISTING_SOURCES = (
    ("agibot", "agibot", 6),
    ("agibot_beta", "agibot_beta", 1),
    ("robotwin_aloha", "robotwin", 1),
    ("robotwin_ur5", "robotwin", 1),
    ("robotwin_arx", "robotwin", 1),
    ("robotwin_piper", "robotwin", 1),
    ("robotmind_tianyi", "robotmind", 1),
    ("robotmind_arkmobile", "robotmind", 1),
    ("robotmind_ark", "robotmind", 1),
    ("robotmind_tienkung", "robotmind", 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        default="/data/jiyuan.wang/EpisodeCatalogs",
    )
    parser.add_argument(
        "--sample-pool-root",
        default=(
            "/data/jiyuan.wang/SamplePools/_20260727/"
            "ten_source_smoke_v1"
        ),
    )
    parser.add_argument(
        "--openvid-catalog",
        default=(
            "/data/jiyuan.wang/EpisodeCatalogs/_20260730/openvid/"
            "openvid_1m_full_20260730.episode_catalog.jsonl"
        ),
    )
    parser.add_argument(
        "--openvid-index",
        default=(
            "/data/jiyuan.wang/EpisodeCatalogs/_20260730/openvid/"
            "openvid_1m_full_20260730.episode_catalog.index.sqlite"
        ),
    )
    parser.add_argument(
        "--openvid-pool",
        default=(
            "/data/jiyuan.wang/thinkerwm-datafactory-dev-20260729/"
            "outputs/openvid_agibot_beta_mix_gate_20260730/"
            "openvid.mix_gate.sample_pool.jsonl"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--quota-batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def _pool_path(root: Path, dataset: str) -> Path:
    return root / f"{dataset}.engineering_smoke.sample_pool.jsonl"


def _json_first(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _catalog_metadata(path: Path) -> dict[str, Any]:
    return json.loads(
        path.with_suffix(".meta.json").read_text(encoding="utf-8")
    )


def _adapter_options(
    *,
    ratio: int,
    absent: bool,
) -> dict[str, Any]:
    options = {
        "num_video_frames": 48,
        "action_video_fps_ratio": int(ratio),
        "video_loss_weight": 1.0,
        "action_loss_weight": 0.0,
        "state_action_policy": "absent" if absent else "real",
    }
    if absent:
        options.update(
            {
                "allow_missing_state_action": True,
                "placeholder_state_action_dim": 26,
            }
        )
    return options


def _sources(args: argparse.Namespace) -> list[IndexedTrainingSource]:
    catalog_root = Path(args.catalog_root).resolve()
    pool_root = Path(args.sample_pool_root).resolve()
    sources = []
    for dataset, family, ratio in EXISTING_SOURCES:
        pool = _pool_path(pool_root, dataset)
        if not pool.is_file():
            raise FileNotFoundError(pool)
        header = _json_first(pool)
        sources.append(
            IndexedTrainingSource(
                name=dataset,
                dataset=dataset,
                family=family,
                sample_pool=pool,
                weight=float(header["meta"]["catalog_episodes"]),
                catalog_root=catalog_root,
                output_adapter=MOTUS_MODEL_BATCH_V1,
                adapter_options=_adapter_options(
                    ratio=ratio,
                    absent=False,
                ),
            )
        )

    openvid_catalog = Path(args.openvid_catalog).resolve()
    openvid_index = Path(args.openvid_index).resolve()
    openvid_pool = Path(args.openvid_pool).resolve()
    for path in (openvid_catalog, openvid_index, openvid_pool):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = _catalog_metadata(openvid_catalog)
    release = EpisodeDatasetRelease(
        dataset="openvid",
        release="20260730_full_v1_gate",
        catalog_path=str(openvid_catalog),
        index_path=str(openvid_index),
        default_video_size=(int(args.height), int(args.width)),
        default_num_workers=0,
        default_queue_size=int(args.queue_size),
        action_magnitude_guard_sources=None,
    )
    sources.append(
        IndexedTrainingSource(
            name="openvid",
            dataset="openvid",
            family="openvid",
            sample_pool=openvid_pool,
            weight=float(metadata["episodes"]),
            release=release,
            verify_sha256=False,
            output_adapter=MOTUS_MODEL_BATCH_V1,
            adapter_options=_adapter_options(ratio=1, absent=True),
        )
    )
    return sources


def _true_weight_gate(
    sources: list[IndexedTrainingSource],
    *,
    args: argparse.Namespace,
) -> None:
    pipeline = open_mixed_training_data(
        sources,
        batch_size=int(args.batch_size),
        batches_per_epoch=int(args.quota_batches),
        video_size=(int(args.height), int(args.width)),
        seed=int(args.seed),
        rank=0,
        world_size=1,
        queue_enabled=False,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    try:
        snapshot = pipeline.snapshot()
        quotas = snapshot["mix_epoch_quotas"]
        weights = snapshot["mix_normalized_weights"]
        if len(quotas) != 11 or set(quotas) != {
            source.name for source in sources
        }:
            raise RuntimeError("eleven-source quota keys are incomplete")
        if sum(quotas.values()) != int(args.quota_batches):
            raise RuntimeError("eleven-source quotas do not fill the epoch")
        if any(int(value) <= 0 for value in quotas.values()):
            raise RuntimeError(
                f"quota epoch does not cover every child: {quotas}"
            )
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
            raise RuntimeError("normalized source weights do not sum to one")
        print(
            "ELEVEN_SOURCE_TRUE_WEIGHTS "
            + json.dumps(weights, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        print(
            "ELEVEN_SOURCE_TRUE_QUOTAS "
            + json.dumps(quotas, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    finally:
        pipeline.close()
    print("ELEVEN_SOURCE_METADATA_BINDING_OK", flush=True)


def _validate_batch(
    batch: dict[str, Any],
    *,
    source: IndexedTrainingSource,
    batch_size: int,
    size: tuple[int, int],
) -> None:
    ratio = int(source.adapter_options["action_video_fps_ratio"])
    validate_motus_model_batch_v1(
        batch,
        expected_batch_size=batch_size,
        num_video_frames=48,
        video_action_freq_ratio=ratio,
    )
    if tuple(batch["first_frame"].shape) != (batch_size, 3, *size):
        raise RuntimeError(f"{source.name} first-frame shape changed")
    if tuple(batch["video_frames"].shape) != (
        batch_size,
        48,
        3,
        *size,
    ):
        raise RuntimeError(f"{source.name} video shape changed")
    if set(batch["source_datasets"]) != {source.family}:
        raise RuntimeError(f"{source.name} source family changed")
    if set(batch["source_children"]) != {source.name}:
        raise RuntimeError(f"{source.name} source child changed")
    if not torch.equal(
        batch["action_loss_weights"],
        torch.zeros(batch_size),
    ):
        raise RuntimeError(f"{source.name} Stage1 action loss changed")
    if not torch.equal(
        batch["video_loss_weights"],
        torch.ones(batch_size),
    ):
        raise RuntimeError(f"{source.name} video loss changed")
    if not all(str(text).strip() for text in batch["t5_input_messages"]):
        raise RuntimeError(f"{source.name} returned empty text")

    if source.name == "openvid":
        if set(batch["state_action_policies"]) != {"absent"}:
            raise RuntimeError("OpenVid state/action policy changed")
        if torch.count_nonzero(batch["initial_state"]).item() != 0:
            raise RuntimeError("OpenVid placeholder state is not zero")
        if torch.count_nonzero(batch["action_sequence"]).item() != 0:
            raise RuntimeError("OpenVid placeholder action is not zero")
        if not all(
            item["synthetic_state_action"]
            for item in batch["metadata"]
        ):
            raise RuntimeError("OpenVid synthetic audit flag is absent")
    else:
        if set(batch["state_action_policies"]) != {"real"}:
            raise RuntimeError(f"{source.name} real policy changed")
        if not torch.isfinite(batch["initial_state"]).all():
            raise RuntimeError(f"{source.name} state is non-finite")
        if not torch.isfinite(batch["action_sequence"]).all():
            raise RuntimeError(f"{source.name} action is non-finite")
        if "metadata" in batch:
            raise RuntimeError(
                f"{source.name} batch gained OpenVid-only metadata"
            )


def _balanced_queue_gate(
    sources: list[IndexedTrainingSource],
    *,
    args: argparse.Namespace,
) -> None:
    balanced_sources = [
        replace(source, weight=1.0)
        for source in sources
    ]
    size = (int(args.height), int(args.width))
    pipeline = open_mixed_training_data(
        balanced_sources,
        batch_size=int(args.batch_size),
        batches_per_epoch=len(balanced_sources),
        video_size=size,
        seed=int(args.seed),
        rank=0,
        world_size=1,
        queue_enabled=True,
        num_workers=0,
        pin_memory=False,
        queue_size=int(args.queue_size),
        queue_clone_tensors=False,
        queue_timeout_seconds=600.0,
        shuffle=True,
        drop_last=True,
    )
    by_name = {source.name: source for source in balanced_sources}
    observed = Counter()
    try:
        with pipeline:
            for step in range(len(balanced_sources)):
                batch, stats = pipeline.next()
                source_name = str(stats["mix_source_name"])
                _validate_batch(
                    batch,
                    source=by_name[source_name],
                    batch_size=int(args.batch_size),
                    size=size,
                )
                observed[source_name] += 1
                print(
                    f"ELEVEN_SOURCE_BATCH_OK step={step} "
                    f"source={source_name} "
                    f"episode={batch['episode_ids'][0]} "
                    f"video={tuple(batch['video_frames'].shape)} "
                    f"state={tuple(batch['initial_state'].shape)} "
                    f"action={tuple(batch['action_sequence'].shape)}",
                    flush=True,
                )
        snapshot = pipeline.snapshot()
    finally:
        pipeline.close()

    expected = {source.name: 1 for source in balanced_sources}
    if dict(observed) != expected:
        raise RuntimeError(
            f"balanced observed={dict(observed)} expected={expected}"
        )
    if snapshot["ready_queue_producer_exception_type"] is not None:
        raise RuntimeError(
            "eleven-source queue producer failed: "
            f"{snapshot['ready_queue_producer_exception_message']}"
        )
    print(
        "ELEVEN_SOURCE_BALANCED_QUEUE_OK "
        + json.dumps(
            {
                "observed": dict(observed),
                "produced": snapshot["ready_queue_produced_count"],
                "consumed": snapshot["ready_queue_consumed_count"],
                "current_size": snapshot["ready_queue_current_size"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    for name in (
        "batch_size",
        "height",
        "width",
        "queue_size",
        "quota_batches",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    sources = _sources(args)
    _true_weight_gate(sources, args=args)
    if args.metadata_only:
        return 0
    _balanced_queue_gate(sources, args=args)
    print("ELEVEN_SOURCE_INDEXED_MIX_AUDIT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
