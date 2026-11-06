#!/usr/bin/env python3
"""Audit EgoVerse and all registered children through the indexed queue path."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

from motus_data import (
    MOTUS_MODEL_BATCH_V1,
    EpisodeDatasetRelease,
    EpisodeRepository,
    IndexedTrainingSource,
    get_episode_dataset_release,
    open_mixed_training_data,
    open_training_data,
    validate_motus_model_batch_v1,
)
from motus_data.index_contracts import MOTUS_SAMPLE_POOL_HEADER_V2


EGOVERSE_DATA_VERSION = "20260730_first6_v1"
EGOVERSE_PROFILES = (
    "aria_bimanual",
    "aria_left_arm",
    "aria_right_arm",
    "eva_bimanual",
    "scale_bimanual",
    "mecka_freeform_bimanual",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    family: str
    action_video_fps_ratio: int
    absent_state_action: bool = False
    egoverse_profile: str | None = None


SOURCE_SPECS = (
    SourceSpec("agibot", "agibot", 6),
    SourceSpec("agibot_beta", "agibot_beta", 1),
    SourceSpec("robotwin_aloha", "robotwin", 1),
    SourceSpec("robotwin_ur5", "robotwin", 1),
    SourceSpec("robotwin_arx", "robotwin", 1),
    SourceSpec("robotwin_piper", "robotwin", 1),
    SourceSpec("robotmind_tianyi", "robotmind", 1),
    SourceSpec("robotmind_arkmobile", "robotmind", 1),
    SourceSpec("robotmind_ark", "robotmind", 1),
    SourceSpec("robotmind_tienkung", "robotmind", 1),
    SourceSpec("robotmind_agilex", "robotmind", 1),
    SourceSpec("openvid", "openvid", 1, absent_state_action=True),
    *(
        SourceSpec(
            f"egoverse_{profile}",
            "egoverse",
            1,
            egoverse_profile=profile,
        )
        for profile in EGOVERSE_PROFILES
    ),
)


@dataclass(frozen=True)
class ResolvedSource:
    spec: SourceSpec
    release: EpisodeDatasetRelease
    catalog_path: Path
    index_path: Path
    catalog_metadata: dict[str, Any]
    sample_pool_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        default="/data/jiyuan.wang/EpisodeCatalogs",
    )
    parser.add_argument(
        "--egoverse-catalog-root",
        default=(
            "/data/jiyuan.wang/EpisodeCatalogs/_20260730/"
            "egoverse_first6_v1"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-child", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches-per-child", type=int, default=4)
    parser.add_argument("--egoverse-probes-per-child", type=int, default=3)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--quota-batches", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--force-pools", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--registered-release-only", action="store_true")
    return parser.parse_args()


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _metadata_path(catalog_path: Path) -> Path:
    return catalog_path.with_suffix(".meta.json")


def _egoverse_release(
    spec: SourceSpec,
    *,
    egoverse_catalog_root: Path,
    video_size: tuple[int, int],
    queue_size: int,
) -> EpisodeDatasetRelease:
    profile = str(spec.egoverse_profile)
    stem = f"egoverse_{profile}.{EGOVERSE_DATA_VERSION}.episode_catalog"
    return EpisodeDatasetRelease(
        dataset=spec.name,
        release=EGOVERSE_DATA_VERSION,
        catalog_path=str(egoverse_catalog_root / f"{stem}.jsonl"),
        index_path=str(egoverse_catalog_root / f"{stem}.index.sqlite"),
        default_video_size=video_size,
        default_num_workers=0,
        default_queue_size=queue_size,
        action_magnitude_guard_threshold=None,
    )


def _resolve_sources(args: argparse.Namespace) -> list[ResolvedSource]:
    catalog_root = Path(args.catalog_root).resolve()
    egoverse_catalog_root = Path(args.egoverse_catalog_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    pool_dir = output_dir / "sample_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    size = (int(args.height), int(args.width))
    resolved = []
    for spec in SOURCE_SPECS:
        release = (
            _egoverse_release(
                spec,
                egoverse_catalog_root=egoverse_catalog_root,
                video_size=size,
                queue_size=int(args.queue_size),
            )
            if spec.egoverse_profile is not None
            else get_episode_dataset_release(spec.name)
        )
        catalog, index = release.resolve_paths(catalog_root)
        if index is None:
            raise RuntimeError(f"{spec.name} release has no SQLite index")
        for path in (catalog, index, _metadata_path(catalog)):
            if not path.is_file():
                raise FileNotFoundError(path)
        metadata = json.loads(
            _metadata_path(catalog).read_text(encoding="utf-8")
        )
        expected_child = spec.name
        actual_child = str(
            metadata.get("child", metadata.get("source_child"))
        )
        if actual_child != expected_child:
            raise RuntimeError(
                f"{spec.name} Catalog child={actual_child!r}"
            )
        resolved.append(
            ResolvedSource(
                spec=spec,
                release=release,
                catalog_path=catalog,
                index_path=index,
                catalog_metadata=metadata,
                sample_pool_path=(
                    pool_dir / f"{spec.name}.runtime_gate.sample_pool.jsonl"
                ),
            )
        )
    return resolved


def _episode_id_at(
    connection: sqlite3.Connection,
    ordinal: int,
) -> str:
    row = connection.execute(
        "SELECT episode_id FROM episodes WHERE ordinal=?",
        (int(ordinal),),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"SQLite index has no ordinal={ordinal}")
    return str(row[0])


def _text_indices(episode: Any, *, start: int, stop: int) -> list[int]:
    items = [
        item
        for item in episode.read_text(range(episode.info.text_count))
        if item.text.strip()
    ]
    if not items:
        return []
    episode_scope = next(
        (item for item in items if item.frame_range is None),
        None,
    )
    matching = next(
        (
            item
            for item in items
            if item.frame_range is not None
            and item.frame_range[0] <= start
            and stop <= item.frame_range[1]
        ),
        None,
    )
    selected = []
    if episode_scope is not None:
        selected.append(int(episode_scope.index))
    if matching is not None and matching.index not in selected:
        selected.append(int(matching.index))
    if not selected:
        selected.append(int(items[0].index))
    return selected


def _selection_for_episode(
    episode: Any,
    *,
    spec: SourceSpec,
    num_video_frames: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    selected_video_count = int(num_video_frames) + 1
    max_start = int(episode.info.video_count) - selected_video_count
    if spec.absent_state_action:
        if episode.info.state_count != 0 or episode.info.action_count != 0:
            raise RuntimeError(
                f"{spec.name} absent policy conflicts with Catalog counts"
            )
    else:
        ratio = int(spec.action_video_fps_ratio)
        required_actions = int(num_video_frames) * ratio
        if episode.info.state_count <= 0 or episode.info.action_count <= 0:
            return None
        max_start = min(
            max_start,
            (int(episode.info.state_count) - 1) // ratio,
            (
                int(episode.info.action_count)
                - 1
                - required_actions
            )
            // ratio,
        )
    if max_start < 0:
        return None
    start = rng.randint(0, max_start)
    stop = start + selected_video_count
    text_indices = _text_indices(episode, start=start, stop=stop)
    if not text_indices:
        return None
    if spec.absent_state_action:
        state_indices: list[int] = []
        action_indices: list[int] = []
    else:
        ratio = int(spec.action_video_fps_ratio)
        state_start = start * ratio
        state_indices = [state_start]
        action_indices = list(
            range(
                state_start + 1,
                state_start + 1 + int(num_video_frames) * ratio,
            )
        )
    return {
        "episode_id": episode.episode_id,
        "video_frame_indices": list(range(start, stop)),
        "state_indices": state_indices,
        "action_indices": action_indices,
        "text_indices": text_indices,
    }


def _candidate_ordinals(
    count: int,
    *,
    samples: int,
    rng: random.Random,
) -> list[int]:
    target = min(count, max(samples * 64, 512))
    values = rng.sample(range(count), target)
    # Deterministic edge coverage catches early/late Catalog locator issues.
    for value in (0, count // 2, count - 1):
        if value not in values:
            values.append(value)
    return values


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
    temporary.replace(path)


def _prepare_pool(
    source: ResolvedSource,
    *,
    samples: int,
    batch_size: int,
    num_video_frames: int,
    seed: int,
    force: bool,
) -> list[dict[str, Any]]:
    path = source.sample_pool_path
    if path.exists() and not force:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != samples + 1:
            raise RuntimeError(
                f"existing pool {path} has {len(rows) - 1} samples, "
                f"expected {samples}; use --force-pools"
            )
        if rows[0].get("catalog_id") != source.catalog_metadata["catalog_id"]:
            raise RuntimeError(
                f"existing pool {path} is bound to another Catalog"
            )
        episode_counts = Counter(
            row["episode_id"] for row in rows[1:]
        )
        if any(count != batch_size for count in episode_counts.values()):
            raise RuntimeError(
                f"existing pool {path} is not grouped in batch_size="
                f"{batch_size} same-Episode selections; use --force-pools"
            )
        return rows[1:]

    repository = EpisodeRepository(
        source.catalog_path,
        index_path=source.index_path,
        verify_sha256=False,
    )
    target_episodes = samples // batch_size
    rng = random.Random(_stable_seed(seed, source.spec.name))
    selections = []
    seen_episode_ids = set()
    uri = f"{source.index_path.as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        for ordinal in _candidate_ordinals(
            len(repository),
            samples=target_episodes,
            rng=rng,
        ):
            episode = repository.get_episode(
                _episode_id_at(connection, ordinal)
            )
            if episode.episode_id in seen_episode_ids:
                continue
            group = []
            seen_starts = set()
            for _ in range(batch_size * 32):
                selection = _selection_for_episode(
                    episode,
                    spec=source.spec,
                    num_video_frames=num_video_frames,
                    rng=rng,
                )
                if selection is None:
                    break
                start = int(selection["video_frame_indices"][0])
                if start in seen_starts:
                    continue
                seen_starts.add(start)
                group.append(selection)
                if len(group) == batch_size:
                    break
            if len(group) != batch_size:
                continue
            seen_episode_ids.add(episode.episode_id)
            selections.extend(group)
            if len(seen_episode_ids) == target_episodes:
                break
    finally:
        connection.close()
        repository.close()
    if len(selections) != samples:
        raise RuntimeError(
            f"{source.spec.name} produced only {len(selections)}/{samples} "
            "eligible distinct-Episode engineering selections"
        )
    header = {
        "format": MOTUS_SAMPLE_POOL_HEADER_V2,
        "pool_id": f"{source.spec.name}_runtime_gate_20260731",
        "catalog_id": source.catalog_metadata["catalog_id"],
        "samples": len(selections),
        "meta": {
            "purpose": "engineering_smoke_only",
            "production_sampling_policy": False,
            "catalog_episodes": int(
                source.catalog_metadata.get(
                    "episodes",
                    source.catalog_metadata.get("record_count"),
                )
            ),
            "num_video_frames": int(num_video_frames),
            "action_video_fps_ratio": int(
                source.spec.action_video_fps_ratio
            ),
            "distinct_episode_ids": len(seen_episode_ids),
            "same_episode_batch_size": int(batch_size),
            "seed": int(seed),
        },
    }
    _write_jsonl(path, [header, *selections])
    print(
        f"POOL_READY child={source.spec.name} samples={len(selections)} "
        f"distinct_episodes={len(seen_episode_ids)} path={path}",
        flush=True,
    )
    return selections


def _adapter_options(spec: SourceSpec, num_video_frames: int) -> dict[str, Any]:
    options = {
        "num_video_frames": int(num_video_frames),
        "action_video_fps_ratio": int(spec.action_video_fps_ratio),
        "video_loss_weight": 1.0,
        "action_loss_weight": 0.0,
        "state_action_policy": (
            "absent" if spec.absent_state_action else "real"
        ),
    }
    if spec.absent_state_action:
        options.update(
            {
                "allow_missing_state_action": True,
                "placeholder_state_action_dim": 26,
            }
        )
    return options


def _indexed_source(
    source: ResolvedSource,
    *,
    num_video_frames: int,
) -> IndexedTrainingSource:
    count = int(
        source.catalog_metadata.get(
            "episodes",
            source.catalog_metadata.get("record_count"),
        )
    )
    return IndexedTrainingSource(
        name=source.spec.name,
        dataset=source.spec.name,
        family=source.spec.family,
        sample_pool=source.sample_pool_path,
        weight=float(count),
        release=source.release,
        verify_sha256=False,
        output_adapter=MOTUS_MODEL_BATCH_V1,
        adapter_options=_adapter_options(
            source.spec,
            num_video_frames=num_video_frames,
        ),
    )


def _validate_batch(
    batch: dict[str, Any],
    *,
    source: ResolvedSource,
    batch_size: int,
    num_video_frames: int,
    size: tuple[int, int],
) -> None:
    spec = source.spec
    validate_motus_model_batch_v1(
        batch,
        expected_batch_size=batch_size,
        num_video_frames=num_video_frames,
        video_action_freq_ratio=spec.action_video_fps_ratio,
    )
    if tuple(batch["first_frame"].shape) != (batch_size, 3, *size):
        raise RuntimeError(f"{spec.name} first-frame shape changed")
    if tuple(batch["video_frames"].shape) != (
        batch_size,
        num_video_frames,
        3,
        *size,
    ):
        raise RuntimeError(f"{spec.name} video shape changed")
    if set(batch["source_datasets"]) != {spec.family}:
        raise RuntimeError(
            f"{spec.name} family={set(batch['source_datasets'])}"
        )
    if set(batch["source_children"]) != {spec.name}:
        raise RuntimeError(
            f"{spec.name} children={set(batch['source_children'])}"
        )
    if not torch.isfinite(batch["first_frame"]).all():
        raise RuntimeError(f"{spec.name} first frame is non-finite")
    if not torch.isfinite(batch["video_frames"]).all():
        raise RuntimeError(f"{spec.name} video is non-finite")
    if not torch.isfinite(batch["initial_state"]).all():
        raise RuntimeError(f"{spec.name} state is non-finite")
    if not torch.isfinite(batch["action_sequence"]).all():
        raise RuntimeError(f"{spec.name} action is non-finite")
    if not torch.equal(
        batch["action_loss_weights"],
        torch.zeros(batch_size, dtype=torch.float32),
    ):
        raise RuntimeError(f"{spec.name} Stage1 action loss is not zero")
    if not torch.equal(
        batch["video_loss_weights"],
        torch.ones(batch_size, dtype=torch.float32),
    ):
        raise RuntimeError(f"{spec.name} video loss is not one")
    if not all(str(text).strip() for text in batch["t5_input_messages"]):
        raise RuntimeError(f"{spec.name} returned empty text")
    if spec.absent_state_action:
        if set(batch["state_action_policies"]) != {"absent"}:
            raise RuntimeError(f"{spec.name} absent policy changed")
        if torch.count_nonzero(batch["initial_state"]).item() != 0:
            raise RuntimeError(f"{spec.name} placeholder state is nonzero")
        if torch.count_nonzero(batch["action_sequence"]).item() != 0:
            raise RuntimeError(f"{spec.name} placeholder action is nonzero")
        if not all(
            item["synthetic_state_action"] for item in batch["metadata"]
        ):
            raise RuntimeError(f"{spec.name} synthetic marker is absent")
    else:
        if set(batch["state_action_policies"]) != {"real"}:
            raise RuntimeError(f"{spec.name} real policy changed")
        if "metadata" in batch:
            raise RuntimeError(
                f"{spec.name} gained absent-source-only metadata"
            )


def _probe_egoverse(
    sources: list[ResolvedSource],
    selections_by_child: dict[str, list[dict[str, Any]]],
    *,
    probes_per_child: int,
    size: tuple[int, int],
) -> None:
    for source in sources:
        if source.spec.egoverse_profile is None:
            continue
        repository = EpisodeRepository(
            source.catalog_path,
            index_path=source.index_path,
            verify_sha256=False,
        )
        try:
            unique_selections = []
            seen_episode_ids = set()
            for selection in selections_by_child[source.spec.name]:
                if selection["episode_id"] in seen_episode_ids:
                    continue
                seen_episode_ids.add(selection["episode_id"])
                unique_selections.append(selection)
            for probe, selection in enumerate(
                unique_selections[:probes_per_child]
            ):
                episode = repository.get_episode(selection["episode_id"])
                video_indices = selection["video_frame_indices"]
                sparse = {
                    "episode_id": episode.episode_id,
                    "video_frame_indices": [
                        video_indices[0],
                        video_indices[len(video_indices) // 2],
                        video_indices[-1],
                    ],
                    "state_indices": selection["state_indices"],
                    "action_indices": [
                        selection["action_indices"][0],
                        selection["action_indices"][-1],
                    ],
                    "text_indices": selection["text_indices"],
                }
                payload = episode.read_selection(
                    sparse,
                    video_output_format="rgb_float32_chw",
                    video_size=size,
                )
                if tuple(payload.video.shape) != (3, 3, *size):
                    raise RuntimeError(
                        f"{source.spec.name} sparse video shape changed"
                    )
                for tensor_name, tensor in (
                    ("video", payload.video),
                    ("state", payload.state),
                    ("action", payload.action),
                ):
                    if not torch.isfinite(tensor).all():
                        raise RuntimeError(
                            f"{source.spec.name} {tensor_name} is non-finite"
                        )
                print(
                    f"EGOVERSE_INDEXED_PROBE_OK child={source.spec.name} "
                    f"probe={probe} episode={episode.episode_id} "
                    f"video={tuple(payload.video.shape)} "
                    f"state={tuple(payload.state.shape)} "
                    f"action={tuple(payload.action.shape)} "
                    f"text_items={len(payload.text)}",
                    flush=True,
                )
        finally:
            repository.close()


def _egoverse_child_queue_gate(
    sources: list[ResolvedSource],
    *,
    args: argparse.Namespace,
) -> None:
    size = (int(args.height), int(args.width))
    for source in sources:
        if source.spec.egoverse_profile is None:
            continue
        pipeline = open_training_data(
            source.spec.name,
            release=source.release,
            sample_pool=source.sample_pool_path,
            batch_size=int(args.batch_size),
            output_adapter=MOTUS_MODEL_BATCH_V1,
            adapter_options=_adapter_options(
                source.spec,
                int(args.num_video_frames),
            ),
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
            verify_sha256=False,
        )
        observed = set()
        try:
            with pipeline:
                for step in range(int(args.batches_per_child)):
                    batch, _ = pipeline.next()
                    _validate_batch(
                        batch,
                        source=source,
                        batch_size=int(args.batch_size),
                        num_video_frames=int(args.num_video_frames),
                        size=size,
                    )
                    observed.update(batch["episode_ids"])
                    print(
                        f"EGOVERSE_CHILD_QUEUE_BATCH_OK "
                        f"child={source.spec.name} step={step} "
                        f"episodes={batch['episode_ids']}",
                        flush=True,
                    )
            snapshot = pipeline.snapshot()
        finally:
            pipeline.close()
        if len(observed) != int(args.batches_per_child):
            raise RuntimeError(
                f"{source.spec.name} observed {len(observed)} distinct "
                f"Episodes, expected {int(args.batches_per_child)}"
            )
        if snapshot["ready_queue_producer_exception_type"] is not None:
            raise RuntimeError(
                f"{source.spec.name} queue producer failed: "
                f"{snapshot['ready_queue_producer_exception_message']}"
            )
        print(
            f"EGOVERSE_CHILD_QUEUE_OK child={source.spec.name} "
            f"distinct_episodes={len(observed)} "
            f"produced={snapshot['ready_queue_produced_count']} "
            f"consumed={snapshot['ready_queue_consumed_count']}",
            flush=True,
        )


def _true_weight_gate(
    sources: list[ResolvedSource],
    *,
    args: argparse.Namespace,
) -> None:
    indexed = [
        _indexed_source(
            source,
            num_video_frames=int(args.num_video_frames),
        )
        for source in sources
    ]
    pipeline = open_mixed_training_data(
        indexed,
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
    finally:
        pipeline.close()
    expected = {source.spec.name for source in sources}
    if set(quotas) != expected or set(weights) != expected:
        raise RuntimeError("true-weight gate is missing source children")
    if sum(int(value) for value in quotas.values()) != int(
        args.quota_batches
    ):
        raise RuntimeError("true-weight quotas do not fill the epoch")
    if any(int(value) <= 0 for value in quotas.values()):
        raise RuntimeError(
            f"true-weight epoch does not cover every child: {quotas}"
        )
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise RuntimeError("normalized source weights do not sum to one")
    print(
        "ALL_SOURCE_TRUE_WEIGHTS "
        + json.dumps(weights, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "ALL_SOURCE_TRUE_QUOTAS "
        + json.dumps(quotas, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print("ALL_SOURCE_METADATA_BINDING_OK", flush=True)


def _registered_egoverse_release_gate(
    sources: list[ResolvedSource],
    *,
    args: argparse.Namespace,
) -> None:
    size = (int(args.height), int(args.width))
    for source in sources:
        if source.spec.egoverse_profile is None:
            continue
        # Deliberately omit release/catalog/index/root arguments. This is the
        # exact default-registry call expected from a downstream trainer.
        pipeline = open_training_data(
            source.spec.name,
            sample_pool=source.sample_pool_path,
            batch_size=int(args.batch_size),
            output_adapter=MOTUS_MODEL_BATCH_V1,
            adapter_options=_adapter_options(
                source.spec,
                int(args.num_video_frames),
            ),
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
            verify_sha256=False,
        )
        try:
            with pipeline:
                batch, stats = pipeline.next()
            _validate_batch(
                batch,
                source=source,
                batch_size=int(args.batch_size),
                num_video_frames=int(args.num_video_frames),
                size=size,
            )
        finally:
            pipeline.close()
        print(
            f"EGOVERSE_REGISTERED_RELEASE_OK child={source.spec.name} "
            f"episodes={batch['episode_ids']} "
            f"produced={stats['ready_queue_produced_count']} "
            f"consumed={stats['ready_queue_consumed_count']}",
            flush=True,
        )
    print("EGOVERSE_REGISTERED_FIRST6_GATE_OK", flush=True)


def _balanced_mix_gate(
    sources: list[ResolvedSource],
    *,
    args: argparse.Namespace,
) -> None:
    indexed = [
        replace(
            _indexed_source(
                source,
                num_video_frames=int(args.num_video_frames),
            ),
            weight=1.0,
        )
        for source in sources
    ]
    batches_per_child = int(args.batches_per_child)
    total_batches = len(indexed) * batches_per_child
    size = (int(args.height), int(args.width))
    pipeline = open_mixed_training_data(
        indexed,
        batch_size=int(args.batch_size),
        batches_per_epoch=total_batches,
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
    by_name = {source.spec.name: source for source in sources}
    observed_batches: Counter[str] = Counter()
    observed_episodes: dict[str, set[str]] = {
        source.spec.name: set() for source in sources
    }
    family_batches: Counter[str] = Counter()
    try:
        with pipeline:
            for step in range(total_batches):
                batch, stats = pipeline.next()
                name = str(stats["mix_source_name"])
                source = by_name[name]
                _validate_batch(
                    batch,
                    source=source,
                    batch_size=int(args.batch_size),
                    num_video_frames=int(args.num_video_frames),
                    size=size,
                )
                observed_batches[name] += 1
                observed_episodes[name].update(batch["episode_ids"])
                family_batches[source.spec.family] += 1
                print(
                    f"ALL_SOURCE_MIX_BATCH_OK step={step} child={name} "
                    f"family={source.spec.family} "
                    f"episodes={batch['episode_ids']}",
                    flush=True,
                )
        snapshot = pipeline.snapshot()
    finally:
        pipeline.close()

    expected_batches = {
        source.spec.name: batches_per_child for source in sources
    }
    if dict(observed_batches) != expected_batches:
        raise RuntimeError(
            f"child coverage={dict(observed_batches)} "
            f"expected={expected_batches}"
        )
    expected_episode_count = batches_per_child
    wrong_episode_counts = {
        name: len(values)
        for name, values in observed_episodes.items()
        if len(values) != expected_episode_count
    }
    if wrong_episode_counts:
        raise RuntimeError(
            "mix did not consume every distinct engineering sample: "
            f"{wrong_episode_counts}"
        )
    expected_families = {source.spec.family for source in sources}
    if set(family_batches) != expected_families:
        raise RuntimeError(
            f"family coverage={dict(family_batches)} "
            f"expected={sorted(expected_families)}"
        )
    if snapshot["ready_queue_producer_exception_type"] is not None:
        raise RuntimeError(
            "all-source queue producer failed: "
            f"{snapshot['ready_queue_producer_exception_message']}"
        )
    if int(snapshot["ready_queue_current_size"]) != 0:
        raise RuntimeError("all-source queue did not drain")
    print(
        "ALL_SOURCE_BALANCED_QUEUE_OK "
        + json.dumps(
            {
                "child_batches": dict(observed_batches),
                "child_distinct_episodes": {
                    name: len(values)
                    for name, values in observed_episodes.items()
                },
                "family_batches": dict(family_batches),
                "produced": snapshot["ready_queue_produced_count"],
                "consumed": snapshot["ready_queue_consumed_count"],
                "current_size": snapshot["ready_queue_current_size"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    print("ALL_SOURCE_INDEXED_MIX_AUDIT_OK", flush=True)


def main() -> int:
    args = _parse_args()
    for name in (
        "samples_per_child",
        "batch_size",
        "batches_per_child",
        "num_video_frames",
        "height",
        "width",
        "queue_size",
        "quota_batches",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.egoverse_probes_per_child) < 0:
        raise ValueError("--egoverse-probes-per-child must be >= 0")
    required_samples = int(args.batch_size) * int(args.batches_per_child)
    if int(args.samples_per_child) != required_samples:
        raise ValueError(
            "--samples-per-child must equal "
            "--batch-size * --batches-per-child so the coverage gate consumes "
            "every engineering sample exactly once"
        )
    if int(args.egoverse_probes_per_child) > int(args.samples_per_child):
        raise ValueError(
            "--egoverse-probes-per-child cannot exceed samples per child"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _resolve_sources(args)
    selections_by_child = {
        source.spec.name: _prepare_pool(
            source,
            samples=int(args.samples_per_child),
            batch_size=int(args.batch_size),
            num_video_frames=int(args.num_video_frames),
            seed=int(args.seed),
            force=bool(args.force_pools),
        )
        for source in sources
    }
    summary = {
        "format": "motus_all_source_runtime_gate_plan_v1",
        "sources": [
            {
                "child": source.spec.name,
                "family": source.spec.family,
                "catalog_id": source.catalog_metadata["catalog_id"],
                "catalog_episodes": int(
                    source.catalog_metadata.get(
                        "episodes",
                        source.catalog_metadata.get("record_count"),
                    )
                ),
                "sample_pool": str(source.sample_pool_path),
                "engineering_samples": len(
                    selections_by_child[source.spec.name]
                ),
            }
            for source in sources
        ],
    }
    (output_dir / "runtime_gate_plan.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"ALL_SOURCE_POOLS_READY children={len(sources)} "
        f"samples={sum(len(v) for v in selections_by_child.values())}",
        flush=True,
    )
    if args.prepare_only:
        print("ALL_SOURCE_RUNTIME_PREPARE_OK", flush=True)
        return 0
    if args.registered_release_only:
        _registered_egoverse_release_gate(sources, args=args)
        return 0

    _probe_egoverse(
        sources,
        selections_by_child,
        probes_per_child=int(args.egoverse_probes_per_child),
        size=(int(args.height), int(args.width)),
    )
    _egoverse_child_queue_gate(sources, args=args)
    _true_weight_gate(sources, args=args)
    _balanced_mix_gate(sources, args=args)
    _registered_egoverse_release_gate(sources, args=args)
    print("EGOVERSE_ALL_SOURCE_RUNTIME_GATE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
