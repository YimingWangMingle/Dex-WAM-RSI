#!/usr/bin/env python3
"""Template: plug a custom Episode dataset into ``motus_data.open_dataset``.

This file is the complete "plugin unit" for a new dataset source. A third
party copies it, renames ``custom_demo``, and fills in real decoding logic;
no motus_data internals are modified. Run it directly to see the flow work
end to end on synthetic data:

    python examples/custom_episode_source.py

The four steps of the plugin contract:

1. Subclass ``EpisodeSourceReader`` (the unified interface) and implement
   ``schemas`` / ``read_video`` / ``read_state`` / ``read_action``. The class
   attribute ``source_dataset`` must equal the ``source`` field published by
   the Catalog metadata — that name is how records are dispatched.
2. Produce a contract-conformant Catalog: ``*.episode_catalog.jsonl`` rows
   (``motus_episode_catalog_v2``), a sibling ``*.meta.json`` whose ``source``
   matches step 1, and a SQLite offset index via ``build_catalog_offset_index``.
3. Register the reader once at import time:
   ``register_episode_source_reader("custom_demo", CustomEpisodeReader)``.
4. Register (or reference) a release, then call ``open_dataset("custom_demo")``
   — no ``readers=`` needed, and ``open_training_data`` /
   ``open_mixed_training_data`` pick the reader up automatically.

Maintainer adoption checklist (to make this source built-in):
- Copy the reader class + its ``register_episode_source_reader`` line into
  ``motus_data/source_readers.py`` verbatim (or a new module imported from
  ``motus_data.repository._ensure_builtin_source_readers``).
- Add one release entry to ``motus_data/releases.yaml``.
- Run the test suite. Nothing else needs to change.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import motus_data
from motus_data import (
    EpisodeDatasetRelease,
    EpisodeRepositoryError,
    EpisodeSourceReader,
    ModalitySchema,
    build_catalog_offset_index,
    register_episode_dataset_release,
    register_episode_source_reader,
)

SOURCE_DATASET = "custom_demo"
STATE_ACTION_DIM = 7
VIDEO_FRAMES = 8
VIDEO_HEIGHT = 12
VIDEO_WIDTH = 16


# ---------------------------------------------------------------------------
# Step 1: the unified reader interface. Replace the demo tensor-file decoding
# below with the real storage decoding for your dataset (mp4, HDF5, tar, ...).
# Readers see the *normalized* Catalog record: each modality is
# {"count"/"frame_count": int, "locator": {"storage_backend", "uri", ...},
# "metadata": {...}} and texts live under record["text"]["items"].
# ---------------------------------------------------------------------------


class CustomEpisodeReader(EpisodeSourceReader):
    """Read demo episodes stored as one frames tensor and one timeline tensor."""

    source_dataset = SOURCE_DATASET

    @staticmethod
    def _file_uri(record: Mapping[str, Any], modality: str) -> str:
        locator = record[modality]["locator"]
        if locator["storage_backend"] != "file":
            raise EpisodeRepositoryError(
                f"{record['source_dataset']} {modality} requires file storage"
            )
        return str(locator["uri"])

    @staticmethod
    def _load_tensor(uri: str) -> torch.Tensor:
        try:
            return torch.load(uri, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            return torch.load(uri, map_location="cpu")

    @staticmethod
    def _take(timeline: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        if not indices:
            return timeline.new_empty((0, int(timeline.shape[-1])))
        return timeline[torch.tensor(indices, dtype=torch.long)].clone()

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_decoded_video_frame",
                {"formats": ["rgb_uint8_hwc", "rgb_float32_chw"]},
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (STATE_ACTION_DIM,),
                "zero_based_canonical_proprio_timestep",
                {"normalization": "none"},
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (STATE_ACTION_DIM,),
                "zero_based_canonical_action_timestep",
                {"normalization": "none"},
            ),
            "text": ModalitySchema(
                "text",
                len(record["text"]["items"]),
                "structured_text_item",
                (),
                "zero_based_catalog_text_item",
                {},
            ),
        }

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        if output_format not in {"rgb_uint8_hwc", "rgb_float32_chw"}:
            raise EpisodeRepositoryError(
                f"unsupported video output_format={output_format!r}"
            )
        frames = self._load_tensor(self._file_uri(record, "video"))
        selected = frames[torch.tensor(indices, dtype=torch.long)].clone()
        if video_size is None and output_format == "rgb_uint8_hwc":
            return selected
        chw = selected.permute(0, 3, 1, 2).float().div_(255.0)
        if video_size is not None:
            chw = F.interpolate(
                chw,
                size=tuple(int(value) for value in video_size),
                mode="bilinear",
                align_corners=False,
            )
        if output_format == "rgb_float32_chw":
            return chw
        return (
            chw.mul(255.0)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    def _timeline(self, record: Mapping[str, Any]) -> torch.Tensor:
        timeline = self._load_tensor(self._file_uri(record, "state"))
        expected = (
            int(record["state"]["count"]),
            STATE_ACTION_DIM,
        )
        if tuple(timeline.shape) != expected:
            raise EpisodeRepositoryError(
                "custom_demo timeline does not match Catalog counts"
            )
        return timeline

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._take(self._timeline(record), indices)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._take(self._timeline(record), indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Optional override: state and action share one timeline file here,
        # so load it once instead of twice (the base class would call
        # read_state/read_action separately).
        timeline = self._timeline(record)
        return self._take(timeline, state_indices), self._take(
            timeline, action_indices
        )


# ---------------------------------------------------------------------------
# Step 3: register the reader once, under the Catalog ``source`` name.
# ---------------------------------------------------------------------------

register_episode_source_reader(SOURCE_DATASET, CustomEpisodeReader)


# ---------------------------------------------------------------------------
# Demo scaffolding: synthetic episode payloads + a contract-conformant V2
# Catalog. Real integrations replace this with their offline Catalog build.
# ---------------------------------------------------------------------------


def _write_demo_episode(root: Path, ordinal: int) -> dict[str, Any]:
    episode_dir = root / f"episode_{ordinal:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(ordinal)
    frames = torch.randint(
        0,
        256,
        (VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    timeline = torch.randn(
        VIDEO_FRAMES, STATE_ACTION_DIM, generator=generator
    ).float()
    frames_path = episode_dir / "frames.pt"
    timeline_path = episode_dir / "timeline.pt"
    torch.save(frames, frames_path)
    torch.save(timeline, timeline_path)
    return {
        "episode_id": f"{SOURCE_DATASET}:episode_{ordinal:06d}",
        "robot": "custom_demo_arm",
        "video": {
            "count": VIDEO_FRAMES,
            "storage": {"type": "file", "path": str(frames_path)},
        },
        "state": {
            "count": VIDEO_FRAMES,
            "storage": {"type": "file", "path": str(timeline_path)},
            "meta": {"dimension": STATE_ACTION_DIM},
        },
        "action": {
            "count": VIDEO_FRAMES,
            "storage": {"type": "file", "path": str(timeline_path)},
            "meta": {"dimension": STATE_ACTION_DIM},
        },
        "texts": [
            {
                "index": 0,
                "kind": "instruction",
                "text": f"demo task {ordinal}",
                "video_range": [0, VIDEO_FRAMES],
            }
        ],
        "meta": {"demo": True},
    }


def _write_demo_catalog(root: Path, episodes: int) -> tuple[Path, Path]:
    catalog_path = root / f"{SOURCE_DATASET}.episode_catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as handle:
        for ordinal in range(episodes):
            record = _write_demo_episode(root, ordinal)
            handle.write(json.dumps(record) + "\n")
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    metadata = {
        "format": "motus_episode_catalog_v2",
        "catalog_id": f"{SOURCE_DATASET}.demo_v1",
        "source": SOURCE_DATASET,
        "child": "demo_split",
        "episodes": episodes,
        "sha256": digest,
    }
    metadata_path = catalog_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    index_path = Path(
        build_catalog_offset_index(catalog_path)["index_path"]
    )
    return catalog_path, index_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="motus_custom_demo_") as temporary:
        root = Path(temporary)
        catalog_path, index_path = _write_demo_catalog(root, episodes=2)

        # Step 4: register the release, then open by dataset name only.
        register_episode_dataset_release(
            EpisodeDatasetRelease(
                dataset=SOURCE_DATASET,
                release="demo_v1",
                catalog_path=str(catalog_path),
                index_path=str(index_path),
            ),
            default=True,
        )
        repository = motus_data.open_dataset(SOURCE_DATASET)
        try:
            print(f"catalog_id={repository.catalog_id} episodes={len(repository)}")
            for episode_id in repository.iter_episode_ids():
                episode = repository.get_episode(episode_id)
                video = episode.read_video(
                    [0, 2, 4], output_format="rgb_float32_chw"
                )
                state = episode.read_state([0, 1])
                action = episode.read_action([1, 2])
                texts = episode.read_text([0])
                print(
                    f"{episode_id} robot={episode.info.robot_name} "
                    f"video={tuple(video.shape)} state={tuple(state.shape)} "
                    f"action={tuple(action.shape)} text={texts[0].text!r}"
                )
        finally:
            repository.close()
    print("custom_demo source reader works via open_dataset without readers=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
