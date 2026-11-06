from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import motus_data
from motus_data import (
    EpisodeDatasetRelease,
    EpisodeSourceReader,
    ModalitySchema,
    open_dataset,
    register_episode_source_reader,
)
from motus_data.catalog.sqlite_index import build_catalog_offset_index
from motus_data.repository import EpisodeRepositoryError


def _stub_schemas(record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
    return {
        "video": ModalitySchema(
            "video",
            int(record["video"]["frame_count"]),
            "uint8",
            (2, 2, 3),
            "zero_based_stub_frame",
            {},
        ),
        "state": ModalitySchema(
            "state",
            int(record["state"]["count"]),
            "float32",
            (2,),
            "zero_based_stub_timestep",
            {},
        ),
        "action": ModalitySchema(
            "action",
            int(record["action"]["count"]),
            "float32",
            (2,),
            "zero_based_stub_timestep",
            {},
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


class _StubReader(EpisodeSourceReader):
    """Sentinel reader: returns filled tensors so tests can see dispatch."""

    source_dataset = "test_plugin_stub"
    state_fill = 7.0
    action_fill = 8.0

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        return _stub_schemas(record)

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        return torch.full((len(indices), 2, 2, 3), 255, dtype=torch.uint8)

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return torch.full((len(indices), 2), self.state_fill)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return torch.full((len(indices), 2), self.action_fill)


class _OverrideReader(_StubReader):
    state_fill = 9.0
    action_fill = 9.5


def _write_catalog(root: Path, source: str) -> tuple[Path, Path]:
    """Write a minimal contract-conformant V2 Catalog + metadata + index."""

    catalog_path = root / f"{source}.episode_catalog.jsonl"
    modality = {
        "count": 4,
        "storage": {"type": "file", "path": str(root / "payload.pt")},
    }
    record = {
        "episode_id": f"{source}:episode_000000",
        "robot": "stub_robot",
        "video": {"count": 4, "storage": modality["storage"]},
        "state": modality,
        "action": modality,
        "texts": [{"index": 0, "kind": "instruction", "text": "stub task"}],
    }
    with catalog_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    metadata = {
        "format": "motus_episode_catalog_v2",
        "catalog_id": f"{source}.test_v1",
        "source": source,
        "child": "test_split",
        "episodes": 1,
        "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    }
    catalog_path.with_suffix(".meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    index_path = Path(build_catalog_offset_index(catalog_path)["index_path"])
    return catalog_path, index_path


def _release(dataset: str, catalog_path: Path, index_path: Path) -> EpisodeDatasetRelease:
    return EpisodeDatasetRelease(
        dataset=dataset,
        release="test_v1",
        catalog_path=str(catalog_path),
        index_path=str(index_path),
    )


class SourceReaderRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def _open(self, source: str, dataset: str, **kwargs):
        catalog_path, index_path = _write_catalog(self.root, source)
        return open_dataset(
            dataset,
            release=_release(dataset, catalog_path, index_path),
            **kwargs,
        )

    def test_registered_reader_resolves_without_readers_arg(self) -> None:
        register_episode_source_reader("test_plugin_reg", _StubReader)
        repository = self._open("test_plugin_reg", "test_plugin_reg_ds")
        try:
            episode = repository.get_episode("test_plugin_reg:episode_000000")
            state = episode.read_state([0, 1])
            action = episode.read_action([2])
            video = episode.read_video([0, 3])
            self.assertTrue(torch.all(state == 7.0))
            self.assertTrue(torch.all(action == 8.0))
            self.assertEqual(tuple(video.shape), (2, 2, 2, 3))
            self.assertEqual(episode.read_text([0])[0].text, "stub task")
        finally:
            repository.close()

    def test_per_call_readers_override_registry(self) -> None:
        register_episode_source_reader("test_plugin_ovr", _StubReader)
        repository = self._open(
            "test_plugin_ovr",
            "test_plugin_ovr_ds",
            readers={"test_plugin_ovr": _OverrideReader()},
        )
        try:
            episode = repository.get_episode("test_plugin_ovr:episode_000000")
            self.assertTrue(torch.all(episode.read_state([0]) == 9.0))
            self.assertTrue(torch.all(episode.read_action([0]) == 9.5))
        finally:
            repository.close()

    def test_duplicate_registration_raises_and_replace_works(self) -> None:
        register_episode_source_reader("test_plugin_dup", _StubReader)
        with self.assertRaises(KeyError):
            register_episode_source_reader("test_plugin_dup", _StubReader)
        register_episode_source_reader(
            "test_plugin_dup", _OverrideReader, replace=True
        )
        snapshot = motus_data.EPISODE_SOURCE_READERS.snapshot()
        self.assertIsInstance(snapshot["test_plugin_dup"], _OverrideReader)

    def test_class_registration_instantiates_per_snapshot(self) -> None:
        register_episode_source_reader("test_plugin_cls", _StubReader)
        first = motus_data.EPISODE_SOURCE_READERS.snapshot()
        second = motus_data.EPISODE_SOURCE_READERS.snapshot()
        self.assertIsInstance(first["test_plugin_cls"], _StubReader)
        self.assertIsNot(
            first["test_plugin_cls"], second["test_plugin_cls"]
        )

    def test_invalid_reader_and_empty_name_raise(self) -> None:
        with self.assertRaises(TypeError):
            register_episode_source_reader("test_plugin_bad", object())
        with self.assertRaises(ValueError):
            register_episode_source_reader("  ", _StubReader)

    def test_unknown_source_error_lists_registered_names(self) -> None:
        with self.assertRaises(EpisodeRepositoryError) as context:
            self._open("test_plugin_missing", "test_plugin_missing_ds")
        message = str(context.exception)
        self.assertIn("test_plugin_missing", message)
        self.assertIn("registered=", message)

    def test_builtin_readers_still_resolve(self) -> None:
        register_episode_source_reader("test_plugin_builtin", _StubReader)
        repository = self._open("test_plugin_builtin", "test_plugin_builtin_ds")
        try:
            names = motus_data.episode_source_readers()
            for builtin in (
                "agibot",
                "agibot_beta",
                "egoverse",
                "openvid",
                "robotmind",
                "robotwin",
            ):
                self.assertIn(builtin, names)
        finally:
            repository.close()


if __name__ == "__main__":
    unittest.main()
