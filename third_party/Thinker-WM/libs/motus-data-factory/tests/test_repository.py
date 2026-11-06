from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from motus_data.datasets.agibot_common import (
    AGIBOT_ACTION_MEAN,
    AGIBOT_ACTION_STD,
    normalize_agibot_eef_actions,
)
from motus_data.catalog.sqlite_index import (
    CatalogOffsetIndexError,
    build_catalog_offset_index,
)
from motus_data.repository import (
    AgibotBetaEpisodeReader,
    EpisodeMemoryLimitError,
    EpisodeRepository,
    EpisodeRepositoryError,
)


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _hdf5_bytes(
    timesteps: int,
    outlier_timestep: int | None = None,
    nonfinite_timestep: int | None = None,
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".h5") as temporary:
        with h5py.File(temporary.name, "w") as handle:
            positions = np.zeros((timesteps, 2, 3), dtype=np.float32)
            positions[:, 0, 0] = np.arange(timesteps, dtype=np.float32) / 100
            positions[:, 1, 1] = np.arange(timesteps, dtype=np.float32) / 200
            if outlier_timestep is not None:
                positions[int(outlier_timestep), 0, 0] = 1000.0
            if nonfinite_timestep is not None:
                positions[int(nonfinite_timestep), 0, 0] = np.nan
            orientations = np.zeros((timesteps, 2, 4), dtype=np.float32)
            orientations[..., 3] = 1
            effectors = np.full((timesteps, 2), 0.5, dtype=np.float32)
            effectors[:, 0] += np.arange(timesteps, dtype=np.float32) / 1000
            handle.create_dataset("state/end/position", data=positions)
            handle.create_dataset("state/end/orientation", data=orientations)
            handle.create_dataset("state/effector/position", data=effectors)
        temporary.seek(0)
        return temporary.read()


class RepositoryFixture:
    def __init__(
        self,
        root: Path,
        protocol_version: str = "v1",
        outlier_timestep: int | None = None,
        nonfinite_timestep: int | None = None,
    ):
        self.root = root
        self.timesteps = 6
        self.episode_id = "agibot_beta:3:7"
        self.tar_path = root / "episode.tar"
        with tarfile.open(self.tar_path, "w") as archive:
            _add_bytes(archive, "7/videos/head_color.mp4", b"fixture-video")
            _add_bytes(
                archive,
                "3/7/proprio_stats.h5",
                _hdf5_bytes(
                    self.timesteps,
                    outlier_timestep=outlier_timestep,
                    nonfinite_timestep=nonfinite_timestep,
                ),
            )
        with tarfile.open(self.tar_path, "r") as archive:
            video_member = archive.getmember("7/videos/head_color.mp4")
            proprio_member = archive.getmember("3/7/proprio_stats.h5")

        def locator(member):
            return {
                "storage_backend": "tar_member",
                "uri": str(self.tar_path),
                "member": member.name,
                "offset_data": member.offset_data,
                "size": member.size,
            }

        v1_texts = [
            {
                "index": 0,
                "kind": "episode_context",
                "text": "Scene and task.",
            },
            {
                "index": 1,
                "kind": "action_segment",
                "text": "Action one.",
                "frame_range": [0, 3],
                "metadata": {"source_segment_index": 0},
            },
            {
                "index": 2,
                "kind": "action_segment",
                "text": "Action two.",
                "frame_range": [3, 6],
                "metadata": {"source_segment_index": 1},
            },
        ]
        if protocol_version == "v1":
            self.record = {
                "schema_version": "motus_episode_record_v1",
                "episode_id": self.episode_id,
                "source_dataset": "agibot_beta",
                "source_child": "agibot_beta",
                "robot_name": "agibot-a2-d",
                "split": "train",
                "video": {
                    "frame_count": self.timesteps,
                    "locator": locator(video_member),
                },
                "state": {
                    "count": self.timesteps,
                    "locator": locator(proprio_member),
                },
                "action": {
                    "count": self.timesteps,
                    "locator": locator(proprio_member),
                },
                "text": {"items": v1_texts},
                "metadata": {"task_id": 3, "raw_episode_id": 7},
            }
        elif protocol_version == "v2":
            def storage(member):
                return {
                    "type": "tar",
                    "path": str(self.tar_path),
                    "member": member.name,
                    "offset": member.offset_data,
                    "size": member.size,
                }

            self.record = {
                "episode_id": self.episode_id,
                "robot": "agibot-a2-d",
                "video": {
                    "count": self.timesteps,
                    "storage": storage(video_member),
                },
                "state": {
                    "count": self.timesteps,
                    "storage": storage(proprio_member),
                },
                "action": {
                    "count": self.timesteps,
                    "storage": storage(proprio_member),
                },
                "texts": [
                    {
                        "index": item["index"],
                        "kind": item["kind"],
                        "text": item["text"],
                        **(
                            {"video_range": item["frame_range"]}
                            if "frame_range" in item
                            else {}
                        ),
                        **(
                            {"meta": item["metadata"]}
                            if "metadata" in item
                            else {}
                        ),
                    }
                    for item in v1_texts
                ],
                "meta": {"task_id": 3, "raw_episode_id": 7},
            }
        else:
            raise ValueError(f"unsupported fixture protocol={protocol_version}")
        self.catalog_path = root / "fixture.episode_catalog.jsonl"
        payload = (
            json.dumps(self.record, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self.catalog_path.write_bytes(payload)
        if protocol_version == "v1":
            metadata = {
                "schema_version": "motus_episode_catalog_metadata_v1",
                "record_schema_version": "motus_episode_record_v1",
                "catalog_id": "fixture_catalog_v1",
                "source_dataset": "agibot_beta",
                "source_child": "agibot_beta",
                "data_version": "fixture_v1",
                "record_count": 1,
                "statistics": {
                    "catalog_sha256": hashlib.sha256(payload).hexdigest(),
                },
                "split_policy": {
                    "type": "stable_ranked_episode_hash",
                    "seed": 20260720,
                    "test_fraction": 0.01,
                },
            }
        else:
            metadata = {
                "format": "motus_episode_catalog_v2",
                "catalog_id": "fixture_catalog_v2",
                "source": "agibot_beta",
                "child": "agibot_beta",
                "episodes": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        self.catalog_path.with_suffix(".meta.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        self.decode_calls = []

    def video_decoder(self, path: str, indices):
        self.decode_calls.append(tuple(indices))
        self.assert_video_payload(path)
        return {
            int(index): np.full((4, 5, 3), int(index), dtype=np.uint8)
            for index in indices
        }

    @staticmethod
    def assert_video_payload(path: str):
        assert Path(path).read_bytes() == b"fixture-video"

    def repository(self) -> EpisodeRepository:
        self.reader = AgibotBetaEpisodeReader(
            video_decoder=self.video_decoder
        )
        return EpisodeRepository(
            self.catalog_path,
            readers={"agibot_beta": self.reader},
        )


class EpisodeRepositoryTests(unittest.TestCase):
    def test_indexed_repository_is_lazy_and_preserves_episode_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root, protocol_version="v2")
            progress = []
            result = build_catalog_offset_index(
                fixture.catalog_path,
                progress_callback=lambda state: progress.append(dict(state)),
            )
            index_path = Path(result["index_path"])
            self.assertTrue(index_path.is_file())
            self.assertEqual(result["catalog_records"], "1")
            self.assertEqual(progress[-1]["records"], 1)
            self.assertFalse(list(root.glob(".*.tmp.*")))

            reader = AgibotBetaEpisodeReader(
                video_decoder=fixture.video_decoder
            )
            repository = EpisodeRepository(
                fixture.catalog_path,
                readers={"agibot_beta": reader},
                index_path=index_path,
                verify_sha256=False,
                record_cache_size=1,
            )
            self.assertTrue(repository.indexed)
            self.assertIsNone(repository._records)
            self.assertEqual(len(repository), 1)
            self.assertEqual(
                list(repository.iter_episode_ids()),
                [fixture.episode_id],
            )
            episode = repository.get_episode(fixture.episode_id)
            self.assertEqual(episode.info.video_count, fixture.timesteps)
            frames = episode.read_video([4, 1, 4])
            self.assertEqual(frames[:, 0, 0, 0].tolist(), [4, 1, 4])
            self.assertEqual(len(repository._record_cache), 1)
            self.assertEqual(
                [info.episode_id for info in repository.iter_episode_infos()],
                [fixture.episode_id],
            )
            repository.close()

    def test_index_binding_and_optional_full_sha_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root, protocol_version="v2")
            result = build_catalog_offset_index(fixture.catalog_path)
            index_path = Path(result["index_path"])

            original = fixture.catalog_path.read_bytes()
            changed = original.replace(b"agibot-a2-d", b"agibot-a2-x")
            self.assertEqual(len(changed), len(original))
            self.assertNotEqual(changed, original)
            fixture.catalog_path.write_bytes(changed)
            with self.assertRaisesRegex(
                EpisodeRepositoryError, "Catalog SHA256 mismatch"
            ):
                EpisodeRepository(
                    fixture.catalog_path,
                    index_path=index_path,
                    verify_sha256=True,
                )
            fixture.catalog_path.write_bytes(original)

            connection = sqlite3.connect(index_path)
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='catalog_id'",
                ("wrong_catalog",),
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(
                CatalogOffsetIndexError, "binding mismatch"
            ):
                EpisodeRepository(
                    fixture.catalog_path,
                    index_path=index_path,
                    verify_sha256=False,
                )

    def test_split_neutral_v2_repository_and_v1_compatibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            v2 = RepositoryFixture(Path(temp_dir), protocol_version="v2")
            repository = v2.repository()
            self.assertEqual(
                repository.protocol_version,
                "motus_episode_catalog_v2",
            )
            info = repository.get_episode(v2.episode_id).info
            self.assertIsNone(info.split)
            self.assertEqual(len(list(repository.iter_episode_infos())), 1)
            with self.assertRaisesRegex(
                EpisodeRepositoryError, "split filtering is unavailable"
            ):
                list(repository.iter_episode_infos(split="train"))

    def test_complete_timeline_normalization_matches_pre_repository_math(self):
        raw = np.arange(5 * 2 * 7, dtype=np.float32).reshape(5, 2, 7) / 100
        previous = torch.tensor(raw, dtype=torch.float32)
        previous = torch.cat(
            [previous, torch.zeros(previous.shape[0], previous.shape[1], 6)],
            dim=-1,
        )
        previous[:, :, 7:] = AGIBOT_ACTION_MEAN[7:]
        previous = (
            previous - AGIBOT_ACTION_MEAN[None, None, :]
        ) / AGIBOT_ACTION_STD[None, None, :]
        previous = previous.reshape(5, 26).float()

        current = normalize_agibot_eef_actions(raw)
        self.assertTrue(torch.equal(current, previous))

    def test_complete_episode_view_and_exact_independent_reads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RepositoryFixture(Path(temp_dir))
            repository = fixture.repository()
            self.assertEqual(len(repository), 1)

            infos = list(
                repository.iter_episode_infos(
                    split="train",
                    source_child="agibot_beta",
                    robot_name="agibot-a2-d",
                )
            )
            self.assertEqual(len(infos), 1)
            episode = repository.get_episode(fixture.episode_id)
            self.assertEqual(episode.info, infos[0])
            self.assertEqual(episode.video.count, 6)
            self.assertEqual(episode.state.schema.item_shape, (26,))
            self.assertEqual(episode.action.schema.item_shape, (26,))
            self.assertEqual([item.index for item in episode.text.items], [0, 1, 2])

            frames = episode.read_video([3, 1, 3])
            self.assertEqual(tuple(frames.shape), (3, 4, 5, 3))
            self.assertEqual(frames.dtype, torch.uint8)
            self.assertEqual(frames[:, 0, 0, 0].tolist(), [3, 1, 3])
            self.assertEqual(fixture.decode_calls[-1], (1, 3))

            float_frames = episode.video.read(
                [2, 0],
                output_format="rgb_float32_chw",
                video_size=(2, 3),
            )
            self.assertEqual(tuple(float_frames.shape), (2, 3, 2, 3))
            self.assertEqual(float_frames.dtype, torch.float32)
            self.assertTrue(
                torch.allclose(
                    float_frames[:, 0, 0, 0],
                    torch.tensor([2 / 255, 0.0]),
                )
            )

            states = episode.read_state([2, 0, 2])
            actions = episode.read_action([2, 0, 2])
            self.assertEqual(tuple(states.shape), (3, 26))
            self.assertTrue(torch.equal(states, actions))
            self.assertTrue(torch.equal(states[0], states[2]))
            self.assertFalse(torch.equal(states[0], states[1]))

            texts = episode.read_text([2, 0, 2])
            self.assertEqual([item.index for item in texts], [2, 0, 2])
            self.assertEqual(tuple(episode.read_state([]).shape), (0, 26))
            self.assertEqual(tuple(episode.read_action([]).shape), (0, 26))
            self.assertEqual(episode.read_text([]), ())

            with self.assertRaisesRegex(EpisodeRepositoryError, "out of bounds"):
                episode.read_video([6])
            with self.assertRaisesRegex(EpisodeRepositoryError, "must be an integer"):
                episode.read_state([True])
            with self.assertRaises(KeyError):
                repository.get_episode("agibot_beta:missing")

    def test_selection_chunks_and_complete_materialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RepositoryFixture(Path(temp_dir))
            repository = fixture.repository()
            episode = repository.get_episode(fixture.episode_id)

            selection = {
                "episode_id": fixture.episode_id,
                "video_frame_indices": [4, 1, 4],
                "state_indices": [0, 2],
                "action_indices": [3, 1, 3],
                "text_indices": [2, 0, 2],
            }
            selected = episode.read_selection(selection)
            self.assertEqual(selected.video_frame_indices, (4, 1, 4))
            self.assertEqual(selected.video[:, 0, 0, 0].tolist(), [4, 1, 4])
            self.assertEqual(tuple(selected.state.shape), (2, 26))
            self.assertEqual(tuple(selected.action.shape), (3, 26))
            self.assertEqual([item.index for item in selected.text], [2, 0, 2])

            chunks = list(episode.video.iter_chunks(chunk_size=2, start=1, stop=6))
            self.assertEqual(
                [chunk.indices for chunk in chunks],
                [(1, 2), (3, 4), (5,)],
            )
            self.assertEqual(
                torch.cat([chunk.frames for chunk in chunks])[:, 0, 0, 0].tolist(),
                [1, 2, 3, 4, 5],
            )

            state_action_text = repository.load_full_episode(fixture.episode_id)
            self.assertIsNone(state_action_text.video)
            self.assertEqual(tuple(state_action_text.state.shape), (6, 26))
            self.assertEqual(tuple(state_action_text.action.shape), (6, 26))
            self.assertEqual(len(state_action_text.text), 3)

            required_bytes = 6 * 4 * 5 * 3
            with self.assertRaises(EpisodeMemoryLimitError):
                episode.materialize_video(max_bytes=required_bytes - 1)
            full_video = episode.materialize_video(max_bytes=required_bytes)
            self.assertEqual(tuple(full_video.shape), (6, 4, 5, 3))
            self.assertEqual(full_video[:, 0, 0, 0].tolist(), list(range(6)))

            complete = episode.materialize(
                modalities=("video", "state", "action", "text"),
                max_video_bytes=required_bytes,
            )
            self.assertEqual(tuple(complete.video.shape), (6, 4, 5, 3))
            self.assertEqual(tuple(complete.state.shape), (6, 26))
            self.assertEqual(tuple(complete.action.shape), (6, 26))
            self.assertEqual(len(complete.text), 3)

    def test_multiple_selections_union_each_physical_modality_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RepositoryFixture(Path(temp_dir))
            repository = fixture.repository()
            episode = repository.get_episode(fixture.episode_id)
            selections = [
                {
                    "episode_id": fixture.episode_id,
                    "video_frame_indices": [0, 2, 0],
                    "state_indices": [0, 2, 0],
                    "action_indices": [1, 3],
                    "text_indices": [0, 1],
                },
                {
                    "episode_id": fixture.episode_id,
                    "video_frame_indices": [2, 4, 2],
                    "state_indices": [4, 2],
                    "action_indices": [3, 5, 3],
                    "text_indices": [2, 0, 2],
                },
            ]
            fixture.decode_calls.clear()
            with mock.patch.object(
                fixture.reader,
                "_canonical_timeline",
                wraps=fixture.reader._canonical_timeline,
            ) as timeline:
                samples = episode.read_selections(selections)

            self.assertEqual(fixture.decode_calls, [(0, 2, 4)])
            self.assertEqual(timeline.call_count, 1)
            self.assertEqual(samples[0].video[:, 0, 0, 0].tolist(), [0, 2, 0])
            self.assertEqual(samples[1].video[:, 0, 0, 0].tolist(), [2, 4, 2])
            self.assertTrue(torch.equal(samples[0].state[0], samples[0].state[2]))
            self.assertTrue(torch.equal(samples[1].action[0], samples[1].action[2]))
            self.assertEqual([item.index for item in samples[1].text], [2, 0, 2])

    def test_catalog_integrity_is_checked_before_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RepositoryFixture(Path(temp_dir))
            fixture.catalog_path.write_bytes(
                fixture.catalog_path.read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(EpisodeRepositoryError, "SHA256 mismatch"):
                fixture.repository()


if __name__ == "__main__":
    unittest.main()
