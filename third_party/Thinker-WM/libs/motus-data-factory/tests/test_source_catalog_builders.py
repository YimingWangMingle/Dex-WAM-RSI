from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from motus_data.catalog.source_families import (
    build_source_family_episode_catalog,
)
from motus_data.catalog.agibot import build_agibot_episode_catalog
from motus_data.catalog.robotmind import build_robotmind_episode_catalog
from motus_data.datasets.robotmind import ROBOTMIND_SCHEMAS
from motus_data.repository import EpisodeRepository
from motus_data.source_readers import AgibotEpisodeReader
from tests.test_repository import _hdf5_bytes


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_robotmind_agilex_episode(
    root: Path,
    *,
    count: int = 6,
    mismatched_key: str | None = None,
) -> Path:
    task_dir = root / "move_object_with_both_arms"
    task_dir.mkdir(parents=True)
    (task_dir / "zh_description.txt").write_text(
        "双臂移动物体。",
        encoding="utf-8",
    )
    h5_path = (
        task_dir
        / "success_episodes"
        / "0001"
        / "data"
        / "trajectory.hdf5"
    )
    h5_path.parent.mkdir(parents=True)
    schema = ROBOTMIND_SCHEMAS["agilex"]
    with h5py.File(h5_path, "w") as handle:
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        images = handle.create_dataset(
            schema.rgb_key,
            shape=(count,),
            dtype=dtype,
        )
        for index in range(count):
            image = Image.fromarray(
                np.full((4, 5, 3), index, dtype=np.uint8)
            )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            images[index] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        dimensions = (6, 1, 6, 1)
        for (_, key), dimension in zip(
            schema.state_action_keys,
            dimensions,
            strict=True,
        ):
            rows = count - 1 if key == mismatched_key else count
            values = np.arange(rows * dimension, dtype=np.float32).reshape(
                rows,
                dimension,
            )
            handle.create_dataset(key, data=values)
        metadata = handle.create_group("metadata")
        metadata.attrs["trajectory_length"] = count
        metadata.attrs["language_instruction"] = "Move the object with both arms."
    return h5_path


class SourceCatalogBuilderTests(unittest.TestCase):
    def test_agibot_catalog_preserves_text_ranges_and_independent_timelines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation_root = root / "annotations"
            annotation_root.mkdir()
            data_root = root / "data"
            video_path = (
                data_root
                / "observations"
                / "3"
                / "7"
                / "videos_5fps"
                / "head_color.mp4"
            )
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"fixture-video")
            proprio_path = (
                data_root
                / "proprio_stats"
                / "3"
                / "7"
                / "proprio_stats.h5"
            )
            proprio_path.parent.mkdir(parents=True)
            proprio_path.write_bytes(_hdf5_bytes(11))
            annotation = [
                {
                    "task_id": 3,
                    "episode_id": 7,
                    "task_name": "Open the drawer.",
                    "init_scene_text": "A closed drawer.",
                    "action_video_fps_ratio": 2,
                    "effective_frames": [0, 5],
                    "label_info": {
                        "flatten_action_config": {
                            "0": {"action_text": "Reach for the handle."},
                            "3": {"action_text": "Pull the drawer."},
                            "4": {"action_text": "", "skill": ""},
                        }
                    },
                }
            ]
            (annotation_root / "task_3.json").write_text(
                json.dumps(annotation), encoding="utf-8"
            )
            output = root / "agibot.episode_catalog.jsonl"
            with mock.patch(
                "motus_data.catalog.agibot._video_count",
                return_value=6,
            ):
                metadata = build_agibot_episode_catalog(
                    annotation_folders=[str(annotation_root)],
                    data_root_folders=[str(data_root)],
                    dataset_names=["agibot-world"],
                    output_path=output,
                    catalog_id="agibot_fixture",
                    data_version="fixture",
                )
            self.assertEqual(metadata["episodes"], 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["video"]["count"], 6)
            self.assertEqual(record["state"]["count"], 11)
            self.assertEqual(
                [item.get("video_range") for item in record["texts"]],
                [None, [0, 3], [3, 4], [4, 6]],
            )
            self.assertEqual(record["texts"][-1]["text"], "Action: \n")
            self.assertEqual(
                record["meta"]["timeline_mapping"]["ratio"], 2
            )
            self.assertEqual(
                record["meta"]["timeline_mapping"]["partial_tail_steps"], 1
            )
            self.assertEqual(
                record["meta"]["effective_video_frames"], [0, 5]
            )

            def video_decoder(path: str, indices):
                return {
                    int(index): np.full(
                        (4, 5, 3), int(index), dtype=np.uint8
                    )
                    for index in indices
                }

            repository = EpisodeRepository(
                output,
                readers={
                    "agibot": AgibotEpisodeReader(
                        video_decoder=video_decoder
                    )
                },
            )
            try:
                episode = repository.get_episode("agibot:3:7")
                selection = episode.read_selection(
                    {
                        "episode_id": "agibot:3:7",
                        "video_frame_indices": [1, 2],
                        "state_indices": [2],
                        "action_indices": [3, 4],
                        "text_indices": [0, 1],
                    }
                )
                self.assertEqual(tuple(selection.video.shape), (2, 4, 5, 3))
                self.assertEqual(tuple(selection.state.shape), (1, 26))
                self.assertEqual(tuple(selection.action.shape), (2, 26))
            finally:
                repository.close()

    def test_robotwin_manifest_is_only_an_offline_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_path = root / "episode.txt"
            meta_path.write_text(
                "Move the red block.\nThen close the drawer.\n",
                encoding="utf-8",
            )
            base = {
                "source_dataset": "robotwin",
                "task": "fixture_task",
                "text": {"target": "Move the red block."},
                "video": {"uri": "/raw/fixture.mp4"},
                "state": {"uri": "/raw/fixture.pt"},
                "action": {"uri": "/raw/fixture.pt"},
                "robot": {
                    "name": "aloha-agilex",
                    "state_dim": 14,
                    "action_dim": 14,
                },
                "metadata": {
                    "data_split": "clean",
                    "episode_name": "0",
                    "meta_path": str(meta_path),
                    "dataset_root": "/raw/robotwin/aloha",
                    "effective_total_frames": 140,
                },
            }
            manifest = root / "robotwin.jsonl"
            _write_rows(
                manifest,
                [
                    {**base, "id": "window0"},
                    {**base, "id": "window1"},
                ],
            )
            output = root / "robotwin.episode_catalog.jsonl"
            progress = []
            metadata = build_source_family_episode_catalog(
                manifest_path=manifest,
                output_path=output,
                family="robotwin",
                source_child="robotwin_aloha",
                catalog_id="robotwin_aloha_fixture",
                data_version="fixture",
                verify_source=False,
                progress_callback=lambda state: progress.append(dict(state)),
            )
            self.assertEqual(metadata["episodes"], 1)
            self.assertTrue(progress[-1]["done"])
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["video"]["count"], 140)
            self.assertEqual(record["state"]["meta"]["dimension"], 14)
            self.assertEqual(
                [item["text"] for item in record["texts"]],
                ["Move the red block.", "Then close the drawer."],
            )
            self.assertNotIn(
                str(manifest),
                json.dumps(record, ensure_ascii=False),
            )

    def test_robotmind_builder_verifies_raw_alignment_and_keeps_languages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "trajectory.hdf5"
            with h5py.File(h5_path, "w") as handle:
                dtype = h5py.vlen_dtype(np.dtype("uint8"))
                images = handle.create_dataset(
                    "camera/rgb", shape=(6,), dtype=dtype
                )
                for index in range(6):
                    image = Image.fromarray(
                        np.full((4, 5, 3), index, dtype=np.uint8)
                    )
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    images[index] = np.frombuffer(
                        buffer.getvalue(), dtype=np.uint8
                    )
                handle.create_dataset(
                    "puppet/left",
                    data=np.arange(12, dtype=np.float32).reshape(6, 2),
                )
                handle.create_dataset(
                    "puppet/right",
                    data=np.arange(6, dtype=np.float32).reshape(6, 1),
                )
                metadata_group = handle.create_group("metadata")
                metadata_group.attrs["trajectory_length"] = 6
            base = {
                "source_dataset": "robotmind",
                "task": "fixture_task",
                "text": {
                    "en": "Move the object.",
                    "zh": "移动物体。",
                    "target": "Move the object.",
                },
                "video": {"uri": str(h5_path), "key": "camera/rgb"},
                "state": {
                    "uri": str(h5_path),
                    "keys": {
                        "left": "puppet/left",
                        "right": "puppet/right",
                    },
                },
                "action": {
                    "uri": str(h5_path),
                    "keys": {
                        "left": "puppet/left",
                        "right": "puppet/right",
                    },
                },
                "robot": {
                    "name": "robotmind-fixture",
                    "state_dim": 3,
                    "action_dim": 3,
                },
                "metadata": {
                    "episode_name": "episode0",
                    "total_frames": 6,
                    "state_action_keys_order": ["left", "right"],
                },
            }
            manifest = root / "robotmind.jsonl"
            _write_rows(
                manifest,
                [
                    {**base, "id": "window0"},
                    {**base, "id": "window1"},
                ],
            )
            output = root / "robotmind.episode_catalog.jsonl"
            metadata = build_source_family_episode_catalog(
                manifest_path=manifest,
                output_path=output,
                family="robotmind",
                source_child="robotmind_fixture",
                catalog_id="robotmind_fixture",
                data_version="fixture",
                verify_source=True,
            )
            self.assertEqual(metadata["episodes"], 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["kind"] for item in record["texts"]],
                ["instruction_en", "instruction_zh"],
            )
            self.assertEqual(
                record["state"]["meta"]["keys"],
                ["puppet/left", "puppet/right"],
            )
            repository = EpisodeRepository(output)
            try:
                episode = repository.get_episode(record["episode_id"])
                self.assertEqual(episode.info.state_count, 6)
                self.assertEqual(episode.state.schema.item_shape, (3,))
            finally:
                repository.close()

    def test_robotmind_builder_rejects_misaligned_modalities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "bad.hdf5"
            with h5py.File(h5_path, "w") as handle:
                dtype = h5py.vlen_dtype(np.dtype("uint8"))
                handle.create_dataset("camera/rgb", shape=(6,), dtype=dtype)
                handle.create_dataset("puppet/left", shape=(5, 2))
            row = {
                "id": "bad",
                "source_dataset": "robotmind",
                "task": "bad",
                "text": {"en": "bad"},
                "video": {"uri": str(h5_path), "key": "camera/rgb"},
                "state": {
                    "uri": str(h5_path),
                    "keys": {"left": "puppet/left"},
                },
                "action": {
                    "uri": str(h5_path),
                    "keys": {"left": "puppet/left"},
                },
                "robot": {
                    "name": "robotmind-bad",
                    "state_dim": 2,
                    "action_dim": 2,
                },
                "metadata": {
                    "episode_name": "0",
                    "total_frames": 5,
                    "state_action_keys_order": ["left"],
                },
            }
            manifest = root / "bad.jsonl"
            _write_rows(manifest, [row])
            with self.assertRaisesRegex(ValueError, "not exactly aligned"):
                build_source_family_episode_catalog(
                    manifest_path=manifest,
                    output_path=root / "bad.episode_catalog.jsonl",
                    family="robotmind",
                    source_child="robotmind_bad",
                    catalog_id="bad",
                    data_version="bad",
                    verify_source=True,
                )

    def test_robotmind_direct_agilex_builder_needs_no_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "agilex"
            h5_path = _write_robotmind_agilex_episode(data_root)
            output = root / "robotmind_agilex.episode_catalog.jsonl"
            progress = []
            metadata = build_robotmind_episode_catalog(
                data_root=data_root,
                embodiment="agilex",
                output_path=output,
                catalog_id="robotmind_agilex_fixture",
                data_version="fixture",
                progress_callback=lambda state: progress.append(dict(state)),
            )
            self.assertEqual(metadata["episodes"], 1)
            self.assertEqual(metadata["child"], "robotmind_agilex")
            self.assertEqual(
                metadata["meta"]["build"]["source_type"],
                "raw_episode_hdf5",
            )
            self.assertNotIn(
                "manifest",
                json.dumps(metadata, ensure_ascii=False).lower(),
            )
            self.assertTrue(progress[-1]["done"])

            record = json.loads(output.read_text(encoding="utf-8"))
            schema = ROBOTMIND_SCHEMAS["agilex"]
            self.assertEqual(record["robot"], "robotmind-agilex")
            self.assertEqual(record["video"]["count"], 6)
            self.assertEqual(record["video"]["storage"]["path"], str(h5_path))
            self.assertEqual(record["video"]["meta"]["key"], schema.rgb_key)
            self.assertEqual(
                record["video"]["meta"]["native_shape_hwc"],
                [4, 5, 3],
            )
            self.assertEqual(record["state"]["meta"]["dimension"], 14)
            self.assertEqual(
                record["state"]["meta"]["keys"],
                [key for _, key in schema.state_action_keys],
            )
            self.assertEqual(
                [item["kind"] for item in record["texts"]],
                ["instruction_en", "instruction_zh"],
            )

            repository = EpisodeRepository(output)
            try:
                episode = repository.get_episode(record["episode_id"])
                selection = episode.read_selection(
                    {
                        "episode_id": record["episode_id"],
                        "video_frame_indices": [1, 4],
                        "state_indices": [0, 3],
                        "action_indices": [2, 5],
                        "text_indices": [0, 1],
                    }
                )
                self.assertEqual(tuple(selection.video.shape), (2, 4, 5, 3))
                self.assertEqual(tuple(selection.state.shape), (2, 14))
                self.assertEqual(tuple(selection.action.shape), (2, 14))
                self.assertEqual(len(selection.text), 2)
            finally:
                repository.close()

    def test_robotmind_direct_builder_rejects_misaligned_agilex_episode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "agilex"
            schema = ROBOTMIND_SCHEMAS["agilex"]
            _write_robotmind_agilex_episode(
                data_root,
                mismatched_key=schema.state_action_keys[-1][1],
            )
            with self.assertRaisesRegex(ValueError, "not exactly aligned"):
                build_robotmind_episode_catalog(
                    data_root=data_root,
                    embodiment="agilex",
                    output_path=root / "bad.episode_catalog.jsonl",
                    catalog_id="bad",
                    data_version="bad",
                )


if __name__ == "__main__":
    unittest.main()
