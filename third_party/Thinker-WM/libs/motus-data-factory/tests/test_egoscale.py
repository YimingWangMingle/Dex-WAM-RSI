from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

from motus_data.batch_adapters import IndexedBatchCollator, build_indexed_batch_adapter
from motus_data.catalog.egoscale import (
    EGOSCALE_PARQUET_COLUMNS,
    build_egoscale_episode_catalog,
    egoscale_shard_index,
    egoscale_source_group_key,
    select_egoscale_episode_dirs,
)
from motus_data.contracts import validate_motus_native_multimodal_batch_v1
from motus_data.egoscale import EgoScaleEpisodeReader, compose_egoscale_text
from motus_data.repository import EpisodeRepository, TextItem
from motus_data.sample_pool import SamplePoolDataset


PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


class EgoScaleTextComposerTests(unittest.TestCase):
    """compose_egoscale_text：episode 级字段 + 窗口内全部片段（所有字段）。"""

    @staticmethod
    def _items() -> list[TextItem]:
        return [
            TextItem(0, "task_name_en", "Pick and place cup", None, {}),
            TextItem(1, "task_description_en", "Pick up the cup.", None, {}),
            TextItem(2, "task_detail_en", "Use the right hand", None, {}),
            TextItem(3, "task_label", "kitchen_skills", None, {}),
            TextItem(
                4,
                "fine_action_en",
                "reach for the cup",
                (100, 200),
                {
                    "action": "reach",
                    "hand": "right",
                    "scene_semantic": "kitchen",
                    "camera_motion": "static",
                },
            ),
            TextItem(
                5,
                "fine_action_en",
                "grasp the cup",
                (200, 260),
                {"action": "grasp", "hand": "right", "scene_semantic": "", "camera_motion": ""},
            ),
        ]

    def test_all_context_fields_and_all_segment_fields(self) -> None:
        self.assertEqual(
            compose_egoscale_text(self._items()),
            "Task name: Pick and place cup.\n"
            "Task: Pick up the cup.\n"
            "Task detail: Use the right hand.\n"
            "Task label: kitchen_skills.\n"
            "Action: reach for the cup.\n"
            "Action label: reach.\n"
            "Hand: right.\n"
            "Scene semantic: kitchen.\n"
            "Camera motion: static.\n"
            "Action: grasp the cup.\n"
            "Action label: grasp.\n"
            "Hand: right.",
        )

    def test_missing_context_and_empty_meta_are_skipped(self) -> None:
        items = [
            TextItem(1, "task_description_en", "Pick up the cup.", None, {}),
            TextItem(5, "fine_action_en", "grasp the cup", (200, 260), {}),
        ]
        self.assertEqual(
            compose_egoscale_text(items),
            "Task: Pick up the cup.\nAction: grasp the cup.",
        )

    def test_unknown_kind_is_kept_verbatim(self) -> None:
        items = [TextItem(0, "mystery_kind", "raw text", None, {})]
        self.assertEqual(compose_egoscale_text(items), "raw text.")

    def test_empty_input_composes_empty(self) -> None:
        self.assertEqual(compose_egoscale_text([]), "")

    def test_registered_for_egoscale_source(self) -> None:
        import motus_data

        self.assertIs(motus_data.resolve_text_composer("egoscale"), compose_egoscale_text)



@unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow is not installed")
class EgoScaleTests(unittest.TestCase):
    EPISODE_NAME = "session_abc123_00000_task_0001_lerobot_v3"

    @classmethod
    def _fixture(cls, root: Path) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        episode = root / cls.EPISODE_NAME
        for directory in (
            "calibration",
            "meta",
            "data/chunk-000",
            "videos/base_0_camera/rgb/image/chunk-000",
            "videos/base_1_camera/rgb/image/chunk-000",
        ):
            (episode / directory).mkdir(parents=True)
        frame_count = 4
        task = {
            "task_index": 0,
            "task_id": "fixture_task_0",
            "task_name": "Fixture task name",
            "task_description": "Perform the fixture task.",
            "task_usable": True,
            "source": {
                "task_label": "fixture_label",
                "detail": "Detailed fixture instruction.",
                "source_start_frame": 100,
                "source_end_frame": 104,
            },
        }
        source_manifest = {
            "data_type": "teleop",
            "cases": [
                {
                    "episode_id": cls.EPISODE_NAME,
                    "frame_count": frame_count,
                    "fps": 30.0,
                    "duration": frame_count / 30.0,
                    "tasks": [task],
                }
            ],
        }
        (episode / "source_manifest.json").write_text(
            json.dumps(source_manifest), encoding="utf-8"
        )
        info = {
            "fps": 30,
            "codebase_version": "v3.0",
            "robot_type": "orb_stereo",
        }
        (episode / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        annotation = {
            "episodes": [
                {
                    "episode_id": cls.EPISODE_NAME,
                    "extras": {
                        "actions": [
                            {
                                "start_frame": 0,
                                "end_frame": 4,
                                "start_seconds": 0.0,
                                "end_seconds": 4 / 30.0,
                                "description": "Move both hands.",
                                "action": "move_both",
                                "hand": "both",
                                "action_index": 0,
                            }
                        ]
                    },
                }
            ]
        }
        (episode / "meta/custom_annotation.json").write_text(
            json.dumps(annotation), encoding="utf-8"
        )
        (episode / "meta/custom_metadata.csv").write_text(
            "episode_index,episode_id\n0,fixture\n", encoding="utf-8"
        )
        for calibration in (
            "undistorted_intrinsics.json",
            "stereo_calibration.json",
        ):
            (episode / "calibration" / calibration).write_text("{}\n", encoding="utf-8")
        for view in ("base_0", "base_1"):
            (
                episode / f"videos/{view}_camera/rgb/image/chunk-000/episode_000000.mp4"
            ).write_bytes(b"fixture-mp4")

        pq.write_table(
            pa.table({"task_index": [0], "task": ["Perform the fixture task."]}),
            episode / "meta/tasks.parquet",
        )
        pq.write_table(
            pa.table({"episode_index": [0], "tasks": [["Perform the fixture task."]]}),
            episode / "meta/episodes.parquet",
        )

        # The production payload uses Arrow list columns and may contain null
        # rows when a hand is outside the camera view.
        pose_type = pa.list_(pa.float32())
        track_type = pa.list_(pa.list_(pa.float32(), 3), 21)
        identity_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        left_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        right_pose = [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
        tracks = [[[float(row), 0.0, 1.0] for _ in range(21)] for row in range(4)]
        tracks[0][0][0] = None
        columns = {name: pa.nulls(frame_count) for name in EGOSCALE_PARQUET_COLUMNS}
        columns.update(
            {
                "episode_index": pa.array([0] * 4, type=pa.int64()),
                "frame_index": pa.array(range(4), type=pa.int64()),
                "timestamp": pa.array(np.arange(4) / 30.0, type=pa.float32()),
                "task_index": pa.array([0] * 4, type=pa.int64()),
                "left_hand/tracks": pa.array(tracks, type=track_type),
                "left_hand/timestamp": pa.array(range(4), type=pa.int64()),
                "right_hand/timestamp": pa.array(range(4), type=pa.int64()),
                "head/pose": pa.array([identity_pose] * 4, type=pose_type),
                "head/valid": pa.array([True, True, False, True]),
                "head/timestamp": pa.array(range(4), type=pa.int64()),
                "left_hand/eepose": pa.array(
                    [left_pose, left_pose, None, left_pose], type=pose_type
                ),
                "right_hand/eepose": pa.array(
                    [None, right_pose, right_pose, right_pose], type=pose_type
                ),
                "left_hand/world_eepose": pa.array(
                    [left_pose, left_pose, None, left_pose], type=pose_type
                ),
                "right_hand/world_eepose": pa.array(
                    [None, right_pose, right_pose, right_pose], type=pose_type
                ),
                "left_hand/mano/valid": pa.array([True, True, False, True]),
                "right_hand/mano/valid": pa.array([False, True, True, True]),
            }
        )
        pq.write_table(
            pa.table(columns),
            episode / "data/chunk-000/episode_000000.parquet",
            row_group_size=2,
        )
        return episode

    @staticmethod
    def _decoder(path: str, indices: list[int]) -> dict[int, np.ndarray]:
        base = 10 if "base_0" in path else 100
        return {
            int(index): np.full((4, 5, 3), base + int(index), dtype=np.uint8)
            for index in set(indices)
        }

    def test_stable_group_sharding_keeps_chunks_together(self) -> None:
        first = "session_abc123_00000_task_0001_0_120_lerobot_v3"
        second = "session_abc123_00000_task_0001_120_240_lerobot_v3"
        self.assertEqual(
            egoscale_source_group_key(first),
            egoscale_source_group_key(second),
        )
        group = egoscale_source_group_key(first)
        self.assertEqual(
            egoscale_shard_index(group, 10),
            egoscale_shard_index(egoscale_source_group_key(second), 10),
        )

    def test_multi_bucket_selection_is_exact_single_bucket_complement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            groups: dict[int, str] = {}
            candidate = 0
            while len(groups) < 10:
                group = f"session_group{candidate:05d}_00000_task_0001"
                bucket = egoscale_shard_index(group, 10)
                groups.setdefault(bucket, group)
                candidate += 1
            for group in groups.values():
                (root / f"{group}_0_120_lerobot_v3").mkdir()
                (root / f"{group}_120_240_lerobot_v3").mkdir()

            first, first_meta = select_egoscale_episode_dirs(
                root,
                shard_count=10,
                shard_index=0,
            )
            remaining, remaining_meta = select_egoscale_episode_dirs(
                root,
                shard_count=10,
                shard_indices=tuple(range(1, 10)),
            )
            first_names = {path.name for path in first}
            remaining_names = {path.name for path in remaining}
            all_names = {path.name for path in root.iterdir()}
            self.assertEqual(len(first_names), 2)
            self.assertEqual(len(remaining_names), 18)
            self.assertFalse(first_names.intersection(remaining_names))
            self.assertEqual(first_names.union(remaining_names), all_names)
            self.assertEqual(first_meta["shard_index"], 0)
            self.assertEqual(first_meta["selected_shard_indices"], [0])
            self.assertEqual(
                remaining_meta["selected_shard_indices"],
                list(range(1, 10)),
            )
            self.assertEqual(
                remaining_meta["selection_mode"],
                "hash_bucket_union",
            )

    def test_multi_bucket_selection_rejects_ambiguous_or_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                select_egoscale_episode_dirs(
                    root,
                    shard_count=10,
                    shard_index=0,
                    shard_indices=(1, 2),
                )
            with self.assertRaisesRegex(ValueError, "duplicates"):
                select_egoscale_episode_dirs(
                    root,
                    shard_count=10,
                    shard_indices=(1, 1),
                )
            with self.assertRaisesRegex(ValueError, "0 <= index"):
                select_egoscale_episode_dirs(
                    root,
                    shard_count=10,
                    shard_indices=(1, 10),
                )

    def test_catalog_episode_access_native_fields_and_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode_dir = self._fixture(root)
            catalog = root / "egoscale.episode_catalog.jsonl"
            metadata = build_egoscale_episode_catalog(
                root,
                catalog,
                catalog_id="egoscale_fixture_v1",
                data_version="fixture",
                shard_count=1,
                shard_index=0,
            )
            self.assertEqual(metadata["source"], "egoscale")
            self.assertEqual(metadata["child"], "egoscale_first500h")
            self.assertEqual(metadata["episodes"], 1)
            self.assertEqual(
                metadata["meta"]["sharding"]["selected_shard_indices"],
                [0],
            )
            record = json.loads(catalog.read_text(encoding="utf-8"))
            self.assertEqual(record["video"]["count"], 4)
            self.assertEqual(record["state"]["meta"]["dimension"], 24)
            self.assertEqual(record["meta"]["source_frame_range"], [100, 104])
            self.assertEqual(len(record["texts"]), 5)

            reader = EgoScaleEpisodeReader(
                camera_mode="stereo_horizontal",
                video_decoder=self._decoder,
            )
            repository = EpisodeRepository(
                catalog,
                readers={"egoscale": reader},
            )
            try:
                episode_id = f"egoscale:{episode_dir.name}"
                episode = repository.get_episode(episode_id)
                video = episode.read_video([2, 0])
                self.assertEqual(tuple(video.shape), (2, 4, 10, 3))
                self.assertEqual(int(video[0, 0, 0, 0]), 12)
                self.assertEqual(int(video[0, 0, 9, 0]), 102)

                state = episode.read_state([0, 1, 2])
                self.assertEqual(tuple(state.shape), (3, 24))
                self.assertEqual(float(state[0, 7]), 1.0)
                self.assertEqual(float(state[0, 15]), 1.0)
                self.assertEqual(float(state[0, 23]), 0.0)
                self.assertTrue(torch.equal(state[0, 16:23], torch.zeros(7)))
                self.assertEqual(float(state[2, 7]), 0.0)
                self.assertTrue(torch.equal(state[2, :7], torch.zeros(7)))

                source = episode.read_source_modalities(
                    {
                        "left_hand/tracks": [0, 3],
                        "right_hand/eepose": [0, 1],
                    }
                )
                self.assertEqual(
                    tuple(source["left_hand/tracks"].shape),
                    (2, 21, 3),
                )
                self.assertEqual(float(source["left_hand/tracks"][0, 0, 0]), 0.0)
                self.assertEqual(float(source["left_hand/tracks"][1, 0, 0]), 3.0)
                self.assertTrue(
                    torch.equal(source["right_hand/eepose"][0], torch.zeros(7))
                )

                rows = [
                    {
                        "format": "motus_sample_pool_v2",
                        "pool_id": "egoscale_fixture_pool",
                        "catalog_id": "egoscale_fixture_v1",
                        "samples": 2,
                    },
                    {
                        "episode_id": episode_id,
                        "video_frame_indices": [0, 1, 2],
                        "state_indices": [0],
                        "action_indices": [0, 1],
                        "text_indices": [1],
                    },
                    {
                        "episode_id": episode_id,
                        "video_frame_indices": [1, 2, 3],
                        "state_indices": [1],
                        "action_indices": [1, 2],
                        "text_indices": [1],
                    },
                ]
                pool = root / "egoscale.sample_pool.jsonl"
                pool.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                dataset = SamplePoolDataset(
                    repository,
                    pool,
                    video_output_format="rgb_float32_chw",
                    video_size=(4, 5),
                )
                adapter = build_indexed_batch_adapter(
                    "motus_native_multimodal_batch_v1",
                    num_video_frames=2,
                    action_video_fps_ratio=1,
                )
                loader = DataLoader(
                    dataset,
                    batch_sampler=dataset.create_batch_sampler(
                        batch_size=2,
                        drop_last=True,
                        shuffle=False,
                    ),
                    collate_fn=IndexedBatchCollator(adapter),
                    num_workers=0,
                )
                batch = next(iter(loader))
                validate_motus_native_multimodal_batch_v1(
                    batch,
                    expected_batch_size=2,
                    num_video_frames=2,
                    video_action_freq_ratio=1,
                    expected_state_dimension=24,
                    expected_action_dimension=24,
                )
                # text_indices=[1]（task_description_en）经注册的 egoscale
                # composer 拼成带标签文本（而非裸 join）。
                self.assertEqual(
                    batch["t5_input_messages"],
                    ["Task: Perform the fixture task."] * 2,
                )
            finally:
                repository.close()

    def test_catalog_accepts_parquet_column_reordering_by_name(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode = self._fixture(root)
            parquet = episode / "data/chunk-000/episode_000000.parquet"
            table = pq.read_table(parquet)
            reordered = [
                *EGOSCALE_PARQUET_COLUMNS[:4],
                "left_hand/timestamp",
                "right_hand/timestamp",
                "head/pose",
                "head/valid",
                "head/timestamp",
                "left_hand/tracks",
                "right_hand/tracks",
                *EGOSCALE_PARQUET_COLUMNS[11:],
            ]
            self.assertEqual(set(reordered), set(EGOSCALE_PARQUET_COLUMNS))
            self.assertNotEqual(tuple(reordered), EGOSCALE_PARQUET_COLUMNS)
            pq.write_table(table.select(reordered), parquet, row_group_size=2)

            catalog = root / "egoscale_reordered.episode_catalog.jsonl"
            metadata = build_egoscale_episode_catalog(
                root,
                catalog,
                catalog_id="egoscale_second1700h_fixture_v1",
                data_version="fixture",
                child="egoscale_second1700h_batch01",
                shard_count=1,
                shard_index=0,
            )
            record = json.loads(catalog.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child"], "egoscale_second1700h_batch01")
            self.assertEqual(record["meta"]["raw_parquet_columns"], reordered)
            self.assertFalse(
                record["meta"]["quality"][
                    "parquet_column_order_matches_canonical"
                ]
            )
            self.assertEqual(
                metadata["meta"]["build"]["warning_counts"],
                {"parquet_column_order_changed": 1},
            )
            self.assertEqual(
                sum(
                    metadata["meta"]["build"][
                        "parquet_column_order_fingerprint_counts"
                    ].values()
                ),
                1,
            )

    def test_catalog_rejects_missing_parquet_column(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode = self._fixture(root)
            parquet = episode / "data/chunk-000/episode_000000.parquet"
            table = pq.read_table(parquet)
            missing_name = EGOSCALE_PARQUET_COLUMNS[-1]
            pq.write_table(
                table.select(
                    [name for name in table.column_names if name != missing_name]
                ),
                parquet,
                row_group_size=2,
            )
            with self.assertRaisesRegex(
                ValueError,
                rf"missing=\['{missing_name}'\] unexpected=\[\]",
            ):
                build_egoscale_episode_catalog(
                    root,
                    root / "egoscale_missing.episode_catalog.jsonl",
                    catalog_id="egoscale_missing_fixture_v1",
                    data_version="fixture",
                    shard_count=1,
                    shard_index=0,
                )

    @staticmethod
    def _fake_catalog_record(episode: Path) -> tuple[dict, dict]:
        return (
            {"episode_id": f"egoscale:{episode.name}"},
            {
                "frames": 10,
                "texts": 2,
                "fine_actions": 1,
                "annotation_coverage": 1.0,
                "null_value_columns": [],
                "parquet_column_order_matches_canonical": True,
                "parquet_column_order_fingerprint": "fixture_order",
                "annotation_uses_source_group_id": False,
                "task_usable": True,
            },
        )

    def test_catalog_resume_truncates_uncommitted_tail_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            names = [
                f"session_resume{i:05d}_00000_task_0001_lerobot_v3"
                for i in range(5)
            ]
            for name in names:
                (root / name).mkdir()
            catalog = root / "egoscale_resume.episode_catalog.jsonl"
            calls = 0

            def interrupt_after_uncommitted_record(
                episode: Path, *, published_root: Path
            ):
                del published_root
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("simulated interruption")
                return self._fake_catalog_record(episode)

            with mock.patch(
                "motus_data.catalog.egoscale._episode_record",
                side_effect=interrupt_after_uncommitted_record,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    build_egoscale_episode_catalog(
                        root,
                        catalog,
                        catalog_id="egoscale_resume_fixture_v1",
                        data_version="fixture",
                        shard_count=1,
                        shard_index=0,
                        resume=True,
                        checkpoint_interval_episodes=2,
                    )

            partial = root / f".{catalog.name}.partial"
            checkpoint_path = root / f".{catalog.name}.checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["completed_episodes"], 2)
            self.assertEqual(len(partial.read_text(encoding="utf-8").splitlines()), 3)

            progress = []
            with mock.patch(
                "motus_data.catalog.egoscale._episode_record",
                side_effect=lambda episode, *, published_root: (
                    self._fake_catalog_record(episode)
                ),
            ):
                metadata = build_egoscale_episode_catalog(
                    root,
                    catalog,
                    catalog_id="egoscale_resume_fixture_v1",
                    data_version="fixture",
                    shard_count=1,
                    shard_index=0,
                    resume=True,
                    checkpoint_interval_episodes=2,
                    progress_callback=progress.append,
                )

            rows = [
                json.loads(line)
                for line in catalog.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["episode_id"] for row in rows],
                [f"egoscale:{name}" for name in names],
            )
            self.assertEqual(metadata["episodes"], 5)
            self.assertEqual(metadata["meta"]["build"]["frames"], 50)
            self.assertTrue(metadata["meta"]["build"]["resumed"])
            self.assertEqual(
                metadata["meta"]["build"]["resumed_from_episode"], 2
            )
            self.assertEqual(metadata["meta"]["build"]["resume_count"], 1)
            self.assertEqual(progress[0]["processed"], 2)
            self.assertFalse(partial.exists())
            self.assertFalse(checkpoint_path.exists())

    def test_catalog_resume_rejects_changed_episode_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "session_resume00000_00000_task_0001_lerobot_v3"
            second = root / "session_resume00001_00000_task_0001_lerobot_v3"
            first.mkdir()
            catalog = root / "egoscale_resume_mismatch.episode_catalog.jsonl"

            with mock.patch(
                "motus_data.catalog.egoscale._episode_record",
                side_effect=RuntimeError("simulated interruption"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    build_egoscale_episode_catalog(
                        root,
                        catalog,
                        catalog_id="egoscale_resume_mismatch_v1",
                        data_version="fixture",
                        shard_count=1,
                        shard_index=0,
                        resume=True,
                        checkpoint_interval_episodes=1,
                    )

            with self.assertRaisesRegex(RuntimeError, "rerun with resume=True"):
                build_egoscale_episode_catalog(
                    root,
                    catalog,
                    catalog_id="egoscale_resume_mismatch_v1",
                    data_version="fixture",
                    shard_count=1,
                    shard_index=0,
                )

            second.mkdir()
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                build_egoscale_episode_catalog(
                    root,
                    catalog,
                    catalog_id="egoscale_resume_mismatch_v1",
                    data_version="fixture",
                    shard_count=1,
                    shard_index=0,
                    resume=True,
                    checkpoint_interval_episodes=1,
                )

    def test_reader_options_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_mode"):
            EgoScaleEpisodeReader(camera_mode="left")
        with self.assertRaisesRegex(ValueError, "pose_profile"):
            EgoScaleEpisodeReader(pose_profile="unsupported")
        with self.assertRaisesRegex(ValueError, "resize_mode"):
            EgoScaleEpisodeReader(resize_mode="crop")


if __name__ == "__main__":
    unittest.main()
