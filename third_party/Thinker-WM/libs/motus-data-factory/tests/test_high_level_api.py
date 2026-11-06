from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from motus_data import (
    DEFAULT_EPISODE_CATALOG_ROOT,
    EpisodeDatasetRelease,
    IndexedEpisodeBatch,
    MotusModelBatchV1Adapter,
    episode_dataset_releases,
    get_episode_dataset_release,
    open_dataset,
    open_training_data,
)
from motus_data.catalog.sqlite_index import build_catalog_offset_index
from motus_data.repository import AgibotBetaEpisodeReader
from tests.test_repository import RepositoryFixture


def _write_pool(path: Path, catalog_id: str, episode_id: str) -> None:
    selections = [
        {
            "episode_id": episode_id,
            "video_frame_indices": [0, 1, 2],
            "state_indices": [0],
            "action_indices": [1, 2],
            "text_indices": [0],
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [1, 2, 3],
            "state_indices": [1],
            "action_indices": [2, 3],
            "text_indices": [0, 1],
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [2, 3, 4],
            "state_indices": [2],
            "action_indices": [3, 4],
            "text_indices": [0, 1],
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [3, 4, 5],
            "state_indices": [3],
            "action_indices": [4, 5],
            "text_indices": [0, 2],
        },
    ]
    rows = [
        {
            "format": "motus_sample_pool_v2",
            "pool_id": "high_level_api_fixture",
            "catalog_id": catalog_id,
            "samples": len(selections),
        },
        *selections,
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_release(root: Path):
    fixture = RepositoryFixture(root, protocol_version="v2")
    index_result = build_catalog_offset_index(fixture.catalog_path)
    release = EpisodeDatasetRelease(
        dataset="test_agibot_beta",
        release="fixture_v2",
        catalog_path=str(fixture.catalog_path),
        index_path=str(index_result["index_path"]),
        default_video_size=(2, 3),
        default_queue_size=1,
        action_magnitude_guard_sources=("agibot_beta",),
    )
    reader = AgibotBetaEpisodeReader(
        video_decoder=fixture.video_decoder
    )
    return fixture, release, reader


class HighLevelApiTests(unittest.TestCase):
    def _fixture_release(self, root: Path):
        return _fixture_release(root)

    def test_builtin_release_resolves_from_default_and_explicit_root(self):
        release = get_episode_dataset_release("agibot_beta")
        self.assertEqual(release.release, "20260723_concise_v2")
        self.assertIn(
            "agibot_beta:20260723_concise_v2",
            episode_dataset_releases("agibot_beta"),
        )
        catalog, index = release.resolve_paths()
        self.assertEqual(
            catalog,
            DEFAULT_EPISODE_CATALOG_ROOT
            / "_20260723/agibot_beta/agibot_beta_full_20260723_concise_v2."
            "episode_catalog.jsonl",
        )
        self.assertEqual(
            index,
            DEFAULT_EPISODE_CATALOG_ROOT
            / "_20260723/agibot_beta/agibot_beta_full_20260723_concise_v2."
            "episode_catalog.index.sqlite",
        )
        catalog, index = release.resolve_paths("/tmp/catalog-root")
        self.assertEqual(
            catalog,
            Path("/tmp/catalog-root/_20260723/agibot_beta/"
                 "agibot_beta_full_20260723_concise_v2."
                 "episode_catalog.jsonl"),
        )
        self.assertEqual(
            index,
            Path("/tmp/catalog-root/_20260723/agibot_beta/"
                 "agibot_beta_full_20260723_concise_v2."
                 "episode_catalog.index.sqlite"),
        )

    def test_builtin_agibot_beta_profile_releases_resolve_reader_options(self):
        expected = {
            "agibot_beta_parallel_gripper": "parallel_gripper",
            "agibot_beta_dexterous_hand": "dexterous_hand",
        }
        for dataset, profile in expected.items():
            with self.subTest(dataset=dataset):
                release = get_episode_dataset_release(dataset)
                self.assertEqual(release.release, "20260805_full_v1")
                self.assertEqual(
                    release.action_magnitude_guard_threshold, 20.0
                )
                options = release.reader_options["agibot_beta"]
                self.assertIn(profile, options["normalization_artifacts"])
        dexterous = get_episode_dataset_release(
            "agibot_beta_dexterous_hand"
        )
        self.assertEqual(
            dexterous.reader_options["agibot_beta"]["excluded_episode_ids"],
            ["agibot_beta:554:913085"],
        )

    def test_builtin_robotmind_agilex_release_resolves(self):
        release = get_episode_dataset_release("robotmind_agilex")
        self.assertEqual(release.release, "20260728_full_v1")
        self.assertIn(
            "robotmind_agilex:20260728_full_v1",
            episode_dataset_releases("robotmind_agilex"),
        )
        catalog, index = release.resolve_paths("/tmp/catalog-root")
        self.assertEqual(
            catalog,
            Path(
                "/tmp/catalog-root/_20260728/robotmind_agilex/"
                "robotmind_agilex.20260728_full_v1.episode_catalog.jsonl"
            ),
        )
        self.assertEqual(
            index,
            Path(
                "/tmp/catalog-root/_20260728/robotmind_agilex/"
                "robotmind_agilex.20260728_full_v1."
                "episode_catalog.index.sqlite"
            ),
        )
        self.assertEqual(
            release.action_magnitude_guard_sources,
            ("robotmind",),
        )

    def test_builtin_egoscale_first500h_shard_release_resolves(self):
        dataset = "egoscale_first500h_shard00"
        release = get_episode_dataset_release(dataset)
        self.assertEqual(release.release, "20260816_shards_v1")
        self.assertIn(
            f"{dataset}:20260816_shards_v1",
            episode_dataset_releases(dataset),
        )
        catalog, index = release.resolve_paths("/tmp/catalog-root")
        expected = Path(
            "/tmp/catalog-root/_20260816/egoscale_first500h_shards_v1/"
            "egoscale_first500h.shard00_of10.20260816_shards_v1."
            "episode_catalog"
        )
        self.assertEqual(catalog, Path(str(expected) + ".jsonl"))
        self.assertEqual(index, Path(str(expected) + ".index.sqlite"))
        self.assertEqual(release.default_video_size, (480, 640))
        self.assertEqual(release.default_num_workers, 0)
        self.assertIsNone(release.action_magnitude_guard_threshold)
        self.assertEqual(
            release.reader_options["egoscale"],
            {
                "camera_mode": "base_0",
                "pose_profile": "world_pose_with_validity",
                "resize_mode": "pad",
            },
        )

        remaining_dataset = "egoscale_first500h_remaining90pct"
        remaining = get_episode_dataset_release(remaining_dataset)
        self.assertEqual(remaining.release, "20260816_shards_v1")
        remaining_catalog, remaining_index = remaining.resolve_paths(
            "/tmp/catalog-root"
        )
        remaining_base = Path(
            "/tmp/catalog-root/_20260816/egoscale_first500h_shards_v1/"
            "egoscale_first500h.remaining90pct.20260816_shards_v1."
            "episode_catalog"
        )
        self.assertEqual(
            remaining_catalog,
            Path(str(remaining_base) + ".jsonl"),
        )
        self.assertEqual(
            remaining_index,
            Path(str(remaining_base) + ".index.sqlite"),
        )
        self.assertEqual(
            remaining.reader_options,
            release.reader_options,
        )

    def test_builtin_egoscale_second1700h_batch01_release_resolves(self):
        dataset = "egoscale_second1700h_batch01"
        release = get_episode_dataset_release(dataset)
        self.assertEqual(release.release, "20260820_full_v1")
        self.assertIn(
            f"{dataset}:20260820_full_v1",
            episode_dataset_releases(dataset),
        )
        catalog, index = release.resolve_paths("/tmp/catalog-root")
        base = Path(
            "/tmp/catalog-root/_20260820/"
            "egoscale_second1700h_batch01_full_v1/"
            "egoscale_second1700h_batch01.full.20260820_full_v1."
            "episode_catalog"
        )
        self.assertEqual(catalog, Path(str(base) + ".jsonl"))
        self.assertEqual(index, Path(str(base) + ".index.sqlite"))
        self.assertEqual(release.default_video_size, (480, 640))
        self.assertEqual(release.default_num_workers, 0)
        self.assertIsNone(release.action_magnitude_guard_threshold)
        self.assertEqual(
            release.reader_options["egoscale"],
            {
                "camera_mode": "base_0",
                "pose_profile": "world_pose_with_validity",
                "resize_mode": "pad",
            },
        )

    def test_builtin_egoscale_second1700h_batch02_release_resolves(self):
        dataset = "egoscale_second1700h_batch02"
        release = get_episode_dataset_release(dataset)
        self.assertEqual(release.release, "20260821_full_v1")
        self.assertIn(
            f"{dataset}:20260821_full_v1",
            episode_dataset_releases(dataset),
        )
        catalog, index = release.resolve_paths("/tmp/catalog-root")
        base = Path(
            "/tmp/catalog-root/_20260821/"
            "egoscale_second1700h_batch02_full_v1/"
            "egoscale_second1700h_batch02.full.20260821_full_v1."
            "episode_catalog"
        )
        self.assertEqual(catalog, Path(str(base) + ".jsonl"))
        self.assertEqual(index, Path(str(base) + ".index.sqlite"))
        self.assertEqual(release.default_video_size, (480, 640))
        self.assertEqual(release.default_num_workers, 0)
        self.assertIsNone(release.action_magnitude_guard_threshold)
        self.assertEqual(
            release.reader_options["egoscale"],
            {
                "camera_mode": "base_0",
                "pose_profile": "world_pose_with_validity",
                "resize_mode": "pad",
            },
        )

    def test_builtin_openvid_release_resolves(self):
        release = get_episode_dataset_release("openvid")
        self.assertEqual(release.release, "20260730_full_v1")
        self.assertIn(
            "openvid:20260730_full_v1",
            episode_dataset_releases("openvid"),
        )
        catalog, index = release.resolve_paths("/tmp/catalog-root")
        self.assertEqual(
            catalog,
            Path(
                "/tmp/catalog-root/_20260730/openvid/"
                "openvid_1m_full_20260730.episode_catalog.jsonl"
            ),
        )
        self.assertEqual(
            index,
            Path(
                "/tmp/catalog-root/_20260730/openvid/"
                "openvid_1m_full_20260730.episode_catalog.index.sqlite"
            ),
        )
        self.assertIsNone(release.action_magnitude_guard_sources)

    def test_builtin_libero_releases_resolve(self):
        children = (
            "libero_10",
            "libero_goal",
            "libero_object",
            "libero_spatial",
        )
        expected_root = Path(
            "/tmp/catalog-root/_20260810/libero_parquet_v1"
        )
        for child in children:
            with self.subTest(child=child):
                release = get_episode_dataset_release(child)
                self.assertEqual(release.release, "20260810_full_v1")
                self.assertIn(
                    f"{child}:20260810_full_v1",
                    episode_dataset_releases(child),
                )
                catalog, index = release.resolve_paths("/tmp/catalog-root")
                stem = f"{child}.libero_parquet_v1.episode_catalog"
                self.assertEqual(catalog, expected_root / f"{stem}.jsonl")
                self.assertEqual(
                    index,
                    expected_root / f"{stem}.index.sqlite",
                )
                self.assertEqual(release.default_video_size, (480, 640))
                self.assertEqual(release.default_num_workers, 0)
                self.assertEqual(
                    release.action_magnitude_guard_sources,
                    ("libero",),
                )
                self.assertEqual(
                    release.reader_options,
                    {
                        "libero": {
                            "camera_mode": "main",
                            "resize_mode": "pad",
                            "normalization_mode": "zscore",
                            "statistics_path": (
                                "/data/chen.liao/Libero_parquet/"
                                "meta/stats.json"
                            ),
                        }
                    },
                )

    def test_builtin_egoverse_first6_releases_resolve(self):
        profiles = (
            "aria_bimanual",
            "aria_left_arm",
            "aria_right_arm",
            "eva_bimanual",
            "scale_bimanual",
            "mecka_freeform_bimanual",
        )
        for profile in profiles:
            dataset = f"egoverse_{profile}"
            with self.subTest(dataset=dataset):
                release = get_episode_dataset_release(dataset)
                self.assertEqual(release.release, "20260730_first6_v1")
                self.assertIn(
                    f"{dataset}:20260730_first6_v1",
                    episode_dataset_releases(dataset),
                )
                catalog, index = release.resolve_paths("/tmp/catalog-root")
                stem = (
                    f"{dataset}.20260730_first6_v1.episode_catalog"
                )
                expected_root = Path(
                    "/tmp/catalog-root/_20260730/"
                    "egoverse_first6_v1"
                )
                self.assertEqual(catalog, expected_root / f"{stem}.jsonl")
                self.assertEqual(
                    index,
                    expected_root / f"{stem}.index.sqlite",
                )
                self.assertEqual(release.default_video_size, (480, 640))
                self.assertEqual(release.default_num_workers, 0)
                self.assertIsNone(
                    release.action_magnitude_guard_threshold
                )

    def test_open_dataset_hides_release_and_index_construction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture, release, reader = self._fixture_release(Path(temp_dir))
            repository = open_dataset(
                "test_agibot_beta",
                release=release,
                readers={"agibot_beta": reader},
            )
            try:
                self.assertTrue(repository.indexed)
                self.assertEqual(len(repository), 1)
                episode = repository.get_episode(fixture.episode_id)
                self.assertEqual(episode.video.count, fixture.timesteps)
                self.assertEqual(
                    tuple(episode.read_video([0, 2]).shape),
                    (2, 4, 5, 3),
                )
            finally:
                repository.close()

    def test_open_training_data_queue_returns_model_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture, release, reader = self._fixture_release(root)
            pool_path = root / "train.sample_pool.jsonl"
            _write_pool(pool_path, "fixture_catalog_v2", fixture.episode_id)
            adapter = MotusModelBatchV1Adapter(
                num_video_frames=2,
                action_video_fps_ratio=1,
                action_loss_weight=0.0,
            )
            pipeline = open_training_data(
                "test_agibot_beta",
                release=release,
                sample_pool=pool_path,
                batch_size=2,
                output_adapter=adapter,
                readers={"agibot_beta": reader},
                queue_size=1,
                shuffle=False,
            )
            with pipeline:
                batch, stats = pipeline.next()
                self.assertEqual(
                    tuple(batch["first_frame"].shape), (2, 3, 2, 3)
                )
                self.assertEqual(
                    tuple(batch["video_frames"].shape), (2, 2, 3, 2, 3)
                )
                self.assertEqual(
                    tuple(batch["initial_state"].shape), (2, 26)
                )
                self.assertEqual(
                    tuple(batch["action_sequence"].shape), (2, 2, 26)
                )
                self.assertEqual(
                    batch["frame_indices"],
                    [[0, 1, 2], [1, 2, 3]],
                )
                self.assertTrue(stats["ready_queue_enabled"])
                self.assertEqual(pipeline.rank, 0)
                self.assertEqual(pipeline.world_size, 1)
            self.assertTrue(pipeline._closed)

    def test_open_training_data_can_return_raw_indexed_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture, release, reader = self._fixture_release(root)
            pool_path = root / "raw.sample_pool.jsonl"
            _write_pool(pool_path, "fixture_catalog_v2", fixture.episode_id)
            pipeline = open_training_data(
                "test_agibot_beta",
                release=release,
                sample_pool=pool_path,
                batch_size=2,
                output_adapter=None,
                queue_enabled=False,
                readers={"agibot_beta": reader},
                shuffle=False,
            )
            try:
                batch, stats = pipeline.next()
                self.assertIsInstance(batch, IndexedEpisodeBatch)
                self.assertEqual(batch.batch_size, 2)
                self.assertFalse(stats["ready_queue_enabled"])
                self.assertEqual(stats["data_epoch"], 0)
            finally:
                pipeline.close()


class _FixtureGeneratedPool:
    """Test-owned in-memory pool (no file) satisfying SupportsSamplePool."""

    def __init__(self, episode_id: str):
        self._episode_id = episode_id

    @property
    def pool_id(self) -> str:
        return "fixture_generated_pool"

    @property
    def episode_groups(self):
        return {self._episode_id: (0, 1, 2, 3)}

    @property
    def selections(self):
        return tuple(
            {
                "episode_id": self._episode_id,
                "video_frame_indices": [index, index + 1, index + 2],
                "state_indices": [index],
                "action_indices": [index + 1, index + 2],
                "text_indices": [0],
            }
            for index in range(4)
        )

    def __len__(self) -> int:
        return 4


class CustomPoolInjectionApiTests(unittest.TestCase):
    def test_open_training_data_accepts_pool_object_and_factory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture, release, reader = _fixture_release(root)
            adapter = MotusModelBatchV1Adapter(
                num_video_frames=2,
                action_video_fps_ratio=1,
                action_loss_weight=0.0,
            )

            pipeline = open_training_data(
                "test_agibot_beta",
                release=release,
                sample_pool=_FixtureGeneratedPool(fixture.episode_id),
                batch_size=2,
                output_adapter=adapter,
                queue_enabled=False,
                readers={"agibot_beta": reader},
                shuffle=False,
            )
            with pipeline:
                batch, _ = pipeline.next()
                self.assertEqual(batch["frame_indices"], [[0, 1, 2], [1, 2, 3]])
                self.assertIsInstance(
                    pipeline.sample_pool,
                    _FixtureGeneratedPool,
                )

            calls = []

            def factory(*, repository, seed):
                calls.append((repository.catalog_id, seed))
                return _FixtureGeneratedPool(fixture.episode_id)

            pipeline = open_training_data(
                "test_agibot_beta",
                release=release,
                sample_pool=factory,
                batch_size=2,
                output_adapter=adapter,
                queue_enabled=False,
                readers={"agibot_beta": reader},
                shuffle=False,
                seed=11,
            )
            with pipeline:
                batch, _ = pipeline.next()
                self.assertEqual(batch["frame_indices"], [[0, 1, 2], [1, 2, 3]])
            self.assertEqual(calls, [("fixture_catalog_v2", 11)])


if __name__ == "__main__":
    unittest.main()
