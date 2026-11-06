from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from motus_data.datasets.action_safety import (
    ActionMagnitudeOutlier,
    ActionStateNonFinite,
)
from motus_data.sample_pool import (
    IndexedEpisodeBatch,
    SamplePool,
    SamplePoolDataset,
    SamplePoolError,
    SupportsSamplePool,
    resolve_sample_pool,
)
from tests.test_repository import RepositoryFixture


def _header(catalog_id: str, count: int, protocol_version: str = "v1") -> dict:
    if protocol_version == "v1":
        return {
            "schema_version": "motus_sample_pool_header_v1",
            "record_type": "sample_pool_header",
            "record_schema_version": "motus_sample_selection_v1",
            "sample_pool_id": "fixture_pool_v1",
            "catalog_id": catalog_id,
            "source_dataset": "agibot_beta",
            "source_child": "agibot_beta",
            "split": "train",
            "record_count": count,
        }
    if protocol_version == "v2":
        return {
            "format": "motus_sample_pool_v2",
            "pool_id": "fixture_pool_v2",
            "catalog_id": catalog_id,
            "samples": count,
        }
    raise ValueError(f"unsupported protocol_version={protocol_version}")


def _selections(episode_id: str) -> list[dict]:
    return [
        {
            "episode_id": episode_id,
            "video_frame_indices": [0, 1, 0],
            "state_indices": [0],
            "action_indices": [1, 0],
            "text_indices": [0, 1],
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [1, 2],
            "state_indices": [2, 1],
            "action_indices": [2],
            "text_indices": [1, 0],
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [3, 4],
            "state_indices": [],
            "action_indices": [4, 3],
            "text_indices": [],
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [4, 5, 4],
            "state_indices": [5],
            "action_indices": [],
            "text_indices": [2, 2],
        },
    ]


def _write_pool(path: Path, header: dict, selections: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in [header, *selections]),
        encoding="utf-8",
    )


class SamplePoolDatasetTests(unittest.TestCase):
    def test_action_magnitude_guard_rejects_before_video_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root, outlier_timestep=1)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )

            dataset = SamplePoolDataset(repository, pool_path)
            fixture.decode_calls.clear()
            with self.assertRaises(ActionMagnitudeOutlier) as raised:
                dataset.get_grouped_samples([0, 1])

            self.assertGreater(
                raised.exception.details["observed_abs_max"],
                raised.exception.details["threshold"],
            )
            self.assertEqual(fixture.decode_calls, [])

    def test_action_magnitude_guard_can_be_explicitly_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root, outlier_timestep=1)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )

            dataset = SamplePoolDataset(
                repository,
                pool_path,
                action_magnitude_guard_threshold=None,
            )
            samples = dataset.get_grouped_samples([0, 1])
            self.assertEqual(len(samples), 2)
            self.assertEqual(fixture.decode_calls, [(0, 1, 2)])

    def test_action_magnitude_guard_configuration_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )

            with self.assertRaisesRegex(SamplePoolError, "must be positive"):
                SamplePoolDataset(
                    repository,
                    pool_path,
                    action_magnitude_guard_threshold=0,
                )

    def test_nonfinite_guard_cannot_be_disabled_with_magnitude_guard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root, nonfinite_timestep=1)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )

            dataset = SamplePoolDataset(
                repository,
                pool_path,
                action_magnitude_guard_threshold=None,
            )
            with self.assertRaises(ActionStateNonFinite):
                dataset.get_grouped_samples([0, 1])
            self.assertEqual(fixture.decode_calls, [])

    def test_split_neutral_v2_pool_binds_and_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root, protocol_version="v2")
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture_v2.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections), "v2"),
                selections,
            )
            pool = SamplePool(pool_path, repository)
            self.assertNotIn("split", pool.header)
            self.assertEqual(
                set(pool.header),
                {"format", "pool_id", "catalog_id", "samples"},
            )
            dataset = SamplePoolDataset(repository, pool)
            batch = dataset.get_grouped_batch([0, 1])
            self.assertIsNone(batch.info.split)
            self.assertEqual(batch.batch_size, 2)

    def test_pool_binding_dataloader_and_grouped_union_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )

            pool = SamplePool(pool_path, repository)
            self.assertEqual(len(pool), 4)
            self.assertEqual(
                pool.episode_groups,
                {fixture.episode_id: (0, 1, 2, 3)},
            )
            dataset = SamplePoolDataset(repository, pool)
            sampler = dataset.create_batch_sampler(
                batch_size=2,
                drop_last=True,
                shuffle=False,
                seed=17,
            )
            self.assertEqual(list(sampler), [[0, 1], [2, 3]])

            fixture.decode_calls.clear()
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=dataset.collate_fn,
                num_workers=0,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 2)
            self.assertTrue(
                all(isinstance(batch, IndexedEpisodeBatch) for batch in batches)
            )
            self.assertTrue(all(batch.batch_size == 2 for batch in batches))
            self.assertTrue(
                all(batch.episode_id == fixture.episode_id for batch in batches)
            )
            self.assertEqual(fixture.decode_calls, [(0, 1, 2), (3, 4, 5)])
            self.assertEqual(
                batches[0].samples[0].video[:, 0, 0, 0].tolist(),
                [0, 1, 0],
            )
            self.assertEqual(
                batches[1].samples[1].video[:, 0, 0, 0].tolist(),
                [4, 5, 4],
            )

    def test_distributed_sampler_shards_complete_batches_and_set_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )
            dataset = SamplePoolDataset(repository, pool_path)
            rank0 = dataset.create_batch_sampler(
                batch_size=1,
                shuffle=True,
                num_replicas=2,
                rank=0,
                seed=23,
            )
            rank1 = dataset.create_batch_sampler(
                batch_size=1,
                shuffle=True,
                num_replicas=2,
                rank=1,
                seed=23,
            )
            epoch0_rank0 = [batch[0] for batch in rank0]
            epoch0_rank1 = [batch[0] for batch in rank1]
            self.assertEqual(len(rank0), len(rank1))
            self.assertEqual(set(epoch0_rank0) & set(epoch0_rank1), set())
            self.assertEqual(
                set(epoch0_rank0) | set(epoch0_rank1),
                set(range(len(selections))),
            )

            rank0.set_epoch(1)
            rank1.set_epoch(1)
            epoch1_rank0 = [batch[0] for batch in rank0]
            epoch1_rank1 = [batch[0] for batch in rank1]
            self.assertEqual(set(epoch1_rank0) & set(epoch1_rank1), set())
            self.assertEqual(
                set(epoch1_rank0) | set(epoch1_rank1),
                set(range(len(selections))),
            )
            self.assertNotEqual(
                (epoch0_rank0, epoch0_rank1),
                (epoch1_rank0, epoch1_rank1),
            )

    def test_pool_rejects_wrong_catalog_count_and_out_of_bounds_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"

            _write_pool(
                pool_path,
                _header("wrong_catalog", len(selections)),
                selections,
            )
            with self.assertRaisesRegex(SamplePoolError, "does not match"):
                SamplePool(pool_path, repository)

            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections) + 1),
                selections,
            )
            with self.assertRaisesRegex(SamplePoolError, "record count mismatch"):
                SamplePool(pool_path, repository)

            invalid = _selections(fixture.episode_id)
            invalid[0]["video_frame_indices"] = [6]
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(invalid)),
                invalid,
            )
            with self.assertRaisesRegex(SamplePoolError, "out of bounds"):
                SamplePool(pool_path, repository)


class _ToyGeneratedPool:
    """Test-owned runtime pool: windows generated in memory, shifted per epoch."""

    def __init__(self, episode_id: str, catalog_id: str):
        self._episode_id = episode_id
        self._pool_id = f"toy_generated_{catalog_id}"
        self.epoch = 0
        self.set_epoch_calls: list[int] = []

    @property
    def pool_id(self) -> str:
        return self._pool_id

    @property
    def episode_groups(self):
        return {self._episode_id: (0, 1, 2, 3)}

    @property
    def selections(self):
        return tuple(self._selection(index) for index in range(4))

    def _selection(self, index: int) -> dict:
        start = (index + self.epoch) % 3
        return {
            "episode_id": self._episode_id,
            "video_frame_indices": [start, start + 1],
            "state_indices": [start],
            "action_indices": [start + 1],
            "text_indices": [0],
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.set_epoch_calls.append(int(epoch))

    def __len__(self) -> int:
        return 4


class CustomPoolInjectionTests(unittest.TestCase):
    def test_protocol_object_feeds_dataset_and_grouped_reads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            pool = _ToyGeneratedPool(fixture.episode_id, repository.catalog_id)

            self.assertIsInstance(pool, SupportsSamplePool)
            dataset = SamplePoolDataset(repository, pool)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(
                dataset.episode_groups,
                {fixture.episode_id: (0, 1, 2, 3)},
            )
            fixture.decode_calls.clear()
            samples = dataset.get_grouped_samples([0, 1])
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].video[:, 0, 0, 0].tolist(), [0, 1])
            self.assertEqual(fixture.decode_calls, [(0, 1, 2)])

    def test_resolve_sample_pool_accepts_object_factory_and_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            pool = _ToyGeneratedPool(fixture.episode_id, repository.catalog_id)

            self.assertIs(resolve_sample_pool(pool, repository), pool)

            calls = []

            def factory(*, repository, seed):
                calls.append((repository, seed))
                return pool

            self.assertIs(resolve_sample_pool(factory, repository, seed=7), pool)
            self.assertEqual(calls, [(repository, 7)])

            with self.assertRaisesRegex(SamplePoolError, "SupportsSamplePool"):
                resolve_sample_pool(lambda **_: object(), repository)

            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )
            loaded = resolve_sample_pool(pool_path, repository)
            self.assertIsInstance(loaded, SamplePool)

    def test_set_epoch_forwarded_sampler_to_dataset_to_pool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            pool = _ToyGeneratedPool(fixture.episode_id, repository.catalog_id)
            dataset = SamplePoolDataset(repository, pool)
            sampler = dataset.create_batch_sampler(
                batch_size=2,
                shuffle=False,
                seed=3,
            )

            sampler.set_epoch(5)
            self.assertEqual(pool.set_epoch_calls, [5])
            self.assertEqual(pool.epoch, 5)
            self.assertEqual(
                pool.selections[0]["video_frame_indices"],
                [2, 3],
            )

            dataset.set_epoch(7)
            self.assertEqual(pool.set_epoch_calls, [5, 7])
            self.assertEqual(
                pool.selections[0]["video_frame_indices"],
                [1, 2],
            )

    def test_static_pool_epoch_forwarding_is_a_safe_noop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root)
            repository = fixture.repository()
            selections = _selections(fixture.episode_id)
            pool_path = root / "fixture.sample_pool.jsonl"
            _write_pool(
                pool_path,
                _header(repository.catalog_id, len(selections)),
                selections,
            )
            dataset = SamplePoolDataset(repository, pool_path)
            sampler = dataset.create_batch_sampler(
                batch_size=2,
                shuffle=False,
                seed=3,
            )
            sampler.set_epoch(2)
            self.assertEqual(sampler.epoch, 2)


if __name__ == "__main__":
    unittest.main()
