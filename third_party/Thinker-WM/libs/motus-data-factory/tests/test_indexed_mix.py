from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path

from motus_data import (
    CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP,
    EpisodeDatasetRelease,
    IndexedTrainingSource,
    MotusModelBatchV1Adapter,
    WeightedIndexedBatchStream,
    open_mixed_training_data,
)
from motus_data.catalog.sqlite_index import build_catalog_offset_index
from motus_data.repository import AgibotBetaEpisodeReader
from tests.test_repository import RepositoryFixture


def _write_pool(
    path: Path,
    *,
    pool_id: str,
    catalog_id: str,
    episode_id: str,
) -> None:
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
            "pool_id": pool_id,
            "catalog_id": catalog_id,
            "samples": len(selections),
        },
        *selections,
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_fixture(root: Path, name: str):
    source_root = root / name
    source_root.mkdir()
    fixture = RepositoryFixture(source_root, protocol_version="v2")
    index = build_catalog_offset_index(fixture.catalog_path)
    release = EpisodeDatasetRelease(
        dataset=name,
        release="fixture_v2",
        catalog_path=str(fixture.catalog_path),
        index_path=str(index["index_path"]),
        default_video_size=(2, 3),
        action_magnitude_guard_sources=("agibot_beta",),
    )
    pool_path = source_root / f"{name}.sample_pool.jsonl"
    _write_pool(
        pool_path,
        pool_id=f"{name}_pool",
        catalog_id="fixture_catalog_v2",
        episode_id=fixture.episode_id,
    )
    reader = AgibotBetaEpisodeReader(video_decoder=fixture.video_decoder)
    return IndexedTrainingSource(
        name=name,
        dataset=name,
        release=release,
        sample_pool=pool_path,
        readers={"agibot_beta": reader},
    )


class _NoOpChild:
    def next(self):
        return None, {}

    def set_epoch(self, _epoch: int) -> None:
        return None


def _schedule_source(name: str, weight: float) -> IndexedTrainingSource:
    return IndexedTrainingSource(
        name=name,
        dataset=name,
        sample_pool=lambda **_kwargs: None,
        weight=weight,
    )


def _rank_schedules(
    weights,
    *,
    world_size: int,
    batches_per_epoch: int,
    seed: int = 17,
    epoch: int = 0,
    shuffle: bool = True,
):
    sources = [
        _schedule_source(f"source_{index}", weight)
        for index, weight in enumerate(weights)
    ]
    schedules = []
    for rank in range(world_size):
        stream = WeightedIndexedBatchStream(
            [_NoOpChild() for _ in sources],
            sources,
            batches_per_epoch=batches_per_epoch,
            seed=seed,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
            cross_rank_source_policy=(
                CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP
            ),
        )
        stream.set_epoch(epoch)
        schedules.append(stream.source_schedule())
    return schedules


class IndexedMixTests(unittest.TestCase):
    def test_weighted_epoch_quotas_and_queue_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = _source_fixture(root, "source_a")
            source_a = IndexedTrainingSource(
                **{
                    **source_a.__dict__,
                    "weight": 1.0,
                }
            )
            source_b = _source_fixture(root, "source_b")
            source_b = IndexedTrainingSource(
                **{
                    **source_b.__dict__,
                    "weight": 3.0,
                }
            )
            adapter = MotusModelBatchV1Adapter(
                num_video_frames=2,
                action_video_fps_ratio=1,
                action_loss_weight=0.0,
            )
            pipeline = open_mixed_training_data(
                [source_a, source_b],
                batch_size=1,
                batches_per_epoch=4,
                output_adapter=adapter,
                video_size=(2, 3),
                queue_enabled=True,
                queue_size=2,
                queue_clone_tensors=True,
                shuffle=True,
                seed=17,
            )
            try:
                self.assertEqual(
                    pipeline.stream.source_quotas(),
                    (1, 3),
                )
                observed = []
                with pipeline:
                    for _ in range(4):
                        batch, stats = pipeline.next()
                        observed.append(stats["mix_source_name"])
                        self.assertEqual(
                            tuple(batch["first_frame"].shape),
                            (1, 3, 2, 3),
                        )
                        self.assertTrue(stats["ready_queue_enabled"])
                self.assertEqual(
                    dict(sorted(pipeline._consumed_by_source.items())),
                    {"source_a": 1, "source_b": 3},
                )
                self.assertCountEqual(
                    observed,
                    ["source_a", "source_b", "source_b", "source_b"],
                )
            finally:
                pipeline.close()
            self.assertTrue(pipeline._closed)
            self.assertTrue(all(child._closed for child in pipeline.children))

    def test_queued_batch_transform_runs_in_producer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _source_fixture(Path(temp_dir), "source_a")
            producer_threads = []

            def attach_marker(batch):
                producer_threads.append(threading.current_thread().name)
                transformed = dict(batch)
                transformed["producer_marker"] = "attached"
                return transformed

            pipeline = open_mixed_training_data(
                [source],
                batch_size=1,
                batches_per_epoch=1,
                output_adapter=MotusModelBatchV1Adapter(
                    num_video_frames=2,
                    action_video_fps_ratio=1,
                    action_loss_weight=0.0,
                ),
                video_size=(2, 3),
                queue_enabled=True,
                queue_size=1,
                queue_clone_tensors=True,
                batch_transform=attach_marker,
                shuffle=False,
            )
            try:
                batch, stats = pipeline.next()
                self.assertEqual(batch["producer_marker"], "attached")
                self.assertTrue(producer_threads)
                self.assertTrue(
                    producer_threads[0].startswith("ReadyBatchQueueProducer-")
                )
                self.assertGreaterEqual(stats["mix_batch_transform_seconds"], 0.0)
                self.assertIn(
                    "ready_queue_producer_transform_seconds", stats
                )
                self.assertTrue(
                    pipeline.snapshot()["mix_batch_transform_enabled"]
                )
            finally:
                pipeline.close()

    def test_all_ranks_share_source_schedule_but_not_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [
                _source_fixture(root, "source_a"),
                _source_fixture(root, "source_b"),
            ]
            rank_pipelines = [
                open_mixed_training_data(
                    sources,
                    batch_size=1,
                    batches_per_epoch=4,
                    output_adapter=None,
                    queue_enabled=False,
                    shuffle=False,
                    rank=rank,
                    world_size=2,
                )
                for rank in range(2)
            ]
            try:
                schedules = [[], []]
                selected_frames = [[], []]
                for _ in range(4):
                    for rank, pipeline in enumerate(rank_pipelines):
                        batch, stats = pipeline.next()
                        schedules[rank].append(stats["mix_source_name"])
                        selected_frames[rank].append(
                            batch.samples[0].video_frame_indices
                        )
                self.assertEqual(schedules[0], schedules[1])
                self.assertEqual(
                    schedules[0],
                    ["source_a", "source_a", "source_b", "source_b"],
                )
                for left, right in zip(
                    selected_frames[0], selected_frames[1]
                ):
                    self.assertNotEqual(left, right)
            finally:
                for pipeline in rank_pipelines:
                    pipeline.close()

    def test_weighted_per_step_distributes_global_rank_slots(self):
        weights = (0.6, 0.3, 0.1)
        world_size = 4
        batches_per_epoch = 100
        schedules = _rank_schedules(
            weights,
            world_size=world_size,
            batches_per_epoch=batches_per_epoch,
        )

        normalized = tuple(weight / sum(weights) for weight in weights)
        observed = Counter()
        for step in range(batches_per_epoch):
            row = Counter(schedule[step] for schedule in schedules)
            self.assertEqual(sum(row.values()), world_size)
            for source_index, weight in enumerate(normalized):
                exact = weight * world_size
                self.assertIn(
                    row[source_index],
                    {math.floor(exact), math.ceil(exact)},
                )
            observed.update(row)

        total_slots = batches_per_epoch * world_size
        for source_index, weight in enumerate(normalized):
            self.assertLessEqual(
                abs(observed[source_index] - weight * total_slots),
                1.0,
            )
        self.assertTrue(all(len(set(schedule)) > 1 for schedule in schedules))
        self.assertGreater(len(set(schedules)), 1)

    def test_weighted_per_step_current_14_source_weights_are_reproducible(self):
        weights = (
            0.628653,
            0.0292421,
            *([0.0203509] * 6),
            0.0199133,
            0.1800862,
            *([0.005] * 4),
        )
        world_size = 24
        batches_per_epoch = 1000
        schedules = _rank_schedules(
            weights,
            world_size=world_size,
            batches_per_epoch=batches_per_epoch,
            seed=20260818,
        )
        repeated = _rank_schedules(
            weights,
            world_size=world_size,
            batches_per_epoch=batches_per_epoch,
            seed=20260818,
        )
        self.assertEqual(schedules, repeated)

        normalized = tuple(weight / sum(weights) for weight in weights)
        observed = Counter(
            source_index
            for schedule in schedules
            for source_index in schedule
        )
        total_slots = world_size * batches_per_epoch
        for source_index, weight in enumerate(normalized):
            self.assertLessEqual(
                abs(observed[source_index] - weight * total_slots),
                1.0,
            )
        self.assertEqual(len(set(schedules)), world_size)

        epoch_one = _rank_schedules(
            weights,
            world_size=world_size,
            batches_per_epoch=batches_per_epoch,
            seed=20260818,
            epoch=1,
        )
        self.assertNotEqual(schedules, epoch_one)
        epoch_one_observed = Counter(
            value for schedule in epoch_one for value in schedule
        )
        for source_index, weight in enumerate(normalized):
            self.assertLessEqual(
                abs(epoch_one_observed[source_index] - weight * total_slots),
                1.0,
            )

    def test_weighted_per_step_pipeline_preserves_distributed_sample_shards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = _source_fixture(root, "source_a")
            source_a = IndexedTrainingSource(
                **{**source_a.__dict__, "weight": 3.0}
            )
            source_b = _source_fixture(root, "source_b")
            source_b = IndexedTrainingSource(
                **{**source_b.__dict__, "weight": 1.0}
            )
            pipelines = [
                open_mixed_training_data(
                    [source_a, source_b],
                    batch_size=1,
                    batches_per_epoch=4,
                    output_adapter=None,
                    queue_enabled=True,
                    queue_size=2,
                    shuffle=True,
                    seed=23,
                    rank=rank,
                    world_size=2,
                    cross_rank_source_policy=(
                        CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP
                    ),
                )
                for rank in range(2)
            ]
            try:
                observed = Counter()
                for _step in range(4):
                    rank_batches = []
                    for pipeline in pipelines:
                        batch, stats = pipeline.next()
                        rank_batches.append((batch, stats))
                        observed[stats["mix_source_name"]] += 1
                        self.assertEqual(
                            stats["mix_cross_rank_source_policy"],
                            CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP,
                        )
                        self.assertTrue(stats["ready_queue_enabled"])
                    left_batch, left_stats = rank_batches[0]
                    right_batch, right_stats = rank_batches[1]
                    if (
                        left_stats["mix_source_name"]
                        == right_stats["mix_source_name"]
                    ):
                        self.assertNotEqual(
                            left_batch.samples[0].video_frame_indices,
                            right_batch.samples[0].video_frame_indices,
                        )
                self.assertEqual(
                    observed,
                    Counter({"source_a": 6, "source_b": 2}),
                )
                for rank, pipeline in enumerate(pipelines):
                    snapshot = pipeline.snapshot()
                    self.assertEqual(snapshot["mix_rank"], rank)
                    self.assertEqual(snapshot["mix_world_size"], 2)
                    self.assertEqual(
                        snapshot["mix_cross_rank_source_policy"],
                        CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP,
                    )
            finally:
                for pipeline in pipelines:
                    pipeline.close()

    def test_cross_rank_source_policy_validation(self):
        source = _schedule_source("source", 1.0)
        with self.assertRaisesRegex(ValueError, "cross_rank_source_policy"):
            WeightedIndexedBatchStream(
                [_NoOpChild()],
                [source],
                batches_per_epoch=1,
                seed=0,
                shuffle=True,
                cross_rank_source_policy="unknown",
            )

    def test_source_validation_rejects_ambiguous_configuration(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            IndexedTrainingSource(
                name="bad",
                dataset="bad",
                sample_pool="/tmp/pool",
                weight=0,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            from motus_data.indexed_mix import WeightedIndexedBatchStream

            source = IndexedTrainingSource(
                name="duplicate",
                dataset="fixture",
                sample_pool="/tmp/pool",
            )
            WeightedIndexedBatchStream(
                [object(), object()],
                [source, source],
                batches_per_epoch=1,
                seed=0,
                shuffle=False,
            )

    def test_omitted_weights_follow_sample_pool_sizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [
                _source_fixture(root, "source_a"),
                _source_fixture(root, "source_b"),
            ]
            pipeline = open_mixed_training_data(
                sources,
                batch_size=1,
                batches_per_epoch=4,
                output_adapter=None,
                queue_enabled=False,
                shuffle=False,
            )
            try:
                self.assertEqual(
                    tuple(source.weight for source in pipeline.sources),
                    (4.0, 4.0),
                )
                self.assertEqual(pipeline.stream.source_quotas(), (2, 2))
            finally:
                pipeline.close()

    def test_child_adapter_options_override_mix_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _source_fixture(root, "source_a")
            source = IndexedTrainingSource(
                **{
                    **source.__dict__,
                    "output_adapter": "motus_model_batch_v1",
                    "adapter_options": {
                        "num_video_frames": 2,
                        "action_video_fps_ratio": 1,
                        "action_loss_weight": 0.0,
                    },
                }
            )
            pipeline = open_mixed_training_data(
                [source],
                batch_size=1,
                batches_per_epoch=1,
                output_adapter=None,
                queue_enabled=False,
                shuffle=False,
            )
            try:
                batch, _ = pipeline.next()
                self.assertEqual(
                    tuple(batch["action_sequence"].shape),
                    (1, 2, 26),
                )
            finally:
                pipeline.close()

    def test_mixed_static_and_custom_generated_pools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_source = _source_fixture(root, "source_a")
            generated_source = _source_fixture(root, "source_b")
            generated_pool = _FixtureGeneratedPool(
                "agibot_beta:3:7",
                pool_id="source_b_generated",
            )
            generated_source = IndexedTrainingSource(
                **{
                    **generated_source.__dict__,
                    "sample_pool": generated_pool,
                }
            )
            adapter = MotusModelBatchV1Adapter(
                num_video_frames=2,
                action_video_fps_ratio=1,
                action_loss_weight=0.0,
            )
            pipeline = open_mixed_training_data(
                [static_source, generated_source],
                batch_size=1,
                batches_per_epoch=4,
                output_adapter=adapter,
                video_size=(2, 3),
                queue_enabled=False,
                shuffle=False,
                seed=17,
            )
            try:
                # default weights resolve len() for both pool kinds
                self.assertEqual(
                    tuple(source.weight for source in pipeline.sources),
                    (4.0, 4.0),
                )
                observed = []
                for _ in range(4):
                    batch, stats = pipeline.next()
                    observed.append(stats["mix_source_name"])
                    self.assertEqual(
                        tuple(batch["first_frame"].shape),
                        (1, 3, 2, 3),
                    )
                self.assertEqual(observed, ["source_a", "source_a",
                                            "source_b", "source_b"])
            finally:
                pipeline.close()


class _FixtureGeneratedPool:
    """Test-owned in-memory pool (no file) satisfying SupportsSamplePool."""

    def __init__(self, episode_id: str, pool_id: str):
        self._episode_id = episode_id
        self._pool_id = pool_id

    @property
    def pool_id(self) -> str:
        return self._pool_id

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


if __name__ == "__main__":
    unittest.main()
