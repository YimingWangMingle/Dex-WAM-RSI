from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

from motus_data.batch_adapters import (
    BatchAdapterError,
    IndexedBatchCollator,
    MotusModelBatchV1Adapter,
    _compose_t5_text,
    build_indexed_batch_adapter,
    indexed_batch_adapter_profiles,
    register_indexed_batch_adapter,
)
from motus_data.contracts import validate_motus_model_batch_v1
from motus_data.repository import TextItem
from motus_data.sample_pool import SamplePoolDataset
from tests.test_repository import RepositoryFixture


def _write_pool(path: Path, catalog_id: str, episode_id: str) -> None:
    rows = [
        {
            "schema_version": "motus_sample_pool_header_v1",
            "record_type": "sample_pool_header",
            "record_schema_version": "motus_sample_selection_v1",
            "sample_pool_id": "adapter_fixture_v1",
            "catalog_id": catalog_id,
            "source_dataset": "agibot_beta",
            "source_child": "agibot_beta",
            "split": "train",
            "record_count": 2,
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [0, 1, 2],
            "state_indices": [0],
            "action_indices": [1, 2],
            "text_indices": [0, 1],
        },
        {
            "episode_id": episode_id,
            "video_frame_indices": [2, 3, 4],
            "state_indices": [2],
            "action_indices": [3, 4],
            "text_indices": [0, 2],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BatchAdapterTests(unittest.TestCase):
    def test_motus_profile_is_a_replaceable_dataloader_collator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RepositoryFixture(Path(temp_dir))
            repository = fixture.repository()
            pool_path = Path(temp_dir) / "adapter.sample_pool.jsonl"
            _write_pool(
                pool_path, repository.catalog_id, fixture.episode_id
            )
            dataset = SamplePoolDataset(
                repository,
                pool_path,
                video_output_format="rgb_float32_chw",
                video_size=(2, 3),
            )
            sampler = dataset.create_batch_sampler(
                batch_size=2, drop_last=True, shuffle=False
            )
            adapter = build_indexed_batch_adapter(
                "motus_model_batch_v1",
                num_video_frames=2,
                action_video_fps_ratio=1,
                action_loss_weight=0.0,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=IndexedBatchCollator(adapter),
                num_workers=0,
            )
            batch = next(iter(loader))
            validate_motus_model_batch_v1(
                batch,
                expected_batch_size=2,
                num_video_frames=2,
                video_action_freq_ratio=1,
            )
            self.assertEqual(tuple(batch["first_frame"].shape), (2, 3, 2, 3))
            self.assertEqual(tuple(batch["video_frames"].shape), (2, 2, 3, 2, 3))
            self.assertEqual(tuple(batch["initial_state"].shape), (2, 26))
            self.assertEqual(tuple(batch["action_sequence"].shape), (2, 2, 26))
            self.assertEqual(
                batch["t5_input_messages"],
                ["Scene and task.Action one.", "Scene and task.Action two."],
            )
            self.assertTrue(
                torch.equal(batch["action_loss_weights"], torch.zeros(2))
            )
            self.assertNotIn("metadata", batch)
            self.assertEqual(fixture.decode_calls, [(0, 1, 2, 3, 4)])

    def test_profile_rejects_incompatible_indices_without_repair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RepositoryFixture(Path(temp_dir))
            repository = fixture.repository()
            pool_path = Path(temp_dir) / "adapter.sample_pool.jsonl"
            _write_pool(
                pool_path, repository.catalog_id, fixture.episode_id
            )
            dataset = SamplePoolDataset(
                repository,
                pool_path,
                video_output_format="rgb_float32_chw",
            )
            indexed = dataset.get_grouped_batch([0, 1])
            adapter = MotusModelBatchV1Adapter(num_video_frames=3)
            with self.assertRaisesRegex(
                BatchAdapterError, "requires 4 selected video frames"
            ):
                adapter.adapt(indexed)

    def test_external_profile_registration_does_not_modify_dataset(self):
        profile = "test_episode_ids_v1"

        class EpisodeIdsAdapter:
            profile_name = profile

            @staticmethod
            def adapt(batch):
                return {
                    "episode_id": batch.episode_id,
                    "batch_size": batch.batch_size,
                }

        register_indexed_batch_adapter(
            profile, EpisodeIdsAdapter, replace=True
        )
        self.assertIn(profile, indexed_batch_adapter_profiles())
        adapter = build_indexed_batch_adapter(profile)
        self.assertEqual(adapter.profile_name, profile)


class TextComposerHookTests(unittest.TestCase):
    """_compose_t5_text：注册 composer 的源走源拼装，其余源回退裸 join。"""

    @staticmethod
    def _sample(source_dataset: str, texts: tuple) -> Any:
        return SimpleNamespace(
            info=SimpleNamespace(source_dataset=source_dataset),
            text=texts,
        )

    def test_registered_source_uses_composer(self):
        sample = self._sample(
            "egoscale",
            (
                TextItem(0, "task_description_en", "Do the task", None, {}),
                TextItem(
                    1,
                    "fine_action_en",
                    "reach for the cup",
                    (0, 10),
                    {"action": "reach", "hand": "left"},
                ),
            ),
        )
        self.assertEqual(
            _compose_t5_text(sample),
            "Task: Do the task.\n"
            "Action: reach for the cup.\n"
            "Action label: reach.\n"
            "Hand: left.",
        )

    def test_unregistered_source_falls_back_to_plain_join(self):
        sample = self._sample(
            "agibot_beta",
            (
                TextItem(0, "episode_context", "Scene.\nTask: t.\n", None, {}),
                TextItem(1, "action_segment", "Action: a.\n", (0, 5), {}),
            ),
        )
        self.assertEqual(
            _compose_t5_text(sample),
            "Scene.\nTask: t.\nAction: a.",
        )


if __name__ == "__main__":
    unittest.main()
