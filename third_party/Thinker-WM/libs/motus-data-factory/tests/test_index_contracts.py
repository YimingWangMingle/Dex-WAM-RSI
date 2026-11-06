from __future__ import annotations

import copy
import unittest

from motus_data.index_contracts import (
    IndexContractError,
    MOTUS_EPISODE_CATALOG_METADATA_V1,
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    MOTUS_EPISODE_RECORD_V1,
    MOTUS_EPISODE_RECORD_V2,
    MOTUS_SAMPLE_POOL_HEADER_V1,
    MOTUS_SAMPLE_POOL_HEADER_V2,
    MOTUS_SAMPLE_SELECTION_V1,
    validate_catalog_metadata,
    validate_agibot_beta_selection,
    validate_agibot_beta_selection_v1,
    validate_catalog_metadata_v1,
    validate_episode_catalog_binding,
    validate_episode_catalog_binding_v1,
    validate_episode_record,
    validate_episode_record_v1,
    validate_pool_catalog_binding,
    validate_pool_catalog_binding_v1,
    validate_sample_pool_header,
    validate_sample_pool_header_v1,
    validate_sample_selection_v1,
    validate_selection_pool_binding,
    validate_selection_pool_binding_v1,
)


def catalog_metadata():
    return {
        "schema_version": MOTUS_EPISODE_CATALOG_METADATA_V1,
        "record_schema_version": MOTUS_EPISODE_RECORD_V1,
        "catalog_id": "agibot_beta_test_v1",
        "source_dataset": "agibot_beta",
        "source_child": "agibot_beta",
        "data_version": "fixture_v1",
        "split_policy": {
            "type": "stable_ranked_episode_hash",
            "seed": 20260720,
            "test_fraction": 0.01,
        },
    }


def episode_record():
    locator = {
        "storage_backend": "tar_member",
        "uri": "/fixture/data.tar",
        "member": "episode/data.bin",
        "offset_data": 1024,
        "size": 4096,
    }
    return {
        "schema_version": MOTUS_EPISODE_RECORD_V1,
        "episode_id": "agibot_beta:327:648642",
        "source_dataset": "agibot_beta",
        "source_child": "agibot_beta",
        "robot_name": "agibot-a2-d",
        "split": "train",
        "video": {"frame_count": 120, "locator": dict(locator)},
        "state": {"count": 120, "locator": dict(locator)},
        "action": {"count": 120, "locator": dict(locator)},
        "text": {
            "items": [
                {"index": 0, "kind": "episode_context", "text": "fixture task"},
                {
                    "index": 1,
                    "kind": "action_segment",
                    "text": "first",
                    "frame_range": [0, 60],
                },
                {
                    "index": 2,
                    "kind": "action_segment",
                    "text": "second",
                    "frame_range": [60, 120],
                },
            ]
        },
        "metadata": {"task_id": 327, "raw_episode_id": 648642},
    }


def pool_header():
    return {
        "schema_version": MOTUS_SAMPLE_POOL_HEADER_V1,
        "record_type": "sample_pool_header",
        "record_schema_version": MOTUS_SAMPLE_SELECTION_V1,
        "sample_pool_id": "fixture_pool_v1",
        "catalog_id": "agibot_beta_test_v1",
        "source_dataset": "agibot_beta",
        "source_child": "agibot_beta",
        "split": "train",
    }


def catalog_metadata_v2():
    return {
        "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
        "catalog_id": "agibot_beta_test_v2",
        "source": "agibot_beta",
        "child": "agibot_beta",
        "episodes": 1,
        "sha256": "0" * 64,
    }


def episode_record_v2():
    storage = {
        "type": "tar",
        "path": "/fixture/data.tar",
        "member": "episode/data.bin",
        "offset": 1024,
        "size": 4096,
    }
    return {
        "episode_id": "agibot_beta:327:648642",
        "robot": "agibot-a2-d",
        "video": {"count": 120, "storage": dict(storage)},
        "state": {"count": 120, "storage": dict(storage)},
        "action": {"count": 120, "storage": dict(storage)},
        "texts": [
            {"index": 0, "kind": "episode_context", "text": "fixture task"},
            {
                "index": 1,
                "kind": "action_segment",
                "text": "first",
                "video_range": [0, 60],
            },
            {
                "index": 2,
                "kind": "action_segment",
                "text": "second",
                "video_range": [60, 120],
            },
        ],
        "meta": {"task_id": 327, "raw_episode_id": 648642},
    }


def pool_header_v2():
    return {
        "format": MOTUS_SAMPLE_POOL_HEADER_V2,
        "pool_id": "fixture_pool_v2",
        "catalog_id": "agibot_beta_test_v2",
        "samples": 1,
    }


def selection(start: int = 10):
    return {
        "episode_id": "agibot_beta:327:648642",
        "video_frame_indices": list(range(start, start + 49)),
        "state_indices": [start],
        "action_indices": list(range(start + 1, start + 49)),
        "text_indices": [0, 1],
    }


class IndexContractTests(unittest.TestCase):
    def test_valid_v1_contracts_and_binding(self):
        validate_catalog_metadata_v1(catalog_metadata())
        validate_episode_record_v1(episode_record())
        validate_sample_pool_header_v1(pool_header())
        validate_pool_catalog_binding_v1(pool_header(), catalog_metadata())
        validate_episode_catalog_binding_v1(episode_record(), catalog_metadata())
        validate_selection_pool_binding_v1(selection(), episode_record(), pool_header())

    def test_valid_split_neutral_v2_and_generic_dispatch(self):
        metadata = catalog_metadata_v2()
        episode = episode_record_v2()
        header = pool_header_v2()
        validate_catalog_metadata(metadata)
        validate_episode_record(episode)
        validate_sample_pool_header(header)
        validate_pool_catalog_binding(header, metadata)
        validate_episode_catalog_binding(episode, metadata)
        validate_selection_pool_binding(selection(), episode, header)
        validate_agibot_beta_selection(selection(), episode)

        invalid_metadata = copy.deepcopy(metadata)
        invalid_metadata["split_policy"] = {
            "type": "stable_ranked_episode_hash",
            "seed": 1,
            "test_fraction": 0.1,
        }
        with self.assertRaisesRegex(IndexContractError, "unknown fields"):
            validate_catalog_metadata(invalid_metadata)

        invalid_episode = copy.deepcopy(episode)
        invalid_episode["split"] = "train"
        with self.assertRaisesRegex(IndexContractError, "unknown fields"):
            validate_episode_record(invalid_episode)

        invalid_header = copy.deepcopy(header)
        invalid_header["split"] = "train"
        with self.assertRaisesRegex(IndexContractError, "unknown fields"):
            validate_sample_pool_header(invalid_header)

    def test_protocol_versions_cannot_be_cross_bound(self):
        with self.assertRaisesRegex(IndexContractError, "incompatible"):
            validate_pool_catalog_binding(pool_header(), catalog_metadata_v2())
        with self.assertRaisesRegex(IndexContractError, "incompatible"):
            validate_episode_catalog_binding(
                episode_record(), catalog_metadata_v2()
            )
        with self.assertRaisesRegex(IndexContractError, "incompatible"):
            validate_selection_pool_binding(
                selection(), episode_record_v2(), pool_header()
            )

    def test_generic_selection_preserves_model_owned_modality_indices(self):
        arbitrary = selection()
        arbitrary["video_frame_indices"] = [9, 4, 4]
        arbitrary["state_indices"] = [91, 3]
        arbitrary["action_indices"] = [88, 2, 77]
        arbitrary["text_indices"] = [2, 0]
        validate_sample_selection_v1(arbitrary)

    def test_selection_requires_every_modality_and_rejects_path_leakage(self):
        missing_action = selection()
        del missing_action["action_indices"]
        with self.assertRaisesRegex(IndexContractError, "missing fields"):
            validate_sample_selection_v1(missing_action)

        leaked_path = selection()
        leaked_path["tar_path"] = "/must/not/leak"
        with self.assertRaisesRegex(IndexContractError, "unknown fields"):
            validate_sample_selection_v1(leaked_path)

    def test_agibot_beta_accepts_model_owned_variable_length_indices(self):
        variable = selection()
        variable["video_frame_indices"] = [90, 3, 90]
        variable["state_indices"] = [5, 5, 7]
        variable["action_indices"] = [100, 2]
        variable["text_indices"] = [2, 0, 2]
        validate_agibot_beta_selection_v1(variable, episode_record())

    def test_optional_modalities_may_have_empty_model_owned_indices(self):
        video_only = selection()
        video_only["video_frame_indices"] = [7]
        video_only["state_indices"] = []
        video_only["action_indices"] = []
        video_only["text_indices"] = []
        validate_agibot_beta_selection_v1(video_only, episode_record())

    def test_every_modality_index_is_bounds_checked_without_derivation(self):
        bad_state = selection()
        bad_state["state_indices"] = [120]
        with self.assertRaisesRegex(IndexContractError, "state_indices.*out of bounds"):
            validate_selection_pool_binding_v1(
                bad_state, episode_record(), pool_header()
            )

        bad_text = selection()
        bad_text["text_indices"] = [3]
        with self.assertRaisesRegex(IndexContractError, "text_indices.*out of bounds"):
            validate_selection_pool_binding_v1(
                bad_text, episode_record(), pool_header()
            )

    def test_catalog_rejects_invalid_text_index_space_and_frame_range(self):
        bad_index = episode_record()
        bad_index["text"]["items"][1]["index"] = 7
        with self.assertRaisesRegex(IndexContractError, "list positions"):
            validate_episode_record_v1(bad_index)

        bad_range = episode_record()
        bad_range["text"]["items"][1]["frame_range"] = [0, 121]
        with self.assertRaisesRegex(IndexContractError, "video count"):
            validate_episode_record_v1(bad_range)

        preserved_metadata = episode_record()
        preserved_metadata["text"]["items"][1]["metadata"] = {
            "source_segment_index": 0,
            "action_text": "first",
            "skill": "fixture skill",
        }
        validate_episode_record_v1(preserved_metadata)

    def test_pool_header_must_bind_to_exact_catalog_and_split(self):
        wrong = copy.deepcopy(pool_header())
        wrong["catalog_id"] = "another_catalog"
        with self.assertRaisesRegex(IndexContractError, "does not match"):
            validate_pool_catalog_binding_v1(wrong, catalog_metadata())

        wrong_split = copy.deepcopy(pool_header())
        wrong_split["split"] = "test"
        with self.assertRaisesRegex(IndexContractError, "split=.*does not match"):
            validate_selection_pool_binding_v1(
                selection(), episode_record(), wrong_split
            )


if __name__ == "__main__":
    unittest.main()
    validate_episode_catalog_binding,
    validate_episode_record,
    validate_pool_catalog_binding,
    validate_sample_pool_header,
    validate_selection_pool_binding,
