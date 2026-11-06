from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import h5py
import numpy as np

from motus_data.agibot_beta_profiles import (
    DEXTEROUS_HAND,
    PARALLEL_GRIPPER,
    classify_agibot_beta_effector_shape,
    inspect_agibot_beta_proprio,
    read_agibot_beta_native_timeline,
)
from motus_data.agibot_beta_normalization import (
    build_agibot_beta_normalization_artifact,
)
from motus_data.catalog.agibot_beta_profiles import (
    build_agibot_beta_profile_catalogs,
)
from motus_data.repository import (
    AgibotBetaEpisodeReader,
    EpisodeRepository,
    EpisodeRepositoryError,
)
from motus_data.storage import BoundedFileReader
from examples.audit_agibot_beta_effector_profiles import main as audit_main
from tests.test_repository import RepositoryFixture


def _write_proprio(path: Path, *, timesteps: int, effector_width: int) -> None:
    position = np.zeros((timesteps, 2, 3), dtype=np.float32)
    position[:, 0, 0] = np.arange(timesteps, dtype=np.float32)
    position[:, 1, 1] = np.arange(timesteps, dtype=np.float32) + 10
    orientation = np.zeros((timesteps, 2, 4), dtype=np.float32)
    orientation[..., 3] = 1.0
    if effector_width == 2:
        effector = np.stack(
            (
                np.arange(timesteps, dtype=np.float32) + 20,
                np.arange(timesteps, dtype=np.float32) + 30,
            ),
            axis=1,
        )
    else:
        effector = np.arange(
            timesteps * effector_width, dtype=np.float32
        ).reshape(timesteps, effector_width)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("state/end/position", data=position)
        handle.create_dataset("state/end/orientation", data=orientation)
        handle.create_dataset("state/effector/position", data=effector)


def _write_profile_split_fixture(root: Path):
    source_root = root / "source"
    source_root.mkdir()
    fixture = RepositoryFixture(source_root, protocol_version="v2")
    gripper = copy.deepcopy(fixture.record)
    gripper["episode_id"] = "agibot_beta:327:1"
    gripper["meta"] = {"task_id": 327, "raw_episode_id": 1}
    dexterous = copy.deepcopy(fixture.record)
    dexterous["episode_id"] = "agibot_beta:595:2"
    dexterous["meta"] = {"task_id": 595, "raw_episode_id": 2}
    records = [gripper, dexterous]
    payload = b"".join(
        (
            json.dumps(record, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    catalog = root / "parent.episode_catalog.jsonl"
    catalog.write_bytes(payload)
    metadata = {
        "format": "motus_episode_catalog_v2",
        "catalog_id": "agibot_beta_parent_fixture",
        "source": "agibot_beta",
        "child": "agibot_beta",
        "episodes": len(records),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    catalog.with_suffix(".meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    audit_rows = [
        {
            "status": "ok",
            "episode_id": "agibot_beta:327:1",
            "task_id": 327,
            "timesteps": fixture.timesteps,
            "profile": PARALLEL_GRIPPER,
            "shapes": {
                "state/effector/position": [fixture.timesteps, 2],
            },
        },
        {
            "status": "ok",
            "episode_id": "agibot_beta:595:2",
            "task_id": 595,
            "timesteps": fixture.timesteps,
            "profile": DEXTEROUS_HAND,
            "shapes": {
                "state/effector/position": [fixture.timesteps, 12],
            },
        },
    ]
    audit_payload = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        for row in audit_rows
    )
    audit = root / "audit.jsonl"
    audit.write_bytes(audit_payload)
    summary = root / "audit.summary.json"
    summary.write_text(
        json.dumps(
            {
                "format": "agibot_beta_effector_profile_audit_v1",
                "input_catalog_id": metadata["catalog_id"],
                "input_catalog_sha256": metadata["sha256"],
                "audited_episodes": len(records),
                "status_counts": {"ok": len(records)},
            }
        ),
        encoding="utf-8",
    )
    return catalog, audit, summary, payload


class AgibotBetaEffectorProfileTests(unittest.TestCase):
    def test_shape_classification_is_explicit(self):
        self.assertEqual(
            classify_agibot_beta_effector_shape((8, 2), timesteps=8).name,
            PARALLEL_GRIPPER,
        )
        self.assertEqual(
            classify_agibot_beta_effector_shape((8, 12), timesteps=8).name,
            DEXTEROUS_HAND,
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            classify_agibot_beta_effector_shape((8, 6), timesteps=8)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            classify_agibot_beta_effector_shape((7, 2), timesteps=8)

    def test_parallel_gripper_native_timeline_is_14d_without_padding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gripper.h5"
            _write_proprio(path, timesteps=4, effector_width=2)
            inspection = inspect_agibot_beta_proprio(path, numeric_mode="all")
            self.assertEqual(inspection["profile"], PARALLEL_GRIPPER)
            self.assertEqual(inspection["state_action_dim"], 14)
            timeline, profile = read_agibot_beta_native_timeline(
                path, expected_profile=PARALLEL_GRIPPER
            )
            self.assertEqual(profile.name, PARALLEL_GRIPPER)
            self.assertEqual(timeline.shape, (4, 14))
            np.testing.assert_array_equal(timeline[:, 6], np.arange(4) + 20)
            np.testing.assert_array_equal(timeline[:, 13], np.arange(4) + 30)

    def test_dexterous_native_timeline_is_24d_without_padding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dexterous.h5"
            _write_proprio(path, timesteps=3, effector_width=12)
            inspection = inspect_agibot_beta_proprio(
                path, numeric_mode="effector"
            )
            self.assertEqual(inspection["profile"], DEXTEROUS_HAND)
            self.assertEqual(inspection["state_action_dim"], 24)
            self.assertEqual(
                len(inspection["effector_per_column_statistics"]), 12
            )
            timeline, profile = read_agibot_beta_native_timeline(
                path, expected_profile=DEXTEROUS_HAND
            )
            self.assertEqual(profile.name, DEXTEROUS_HAND)
            self.assertEqual(timeline.shape, (3, 24))
            np.testing.assert_array_equal(timeline[0, 6:12], np.arange(6))
            np.testing.assert_array_equal(timeline[0, 18:24], np.arange(6, 12))
            with self.assertRaisesRegex(ValueError, "differs from Catalog"):
                read_agibot_beta_native_timeline(
                    path, expected_profile=PARALLEL_GRIPPER
                )

    def test_nonfinite_native_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.h5"
            _write_proprio(path, timesteps=2, effector_width=2)
            with h5py.File(path, "r+") as handle:
                handle["state/effector/position"][0, 0] = np.nan
            inspection = inspect_agibot_beta_proprio(
                path, numeric_mode="effector"
            )
            self.assertEqual(inspection["effector_statistics"]["nonfinite"], 1)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_agibot_beta_native_timeline(path)

    def test_read_only_audit_uses_existing_catalog_and_supports_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = RepositoryFixture(root, protocol_version="v2")
            output = root / "audit.jsonl"
            summary = root / "audit.summary.json"
            argv = [
                "audit_agibot_beta_effector_profiles.py",
                "--catalog",
                str(fixture.catalog_path),
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
                "--numeric-mode",
                "all",
                "--progress-every",
                "1",
                "--fsync-every",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(audit_main(), 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["profile"], PARALLEL_GRIPPER)
            self.assertEqual(rows[0]["state_action_dim"], 14)
            report = json.loads(summary.read_text())
            self.assertEqual(report["profile_counts"], {PARALLEL_GRIPPER: 1})
            self.assertEqual(report["task_profile_mismatch_count"], 0)

            with mock.patch.object(sys, "argv", argv + ["--resume"]):
                self.assertEqual(audit_main(), 0)
            self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_profile_catalog_split_is_disjoint_indexed_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, audit, summary, original_payload = (
                _write_profile_split_fixture(root)
            )
            output = root / "profile_release"
            report = build_agibot_beta_profile_catalogs(
                parent_catalog_path=catalog,
                audit_jsonl_path=audit,
                audit_summary_path=summary,
                output_directory=output,
                release="fixture_v1",
                data_version="fixture_data_v1",
            )

            self.assertEqual(catalog.read_bytes(), original_payload)
            self.assertEqual(report["union_episodes"], 2)
            self.assertTrue(report["disjoint_by_construction"])
            episode_sets = []
            expected = {
                PARALLEL_GRIPPER: (
                    "agibot_beta_parallel_gripper",
                    "agibot-beta-parallel-gripper",
                    14,
                    2,
                ),
                DEXTEROUS_HAND: (
                    "agibot_beta_dexterous_hand",
                    "agibot-beta-dexterous-hand",
                    24,
                    12,
                ),
            }
            for profile_name, (child, robot, dimension, width) in expected.items():
                child_report = report["children"][profile_name]
                child_catalog = output / child_report["catalog"]
                metadata = json.loads(
                    (output / child_report["metadata"]).read_text()
                )
                rows = [
                    json.loads(line)
                    for line in child_catalog.read_text().splitlines()
                ]
                self.assertEqual(len(rows), 1)
                self.assertEqual(metadata["child"], child)
                self.assertEqual(metadata["episodes"], 1)
                self.assertEqual(metadata["meta"]["state_action_dim"], dimension)
                self.assertEqual(rows[0]["robot"], robot)
                self.assertEqual(rows[0]["meta"]["state_action_dim"], dimension)
                self.assertEqual(
                    rows[0]["state"]["meta"]["raw_effector_item_shape"],
                    [width],
                )
                index_path = output / child_report["index"]
                with sqlite3.connect(index_path) as connection:
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM episodes"
                        ).fetchone()[0],
                        1,
                    )
                episode_sets.append({rows[0]["episode_id"]})
            self.assertFalse(episode_sets[0] & episode_sets[1])
            self.assertEqual(
                episode_sets[0] | episode_sets[1],
                {"agibot_beta:327:1", "agibot_beta:595:2"},
            )

    def test_profile_catalog_split_failure_does_not_publish_partial_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, audit, summary, original_payload = (
                _write_profile_split_fixture(root)
            )
            audit_rows = audit.read_text().splitlines()
            audit.write_text("\n".join(reversed(audit_rows)) + "\n")
            output = root / "profile_release"
            with self.assertRaisesRegex(ValueError, "episode order differs"):
                build_agibot_beta_profile_catalogs(
                    parent_catalog_path=catalog,
                    audit_jsonl_path=audit,
                    audit_summary_path=summary,
                    output_directory=output,
                    release="fixture_v1",
                    data_version="fixture_data_v1",
                )
            self.assertFalse(output.exists())
            self.assertEqual(catalog.read_bytes(), original_payload)

    def test_normalization_artifact_is_catalog_bound_and_avoids_tiny_scales(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, audit, summary, _ = _write_profile_split_fixture(root)
            release = root / "profile_release"
            split_report = build_agibot_beta_profile_catalogs(
                parent_catalog_path=catalog,
                audit_jsonl_path=audit,
                audit_summary_path=summary,
                output_directory=release,
                release="fixture_v1",
                data_version="fixture_data_v1",
            )
            child_report = split_report["children"][PARALLEL_GRIPPER]
            child_catalog = release / child_report["catalog"]
            output = root / "gripper.normalization.json"
            report = build_agibot_beta_normalization_artifact(
                catalog_path=child_catalog,
                output_path=output,
                checkpoint_every=1,
            )
            artifact = json.loads(output.read_text())
            child_metadata = json.loads(
                (release / child_report["metadata"]).read_text()
            )
            child_record = json.loads(child_catalog.read_text())
            storage = child_record["state"]["storage"]
            with BoundedFileReader(
                storage["path"], storage["offset"], storage["size"]
            ) as member:
                timeline, _ = read_agibot_beta_native_timeline(member)

            self.assertEqual(report["dimension"], 14)
            self.assertEqual(artifact["dimension"], 14)
            self.assertEqual(artifact["timesteps"], timeline.shape[0])
            self.assertEqual(
                artifact["catalog"]["catalog_id"],
                child_metadata["catalog_id"],
            )
            self.assertEqual(
                artifact["catalog"]["sha256"], child_metadata["sha256"]
            )
            np.testing.assert_allclose(
                artifact["mean"], timeline.mean(axis=0), rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                artifact["empirical_std"],
                timeline.astype(np.float64).std(axis=0),
                rtol=1e-6,
                atol=1e-7,
            )
            for is_constant, scale in zip(
                artifact["constant_mask"], artifact["applied_scale"]
            ):
                if is_constant:
                    self.assertEqual(scale, 1.0)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(report["artifact_sha256"], digest)
            self.assertTrue(
                output.with_suffix(output.suffix + ".sha256")
                .read_text()
                .startswith(digest)
            )

    def test_profile_repository_reads_native_normalized_dimension_and_exclusion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, audit, summary, _ = _write_profile_split_fixture(root)
            release = root / "profile_release"
            split_report = build_agibot_beta_profile_catalogs(
                parent_catalog_path=catalog,
                audit_jsonl_path=audit,
                audit_summary_path=summary,
                output_directory=release,
                release="fixture_v1",
                data_version="fixture_data_v1",
            )
            child_report = split_report["children"][PARALLEL_GRIPPER]
            child_catalog = release / child_report["catalog"]
            output = root / "gripper.normalization.json"
            build_agibot_beta_normalization_artifact(
                catalog_path=child_catalog,
                output_path=output,
                checkpoint_every=1,
            )
            artifact = json.loads(output.read_text())
            child_record = json.loads(child_catalog.read_text())
            storage = child_record["state"]["storage"]
            with BoundedFileReader(
                storage["path"], storage["offset"], storage["size"]
            ) as member:
                raw, _ = read_agibot_beta_native_timeline(member)
            expected = (
                raw.astype(np.float64) - np.asarray(artifact["mean"])
            ) / np.asarray(artifact["applied_scale"])

            reader = AgibotBetaEpisodeReader(
                normalization_artifacts={PARALLEL_GRIPPER: output}
            )
            repository = EpisodeRepository(
                child_catalog, readers={"agibot_beta": reader}
            )
            episode_id = str(child_record["episode_id"])
            episode = repository.get_episode(episode_id)
            self.assertEqual(episode.state.schema.item_shape, (14,))
            self.assertEqual(episode.action.schema.item_shape, (14,))
            self.assertEqual(
                episode.state.schema.metadata["effector_profile"],
                PARALLEL_GRIPPER,
            )
            np.testing.assert_allclose(
                episode.read_state([0, 2]).numpy(),
                expected[[0, 2]].astype(np.float32),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                episode.read_action([1, 3]).numpy(),
                expected[[1, 3]].astype(np.float32),
                rtol=1e-6,
                atol=1e-6,
            )
            repository.close()

            excluded_reader = AgibotBetaEpisodeReader(
                normalization_artifacts={PARALLEL_GRIPPER: output},
                excluded_episode_ids=[episode_id],
            )
            excluded_repository = EpisodeRepository(
                child_catalog,
                readers={"agibot_beta": excluded_reader},
            )
            with self.assertRaisesRegex(EpisodeRepositoryError, "quality policy"):
                excluded_repository.get_episode(episode_id).read_state([0])
            excluded_repository.close()


if __name__ == "__main__":
    unittest.main()
