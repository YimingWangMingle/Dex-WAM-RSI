#!/usr/bin/env python3
"""CPU fixtures exercise real parquet/video IO, splits, provenance and RSI consumers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from thinker_wm.data_prepare.robocoliseum_core import (
    STATE_COLUMNS, VIEWS, load_catalog, read_rgb, read_window, scan,
)
from data.robocoliseum import (
    experience_candidate, load_training_row, training_manifest, training_segments,
)
from experience.packages import load_package, ExperienceCatalog, materialize, write_catalog


class RoboColiseumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "source"
        self.task = self.root / "pick_(or)_500"
        (self.task / "meta").mkdir(parents=True)
        (self.task / "data/chunk-000").mkdir(parents=True)
        self.info = dict(codebase_version="v2.1", robot_type="g2a_sim", fps=30,
                         total_episodes=1, chunks_size=1000, splits={"train": "0:1"},
                         data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                         video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                         features={}, instruction_segments={"0": [dict(instruction="pick green block",
                             start_frame_index=1, end_frame_index=5)]})
        for name, dim, prefix in (("observation.state", 109, "state"), ("action", 38, "action")):
            fields = {}
            is_state = prefix == "state"
            for kind, indices in (("position", range(24, 30) if is_state else range(2, 8)),
                                  ("orientation", range(16, 24) if is_state else range(8, 16))):
                key = f"{prefix}/end/{'arm_' if is_state else ''}{kind}"
                fields[key] = dict(indices=list(indices), description="arm_base_link x/y/z/w")
            self.info["features"][name] = dict(shape=[dim], field_descriptions=fields)
        for view in VIEWS:
            key = "observation.images." + view
            self.info["features"][key] = dict(shape=[16, 24, 3])
            path = self.task / "videos/chunk-000" / key / "episode_000000.mp4"
            path.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (24, 16))
            for i in range(8):
                writer.write(np.full((16, 24, 3), i * 20, np.uint8))
            writer.release()
        self.annotation = {"0": dict(episode_status="approved", episode_id=123,
                                     action_steps=[dict(start_frame=1, end_frame=5, is_mistake=False,
                                                        action_text="pick green block")])}
        self.state = np.arange(8 * 109, dtype=np.float32).reshape(8, 109)
        self.action = np.arange(8 * 38, dtype=np.float32).reshape(8, 38)
        self.parquet = self.task / "data/chunk-000/episode_000000.parquet"
        self.write_rows()
        self.write_meta()

    def write_meta(self):
        (self.task / "meta/info.json").write_text(json.dumps(self.info))
        (self.task / "meta/annotations.json").write_text(json.dumps(self.annotation))
        (self.task / "meta/episodes.jsonl").write_text(json.dumps(dict(episode_index=0, length=8)) + "\n")
        (self.task / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"pick"}\n')

    def write_rows(self, frames=None):
        pq.write_table(pa.table({"observation.state": self.state.tolist(), "action": self.action.tolist(),
                                 "frame_index": list(range(8)) if frames is None else frames,
                                 "episode_index": [0] * 8, "timestamp": (np.arange(8) / 30).tolist()}),
                       self.parquet)

    def records(self):
        records, errors = scan(self.root, validate_rows=True, validate_videos=True)
        self.assertEqual(errors, [])
        return records

    def test_real_io_and_arm_frame_mapping(self):
        r = self.records()[0]
        self.assertEqual(r["outcome"], "unverified")
        self.assertEqual(r["instruction_segments"][0]["stop"], 6)
        x = read_window(r, 1, 5, action_offset=0)
        np.testing.assert_array_equal(x["states"], self.state[1:6, STATE_COLUMNS])
        np.testing.assert_array_equal(x["actions"], self.action[1:5])
        self.assertEqual(read_rgb(r, "top_head", [0, 4, 7]).shape, (3, 16, 24, 3))
        np.testing.assert_array_equal(read_window(r, 1, 5, action_offset=1)["actions"], self.action[2:6])
        with self.assertRaises(ValueError):
            read_window(r, 1, 8, action_offset=0)

    def test_incomplete_or_unknown_splits_rejected(self):
        with self.assertRaises(ValueError):
            scan(self.root, split_map={})
        r = self.records()[0]
        for split in ("validation", "val", "test", "holdout", "excluded"):
            r["split"] = split
            with self.assertRaises(ValueError):
                training_segments([r], action_offset=0)
            with self.assertRaises(ValueError):
                experience_candidate([r], Path(self.tmp.name) / "blocked", package_id="blocked")

    def test_parallel_scan_matches_serial(self):
        serial = self.records()
        parallel, errors = scan(self.root, validate_rows=True, validate_videos=True, workers=2)
        self.assertEqual(errors, [])
        self.assertEqual(serial, parallel)
        with self.assertRaises(ValueError):
            scan(self.root, workers=2, split_map={serial[0]["episode_id"]: "train", "unknown": "test"})

    def test_missing_video_and_noncontiguous_rows(self):
        self.write_rows(frames=[0, 1, 2, 3, 4, 5, 6, 9])
        records, errors = scan(self.root, validate_rows=True)
        self.assertFalse(records)
        self.assertIn("contiguous", errors[0]["error"])
        self.write_rows()
        next((self.task / "videos").rglob("*.mp4")).unlink()
        records, errors = scan(self.root)
        self.assertFalse(records)
        self.assertIn("missing video", errors[0]["error"])

    def test_mistake_excludes_action_supervision(self):
        self.annotation["0"]["action_steps"][0]["is_mistake"] = True
        self.write_meta()
        segments = training_segments(self.records(), action_offset=0, allow_reviewed_demonstrations=True)
        self.assertTrue(segments[0].consequence_supervision)
        self.assertFalse(segments[0].action_supervision)

    def test_no_auto_success_or_auto_bc(self):
        r = self.records()
        self.assertFalse(training_segments(r, action_offset=0)[0].action_supervision)
        s = training_segments(r, action_offset=0, allow_reviewed_demonstrations=True)[0]
        self.assertTrue(s.action_supervision)
        self.assertEqual(s.role, "demonstration")
        self.assertEqual(s.outcome, "unverified")

    def test_corruption_and_nonfinite_rejected(self):
        r = self.records()[0]
        self.action[2, 0] = np.nan
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "differs"):
            read_window(r, 1, 5, action_offset=0)
        records, errors = scan(self.root, validate_rows=True)
        self.assertFalse(records)
        self.assertIn("nonfinite", errors[0]["error"])

    def test_range_and_coordinate_contract_rejected(self):
        self.info["instruction_segments"]["0"][0]["end_frame_index"] = 99
        self.write_meta()
        records, errors = scan(self.root)
        self.assertFalse(records)
        self.assertIn("interval", errors[0]["error"])
        fields = self.info["features"]["action"]["field_descriptions"]
        fields["action/end/position"]["description"] = "base_link"
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "arm-base"):
            scan(self.root)

    def test_manifest_and_numeric_reader_connection(self):
        r = self.records()
        m = training_manifest(r, action_offset=0, parent_checkpoint="fixture-parent", data_version="fixture",
                              allow_reviewed_demonstrations=True)
        self.assertEqual({s["mode"] for s in m["rows"]}, {"joint_policy", "action_conditioned_world"})
        x = load_training_row(m["rows"][0])
        self.assertEqual(x["actions"].shape, (5, 38))
        m["rows"][0]["end_step"] = 100
        with self.assertRaises(ValueError):
            load_training_row(m["rows"][0])

    def test_catalog_and_manifest_reject_tampering_and_holdouts(self):
        from dataclasses import replace
        from data.training_manifest import build_manifest
        r = self.records()
        path = Path(self.tmp.name) / "catalog.jsonl"
        path.write_text(json.dumps(r[0]) + "\n")
        self.assertEqual(len(load_catalog(path)), 1)
        r[0]["instruction_segments"][0]["instruction"] = "tampered"
        path.write_text(json.dumps(r[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "digest"):
            load_catalog(path)
        s = training_segments(self.records(), action_offset=0, allow_reviewed_demonstrations=True)[0]
        with self.assertRaises(ValueError):
            build_manifest([], [replace(s, split="validation")], strategy="balanced", rollout_budget=1,
                           parent_checkpoint="fixture", data_version="fixture")

    def test_experience_uses_existing_catalog_and_workspace(self):
        r = self.records()
        out = Path(self.tmp.name) / "packages" / "fixture"
        package = experience_candidate(r, out, package_id="fixture")
        self.assertEqual(load_package(out).manifest_sha256, package.manifest_sha256)
        catalog_file = Path(self.tmp.name) / "experience.json"
        write_catalog(catalog_file, "fixture", out.parent,
                      [package], {"fixture": ["README.md"]},
                      {r[0]["task"]: {"bundle": "fixture", "kind": "original_preparation",
                                       "base_task": r[0]["task"], "reason": "test fixture"}})
        workspace = materialize(ExperienceCatalog.load(catalog_file), r[0]["task"],
                                Path(self.tmp.name) / "episode")
        self.assertIn("not a native task success", workspace.read_text("README.md"))
        # materialize intentionally makes the tree readonly; permit fixture cleanup.
        for p in sorted(workspace.root.rglob("*")):
            p.chmod(0o755 if p.is_dir() else 0o644)
        workspace.root.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
