"""Data-RSI roles: error prefix vs validated recovery vs consequence; test split refused.

Run: python scripts/test_rsi_data_roles.py   (CPU)
"""
from __future__ import annotations

import os
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _rsi_cpu_stubs  # noqa: F401,E402
import numpy as np  # noqa: E402

from thinker_wm.models.temporal_training import MODE_JOINT, MODE_WORLD  # noqa: E402
from data.training_manifest import build_manifest, write_manifest  # noqa: E402
from data.recovery import (ROLE_ERROR, ROLE_NOMINAL, ROLE_RECOVERY, ROLE_UNVERIFIED,  # noqa: E402
                                  collect, segments_from_trajectory)
from execution.trajectory import TrajectoryLogger, read_trajectory  # noqa: E402

VERSIONS = dict(checkpoint="k", action_stats_version="s", camera_transform="c", runtime="r1")


def write_episode(root, name, decisions, last_measured=None, verified=True):
    """decisions: list of (decision_step, action) for chunks of 16 steps.
    Measured step s confirms row s - 1; the default confirms every row."""
    lg = TrajectoryLogger(root, name, VERSIONS)
    end = decisions[-1][0] + 16 if last_measured is None else last_measured
    plan = dict(decisions)
    for s in range(end + 1):
        lg.write("measured", step=s, time_s=s / 25, state=np.zeros(2), executed_row=s - 1 if s else None)
        if s in plan:
            if s > 0:
                lg.write("error", step=s, features=dict(valid_visual=1.0, visual_mse_max=0.01 * s))
                lg.write("decision", step=s, action=plan[s], reason="t")
            lg.write("command", decision_step=s, hash=f"h{s}", source_hash=f"h{s}", postprocess="p",
                     forecast_valid=True, invalid_reason=None, attachments=dict(actions=np.zeros((32, 2))))
    if verified:
        for i, (start, _) in enumerate(decisions):
            stop = min(decisions[i + 1][0] if i + 1 < len(decisions) else end, end)
            if stop > start:
                lg.write("segment_verification", episode_id=name, decision_step=start,
                         end_step=stop - 1, command_hash=f"h{start}", verdict="passed",
                         verifier_revision="test-physical-check/v1", evidence_steps=[stop])
    lg.close()
    return os.path.join(root, name)


class Roles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovery_episode(self):
        p = write_episode(self.root, "ep1", [(0, "continue"), (16, "continue"), (32, "local_recovery"),
                                             (48, "continue")])
        segs = segments_from_trajectory(p, outcome="success", split="development", task="t")
        self.assertEqual([s.role for s in segs], [ROLE_ERROR, ROLE_ERROR, ROLE_RECOVERY, ROLE_RECOVERY])
        self.assertEqual([s.action_supervision for s in segs], [False, False, True, True])
        self.assertTrue(all(s.consequence_supervision for s in segs))
        self.assertEqual((segs[0].decision_step, segs[0].end_step), (0, 15))
        self.assertTrue(segs[0].command_attachment.startswith("attachments/"))

    def test_failed_and_uncertain_episodes_never_supervise_actions(self):
        p = write_episode(self.root, "ep2", [(0, "continue"), (16, "local_recovery")])
        fail = segments_from_trajectory(p, outcome="failure", split="development", task="t")
        unc = segments_from_trajectory(p, outcome="uncertain", split="development", task="t")
        self.assertEqual({s.role for s in fail}, {ROLE_ERROR})
        self.assertEqual({s.role for s in unc}, {ROLE_UNVERIFIED})
        self.assertFalse(any(s.action_supervision for s in fail + unc))

    def test_segments_cover_only_confirmed_rows(self):
        p = write_episode(self.root, "part", [(0, "continue"), (16, "continue")], last_measured=20)
        segs = segments_from_trajectory(p, outcome="success", split="development", task="t")
        self.assertEqual([(s.decision_step, s.end_step) for s in segs], [(0, 15), (16, 19)])
        p = write_episode(self.root, "none", [(0, "continue"), (16, "continue")], last_measured=16)
        segs = segments_from_trajectory(p, outcome="success", split="development", task="t")
        self.assertEqual([(s.decision_step, s.end_step) for s in segs], [(0, 15)])

    def test_nominal_success(self):
        p = write_episode(self.root, "ep3", [(0, "continue"), (16, "continue")])
        segs = segments_from_trajectory(p, outcome="success", split="development", task="t")
        self.assertEqual({s.role for s in segs}, {ROLE_NOMINAL})

    def test_legacy_success_never_supervises_actions(self):
        p = write_episode(self.root, "legacy", [(0, "continue"), (16, "local_recovery")], verified=False)
        segs = segments_from_trajectory(p, outcome="success", split="development", task="t")
        self.assertFalse(any(s.action_supervision for s in segs))
        self.assertTrue(all(s.consequence_supervision for s in segs))

    def test_segment_failure_overrides_episode_success(self):
        p = write_episode(self.root, "mixed", [(0, "continue"), (16, "local_recovery")])
        records = read_trajectory(p)
        next(r for r in records if r["role"] == "segment_verification")["verdict"] = "failed"
        Path(p, "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        segs = segments_from_trajectory(p, outcome="success", split="development", task="t")
        self.assertEqual(segs[0].role, ROLE_ERROR)
        self.assertFalse(segs[0].action_supervision)
        self.assertTrue(segs[1].action_supervision)

    def test_verification_must_bind_exact_segment_and_measured_end(self):
        for key, value in [("command_hash", "wrong"), ("end_step", 18),
                           ("episode_id", "foreign"), ("evidence_steps", [0]),
                           ("verdict", "success"), ("verifier_revision", "")]:
            with self.subTest(key=key):
                p = write_episode(self.root, "bad_" + key, [(0, "continue")])
                records = read_trajectory(p)
                next(r for r in records if r["role"] == "segment_verification")[key] = value
                Path(p, "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                with self.assertRaises(ValueError):
                    segments_from_trajectory(p, outcome="success", split="development", task="t")

    def test_manifest_rejects_unverified_action_flag_and_holdouts(self):
        p = write_episode(self.root, "forged", [(0, "continue")], verified=False)
        segs = segments_from_trajectory(p, outcome="success", split="development", task="t")
        with self.assertRaises(ValueError):
            build_manifest([], [replace(segs[0], action_supervision=True)], strategy="balanced",
                           rollout_budget=1, parent_checkpoint="k", data_version="d")
        for split in ("validation", "holdout", "test", "unknown"):
            with self.assertRaises(ValueError):
                segments_from_trajectory(p, outcome="success", split=split, task="t")

    def test_test_split_refused(self):
        p = write_episode(self.root, "ep4", [(0, "continue")])
        with self.assertRaises(ValueError):
            segments_from_trajectory(p, outcome="success", split="test", task="t")

    def test_manifest_modes_budget_and_immutability(self):
        a = write_episode(self.root, "a", [(0, "continue"), (16, "local_recovery")])
        b = write_episode(self.root, "b", [(0, "continue"), (16, "continue")])
        segs = collect([dict(trajectory=a, outcome="success", split="development", task="t"),
                        dict(trajectory=b, outcome="failure", split="development", task="t")])
        out = {}
        for strategy in ("balanced", "error_weighted"):
            m = build_manifest(["expert.catalog.json"], segs, strategy=strategy, rollout_budget=100.0,
                               parent_checkpoint="k", data_version="d1")
            out[strategy] = m
            self.assertAlmostEqual(sum(r["weight"] for r in m["rows"]), 100.0)
            for r in m["rows"]:
                if r["mode"] == MODE_JOINT:
                    self.assertIn(r["role"], (ROLE_NOMINAL, ROLE_RECOVERY))
            self.assertEqual(sum(r["mode"] == MODE_WORLD for r in m["rows"]), len(segs))
            self.assertEqual(m["expert_catalogs"], ["expert.catalog.json"])
        self.assertNotEqual(out["balanced"]["manifest_sha256"], out["error_weighted"]["manifest_sha256"])
        path = os.path.join(self.root, "m.json")
        write_manifest(out["balanced"], path)
        with self.assertRaises(FileExistsError):
            write_manifest(out["balanced"], path)
        bad = [s for s in segs][:1]
        object.__setattr__(bad[0], "split", "test")
        with self.assertRaises(ValueError):
            build_manifest([], bad, strategy="balanced", rollout_budget=1.0, parent_checkpoint="k",
                           data_version="d")

    def test_logger_roles_and_versions(self):
        with self.assertRaises(ValueError):
            TrajectoryLogger(self.root, "x", dict(checkpoint="k"))
        lg = TrajectoryLogger(self.root, "y", VERSIONS)
        with self.assertRaises(ValueError):
            lg.write("predicted_observation", step=0)
        lg.close()
        recs = read_trajectory(os.path.join(self.root, "y"))
        self.assertEqual(recs[0]["versions"], VERSIONS)
        with self.assertRaises(FileExistsError):
            TrajectoryLogger(self.root, "y", VERSIONS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
