"""RoboDojo field map / HDF5 reader on synthetic episodes, and train == deploy transform.

Run: python scripts/test_robodojo_core.py   (needs h5py + cv2; CPU)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _rsi_cpu_stubs  # noqa: F401,E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from thinker_wm.data_prepare.robodojo_core import (FieldMap, FieldMapError, RoboDojoHDF5, episode_record,  # noqa: E402
                                        pad_or_resize, split_by_scene)
from thinker_wm.runtime.robodojo_adapter import ObservationBinding, RoboDojoObservationAdapter  # noqa: E402

RAW = {"schema_version": 1, "control_hz": 25,
       "rgb": {"views": {"head": "/obs/head/rgb", "wrist": "/obs/wrist/rgb"}, "encoding": "uint8_thwc"},
       "depth": {"views": {"head": "/obs/head/depth", "wrist": "/obs/wrist/depth"}, "unit_to_mm": 1.0,
                 "invalid_values": [0, 65535]},
       "state": {"path": "/robot/state", "columns": [0, 6]},
       "action": {"path": "/robot/action", "semantics": "abs_joint"},
       "instruction": {"attr": "instruction"},
       "layout": {"rows": [["head"], ["wrist"]], "row_hw": [[24, 32], [12, 16]]}}


def write_episode(path: Path, t=12, seed=0):
    g = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["instruction"] = "put the red cup on the plate"
        for v, (h, w) in (("head", (48, 64)), ("wrist", (30, 40))):
            f[f"/obs/{v}/rgb"] = g.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)
            d = g.integers(200, 3000, size=(t, h, w), dtype=np.uint16)
            d[:, 0:4, 0:4] = 0
            d[:, 4:8, 4:8] = 65535
            f[f"/obs/{v}/depth"] = d
        f["/robot/state"] = g.normal(size=(t, 8)).astype(np.float32)
        f["/robot/action"] = g.normal(size=(t, 7)).astype(np.float32)
    return path


class Core(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.fm = FieldMap.from_dict(RAW, digest="test")
        self.io = RoboDojoHDF5(self.fm)
        self.ep = write_episode(self.tmp / "scene_a" / "ep0.hdf5")

    def tearDown(self):
        self._t.cleanup()

    def test_missing_keys_are_errors(self):
        for drop in ("control_hz", "layout", "state"):
            raw = {k: v for k, v in RAW.items() if k != drop}
            with self.assertRaises(FieldMapError):
                FieldMap.from_dict(raw)
        bad = dict(RAW, layout={"rows": [["head", "side"]], "row_hw": [[24, 32]]})
        with self.assertRaises(FieldMapError):
            FieldMap.from_dict(bad)

    def test_compose_and_depth_alignment(self):
        rgb = self.io.read_rgb(str(self.ep), [0, 5, 5])
        self.assertEqual(rgb.shape, (3, 36, 32, 3))
        self.assertTrue((rgb[:, 24:, 16:] == 0).all())          # right-padded short row
        np.testing.assert_array_equal(rgb[1], rgb[2])
        mm, valid = self.io.read_depth_mm(str(self.ep), [0, 5])
        self.assertEqual(mm.shape, (2, 36, 32))
        self.assertTrue(((mm > 0) == valid).all())
        self.assertFalse(valid[:, 24:, 16:].any())
        with h5py.File(self.ep) as f:
            raw = f["/obs/head/depth"][5].astype(np.float32)
        import cv2
        exp = cv2.resize(np.where((raw == 0) | (raw == 65535), 0, raw), (32, 24), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(mm[1, :24], exp)

    def test_rows_and_record(self):
        st = self.io.read_rows(str(self.ep), "state", [0, 3])
        self.assertEqual(st.shape, (2, 6))
        r = episode_record(self.ep, self.fm, robot="rd")
        self.assertEqual(r["video"]["count"], 12)
        self.assertEqual(r["meta"]["scene_id"], "scene_a")
        self.assertEqual(r["texts"][0]["text"], "put the red cup on the plate")
        self.assertEqual(r["action"]["meta"]["semantics"], "abs_joint")

    def test_split_by_scene_no_leak(self):
        recs = []
        for s in range(6):
            for e in range(2):
                recs.append(episode_record(write_episode(self.tmp / f"scene_{s}" / f"ep{e}.hdf5", seed=s * 10 + e),
                                           self.fm, robot="rd"))
        sp = split_by_scene(recs, val_fraction=0.4, seed=3, exclude_scenes=["scene_5"])
        scene = {r["episode_id"]: r["meta"]["scene_id"] for r in recs}
        tr, va = {scene[i] for i in sp["train"]}, {scene[i] for i in sp["val"]}
        self.assertFalse(tr & va)
        self.assertNotIn("scene_5", tr | va)
        self.assertEqual(len(sp["excluded"]), 2)

    def test_runtime_adapter_equals_dataset_transform(self):
        binding = ObservationBinding(rgb_keys={"head": "cams.head.rgb", "wrist": "cams.wrist.rgb"},
                                     state_key="robot.state", state_columns=(0, 6),
                                     depth_keys={"head": "cams.head.depth", "wrist": "cams.wrist.depth"},
                                     depth_unit_to_mm=1.0)
        with h5py.File(self.ep) as f:
            live = {"cams": {v: {"rgb": f[f"/obs/{v}/rgb"][7], "depth": f[f"/obs/{v}/depth"][7]}
                             for v in ("head", "wrist")},
                    "robot": {"state": f["/robot/state"][7]}}
        for mode, hw in (("pad", (40, 40)), ("interpolate", (32, 48))):
            ad = RoboDojoObservationAdapter(self.fm, binding, model_hw=hw, resize_mode=mode, robot_name="rd")
            o, depth = ad.observation(live, step=7)
            chw = torch.from_numpy(self.io.read_rgb(str(self.ep), [7])).permute(0, 3, 1, 2)
            ref = (pad_or_resize(chw, hw, "pad", False).float() / 255.0 if mode == "pad"
                   else pad_or_resize(chw.float() / 255.0, hw, "interpolate", False))
            torch.testing.assert_close(o.rgb, ref[0].clamp(0, 1))
            mm, _ = self.io.read_depth_mm(str(self.ep), [7])
            dref = pad_or_resize(torch.from_numpy(mm).unsqueeze(1), hw, mode, True)[0, 0]
            torch.testing.assert_close(depth, dref)
            self.assertTrue(bool((depth == 0).any()) and float(depth.max()) < 65535)
            torch.testing.assert_close(o.state, torch.from_numpy(self.io.read_rows(str(self.ep), "state", [7])[0]))
            self.assertAlmostEqual(o.time_s, 7 / 25)
            self.assertIn("fm=test", ad.camera_transform)

    def test_joined_paths_canvas_and_path_tail(self):
        raw = dict(RAW, state={"paths": ["/robot/state", "/robot/action"]},
                   action={"paths": ["/robot/action", "/robot/state"], "columns": [0, 9],
                           "semantics": "abs_joint"},
                   scene={"path_tail": 2},
                   layout=dict(RAW["layout"], canvas_hw=[40, 48]))
        fm = FieldMap.from_dict(raw, digest="joined")
        io = RoboDojoHDF5(fm)
        with h5py.File(self.ep) as f:
            s, a = f["/robot/state"][()], f["/robot/action"][()]
        np.testing.assert_array_equal(io.read_rows(str(self.ep), "state", [2]), np.concatenate([s, a], 1)[[2]])
        np.testing.assert_array_equal(io.read_rows(str(self.ep), "action", [2]), np.concatenate([a, s], 1)[[2], :9])
        r = episode_record(self.ep, fm, io, robot="rd")
        self.assertEqual((r["state"]["meta"]["dimension"], r["action"]["meta"]["dimension"]), (15, 9))
        self.assertEqual(r["meta"]["scene_id"], "scene_a/ep0")
        rgb = io.read_rgb(str(self.ep), [5])
        self.assertEqual(rgb.shape, (1, 40, 48, 3))
        np.testing.assert_array_equal(rgb[0, 2:38, 8:40], self.io.read_rgb(str(self.ep), [5])[0])
        self.assertFalse(rgb[0, :2].any() or rgb[0, :, :8].any() or rgb[0, :, 40:].any())
        mm, valid = io.read_depth_mm(str(self.ep), [5])
        np.testing.assert_array_equal(mm[0, 2:38, 8:40], self.io.read_depth_mm(str(self.ep), [5])[0][0])
        self.assertFalse(valid[0, :2].any())
        for bad in ({"path": "/robot/state", "paths": ["/robot/state"]}, {"paths": []}):
            with self.assertRaises(FieldMapError):
                FieldMap.from_dict(dict(RAW, state=bad))
        with self.assertRaises(FieldMapError):
            FieldMap.from_dict(dict(RAW, layout=dict(RAW["layout"], canvas_hw=[30, 48])))

    def test_adapter_requires_explicit_binding(self):
        with self.assertRaises(ValueError):
            RoboDojoObservationAdapter(self.fm, ObservationBinding(rgb_keys={"head": "x"}, state_key="s"),
                                       model_hw=(40, 40), resize_mode="pad", robot_name="rd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
