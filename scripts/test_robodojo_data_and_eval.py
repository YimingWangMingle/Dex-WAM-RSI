#!/usr/bin/env python3
"""CPU checks of depth-label validity and evaluation input routing; no assets."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/Thinker-WM/libs/motus-data-factory"), str(ROOT / "third_party/Thinker-WM/libs/motus-monitor")]

import torch

from thinker_wm.data_prepare.depth_attach import DepthAttacher


def source_function(path, name):
    """Run the production wrapper without importing checkpoint-loading modules."""
    tree = ast.parse((ROOT / path).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    dependencies = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in {"_validation_window", "_history_inputs"}]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *dependencies, fn], type_ignores=[])
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


class DepthIOTests(unittest.TestCase):
    def make_batch(self):
        return {"video_frames": torch.zeros(2, 4, 3, 4, 6),
                "episode_ids": ["valid", "missing"], "frame_indices": [list(range(5))] * 2}

    def make_attacher(self, require=True):
        attacher = DepthAttacher([], z_near=.1, z_far=4., attach_workers=1, require_valid_mask=require)
        raw = torch.full((5, 2, 3), 1000.)
        raw[0, 0, 0] = 0
        raw[1, 0, 0] = float("nan")
        raw[2, 0, 0] = 5000
        raw[3, 0, 0] = 50
        def read(episode, indices):
            return (raw.clone(), True) if episode == "valid" else (None, False)
        attacher.reader = SimpleNamespace(read_mm=read)
        return attacher

    def test_metric_validity_survives_resize_and_encoding(self):
        batch = self.make_attacher().attach(self.make_batch())
        self.assertEqual(tuple(batch["depth_valid_mask"].shape), (2, 5, 4, 6))
        self.assertEqual(batch["depth_present"].tolist(), [True, False])
        self.assertFalse(batch["depth_valid_mask"][0, :4, :2, :2].any())
        self.assertTrue(batch["depth_valid_mask"][0, 4].all())
        self.assertFalse(batch["depth_valid_mask"][1].any())
        self.assertTrue(torch.isfinite(batch["depth_frames"]).all())
        self.assertTrue(torch.isfinite(batch["depth_first"]).all())
        from thinker_wm.models.depth_codec import decode_to_depth_m
        metres, _ = decode_to_depth_m(batch["depth_first"], z_near=.1, z_far=4., in_range="01")
        torch.testing.assert_close(metres[0, 2:, 2:], torch.ones(2, 4))

    def test_attached_batch_is_not_read_twice(self):
        attacher = self.make_attacher()
        batch = attacher.attach(self.make_batch())
        attacher.reader.read_mm = lambda *_: self.fail("duplicate read")
        self.assertIs(attacher.attach(batch), batch)

    def test_rgb_only_rejects_cached_labels_without_validity(self):
        batch = self.make_attacher().attach(self.make_batch())
        del batch["depth_valid_mask"]
        with self.assertRaisesRegex(ValueError, "depth_valid_mask"):
            self.make_attacher().attach(batch)
        self.assertIs(self.make_attacher(require=False).attach(batch), batch)

    def test_partial_attach_rejected(self):
        batch = self.make_batch()
        batch["depth_first"] = torch.zeros(2, 3, 4, 6)
        with self.assertRaisesRegex(RuntimeError, "partially"):
            self.make_attacher().attach(batch)


class EvalRoutingTests(unittest.TestCase):
    def run_eval(self, mode, present=True):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(enable_depth=True, depth_condition_mode=mode)
            def inference_step(self, **kwargs):
                self.received = kwargs
                return torch.zeros(1), torch.zeros(1)
        model = Spy()
        cfg = SimpleNamespace(model=SimpleNamespace(inference=SimpleNamespace(num_inference_timesteps=2)))
        batch = {"first_frame": torch.ones(1, 3, 4, 6), "video_frames": torch.ones(1, 4, 3, 4, 6),
                 "initial_state": torch.tensor([[.1234567] * 14]), "language_embedding": None,
                 "vlm_inputs": None, "vlm_input_messages": ["pick"],
                 "depth_first": torch.ones(1, 3, 4, 6), "depth_present": torch.tensor([present])}
        source_function("third_party/Thinker-WM/train/sample.py", "inference_sample")(model, batch, cfg)
        return model.received

    def test_rgb_only_never_receives_ground_truth_depth(self):
        received = self.run_eval("rgb_only")
        self.assertIsNone(received["depth_first"])
        self.assertEqual(received["state"].dtype, torch.float32)

    def test_legacy_mode_receives_depth_condition(self):
        self.assertIsNotNone(self.run_eval("first_frame")["depth_first"])

    def test_legacy_mode_does_not_silently_drop_depth(self):
        with self.assertRaisesRegex(ValueError, "valid depth conditions"):
            self.run_eval("first_frame", present=False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
