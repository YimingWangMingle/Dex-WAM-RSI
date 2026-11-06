"""Periodic eval: all blocks, history inputs on the main arm, paired no-history arm.

Run: python scripts/test_periodic_eval_history.py   (CPU, fake model)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from thinker_wm.periodic_eval import runner as R  # noqa: E402
from thinker_wm.periodic_eval.config import PeriodicEvalConfig  # noqa: E402

B, T, HW = 3, 16, 8


class FakeModel(torch.nn.Module):
    """Main arm (with history) predicts the GT exactly; the no-history arm is off by 0.1.
    A2V with the GT commands is off by 0.02, with held commands by 0.05."""

    def __init__(self, gt_video, gt_actions):
        super().__init__()
        self.config = SimpleNamespace(temporal_mode="block_causal", ar_packed_teacher_forcing=True,
                                      ar_latents_per_block=1, video_action_freq_ratio=4, num_video_frames=T,
                                      depth_condition_mode="first_frame", enable_depth=False)
        self.gt_video, self.gt_actions, self.calls = gt_video, gt_actions, []

    def inference_step(self, first_frame, history_inputs=None, num_blocks=None, condition_actions=None, **kw):
        n = first_frame.shape[0]
        if condition_actions is None:
            arm = "main" if history_inputs is not None else "nohist"
        else:
            held = bool((condition_actions == condition_actions[:, :1]).all())
            arm = "hold" if held else "a2v"
        start = sum(c["n"] for c in self.calls if c["arm"] == arm)
        self.calls.append(dict(n=n, arm=arm, history=history_inputs, num_blocks=num_blocks,
                               seed=torch.initial_seed(), commands=condition_actions))
        off = dict(main=0.0, nohist=0.1, a2v=0.02, hold=0.05)[arm]
        frames = self.gt_video[start:start + n].permute(0, 2, 1, 3, 4) + off
        return frames, self.gt_actions[start:start + n] + off


class FakeHistory:
    last_metrics = {}

    def attach(self, batch, step):
        self.step = step
        self.last_metrics = dict(history_empty_fraction=0.0)
        out = dict(batch)
        out["history_recent_rgb"] = torch.zeros(len(batch["episode_ids"]), 5, 3, HW, HW)
        return out


class PeriodicEvalHistory(unittest.TestCase):
    def test_history_blocks_and_nohist_arm(self):
        g = torch.Generator().manual_seed(0)
        video = torch.randint(0, 256, (B, T, 3, HW, HW), generator=g, dtype=torch.uint8)
        actions = torch.randn(B, T * 4, 8, generator=g)
        data = dict(tensors=dict(first_frame=video[:, 0], video_frames=video, action_sequence=actions,
                                 initial_state=torch.ones(B, 8)),
                    pils=None,
                    meta=dict(episode_ids=[f"ep{i}" for i in range(B)],
                              frame_indices=[list(range(0, 4 * T + 1, 4))] * B,
                              source_datasets=["robotwin_3d"] * B, vlm_input_messages=["m"] * B,
                              t5_input_messages=["t"] * B))
        model = FakeModel(video.float() / 255.0, actions)
        ev = object.__new__(R.PeriodicEvaluator)
        hist = FakeHistory()
        with tempfile.TemporaryDirectory() as tmp:
            ev.__dict__.update(
                config=OmegaConf.create(dict(model=dict(wm=dict()))),
                cfg=PeriodicEvalConfig(no_history_arm=True, a2v_arm=True, vlm_generate=False), rank=0,
                assignment=SimpleNamespace(source_name="robotwin_3d", kind="k", split="holdout", shard_ordinal=0),
                num_inference_steps=2, guidance_scale=1.0, action_inference_params={}, infer_chunk_size=2,
                _depth_attacher_built=True, _depth_attacher_obj=None, _history_attacher_built=True,
                _history_attacher_obj=hist, _action_dim_labels={}, _eef_display_layouts={},
                _warned_zero_state=set(), media_root=Path(tmp), cache_dir=Path(tmp))
            with mock.patch.object(R.cache_mod, "load_cache", return_value=data), \
                    mock.patch("thinker_wm.periodic_eval.render.write_video"):
                records, summary = ev._run_local_eval(model, global_step=4000)

        self.assertEqual(hist.step, 0)
        arms = {a: [c for c in model.calls if c["arm"] == a] for a in ("main", "nohist", "a2v", "hold")}
        main, ctrl = arms["main"], arms["nohist"]
        self.assertEqual([c["n"] for c in main], [2, 1])          # history sliced per inference chunk
        self.assertEqual(main[0]["history"]["history_recent_rgb"].shape[0], 2)
        self.assertEqual(len(ctrl), 2)
        self.assertTrue(all(c["history"] is not None for c in arms["a2v"] + arms["hold"]))
        self.assertTrue(torch.equal(arms["a2v"][0]["commands"].cpu(), actions[:2]))   # real executed commands
        self.assertEqual({c["num_blocks"] for c in model.calls}, {4})
        self.assertEqual(len({c["seed"] for c in model.calls}), 1)                    # all arms paired
        m = summary["metrics"]
        for b in range(1, 5):
            self.assertAlmostEqual(m[f"video_mse_b{b}_a2v"], 0.02 ** 2, places=6)
            self.assertAlmostEqual(m[f"video_mse_b{b}_a2v_sensitivity"], 0.03 ** 2, places=6)
        self.assertEqual(m["history_empty_fraction"], 0.0)
        for b in range(1, 5):
            self.assertLess(m[f"video_mse_b{b}"], 1e-4)
            self.assertAlmostEqual(m[f"video_mse_b{b}_nohist"], 0.01, places=4)
            self.assertLess(m[f"action_mse_b{b}"], 1e-10)
            self.assertAlmostEqual(m[f"action_mse_b{b}_nohist"], 0.01, places=5)
        self.assertNotIn("video_mse_b5", m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
