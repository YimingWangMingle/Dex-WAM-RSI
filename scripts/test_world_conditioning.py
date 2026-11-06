"""Training/inference contracts of the block-causal UBT_WM (real methods, CPU).

Run: python scripts/test_world_conditioning.py
Window: 8 future frames at ratio 4 -> 3 latents (blocks 0/1/2), 32 actions.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_harness as Hn  # noqa: E402

import torch  # noqa: E402

from thinker_wm.models import temporal_training as TT  # noqa: E402
from thinker_wm.models.temporal_layout import TemporalRequest  # noqa: E402


def capture_calls(model):
    calls = []
    real = model.world_model.forward

    def spy(*a, **k):
        calls.append(dict(video=a[3], action=a[4], **k))
        return real(*a, **k)

    model.world_model.forward = spy
    return calls


def decision(model, cut=1, world=False):
    return TT.StepDecision(
        TemporalRequest(TT.time_grid_from_config(model.config), cut=cut,
                        action_role="world" if world else "policy"),
        TT.MODE_WORLD if world else TT.MODE_JOINT)


class TrainingContracts(unittest.TestCase):
    def test_legacy_mode_untouched(self):
        m = Hn.Harness(Hn.config(temporal_mode="bidirectional")).train()
        out = m.training_step(**Hn.batch(), dropout_seed_step=0)
        self.assertEqual(out["temporal_cut"], 0)
        self.assertTrue(torch.isfinite(out["total_loss"]))

    def test_policy_step_backward(self):
        m = Hn.Harness(Hn.config()).train()
        out = m.training_step(**Hn.batch(), dropout_seed_step=0)
        self.assertEqual(out["temporal_cut"], 1)
        out["total_loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in m.world_model.action_branch.parameters()))

    def test_teacher_forced_block_is_clean_and_unsupervised(self):
        m = Hn.Harness(Hn.config()).train()
        b = Hn.batch()
        ctx = m.prepare_training_context(**b)
        calls = capture_calls(m)
        m.training_step(**b, training_context=ctx, temporal_decision=decision(m, cut=2))
        call = calls[-1]
        clean = ctx["clean_full_latent"]
        torch.testing.assert_close(call["video"][:, :, :2], clean[:, :, :2])
        self.assertFalse(torch.allclose(call["video"][:, :, 2], clean[:, :, 2]))
        torch.testing.assert_close(call["action"][:, :16], ctx["actions"][:, :16])
        self.assertEqual(call["temporal"].cut, 2)

    def test_world_mode_actions_given_and_never_dropped(self):
        cfg = Hn.config(modality_dropout_action=1.0, zero_action_loss_force_max_noise=True)
        m = Hn.Harness(cfg).train()
        b = Hn.batch()
        ctx = m.prepare_training_context(**b)
        calls = capture_calls(m)
        out = m.training_step(**b, training_context=ctx, temporal_decision=decision(m, world=True),
                              action_loss_weights=torch.zeros(2))
        self.assertEqual(out["modality_dropped_action"], 0)
        self.assertEqual(float(out["action_loss"]), 0.0)
        self.assertGreater(float(out["video_loss"]), 0.0)
        torch.testing.assert_close(calls[-1]["action"], ctx["actions"].to(calls[-1]["action"].dtype))

    def test_decisions_are_step_functions(self):
        cfg = Hn.config(ar_cut_probs=[0.5, 0.5], world_mode_prob=0.3)
        a = [TT.sample_step_decision(cfg, s, True, True) for s in range(64)]
        b = [TT.sample_step_decision(cfg, s, True, True) for s in range(64)]
        self.assertEqual([(d.request.cut, d.mode) for d in a], [(d.request.cut, d.mode) for d in b])
        self.assertEqual({d.request.cut for d in a}, {1, 2})
        self.assertIn(TT.MODE_WORLD, {d.mode for d in a})
        self.assertEqual(TT.sample_step_decision(cfg, 3, False, True).request.cut, 1)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            TT.validate_temporal_config(Hn.config(ar_cut_probs=[0.3, 0.3, 0.4]))
        with self.assertRaises(ValueError):
            TT.validate_temporal_config(Hn.config(temporal_mode="bidirectional", history_enable=True))
        with self.assertRaises(ValueError):
            TT.validate_temporal_config(Hn.config(action_t_downsample=24))

    def test_history_training_reaches_encoder(self):
        m = Hn.Harness(Hn.config(history_enable=True)).train()
        b = Hn.batch(history_slots=3)
        out = m.training_step(**b, dropout_seed_step=0)
        out["total_loss"].backward()
        g = m.world_model.history_encoder.logit_bias.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())

    def test_rgb_only_depth_supervises_frame0(self):
        m = Hn.Harness(Hn.config(enable_depth=True, depth_condition_mode="rgb_only")).train()
        b = Hn.batch(depth=True)
        ctx = m.prepare_training_context(**b)
        calls = capture_calls(m)
        out = m.training_step(**b, training_context=ctx, dropout_seed_step=0)
        self.assertGreater(float(out["depth_loss"]), 0.0)
        d_in = calls[-1]["depth"]
        self.assertFalse(torch.allclose(d_in[:, :, 0], ctx["clean_depth_latent"][:, :, 0]))


class InferenceContracts(unittest.TestCase):
    def setUp(self):
        self.m = Hn.Harness(Hn.config(history_enable=True)).eval()
        b = Hn.batch(history_slots=3)
        self.inp = dict(first_frame=b["first_frame"], state=b["state"],
                        vlm_input_messages=b["vlm_input_messages"], num_inference_steps=3,
                        decode_outputs=False)
        self.hist = {k: b[k] for k in b if k.startswith("history_")}
        g = torch.Generator().manual_seed(3)
        self.noise = dict(init_video_noise=torch.randn(2, 8, 3, 4, 4, generator=g),
                          init_action_noise=torch.randn(2, 32, 8, generator=g))
        self.actions = b["actions"]

    def test_capture_and_history_effect(self):
        cap = {}
        with torch.no_grad():
            _, a0 = self.m.inference_step(**self.inp, **self.noise, latent_capture=cap)
            _, a1 = self.m.inference_step(**self.inp, **self.noise, **self.hist)
        self.assertEqual(tuple(cap["video_latent"].shape), (2, 8, 3, 4, 4))
        self.assertEqual(tuple(cap["action_latent"].shape), (2, 32, 8))
        self.assertFalse(torch.allclose(a0, a1))

    def test_world_mode_requires_command_and_keeps_it(self):
        grid = TT.time_grid_from_config(self.m.config)
        world = TemporalRequest(grid, action_role="world")
        with self.assertRaises(ValueError):
            self.m.inference_step(**self.inp, temporal_request=world)
        with torch.no_grad():
            _, a = self.m.inference_step(**self.inp, **self.noise, temporal_request=world,
                                         condition_actions=self.actions, action_smooth_alpha=0.5)
        torch.testing.assert_close(a, self.actions)

    def test_inference_rejects_teacher_forced_cut(self):
        grid = TT.time_grid_from_config(self.m.config)
        with self.assertRaises(ValueError):
            self.m.inference_step(**self.inp, temporal_request=TemporalRequest(grid, cut=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
