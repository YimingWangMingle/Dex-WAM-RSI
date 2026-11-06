"""CPU contract tests for the block-causal WAM path (encoder + uni + decoder + und).

Run: python scripts/test_temporal_causality.py   (torch only; see _rsi_cpu_stubs)

Layout used: ratio 4, t_downsample 4, 1 latent per block, 3 latents
(block 0 = current frame, blocks 1/2 future), 32 action steps
(block 1 = steps 0..15, block 2 = steps 16..31).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_cpu_stubs as S  # noqa: E402

import torch  # noqa: E402

from thinker_wm.models.temporal_attention import (  # noqa: E402
    masked_attention, rope_apply_pos, visibility, wan_block_forward_masked,
)
from thinker_wm.models.temporal_layout import (  # noqa: E402
    ROLE_CLEAN, ROLE_HISTORY, ROLE_OBS, ROLE_TARGET, Segment, TemporalRequest, TimeGrid,
    action_step_roles, history_frame_positions, latent_roles, video_segment,
)

ATOL = 1e-5
GRID = TimeGrid(video_action_freq_ratio=4, action_t_downsample=4)
B, C, F, H, W, A = 2, 8, 3, 4, 4, 32


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        video=torch.randn(B, C, F, H, W, generator=g),
        depth=torch.randn(B, C, F, H, W, generator=g),
        action=torch.randn(B, A, 8, generator=g),
        state=torch.randn(B, 8, generator=g),
    )


def history(seed=5):
    g = torch.Generator().manual_seed(seed)
    return dict(latent=torch.randn(B, C, 3, H, W, generator=g),
                times=torch.tensor([[-8.0, -2.0, -1.0]] * B),
                valid=torch.tensor([[True, True, True], [False, True, True]]),
                kind=torch.tensor([[2, 1, 0]] * B),
                state=torch.randn(B, 3, 8, generator=g))


class Model:
    model = None

    @classmethod
    def get(cls):
        if cls.model is None:
            cls.model = S.tiny_world(enable_depth=True, enable_history=True, recovery_feature_dim=6)
        return cls.model


def run(request, x, hist=None, recovery=None, und=None, t=None):
    m = Model.get()
    t = torch.tensor([[600.0], [800.0]]) if t is None else t
    with torch.no_grad():
        return m(t, None, None, x["video"], x["action"], und_cache=und or S.und_cache(B),
                 depth=x["depth"], state=x["state"], temporal=request, history=hist,
                 recovery=recovery)


def close(a, b):
    return torch.allclose(a, b, atol=ATOL, rtol=0)


class RuleTests(unittest.TestCase):
    def test_roles(self):
        r = TemporalRequest(GRID, cut=2)
        self.assertEqual(latent_roles(r, 3), [ROLE_OBS, ROLE_CLEAN, ROLE_TARGET])
        rw = TemporalRequest(GRID, action_role="world")
        self.assertEqual(set(action_step_roles(rw, A)), {ROLE_OBS})
        rgb = TemporalRequest(GRID, depth_first_frame_conditioned=False)
        self.assertEqual(latent_roles(rgb, 3, depth=True)[0], ROLE_TARGET)

    def test_grid_rejects_straddling_tokens(self):
        with self.assertRaises(ValueError):
            TimeGrid(video_action_freq_ratio=4, action_t_downsample=24)
        with self.assertRaises(ValueError):
            GRID.num_latents(6)

    def test_clean_copy_hidden_from_same_block_target(self):
        q = Segment(torch.tensor([1]), torch.tensor([ROLE_TARGET]), torch.zeros(1, 3, dtype=torch.long))
        k = Segment(torch.tensor([1, 1, 0, -1]),
                    torch.tensor([ROLE_CLEAN, ROLE_OBS, ROLE_OBS, ROLE_HISTORY]),
                    torch.zeros(4, 3, dtype=torch.long))
        self.assertEqual(visibility(q, k).tolist(), [[False, True, True, True]])

    def test_history_positions_negative_and_increasing(self):
        p = history_frame_positions(torch.tensor([[-8.0, -0.1, -0.05]]), GRID)
        self.assertTrue(bool((p < 0).all()))
        self.assertTrue(bool((p[:, 1:] > p[:, :-1]).all()))


class KernelEquivalence(unittest.TestCase):
    def test_rope_pos_matches_wan(self):
        from thinker_wm.wan.modules.model import rope_apply, rope_params
        d = 16
        freqs = torch.cat([rope_params(64, d - 4 * (d // 6)), rope_params(64, 2 * (d // 6)),
                           rope_params(64, 2 * (d // 6))], dim=1)
        x = torch.randn(2, F * H * W, 2, d)
        seg = video_segment([ROLE_OBS] * F, GRID, H, W)
        ref = rope_apply(x, torch.tensor([[F, H, W]] * 2), freqs)
        self.assertTrue(close(rope_apply_pos(x, seg.pos, freqs), ref))

    def test_masked_attention_full_mask_matches_flash(self):
        q, k, v = (torch.randn(2, 7, 2, 8) for _ in range(3))
        ref = S.reference_flash_attention(q, k, v)
        out = masked_attention(q, k, v, torch.ones(1, 7, 7, dtype=torch.bool))
        self.assertTrue(close(out, ref))

    def test_masked_block_full_mask_matches_wan_block(self):
        m = Model.get()
        blk = m.video_branch.encode_blocks[0] if hasattr(m.video_branch, "encode_blocks") else None
        if blk is None:
            blk = next(mod for mod in m.modules() if type(mod).__name__ == "WanAttentionBlock")
        L, D = F * H * W, 64
        x = torch.randn(1, L, D)
        e = torch.randn(1, L, 6, D) * 0.1
        ctx = torch.randn(1, 5, D)
        seg = video_segment([ROLE_OBS] * F, GRID, H, W)
        freqs = m.video_branch.freqs if hasattr(m.video_branch, "freqs") else None
        if freqs is None:
            self.skipTest("video branch exposes no freqs")
        grid = torch.tensor([[F, H, W]])
        with torch.no_grad():
            ref = blk(x, e, torch.tensor([L]), grid, freqs, ctx, None)
            out = wan_block_forward_masked(blk, x, e, seg.pos, freqs, ctx, None,
                                           torch.ones(1, L, L, dtype=torch.bool), None)
        self.assertTrue(close(out, ref))


class CausalityTests(unittest.TestCase):
    def setUp(self):
        self.x = inputs()
        self.req = TemporalRequest(GRID, cut=1)

    def _perturb(self, key, sl):
        y = {k: v.clone() for k, v in self.x.items()}
        y[key][sl] = y[key][sl] + 3.0 * torch.randn_like(y[key][sl])
        return y

    def test_future_block_does_not_leak_backwards(self):
        v0, a0, d0 = run(self.req, self.x)
        for key, sl in (("video", (slice(None), slice(None), 2)),
                        ("depth", (slice(None), slice(None), 2)),
                        ("action", (slice(None), slice(16, 32)))):
            v1, a1, d1 = run(self.req, self._perturb(key, sl))
            self.assertTrue(close(v1[:, :, :2], v0[:, :, :2]), key)
            self.assertTrue(close(d1[:, :, :2], d0[:, :, :2]), key)
            self.assertTrue(close(a1[:, :16], a0[:, :16]), key)

    def test_same_block_is_visible(self):
        v0, a0, _ = run(self.req, self.x)
        v1, a1, _ = run(self.req, self._perturb("video", (slice(None), slice(None), 1)))
        self.assertFalse(close(a1[:, :16], a0[:, :16]))
        v2, _, _ = run(self.req, self._perturb("action", (slice(None), slice(0, 16))))
        self.assertFalse(close(v2[:, :, 1], v0[:, :, 1]))

    def test_current_frame_reads_no_target(self):
        v0, a0, d0 = run(self.req, self.x)
        y = self._perturb("video", (slice(None), slice(None), slice(1, 3)))
        y["action"] = y["action"] + 1.0
        y["depth"][:, :, 1:] += 1.0
        v1, _, d1 = run(self.req, y)
        self.assertTrue(close(v1[:, :, 0], v0[:, :, 0]))
        self.assertTrue(close(d1[:, :, 0], d0[:, :, 0]))

    def test_teacher_forced_block_hidden_from_its_own_targets(self):
        req = TemporalRequest(GRID, cut=2)
        v0, a0, _ = run(req, self.x)
        v1, a1, _ = run(req, self._perturb("video", (slice(None), slice(None), 2)))
        self.assertTrue(close(v1[:, :, :2], v0[:, :, :2]))
        self.assertTrue(close(a1[:, :16], a0[:, :16]))
        v2, a2, _ = run(req, self._perturb("video", (slice(None), slice(None), 1)))
        self.assertFalse(close(v2[:, :, 2], v0[:, :, 2]))

    def test_world_mode_reads_causing_actions_only(self):
        req = TemporalRequest(GRID, action_role="world")
        v0, _, _ = run(req, self.x)
        v1, _, _ = run(req, self._perturb("action", (slice(None), slice(16, 32))))
        self.assertTrue(close(v1[:, :, :2], v0[:, :, :2]))
        v2, _, _ = run(req, self._perturb("action", (slice(None), slice(0, 16))))
        self.assertFalse(close(v2[:, :, 1], v0[:, :, 1]))

    def test_rgb_only_depth_frame0_is_target(self):
        req = TemporalRequest(GRID, depth_first_frame_conditioned=False)
        v0, _, _ = run(req, self.x)
        v1, _, _ = run(req, self._perturb("depth", (slice(None), slice(None), 0)))
        self.assertTrue(close(v1[:, :, 0], v0[:, :, 0]))

    def test_und_padding_ignored(self):
        tok, grid, lens = S.und_cache(B, lens=[3, 5])
        v0, a0, _ = run(self.req, self.x, und=(tok, grid, lens))
        tok2 = tok.clone()
        tok2[0, 3:] += 5.0
        v1, a1, _ = run(self.req, self.x, und=(tok2, grid, lens))
        self.assertTrue(close(v1[0], v0[0]) and close(a1[0], a0[0]))


class CheckpointingTests(unittest.TestCase):
    def test_checkpointed_temporal_path_matches(self):
        x, h = inputs(), history()
        req = TemporalRequest(GRID, cut=2)
        outs, grads = [], []
        for ckpt, every_k in ((False, "1"), (True, "1"), (True, "2")):
            os.environ["MOTUS_UNI_CHECKPOINT_EVERY_K"] = every_k
            try:
                m = S.tiny_world(enable_depth=True, enable_history=True, gradient_checkpointing=ckpt)
            finally:
                os.environ.pop("MOTUS_UNI_CHECKPOINT_EVERY_K")
            self.assertEqual(m.uni_branch.checkpoint_every_k, int(every_k))
            m.train()
            v, a, d = m(torch.tensor([[600.0], [800.0]]), None, None, x["video"], x["action"],
                        und_cache=S.und_cache(B), depth=x["depth"], state=x["state"],
                        temporal=req, history=h)
            (v.square().mean() + a.square().mean() + d.square().mean()).backward()
            outs.append((v.detach(), a.detach(), d.detach()))
            grads.append(m.history_encoder.logit_bias.grad.clone())
        for other in (1, 2):
            for p, q in zip(outs[0], outs[other]):
                self.assertTrue(torch.allclose(p, q, atol=1e-5))
            self.assertTrue(torch.allclose(grads[0], grads[other], atol=1e-5, rtol=1e-4))


class HistoryAndRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.x = inputs()
        self.req = TemporalRequest(GRID, cut=1)

    def test_history_changes_output_and_window_shape(self):
        h = history()
        v0, a0, _ = run(self.req, self.x)
        v1, a1, _ = run(self.req, self.x, hist=h)
        self.assertEqual(v1.shape, v0.shape)
        self.assertFalse(close(v1, v0))

    def test_invalid_history_slot_is_invisible(self):
        h = history()
        v0, a0, _ = run(self.req, self.x, hist=h)
        h2 = {k: v.clone() for k, v in h.items()}
        h2["latent"][1, :, 0] += 4.0          # invalid slot of sample 1
        h2["state"][1, 0] += 4.0
        v1, a1, _ = run(self.req, self.x, hist=h2)
        self.assertTrue(close(v1, v0) and close(a1, a0))
        h3 = {k: v.clone() for k, v in h.items()}
        h3["latent"][0, :, 2] += 4.0          # valid slot of sample 0
        v2, _, _ = run(self.req, self.x, hist=h3)
        self.assertFalse(close(v2[0], v0[0]))
        self.assertTrue(close(v2[1], v0[1]))

    def test_recovery_condition_only_for_valid_samples(self):
        g = torch.Generator().manual_seed(9)
        f = torch.randn(B, 6, generator=g)
        valid = torch.tensor([True, False])
        _, a0, _ = run(self.req, self.x, recovery=dict(features=f, valid=valid))
        f2 = torch.randn(B, 6, generator=g)   # not a constant shift: the encoder LayerNorms its input
        _, a1, _ = run(self.req, self.x, recovery=dict(features=f2, valid=valid))
        self.assertFalse(close(a1[0], a0[0]))
        self.assertTrue(close(a1[1], a0[1]))
        _, a_none, _ = run(self.req, self.x)
        self.assertTrue(close(a0[1], a_none[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
