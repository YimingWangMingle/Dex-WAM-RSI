#!/usr/bin/env python3
"""CPU contracts using production methods, with tiny deterministic test doubles.

Run: python -B scripts/test_robodojo_depth_cpu.py
Only torch is required; AST loading avoids importing WAN/transformers/CUDA kernels.
This tests real prepare/training/sampling code, not a reimplementation of it.
For actual network integration also run test_robodojo_depth_micro.py on CUDA.
"""
import ast
import contextlib
import logging
from pathlib import Path
import sys
import time
import types
import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thinker_wm.models.depth_conditioning import (
    depth_condition_mode, validate_depth_losses, sanitize_depth_targets,
    depth_latent_valid_mask, masked_depth_mse,
)
from thinker_wm.models import temporal_training as TT  # noqa: F401  (used by extracted UBT_WM methods)
from thinker_wm.models import capability_training as CT  # noqa: F401 (production training_step dependency)
from thinker_wm.models.prefix_error_recycling import RGBPrefixErrorBuffer, empty_metrics as prefix_error_metrics  # noqa: F401,E402


def production_class():
    tree = ast.parse((ROOT / 'third_party/Thinker-WM/models/ubt_wm.py').read_text())
    names = {'prepare_training_context', 'training_step', 'inference_step',
             'inference_action_step', '_sample_modality_dropout', '_inference_sigmas',
             '_modality_missing', 'device', '_gather_native_actions',
             '_prepare_history', '_temporal_decision'}
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UBT_WM')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    nodes += [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotusConfig']
    module = types.ModuleType('_robodojo_production_methods')
    sys.modules[module.__name__] = module
    module.__dict__.update(globals())
    module.__dict__.update(logger=logging.getLogger(__name__), UNCONDITIONAL_VLM_MESSAGE='')
    # Restore name after update(globals()) for dataclass introspection.
    module.__dict__['__name__'] = '_robodojo_production_methods'
    source = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes, cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), str(ROOT / 'third_party/Thinker-WM/models/ubt_wm.py'), 'exec'), module.__dict__)
    return module.UBT_WM, module.MotusConfig


Production, MotusConfig = production_class()


class TinyUnd:
    _last_rewrite_cache_stats = (0, 0)
    def __init__(self):
        self.messages = []
    def encode(self, messages, pils):
        self.messages.append(list(messages))
        return torch.tensor([0. if not m else 2. for m in messages])
    def get_uncond_vlm_feature(self):
        return torch.zeros(1, 1, 1), torch.ones(1, dtype=torch.long)
    def precompute_vlm_features(self, messages, pils, **kwargs):
        return torch.ones(len(messages), 1, 1), torch.ones(len(messages), dtype=torch.long)


class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.und_branch = TinyUnd()
        self.calls = []
    def forward(self, t, messages, pils, video, action, **kw):
        depth = kw.get('depth')
        self.calls.append(dict(kw, video=None if video is None else video.detach().clone(),
                               depth=None if depth is None else depth.detach().clone()))
        v = None if video is None else torch.ones_like(video) * self.weight
        a = None if action is None else torch.ones_like(action) * self.weight
        if a is not None:
            if kw.get('und_cache') is not None:
                a = a + kw['und_cache'][:, None, None]
            if depth is not None and kw.get('depth_read_by_action'):
                a = a + depth.mean(dim=(1, 2, 3, 4))[:, None, None]
        d = None if depth is None else torch.ones_like(depth) * self.weight
        self.depth_prediction = d
        if d is not None and d.requires_grad:
            d.retain_grad()
        return v, a, d


class Harness(Production):
    _MODALITY_DROPOUT_SEED_BASE = 20260807
    def __init__(self, mode='rgb_only'):
        nn.Module.__init__(self)
        self.config = MotusConfig(num_video_frames=4, video_action_freq_ratio=1,
            video_height=8, video_width=8, action_dim=2, action_state_dim=2,
            unified_action_dim=2, enable_depth=True, depth_condition_mode=mode,
            depth_share_video_noise=True, independent_modality_noise=True,
            depth_read_by_action=True, action_t_downsample=1)
        self.world_model = TinyWorld()
        self.dtype = torch.float32
        self._action_flow_shift = 1.
        self._modality_dropout_sample_idx = 0
        self._repr_specs = None
        self.fm_train_scheduler_action = None
    def encode_video(self, x):
        first = F.avg_pool3d(x[:, :, :1], (1, 2, 2))
        if x.shape[2] == 1:
            return first
        return torch.cat([first, F.avg_pool3d(x[:, :, 1:], (4, 2, 2))], 2)
    def decode_video(self, z):
        x = torch.cat([z[:, :, :1], z[:, :, 1:].repeat_interleave(4, dim=2)], 2)
        return x.repeat_interleave(2, -1).repeat_interleave(2, -2)
    def _apply_ego4d_action_state_prompt(self, state, actions, **kwargs):
        return state, actions
    def _build_action_validity_mask(self, *args):
        return None
    def _normalize_state_actions(self, state, actions, names):
        return state, actions
    def _denormalize_actions(self, actions, names):
        return actions
    def _scatter_to_unified(self, state, actions, names):
        return state, actions, None
    def _log_depth_latent_stats(self, *args):
        pass
    def _sample_flow_timestep(self, b, scheduler=None):
        return torch.full((b, 1), 500.), torch.full((b, 1, 1, 1, 1), .5), torch.full((b, 1, 1), .5)


def batch():
    torch.manual_seed(123)
    return dict(first_frame=torch.rand(2, 3, 8, 8), video_frames=torch.rand(2, 4, 3, 8, 8),
                state=torch.tensor([[0., 0.], [1., 1.]]), actions=torch.rand(2, 4, 2),
                depth_first=torch.rand(2, 3, 8, 8), depth_frames=torch.rand(2, 4, 3, 8, 8),
                depth_valid_mask=torch.ones(2, 5, 8, 8, dtype=torch.bool),
                vlm_input_messages=['pick', 'place'])


def infer_inputs():
    b = batch()
    return {k: b[k] for k in ('first_frame', 'state', 'vlm_input_messages')}


class DepthTests(unittest.TestCase):
    def test_mask_pool_sanitize_and_gradient(self):
        valid = torch.ones(2, 9, 8, 8, dtype=torch.bool)
        valid[0, 0, 0, 0] = False
        valid[0, 4, 7, 7] = False
        valid[1] = False
        full = torch.ones(2, 3, 9, 8, 8)
        full[0, :, 0, 0, 0] = float('nan')
        clean, effective = sanitize_depth_targets(full, valid)
        self.assertTrue(torch.isfinite(clean).all())
        mask = depth_latent_valid_mask(effective, (2, 3, 3, 4, 4))
        self.assertFalse(mask[0, 0, 0, 0, 0])
        self.assertFalse(mask[0, 0, 1, 3, 3])
        self.assertTrue(mask[0, 0, 2].all())
        pred = torch.zeros(2, 3, 3, 4, 4, requires_grad=True)
        target = torch.ones_like(pred).masked_fill(~mask, float('nan'))
        loss, present = masked_depth_mse(pred, target, mask)
        torch.testing.assert_close(loss, torch.tensor([1., 0.]))
        self.assertEqual(present.tolist(), [True, False])
        loss.sum().backward()
        self.assertEqual(pred.grad.masked_select(~mask.expand_as(pred)).abs().sum(), 0.)
        with self.assertRaises(ValueError):
            sanitize_depth_targets(full, valid.float())

    def test_prepare_requires_mask_and_sanitizes_labels(self):
        model, b = Harness(), batch()
        with self.assertRaisesRegex(ValueError, 'requires depth_valid_mask'):
            model.prepare_training_context(**{k:v for k,v in b.items() if k != 'depth_valid_mask'})
        b['depth_valid_mask'][0, 0, :2, :2] = False
        b['depth_first'][0, :, :2, :2] = float('nan')
        ctx = model.prepare_training_context(**b)
        self.assertTrue(torch.isfinite(ctx['clean_depth_latent']).all())
        self.assertFalse(ctx['depth_latent_valid_mask'][0, 0, 0, 0, 0])
        model.config.depth_edge_loss_enable = True
        with self.assertRaisesRegex(ValueError, 'motion/edge'):
            model.prepare_training_context(**b)

    def test_training_first_frame_loss_and_empty_sample(self):
        model, b = Harness(), batch()
        b['depth_valid_mask'][:, 1:] = False  # only current frame supervised
        b['depth_valid_mask'][1] = False
        ctx = model.prepare_training_context(**b)
        out = model.training_step(b['first_frame'], b['video_frames'], training_context=ctx)
        call = model.world_model.calls[-1]
        torch.testing.assert_close(call['video'][:, :, :1], ctx['clean_full_latent'][:, :, :1])
        self.assertFalse(torch.equal(call['depth'][:, :, :1], ctx['clean_depth_latent'][:, :, :1]))
        self.assertFalse(call['depth_first_frame_conditioned'])
        out['depth_loss'].backward()
        grad = model.world_model.depth_prediction.grad
        self.assertGreater(grad[0, :, :1].abs().sum(), 0.)
        self.assertEqual(grad[:, :, 1:].abs().sum(), 0.)
        self.assertEqual(grad[1].abs().sum(), 0.)
        # One empty sample excluded from denominator, not a factor-of-two dilution.
        self.assertAlmostEqual(out['depth_loss'].item(), 2*out['depth_loss_per_sample_mean'], places=6)

    def test_legacy_loss_and_dropout(self):
        model, b = Harness('first_frame'), batch()
        b.pop('depth_valid_mask')
        ctx = model.prepare_training_context(**b)
        torch.manual_seed(82)
        noise = torch.randn_like(ctx['clean_full_latent'])
        target = noise - ctx['clean_depth_latent']
        torch.manual_seed(82)
        out = model.training_step(b['first_frame'], b['video_frames'], training_context=ctx)
        expected = F.mse_loss(model.world_model.depth_prediction[:, :, 1:], target[:, :, 1:])
        torch.testing.assert_close(out['depth_loss'], expected, rtol=1e-6, atol=0)
        torch.testing.assert_close(model.world_model.calls[-1]['depth'][:, :, :1], ctx['clean_depth_latent'][:, :, :1])
        model.config.modality_dropout_depth = 1.
        out = model.training_step(b['first_frame'], b['video_frames'], training_context=ctx)
        self.assertIsNone(model.world_model.calls[-1]['depth'])
        self.assertEqual(out['depth_loss'], 0.)

    def test_inference_output_and_no_gt(self):
        model, args = Harness().eval(), infer_inputs()
        for key in ('depth_first', 'condition_depth_frames'):
            with self.assertRaisesRegex(ValueError, 'forbids'):
                model.inference_step(**args, **{key: torch.ones(1)}, predict_action=True)
        torch.manual_seed(65)
        full = model.inference_step(**args, predict_action=True, num_inference_steps=2,
                                    return_depth=True, return_depth_first_frame=True)
        self.assertEqual(tuple(full[2].shape), (2, 3, 5, 8, 8))
        self.assertTrue(all(c['depth'] is not None for c in model.world_model.calls))
        torch.manual_seed(65)
        future = model.inference_step(**args, predict_action=True, num_inference_steps=2, return_depth=True)
        torch.testing.assert_close(future[2], full[2][:, :, 1:], rtol=0, atol=0)
        torch.manual_seed(65)
        hidden = model.inference_step(**args, predict_action=True, num_inference_steps=2, return_depth=False)
        self.assertEqual(len(hidden), 2)
        torch.testing.assert_close(hidden[1], full[1], rtol=0, atol=0)
        model.config.depth_condition_mode = 'first_frame'
        model.inference_step(**args, predict_action=True, depth_first=batch()['depth_first'],
                             num_inference_steps=2, return_depth=False)
        self.assertIsNotNone(model.world_model.calls[-1]['depth'])

    def test_prompt_cfg_zero_state_and_legacy_helper(self):
        model, args = Harness().eval(), infer_inputs()
        model.inference_step(**args, predict_action=True, num_inference_steps=1,
                             action_guidance_scale=2., guidance_scale=1.)
        self.assertIn(['', ''], model.world_model.und_branch.messages)
        self.assertEqual(len(model.world_model.calls), 2)
        self.assertTrue(all(c['depth'] is not None for c in model.world_model.calls))
        self.assertTrue(all(not c['depth_first_frame_conditioned'] for c in model.world_model.calls))
        args['vlm_input_messages'] = ['', '']
        model.inference_step(**args, predict_action=True, num_inference_steps=1)
        self.assertEqual(model.world_model.und_branch.messages[-1], ['', ''])
        with self.assertRaisesRegex(ValueError, 'requires conditioning_actions'):
            model.inference_action_step(args['first_frame'])


if __name__ == '__main__':
    unittest.main()
