"""CPU regression: decoded padding, forecast invariance, and metric exclusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
from thinker_wm.models.depth_output import depth_content_mask, mask_decoded_depth
from thinker_wm.models.inference_api import WAMInference, GenerationSettings, Observation
from thinker_wm.runtime.prediction_error import depth_abs_mm, PredictionErrorTracker
from thinker_wm.periodic_eval.runner import PeriodicEvaluator
from thinker_wm.periodic_eval import render

TRANSFORM = 'robotwin_3d/motus_t/pad/480x640'

class DepthOutputTest(unittest.TestCase):
    def test_geometry_and_no_mutation(self):
        mask = depth_content_mask(TRANSFORM, (480, 640))
        self.assertEqual(mask.sum().item(), 480 * 426)
        self.assertTrue(mask[:, 107:533].all())
        for shape in [(3, 480, 640), (2, 3, 4, 480, 640)]:
            x = torch.rand(shape); saved = x.clone()
            y = mask_decoded_depth(x, mask)
            self.assertTrue(torch.equal(y[..., mask], x[..., mask]))
            self.assertEqual(y[..., ~mask].count_nonzero().item(), 0)
            self.assertTrue(torch.equal(x, saved))
        self.assertIsNone(depth_content_mask('robodojo/custom', (480, 640)))
        self.assertIs(mask_decoded_depth(x, None), x)
        self.assertIsNone(mask_decoded_depth(None, mask))
        with self.assertRaises(ValueError):
            depth_content_mask(TRANSFORM, (320, 640))
        # Wide target gives side bars; narrow target gives top/bottom bars.
        for hw in [(481, 641), (640, 320)]:
            from thinker_wm.data_prepare.robotwin_3d import pad_depth_mm_to_size
            expected = pad_depth_mm_to_size(np.ones((1,360,320), dtype=np.uint16), hw)[0] > 0
            actual = depth_content_mask(f'robotwin_3d/motus_t/pad/{hw[0]}x{hw[1]}', hw)
            self.assertTrue(np.array_equal(actual.numpy(), expected))

    def test_joint_a2v_actions_and_latents_unchanged(self):
        cfg = SimpleNamespace(video_action_freq_ratio=4, temporal_mode='block_causal',
                              ar_packed_teacher_forcing=True, ar_latents_per_block=1)
        inf = WAMInference(SimpleNamespace(config=cfg), checkpoint='c', action_stats_version='s',
                           camera_transform=TRANSFORM, settings=GenerationSettings(return_depth=True))
        obs = Observation(rgb=torch.zeros(3,480,640), state=torch.zeros(14), step=0, time_s=0, robot_name="aloha")
        actions = torch.randn(1,16,14); rgb = torch.rand(1,3,4,480,640)
        depth = torch.rand_like(rgb); latent = torch.randn(1,4,2,3,3)
        captured = dict(video_latent=latent, depth_latent=latent)
        with patch.object(inf, '_run', return_value=(rgb, actions, depth, captured)):
            joint = inf.generate_action(obs, None, 'pick')
            a2v = inf.predict_outcome(obs, None, 'pick', actions[0].numpy())
        for fc in [joint.forecast, a2v]:
            np.testing.assert_array_equal(fc.actions, actions[0].numpy())
            self.assertTrue(torch.equal(fc.video_latent, latent))
            self.assertTrue(torch.equal(fc.depth_latent, latent))
            self.assertTrue(torch.equal(fc.frames, rgb))
            self.assertTrue(torch.equal(fc.depth[..., fc.depth_content_mask], depth[..., fc.depth_content_mask]))
            self.assertEqual(fc.depth[..., ~fc.depth_content_mask].count_nonzero(), 0)
            self.assertTrue(fc.meta()['has_depth_content_mask'])
            self.assertNotIn('depth_content_mask', fc.meta())

    def test_geometry_ignores_padding_even_when_measurement_is_positive(self):
        m = depth_content_mask(TRANSFORM, (480,640))
        gt = torch.ones(480,640) * 1000; pred = gt + 10
        pred[~m] = float('nan')
        err, frac = depth_abs_mm(pred, gt, m)
        self.assertEqual(err, 10); self.assertEqual(frac, 1)
        tracker = PredictionErrorTracker(16,1)
        tracker.record_visual(16, torch.zeros(3,480,640), torch.zeros(3,480,640), pred, gt, m)
        self.assertEqual(tracker.features().values['depth_abs_mm_mean'], 10)

    def test_periodic_render_and_metrics(self):
        ev = object.__new__(PeriodicEvaluator)
        ev.assignment = SimpleNamespace(source_name='robotwin_3d')
        ev.config = OmegaConf.create(dict(dataset=dict(sources=[dict(name='robotwin_3d',reader=dict(resize_mode='pad'))]),
                                        depth=dict(z_near=.01,z_far=2.,mode='log')))
        mask = depth_content_mask(TRANSFORM,(480,640))
        gt = torch.full((1,2,3,480,640),.5)
        pred = gt.permute(0,2,1,3,4).clone(); pred[...,~mask] = .9
        original = pred.clone()
        captured = []
        def panel(first, target, prediction):
            captured.append(prediction)
            return prediction
        # Deliberately mark padding valid: layout still excludes it.
        with patch.object(render,'make_compare_frames',side_effect=panel):
            result = ev._depth_panels_and_metrics(render,pred,gt,0,depth_valid_mask=torch.ones(1,2,480,640,dtype=torch.bool), block_frames=1)
        self.assertIsNotNone(result)
        self.assertEqual(result[1]['depth_absrel'],0)
        self.assertTrue((captured[0][:,~mask.numpy(),:] == 0).all())
        self.assertTrue(torch.equal(pred,original))

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
