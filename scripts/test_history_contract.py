"""History contract: window times, sparse/recent encoding, padding, reset, no future rows.

Run: python scripts/test_history_contract.py   (CPU)
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _rsi_cpu_stubs  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _rsi_harness as H  # noqa: E402
from thinker_wm.data_prepare.temporal_windows import (KIND_EPISODE_START, WindowSpec, batch_history_inputs,  # noqa: E402
                                           check_window, decision_points, select_window)
from thinker_wm.models import temporal_training as TT  # noqa: E402
from thinker_wm.models.inference_api import Observation  # noqa: E402
from thinker_wm.runtime.history_buffer import HistoryBuffer  # noqa: E402

SPEC = WindowSpec()


def obs(step, value=0.5, source="measured"):
    return Observation(rgb=torch.full((3, H.H_PIX, H.W_PIX), value), state=torch.full((8,), float(step)),
                       step=step, time_s=step / 25.0, robot_name=None, source=source)


class Windows(unittest.TestCase):
    def test_spec_defaults(self):
        self.assertEqual((SPEC.action_steps, SPEC.future_rows, SPEC.history_slots), (32, 32, 9))
        self.assertAlmostEqual(SPEC.frame_dt, 0.16)

    def test_no_future_rows_any_decision(self):
        n = 600
        for d in decision_points(SPEC, n, stride_rows=7):
            for jitter in (True, False):
                w = select_window(SPEC, n, d, episode_id="ep", seed=3, jitter=jitter)
                check_window(SPEC, w)
                self.assertTrue((w.recent_rows[w.recent_valid] < d).all())
                self.assertTrue((w.sparse_rows[w.sparse_valid] < d).all())
                np.testing.assert_allclose(w.recent_times, -(np.arange(5, 0, -1) * 4) / 25.0)

    def test_jitter_deterministic_and_bounded(self):
        a = select_window(SPEC, 600, 400, episode_id="ep", seed=1)
        b = select_window(SPEC, 600, 400, episode_id="ep", seed=1)
        c = select_window(SPEC, 600, 400, episode_id="ep", seed=2)
        np.testing.assert_array_equal(a.sparse_rows, b.sparse_rows)
        self.assertFalse(np.array_equal(a.sparse_rows, c.sparse_rows))
        nominal = select_window(SPEC, 600, 400, jitter=False)
        dt = np.abs(a.sparse_times[1:] - nominal.sparse_times[1:])
        self.assertTrue((dt <= SPEC.sparse_jitter * SPEC.sparse_stride_s + 0.5 / 25).all())

    def test_episode_start_slot_and_padding(self):
        w = select_window(SPEC, 600, 0)
        self.assertFalse(w.recent_valid.any() or w.sparse_valid.any())
        w = select_window(SPEC, 600, 300, jitter=False)
        self.assertEqual(int(w.sparse_kind[0]), KIND_EPISODE_START)
        self.assertTrue(w.sparse_valid[0])
        self.assertAlmostEqual(float(w.sparse_times[0]), -300 / 25.0)
        w = select_window(SPEC, 600, 30, jitter=False)        # early: sparse slots padded invalid
        self.assertFalse(w.sparse_valid[1:].any())
        self.assertTrue(w.recent_valid[-1])


class Encoding(unittest.TestCase):
    def setUp(self):
        self.m = H.Harness(H.config(history_enable=True))

    def test_sparse_frames_encoded_individually(self):
        g = torch.Generator().manual_seed(0)
        px = torch.rand(1, 3, 3, H.H_PIX, H.W_PIX, generator=g)
        lat = TT.encode_history_frames(self.m.encode_video, px, torch.ones(1, 3, dtype=torch.bool), torch.float32)
        for i in range(3):
            single = self.m.encode_video(px[:, i].unsqueeze(2) * 2 - 1)
            torch.testing.assert_close(lat[:, :, i:i + 1], single)

    def test_recent_clip_is_one_vae_clip_timed_at_latent_last_frame(self):
        self.assertEqual(TT.recent_clip_latent_frames(5), [0, 4])
        g = torch.Generator().manual_seed(1)
        hi = H.history_inputs(1, 2, g)
        ctx = self.m._prepare_history(hi, None)
        self.assertEqual(ctx["latent"].shape[2], 2 + 2)
        torch.testing.assert_close(ctx["times"][0, 2:], hi["history_recent_times"][0, [0, 4]])
        clip = self.m.encode_video(hi["history_recent_rgb"].permute(0, 2, 1, 3, 4) * 2 - 1)
        torch.testing.assert_close(ctx["latent"][:, :, 2:], clip)

    def test_latent_validity_from_frames(self):
        fv = torch.tensor([[False, True, True, True, True], [True, True, True, False, True]])
        torch.testing.assert_close(TT.recent_clip_latent_valid(fv), torch.tensor([[False, True], [True, False]]))

    def test_rejects_non_negative_times(self):
        g = torch.Generator().manual_seed(2)
        hi = H.history_inputs(1, 2, g)
        hi["history_sparse_times"][0, -1] = 0.0
        with self.assertRaises(ValueError):
            self.m._prepare_history(hi, None)

    def test_batch_history_inputs_matches_windows(self):
        ws = [select_window(SPEC, 600, d, jitter=False) for d in (100, 400)]
        rgb = lambda i, rows: torch.rand(len(rows), 3, H.H_PIX, H.W_PIX)  # noqa: E731
        hi = batch_history_inputs(ws, rgb, lambda i, rows: torch.zeros(len(rows), 8))
        self.assertEqual(tuple(hi["history_sparse_rgb"].shape[:2]), (2, 9))
        self.assertEqual(tuple(hi["history_recent_rgb"].shape[:2]), (2, 5))
        ctx = self.m._prepare_history(hi, None)
        self.assertEqual(ctx["latent"].shape[2], 9 + 2)


class Buffer(unittest.TestCase):
    def test_measured_only_and_monotonic(self):
        hb = HistoryBuffer(SPEC, robot_name=None)
        with self.assertRaises(ValueError):
            hb.add(obs(0, source="predicted"))
        hb.add(obs(3))
        with self.assertRaises(ValueError):
            hb.add(obs(3))
        with self.assertRaises(ValueError):
            hb.window(2)                       # only the latest measured step can be queried

    def test_matches_training_window_when_fully_observed(self):
        hb = HistoryBuffer(SPEC, robot_name=None)
        for s in range(0, 401):
            hb.add(obs(s, value=s / 1000))
        hi = hb.history_inputs(400)
        w = select_window(SPEC, 600, 400, jitter=False)
        np.testing.assert_allclose(hi["history_recent_times"][0].numpy(), w.recent_times, rtol=1e-6)
        np.testing.assert_allclose(hi["history_sparse_times"][0].numpy(), w.sparse_times, rtol=1e-6)
        np.testing.assert_array_equal(hi["history_sparse_valid"][0].numpy(), w.sparse_valid)
        np.testing.assert_array_equal(hi["history_recent_valid"][0].numpy(), w.recent_valid)
        expect = np.round(w.recent_rows / 1000 * 255) / 255
        np.testing.assert_allclose(hi["history_recent_rgb"][0, :, 0, 0, 0].numpy(), expect, atol=1e-6)
        self.assertTrue((hi["history_sparse_state"][0, w.sparse_valid, 0] < 400).all())

    def test_missing_rows(self):
        hb = HistoryBuffer(SPEC, robot_name=None)
        for s in range(0, 401, 2):             # only even rows observed
            hb.add(obs(s))
        w = select_window(SPEC, 600, 400, jitter=False)
        hi = hb.history_inputs(400)
        odd_recent = (w.recent_rows % 2 == 1) & w.recent_valid
        self.assertFalse(hi["history_recent_valid"][0][torch.from_numpy(odd_recent)].any())
        sv = hi["history_sparse_valid"][0].numpy()
        t = hi["history_sparse_times"][0].numpy()
        self.assertTrue(sv[w.sparse_valid].all())   # sparse falls back to the previous stored row
        self.assertTrue((np.round(-t[sv] * 25) % 2 == 0).all())

    def test_reset_and_retention(self):
        hb = HistoryBuffer(SPEC, robot_name=None)
        for s in range(0, 1000):
            hb.add(obs(s))
        self.assertLessEqual(len(hb), hb.retention_rows + 2)
        self.assertIn(0, hb._rgb)              # episode-start frame is kept
        hb.reset("next")
        self.assertEqual(len(hb), 0)
        self.assertIsNone(hb.last_step)
        hb.add(obs(0))
        self.assertIsNone(hb.history_inputs(0))


class Attacher(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        import h5py
        self._t = tempfile.TemporaryDirectory()
        self.path = str(Path(self._t.name) / "ep.hdf5")
        t = 400
        with h5py.File(self.path, "w") as f:
            f["/rgb"] = (np.arange(t)[:, None, None, None] % 256 * np.ones((1, 8, 8, 3))).astype(np.uint8)
            f["/state"] = np.repeat(np.arange(t, dtype=np.float32)[:, None], 8, 1)
            f["/action"] = np.zeros((t, 8), np.float32)

    def tearDown(self):
        self._t.cleanup()

    def _attacher(self, seed=5):
        from thinker_wm.data_prepare.history_attach import HistoryAttacher, RoboDojoHistoryReader
        from thinker_wm.data_prepare.robodojo_core import FieldMap
        fm = FieldMap.from_dict({"schema_version": 1, "control_hz": 25, "rgb": {"views": {"c": "/rgb"},
                                 "encoding": "uint8_thwc"}, "state": {"path": "/state"},
                                 "action": {"path": "/action", "semantics": "x"},
                                 "layout": {"rows": [["c"]], "row_hw": [[8, 8]]}})
        rd = RoboDojoHistoryReader(fm)
        rd.register_episode("robodojo:ep", self.path)
        return HistoryAttacher(SPEC, {"robodojo": rd}, seed=seed, sources=["robodojo"])

    def test_attach_matches_window_and_never_reads_future(self):
        d = 300
        batch = dict(video_frames=torch.zeros(2, 8, 3, 8, 8), episode_ids=["robodojo:ep", "other:1"],
                     frame_indices=[list(range(d, d + 33, 4)), list(range(9))],
                     source_datasets=["robodojo", "other"])
        out = self._attacher().attach(batch, step=7)
        self.assertFalse(out["history_sparse_valid"][1].any() or out["history_recent_valid"][1].any())
        rows_recent = out["history_recent_state"][0, :, 0]
        np.testing.assert_array_equal(rows_recent.numpy(), d - np.arange(5, 0, -1) * 4)
        np.testing.assert_allclose(out["history_recent_times"][0].numpy(), -(np.arange(5, 0, -1) * 4) / 25)
        sv = out["history_sparse_valid"][0]
        self.assertTrue(sv.all())
        srows = out["history_sparse_state"][0, :, 0]
        self.assertTrue((srows < d).all())
        np.testing.assert_allclose(out["history_sparse_times"][0][sv].numpy(), ((srows - d) / 25)[sv].numpy(),
                                   rtol=1e-6)
        pix = out["history_recent_rgb"][0, :, 0, 0, 0] * 255
        np.testing.assert_allclose(pix.numpy(), (d - np.arange(5, 0, -1) * 4) % 256, atol=1e-3)
        again = self._attacher().attach(batch, step=7)
        torch.testing.assert_close(again["history_sparse_times"], out["history_sparse_times"])
        other = self._attacher().attach(batch, step=8)
        self.assertFalse(torch.equal(other["history_sparse_times"][0, 1:], out["history_sparse_times"][0, 1:]))
        m = H.Harness(H.config(history_enable=True))
        from thinker_wm.data_prepare.history_attach import history_inputs_from_batch
        ctx = m._prepare_history(history_inputs_from_batch(out), None)
        self.assertEqual(ctx["latent"].shape[2], 9 + 2)

    def test_rejects_data_time_base_mismatch(self):
        batch = dict(video_frames=torch.zeros(1, 8, 3, 8, 8), episode_ids=["robodojo:ep"],
                     frame_indices=[list(range(300, 317, 2))], source_datasets=["robodojo"])
        with self.assertRaisesRegex(ValueError, "spacing"):
            self._attacher().attach(batch, step=0)

    def test_config_kwargs(self):
        from types import SimpleNamespace as NS
        cfg = NS(model=NS(wm=NS(temporal_mode="block_causal", history_enable=True, ar_cut_probs=[0.5, 0.5])),
                 common=NS(control_hz=25))
        kw = TT.motus_temporal_kwargs(cfg)
        self.assertEqual(kw, dict(temporal_mode="block_causal", history_enable=True, ar_cut_probs=[0.5, 0.5],
                                  control_hz=25.0))
        self.assertEqual(TT.motus_temporal_kwargs(NS(model=NS(wm=NS()), common=NS())), {})
        with self.assertRaisesRegex(ValueError, "control_hz"):
            TT.motus_temporal_kwargs(NS(model=NS(wm=NS(temporal_mode="block_causal")), common=NS()))
        from thinker_wm.data_prepare.history_memory import history_window_spec
        with self.assertRaisesRegex(ValueError, "control_hz"):
            history_window_spec(dict(common=dict(), history=dict()))
        from thinker_wm.data_prepare.history_attach import build_history_attacher
        with self.assertRaises(ValueError):
            build_history_attacher(dict(model=dict(wm=dict(history_enable=True))))
        self.assertIsNone(build_history_attacher(dict(model=dict(wm=dict()))))

    def test_finetune_from_pre_rsi_checkpoint_allows_only_history_encoder(self):
        from types import SimpleNamespace as NS
        from thinker_wm.models.ubt_wm import UBT_WM
        m = NS(_PRETRAIN_ALLOWED_MISSING_PREFIXES=UBT_WM._PRETRAIN_ALLOWED_MISSING_PREFIXES)
        UBT_WM._assert_pretrain_load_acceptable(m, ["world_model.history_encoder.logit_bias"], [], [])
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            UBT_WM._assert_pretrain_load_acceptable(m, ["world_model.uni_branch.head.weight"], [], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
