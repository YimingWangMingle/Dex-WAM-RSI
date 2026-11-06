#!/usr/bin/env python3
"""Tests for the subprocess producer transform and CUDA-event timing mode.

    CUDA_VISIBLE_DEVICES= python scripts/test_transform_subprocess.py
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Thinker-WM/libs" / "motus-monitor"))
import torch
from omegaconf import OmegaConf

from thinker_wm.data_prepare import factory
from motus_monitor.timing import TimingStack, timed_section


class _FakeDepth:
    def attach(self, batch):
        out = dict(batch)
        out["depth"] = batch["video"].float().mean(dim=2, keepdim=True) * 0.5
        out["depth_pid"] = os.getpid()
        return out


class _FakeHistory:
    class spec:
        sparse_jitter = 0.0

    def attach(self, batch, step):
        if batch.get("boom"):
            raise ValueError("history read failed")
        out = dict(batch)
        out["history_rgb"] = batch["video"][:, :2].flip(-1).contiguous()
        out["history_sparse_valid"] = torch.tensor([[True, False]] * batch["video"].shape[0])
        return out


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "video": torch.randint(0, 255, (8, 5, 3, 96, 128), generator=g, dtype=torch.uint8),
        "actions": torch.randn(8, 16, 14, generator=g),
        "episode_ids": ["ep_a"] * 8,
        "task": "stack blocks",
        "none_field": None,
    }


def _config(subprocess):
    return OmegaConf.create({"system": {"ready_batch_queue_enabled": True,
                                        "ready_batch_transform_subprocess": subprocess}})


class SubprocessTransformTest(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch.object(factory, "build_depth_attacher", lambda cfg: _FakeDepth()),
                   mock.patch.object(factory, "_background_history_attacher",
                                     lambda cfg, q: _FakeHistory())]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_matches_in_thread_transform(self):
        local = factory._producer_transform(_FakeDepth(), _FakeHistory())
        remote = factory._SubprocessTransform(_config(True))
        self.addCleanup(remote._pool.shutdown)
        for seed in range(3):
            ref, got = local(_batch(seed)), remote(_batch(seed))
            self.assertEqual(ref.keys(), got.keys())
            for k in ref:
                if k == "depth_pid":
                    continue
                if isinstance(ref[k], torch.Tensor):
                    self.assertEqual(ref[k].dtype, got[k].dtype, k)
                    self.assertTrue(torch.equal(ref[k], got[k]), k)
                else:
                    self.assertEqual(ref[k], got[k], k)
            self.assertNotEqual(got["depth_pid"], os.getpid())

    def test_single_persistent_worker(self):
        remote = factory._SubprocessTransform(_config(True))
        self.addCleanup(remote._pool.shutdown)
        pids = {remote(_batch(s))["depth_pid"] for s in range(3)}
        self.assertEqual(len(pids), 1)

    def test_worker_error_propagates(self):
        remote = factory._SubprocessTransform(_config(True))
        self.addCleanup(remote._pool.shutdown)
        bad = dict(_batch(), boom=True)
        with self.assertRaisesRegex(ValueError, "history read failed"):
            remote(bad)
        self.assertIn("history_rgb", remote(_batch()))

    def _captured_transform(self, subprocess):
        captured = {}

        def fake_open(sources, **kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        cfg = OmegaConf.create({
            "common": {"num_video_frames": 5, "video_height": 96, "video_width": 128},
            "training": {"batch_size": 8},
            "dataset": {"sources": []},
            "system": {"ready_batch_queue_enabled": True,
                       "ready_batch_transform_subprocess": subprocess},
        })
        with mock.patch.object(factory, "_indexed_sources", lambda c: [{"name": "s"}]), \
                mock.patch.object(factory, "open_mixed_training_data", fake_open):
            factory.create_training_data_pipeline(cfg)
        return captured["batch_transform"]

    def test_switch_off_keeps_in_thread_function(self):
        t = self._captured_transform(False)
        self.assertNotIsInstance(t, factory._SubprocessTransform)
        self.assertIn("history_rgb", t(_batch()))

    def test_switch_on_uses_subprocess(self):
        t = self._captured_transform(True)
        self.addCleanup(t._pool.shutdown)
        self.assertIsInstance(t, factory._SubprocessTransform)

    def test_no_transform_stays_none(self):
        with mock.patch.object(factory, "build_depth_attacher", lambda cfg: None), \
                mock.patch.object(factory, "_background_history_attacher", lambda cfg, q: None):
            self.assertIsNone(self._captured_transform(True))


class TimingModeTest(unittest.TestCase):
    def test_default_mode_unchanged(self):
        t = TimingStack(sync_cuda=False, enabled=True)
        self.assertFalse(t.use_cuda_events)
        with t.span("outer"):
            with timed_section(t, "inner_seconds"):
                pass
        flat = t.to_flat_dict()
        self.assertEqual(set(flat), {"outer_seconds", "inner_seconds"})
        t.resolve_pending(sync=True)
        self.assertEqual(t.to_flat_dict(), flat)

    def test_events_requested_without_cuda_falls_back(self):
        with mock.patch.object(torch.cuda, "is_available", lambda: False):
            t = TimingStack(enabled=True, use_cuda_events=True)
        self.assertFalse(t.use_cuda_events)
        with t.span("a"):
            pass
        self.assertIn("a_seconds", t.to_flat_dict())
        self.assertNotIn("a_wall_seconds", t.to_flat_dict())

    def test_disabled_is_noop(self):
        t = TimingStack(enabled=False)
        with t.span("a"):
            pass
        self.assertEqual(t.to_flat_dict(), {})

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_events_record_gpu_and_wall(self):
        t = TimingStack(enabled=True, use_cuda_events=True)
        x = torch.randn(2048, 2048, device="cuda")
        with t.span("mm"):
            with timed_section(t, "inner_seconds"):
                for _ in range(10):
                    x = x @ x
        self.assertNotIn("mm_seconds", t.to_flat_dict())
        t.resolve_pending(sync=True)
        flat = t.to_flat_dict()
        self.assertGreater(flat["mm_seconds"], 0.0)
        self.assertGreater(flat["inner_seconds"], 0.0)
        self.assertIn("mm_wall_seconds", flat)


if __name__ == "__main__":
    unittest.main()
