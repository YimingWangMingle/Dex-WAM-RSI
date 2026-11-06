import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import _rsi_cpu_stubs  # noqa
import unittest
from types import SimpleNamespace
import torch

from thinker_wm.models.temporal_training import encode_history_frames
from thinker_wm.runtime.sparse_vae_cache import SparseVAECache
import _rsi_harness as H
from thinker_wm.data_prepare.temporal_windows import WindowSpec
from thinker_wm.models.inference_api import Observation
from thinker_wm.runtime.history_buffer import HistoryBuffer
from thinker_wm.data_prepare.history_memory import EventSpec
import numpy as np


def history_fixture():
    """Model/cache fixture with no simulator or remote policy dependency."""
    model = H.Harness(H.config(num_video_frames=16, action_t_downsample=1, control_hz=50.,
        history_enable=True, ar_packed_teacher_forcing=True,
        ar_target_reads_earlier_targets=False, action_dim=14,
        action_state_dim=14, unified_action_dim=14)).eval()
    spec = WindowSpec(control_hz=50., num_video_frames=16, sparse_jitter=0.,
        include_episode_start=False, require_full_future=False)
    return SimpleNamespace(inf=SimpleNamespace(model=model), history=HistoryBuffer(spec, None))


def measured_observation(step, grip):
    rng = np.random.default_rng(step)
    rgb = torch.from_numpy(rng.random((3, H.H_PIX, H.W_PIX), dtype=np.float32))
    state = torch.linspace(-.5, .5, 14) + .001 * step
    state[[6, 13]] = grip
    return Observation(rgb=rgb, state=state, step=step, time_s=step / 50., robot_name=None)


class Model:
    device = torch.device("cpu")
    dtype = torch.float32
    training = False

    def __init__(self):
        self.encoded = 0
        self.config = SimpleNamespace(history_encode_valid_only=True)

    def encode_video(self, x):
        self.encoded += len(x)
        return torch.nn.functional.avg_pool3d(x, (1, 2, 2))


class Cache(unittest.TestCase):
    def test_original_batch_shape_on_miss_then_no_encode_on_hit(self):
        m, c = Model(), SparseVAECache(8)
        m.config.history_encode_valid_only = False
        px = torch.rand(1, 8, 3, 8, 12)
        valid = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        rows = list(range(8))
        a = c.encode(m, px, valid, rows)
        self.assertEqual(m.encoded, 8)
        b = c.encode(m, px, valid, rows)
        self.assertEqual(m.encoded, 8)
        self.assertTrue(torch.equal(a, b))
        rows[:2] = [1, 0]
        c.encode(m, px[:, [1, 0, 2, 3, 4, 5, 6, 7]], valid, rows)
        self.assertEqual(m.encoded, 16)  # position change reuses original batch shape

    def test_equivalence_reuse_eviction_and_empty(self):
        m, c = Model(), SparseVAECache(4)
        source = torch.rand(20, 3, 8, 12)
        for rows, mask in [([0, 1, 2, 3], [0, 0, 0, 0]), ([1, 2, 3, 4], [1, 1, 0, 0]),
                           ([1, 2, 4, 5], [1, 1, 1, 1]), ([6, 7, 8, 9], [1, 1, 1, 1]),
                           ([1, 2, 4, 5], [1, 1, 1, 1])]:
            px = source[rows][None]
            v = torch.tensor([mask], dtype=torch.bool)
            ref = encode_history_frames(Model().encode_video, px, v, m.dtype, valid_only=True)
            got = c.encode(m, px, v, rows)
            self.assertTrue(torch.equal(ref, got))
            self.assertLessEqual(len(c.entries), 4)
        self.assertGreaterEqual(c.hits, 2)
        c.clear()
        self.assertEqual(c.stats()["entries"], 0)

    def test_training_and_capacity_guard(self):
        m = Model(); m.training = True
        with self.assertRaisesRegex(ValueError, "eval mode"):
            SparseVAECache().encode(m, torch.rand(1, 2, 3, 8, 8), torch.ones(1, 2).bool(), [0, 1])
        with self.assertRaises(ValueError):
            SparseVAECache(0)

    def test_history_times_rebuilt_and_reset_and_model_change(self):
        p = history_fixture()
        p.history = HistoryBuffer(p.history.spec, None, events=EventSpec(enabled=True))
        p.history.enable_sparse_vae_cache = True
        m = p.inf.model
        for s in range(101):
            p.history.add(measured_observation(s, grip=0.0 if s >= 40 else 1.0))
        old_times = None
        for s in (100, 116):
            if s == 116:
                for row in range(101, 117):
                    p.history.add(measured_observation(row, grip=0.0))
            raw = p.history.history_inputs(s)
            ref = m._prepare_history(raw, None)
            got = p.history.context_for(m, s, "checkpoint")
            valid = got["valid"][:, None, :, None, None].expand_as(got["latent"])
            # The tiny VAE uses einsum: changing its batch size may alter FP32
            # accumulation at ~1e-8. Real CUDA VAE/policy checks are separate.
            torch.testing.assert_close(ref["latent"][valid], got["latent"][valid], rtol=1e-6, atol=1e-7)
            for k in ("times", "valid", "kind", "state"):
                torch.testing.assert_close(ref[k], got[k], rtol=0, atol=0)
            if old_times is not None:
                self.assertFalse(torch.equal(old_times, got["times"]))
            old_times = got["times"].clone()
        # Force rebuilding the time/state context with the same sparse slots;
        # changed slot positions intentionally miss the conservative cache.
        p.history._cache_key = None
        p.history.context_for(m, 116, "checkpoint")
        self.assertGreater(p.history.sparse_cache_stats()["hits"], 0)
        p.history.context_for(history_fixture().inf.model, 116, "checkpoint")
        self.assertEqual(p.history.sparse_cache_stats()["hits"], 0)
        p.history.reset("new")
        self.assertEqual(p.history.sparse_cache_stats()["entries"], 0)


if __name__ == "__main__":
    unittest.main()
