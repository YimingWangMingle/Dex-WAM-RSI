"""History read optimisations: one valid-row read per episode and batch,
producer-side attach, valid-only VAE.

Reference for the read is the original code (per sample, one read per slot
group, invalid rows read as frame 0), patched in for the same inputs; outputs
must be equal.
"""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import _rsi_cpu_stubs  # noqa: F401
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from thinker_wm.data_prepare.history_attach import HISTORY_KEYS, HistoryAttacher
from thinker_wm.data_prepare.history_memory import EventSpec, pad_recent_pixels
from thinker_wm.data_prepare.temporal_windows import WindowSpec, select_window
from thinker_wm.models.history_encoder import HistoryEncoder
from thinker_wm.models.temporal_training import encode_history_frames

SPEC = WindowSpec(control_hz=50, num_video_frames=16, sparse_jitter=0)
EVENT = EventSpec(enabled=True, gripper_indices=(0,), after_s=.08)


def _old_read(self, plans, hw):
    res = {}
    for i, (reader, ep, w) in plans.items():
        s, r = reader.rgb(ep, w.sparse_rows.tolist(), hw), reader.rgb(ep, w.recent_rows.tolist(), hw)
        res[i] = None if s is None or r is None else (s, r)
    return res


def _grip(row):
    return 1.0 if (row // 37) % 2 == 0 else 0.0


class Reader:
    """Row-dependent pixels/state so a wrong slot shows up as a different value."""

    def __init__(self, missing=()):
        self.calls, self.missing = [], set(missing)

    def fps(self, ep):
        return 50.0

    def rgb(self, ep, rows, hw):
        self.calls.append(list(rows))
        if ep in self.missing:
            return None
        g = torch.Generator().manual_seed(7)
        base = torch.rand(3, *hw, generator=g)
        return torch.stack([(base + int(r) / 997.0) % 1.0 for r in rows]) if rows else torch.zeros(0, 3, *hw)

    def state(self, ep, rows):
        return torch.tensor([[_grip(int(r)), int(r) / 100.0] for r in rows], dtype=torch.float32)


def _batch(decisions, hw=(8, 12), eps=None):
    b = len(decisions)
    return dict(video_frames=torch.zeros(b, 16, 3, *hw),
                episode_ids=eps or [f"ep{i}" for i in range(b)],
                frame_indices=[list(range(d, d + 65, 4)) for d in decisions],
                source_datasets=["test"] * b)


def _attach(spec, reader, batch, read=None, events=EVENT, strict=True):
    att = HistoryAttacher(spec, {"test": reader}, seed=5, events=events, strict_sources=strict)
    if read is None:
        return att.attach(batch, 0), att
    with mock.patch.object(HistoryAttacher, "_read_rgb", read):
        return att.attach(batch, 0), att


# decision 0: empty history; 8: leading recent padding; then mid/late episode
# with events and the sparse/event budget.
DECISIONS = [0, 4, 8, 16, 20, 60, 120, 333, 900]


SAME_EP = ["ep0"] * len(DECISIONS)
MIXED_EP = ["ep0", "ep1", "ep0", "ep2", "ep1", "ep0", "ep2", "ep1", "ep0"]


class ReadValidRowsOnly(unittest.TestCase):
    def _compare(self, spec, events=EVENT, eps=None):
        old, oa = _attach(spec, Reader(), _batch(DECISIONS, eps=eps), read=_old_read, events=events)
        new, na = _attach(spec, Reader(), _batch(DECISIONS, eps=eps), events=events)
        for k in HISTORY_KEYS:
            self.assertTrue(torch.equal(old[k], new[k]), k)
            self.assertEqual(old[k].dtype, new[k].dtype, k)
        self.assertEqual(oa.last_metrics, na.last_metrics)
        return new

    def test_bitwise_equal_same_and_mixed_episodes(self):
        for eps in (SAME_EP, MIXED_EP):
            self._compare(SPEC, eps=eps)
            self._compare(replace(SPEC, include_episode_start=False), eps=eps)

    def test_one_read_per_episode_in_a_batch(self):
        reader = Reader()
        _attach(SPEC, reader, _batch(DECISIONS, eps=MIXED_EP))
        self.assertEqual(len(reader.calls), 3)
        reader = Reader()
        _attach(SPEC, reader, _batch(DECISIONS, eps=SAME_EP))
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(reader.calls[0], sorted(set(reader.calls[0])))

    def test_empty_episode_next_to_nonempty_one(self):
        reader = Reader()
        out, _ = _attach(SPEC, reader, _batch([0, 333], eps=["ep0", "ep1"]))
        self.assertIn([0], reader.calls)
        self.assertEqual(len(reader.calls), 2)
        self.assertFalse(out["history_recent_valid"][0].any())
        self.assertTrue(out["history_recent_valid"][1].all())

    def test_missing_episode_in_mixed_batch(self):
        with self.assertRaisesRegex(ValueError, "missing history episode ep1"):
            _attach(SPEC, Reader(missing={"ep1"}), _batch(DECISIONS, eps=MIXED_EP))
        old, _ = _attach(SPEC, Reader(missing={"ep1"}), _batch(DECISIONS, eps=MIXED_EP),
                         read=_old_read, strict=False)
        new, _ = _attach(SPEC, Reader(missing={"ep1"}), _batch(DECISIONS, eps=MIXED_EP), strict=False)
        for k in HISTORY_KEYS:
            self.assertTrue(torch.equal(old[k], new[k]), k)

    def test_bitwise_equal_default_spec(self):
        h = self._compare(SPEC)
        self.assertFalse(h["history_recent_valid"][0].any())       # decision 0: all empty
        self.assertFalse(h["history_sparse_valid"][0].any())
        self.assertEqual(h["history_recent_valid"][2].tolist(), [False, False, False, True, True])

    def test_bitwise_equal_episode_start_and_long_recent(self):
        for spec in (replace(SPEC, include_episode_start=False),
                     replace(SPEC, recent_frames=9), replace(SPEC, recent_frames=13),
                     replace(SPEC, recent_frames=9, include_episode_start=False)):
            self._compare(spec)
            self._compare(spec, events=EventSpec(enabled=False))

    def test_one_read_of_sorted_valid_rows(self):
        spec = replace(SPEC, include_episode_start=False)
        reader = Reader()
        out, _ = _attach(spec, reader, _batch([120]))
        w = select_window(replace(spec, require_full_future=False), 121, 120, jitter=False)
        self.assertEqual(len(reader.calls), 1)
        rows = reader.calls[0]
        self.assertEqual(rows, sorted(set(rows)))
        self.assertTrue(set(w.recent_rows.tolist()) <= set(rows))
        n_valid = int(out["history_sparse_valid"].sum() + out["history_recent_valid"].sum())
        self.assertEqual(len(rows), n_valid)
        self.assertLess(len(rows), spec.history_slots + spec.recent_frames)

    def test_all_invalid_window_still_touches_the_episode(self):
        reader = Reader()
        _attach(SPEC, reader, _batch([0]))
        self.assertEqual(reader.calls, [[0]])

    def test_missing_episode_fails_like_before(self):
        with self.assertRaisesRegex(ValueError, "missing history episode"):
            _attach(SPEC, Reader(missing={"ep0"}), _batch([0]))
        with self.assertRaisesRegex(ValueError, "missing history episode"):
            _attach(SPEC, Reader(missing={"ep1"}), _batch([120, 333]))
        old, _ = _attach(SPEC, Reader(missing={"ep1"}), _batch([120, 333]), read=_old_read, strict=False)
        new, _ = _attach(SPEC, Reader(missing={"ep1"}), _batch([120, 333]), strict=False)
        for k in HISTORY_KEYS:
            self.assertTrue(torch.equal(old[k], new[k]), k)

    def test_reader_error_propagates_for_empty_window(self):
        class Broken(Reader):
            def rgb(self, ep, rows, hw):
                raise OSError("corrupt video")
        with self.assertRaisesRegex(OSError, "corrupt"):
            _attach(SPEC, Broken(), _batch([0]))

    def test_internal_recent_hole_restored_before_padding(self):
        reader = Reader()
        w = select_window(SPEC, 500, 400, jitter=False)
        w = replace(w, recent_valid=np.array([False, True, False, True, True]))
        att = HistoryAttacher(SPEC, {"test": reader}, seed=5)
        o_s, o_r = _old_read(att, {0: (reader, "ep", w)}, (4, 4))[0]
        n_s, n_r = att._read_rgb({0: (reader, "ep", w)}, (4, 4))[0]
        self.assertEqual(n_r.shape, o_r.shape)
        self.assertTrue(torch.equal(pad_recent_pixels(o_r, w.recent_valid),
                                    pad_recent_pixels(n_r, w.recent_valid)))
        m = torch.from_numpy(w.sparse_valid)[:, None, None, None]
        self.assertTrue(torch.equal(o_s * m, n_s * m))


class ProducerAttach(unittest.TestCase):
    def _cfg(self, jitter):
        return OmegaConf.create(dict(
            common=dict(control_hz=50, video_action_freq_ratio=4, num_video_frames=16),
            model=dict(wm=dict(history_enable=True)),
            history=dict(enable=True, sources=[], seed=1, sparse_jitter=jitter,
                         include_episode_start=False)))

    def test_background_only_without_jitter_and_with_queue(self):
        from thinker_wm.data_prepare.factory import _background_history_attacher
        self.assertIsNotNone(_background_history_attacher(self._cfg(0.0), True))
        self.assertIsNone(_background_history_attacher(self._cfg(0.0), False))
        self.assertIsNone(_background_history_attacher(self._cfg(0.25), True))

    def test_producer_then_trainer_matches_foreground(self):
        from thinker_wm.data_prepare.factory import _producer_transform
        fg, fg_att = _attach(SPEC, Reader(), _batch(DECISIONS))
        producer = HistoryAttacher(SPEC, {"test": Reader()}, seed=5, events=EVENT, strict_sources=True)
        produced = _producer_transform(None, producer)(_batch(DECISIONS))
        trainer_reader = Reader()
        trainer = HistoryAttacher(SPEC, {"test": trainer_reader}, seed=5, events=EVENT, strict_sources=True)
        seen = trainer.attach(produced, step=4321)
        self.assertEqual(trainer_reader.calls, [])
        for k in HISTORY_KEYS:
            self.assertTrue(torch.equal(fg[k], seen[k]), k)
        self.assertEqual(trainer.last_metrics, fg_att.last_metrics)
        producer.attach(_batch([0, 0]), 0)                    # producer moves on
        self.assertEqual(trainer.last_metrics, fg_att.last_metrics)

    def test_no_transform_when_nothing_attached(self):
        from thinker_wm.data_prepare.factory import _producer_transform
        self.assertIsNone(_producer_transform(None, None))


def _fake_encode(x):
    """Per-sample VAE stand-in: [N,3,1,H,W] -> [N,4,1,H/2,W/2], batch-independent."""
    pooled = torch.nn.functional.avg_pool3d(x.float(), (1, 2, 2))
    return torch.cat([pooled, pooled[:, :1] ** 2], dim=1).to(x.dtype)


class EncodeValidOnly(unittest.TestCase):
    def _inputs(self, valid):
        g = torch.Generator().manual_seed(3)
        b, th = valid.shape
        return torch.rand(b, th, 3, 8, 12, generator=g), valid

    def test_latents_and_tokens_match(self):
        valid = torch.tensor([[False, True, False, True], [False, False, False, False],
                              [True, True, True, True]])
        px, v = self._inputs(valid)
        old = encode_history_frames(_fake_encode, px, v, torch.float32)
        new = encode_history_frames(_fake_encode, px, v, torch.float32, valid_only=True)
        self.assertEqual(old.shape, new.shape)
        m = v[:, None, :, None, None].expand_as(old)
        self.assertTrue(torch.equal(old[m], new[m]))
        self.assertEqual(int(new[~m].count_nonzero()), 0)
        torch.manual_seed(0)
        enc = HistoryEncoder(16, 1, pool=2)
        patch = torch.nn.Conv3d(4, 16, (1, 2, 2), stride=(1, 2, 2))
        times = -torch.arange(4, 0, -1, dtype=torch.float32).expand(3, 4) * .3
        kind = torch.ones(3, 4, dtype=torch.long)
        state = torch.randn(3, 4, 1)
        with torch.no_grad():
            t_old = enc(patch, old, times, v, kind=kind, state=state)
            t_new = enc(patch, new, times, v, kind=kind, state=state)
        self.assertTrue(torch.equal(t_old, t_new))

    def test_all_invalid_batch_keeps_shape(self):
        px, v = self._inputs(torch.zeros(2, 3, dtype=torch.bool))
        calls = []

        def enc(x):
            calls.append(x.shape[0])
            return _fake_encode(x)
        new = encode_history_frames(enc, px, v, torch.float32, valid_only=True)
        self.assertEqual(calls, [1])
        self.assertEqual(new.shape, encode_history_frames(_fake_encode, px, v, torch.float32).shape)
        self.assertEqual(int(new.count_nonzero()), 0)

    def test_only_valid_slots_reach_the_encoder(self):
        valid = torch.tensor([[False, True, False, True], [True, False, False, False]])
        px, v = self._inputs(valid)
        calls = []

        def enc(x):
            calls.append(x.shape[0])
            return _fake_encode(x)
        encode_history_frames(enc, px, v, torch.float32, valid_only=True)
        self.assertEqual(calls, [3])

    def test_switch_reaches_model_config(self):
        from thinker_wm.models import temporal_training as TT
        cfg = OmegaConf.create(dict(model=dict(wm=dict(history_encode_valid_only=True))))
        self.assertEqual(TT.motus_temporal_kwargs(cfg), dict(history_encode_valid_only=True))


if __name__ == "__main__":
    unittest.main()
