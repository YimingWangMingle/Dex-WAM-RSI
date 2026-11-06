"""Prediction alignment: time base, command binding/rewrite/interrupt, expiry, cache.

Run: python scripts/test_prediction_alignment.py   (CPU, tiny real UBT_WM)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _rsi_cpu_stubs  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _rsi_harness as H  # noqa: E402
from thinker_wm.data_prepare.temporal_windows import WindowSpec  # noqa: E402
from thinker_wm.models.inference_api import (KIND_CONDITIONED, KIND_JOINT, Forecast, GenerationSettings,  # noqa: E402
                                  Observation, PolicyOutput, WAMInference, action_hash)
from thinker_wm.runtime.action_postprocess import ActionPostprocessor, PostprocessConfig  # noqa: E402
from data.recovery import segments_from_trajectory  # noqa: E402
from thinker_wm.runtime.episode_runner import ChunkedController, ControllerConfig, EpisodeRunner  # noqa: E402
from thinker_wm.runtime.history_buffer import HistoryBuffer  # noqa: E402
from thinker_wm.runtime.prediction_store import PredictionStore  # noqa: E402
from execution.trajectory import TrajectoryLogger, read_trajectory  # noqa: E402


def obs(step, value=None, robot=None, g=None):
    rgb = torch.full((3, H.H_PIX, H.W_PIX), value) if value is not None else \
        torch.rand(3, H.H_PIX, H.W_PIX, generator=g)
    return Observation(rgb=rgb, state=torch.zeros(8) + step * 0.01, step=step, time_s=step / 25.0,
                       robot_name=robot)


def model(**over):
    m = H.Harness(H.config(**over))
    m.eval()
    return m


class RealInference(unittest.TestCase):
    def setUp(self):
        self.m = model(history_enable=True)
        self.inf = WAMInference(self.m, checkpoint="ckpt-test", action_stats_version="stats-v1",
                                camera_transform="cam-v1",
                                settings=GenerationSettings(num_inference_steps=2, seed=5, decode_outputs=True))

    def test_time_base_and_hash(self):
        o = obs(40, g=torch.Generator().manual_seed(0))
        out = self.inf.generate_action(o, None, "pick")
        fc = out.forecast
        self.assertEqual(out.actions.shape, (32, 8))
        self.assertEqual(fc.kind, KIND_JOINT)
        self.assertEqual(fc.frame_steps, [40 + 4 * n for n in range(1, 9)])
        self.assertEqual(fc.latent_steps, [56, 72])
        self.assertEqual(fc.action_hash, action_hash(out.actions))
        self.assertEqual(tuple(fc.frames.shape[2:3]), (8,))
        self.assertIsNotNone(fc.video_latent)

    def test_deterministic_per_step(self):
        o = obs(8, g=torch.Generator().manual_seed(1))
        a = self.inf.generate_action(o, None, "pick").actions
        b = self.inf.generate_action(o, None, "pick").actions
        np.testing.assert_array_equal(a, b)

    def test_predict_outcome_keeps_plan(self):
        o = obs(0, g=torch.Generator().manual_seed(2))
        plan = np.random.default_rng(0).normal(size=(32, 8)).astype(np.float32)
        fc = self.inf.predict_outcome(o, None, "pick", plan)
        self.assertEqual(fc.kind, KIND_CONDITIONED)
        np.testing.assert_array_equal(fc.actions, plan)
        self.assertEqual(fc.action_hash, action_hash(plan))
        with self.assertRaises(ValueError):
            self.inf.predict_outcome(o, None, "pick", plan[:16])

    def test_visual_compare_requires_decoded_frames(self):
        quiet = WAMInference(self.m, checkpoint="c", action_stats_version="s", camera_transform="t",
                             settings=GenerationSettings(num_inference_steps=2))
        with self.assertRaises(ValueError):
            ChunkedController(quiet, ActionPostprocessor(PostprocessConfig()), "pick")
        ChunkedController(quiet, ActionPostprocessor(PostprocessConfig()), "pick",
                          config=ControllerConfig(compare_visual=False))
        ChunkedController(self.inf, ActionPostprocessor(PostprocessConfig()), "pick")

    def test_refuses_predicted_observation(self):
        o = obs(0, 0.5)
        o.source = "predicted"
        with self.assertRaises(ValueError):
            self.inf.generate_action(o, None, "pick")

    def test_history_buffer_context_and_cache(self):
        hb = HistoryBuffer(WindowSpec(), robot_name=None)
        g = torch.Generator().manual_seed(3)
        for s in range(0, 61):
            hb.add(obs(s, g=g))
        k1 = hb.cache_key(60, "ckpt-test")
        c1 = hb.context_for(self.m, 60, "ckpt-test")
        self.assertIs(hb.context_for(self.m, 60, "ckpt-test"), c1)
        self.assertNotEqual(hb.cache_key(60, "other-ckpt"), k1)
        a_hist = self.inf.generate_action(obs(60, 0.3), hb, "pick").actions
        hb.add(obs(61, g=g))
        self.assertNotEqual(hb.cache_key(61, "ckpt-test"), k1)
        hb2 = HistoryBuffer(WindowSpec(), robot_name=None)
        hb2.add(obs(60, 0.3))
        a_none = self.inf.generate_action(obs(60, 0.3), None, "pick").actions
        self.assertFalse(np.allclose(a_hist, a_none))


def fake_forecast(kind, d, actions, frames=None):
    return Forecast(kind=kind, decision_step=d, decision_time_s=d / 25, actions=actions,
                    action_hash=action_hash(actions), action_stats_version="s", camera_transform="c",
                    checkpoint="k", settings={}, frame_steps=[d + 4 * n for n in range(1, 9)],
                    latent_steps=[d + 16, d + 32], frames=frames)


class FakeInference:
    ratio, num_frames = 4, 8

    def __init__(self, scale=1.0):
        self.calls, self.predict_calls, self.scale = [], [], scale

    def generate_action(self, observation, history, instruction, recovery=None):
        d = observation.step
        self.calls.append(d)
        a = np.full((32, 8), self.scale, dtype=np.float32)
        frames = torch.stack([torch.full((3, H.H_PIX, H.W_PIX), n / 10) for n in range(1, 9)], 1)[None]
        return PolicyOutput(a, fake_forecast(KIND_JOINT, d, a, frames), {})

    def predict_outcome(self, observation, history, instruction, executed_plan):
        d = observation.step
        self.predict_calls.append(d)
        frames = torch.stack([torch.full((3, H.H_PIX, H.W_PIX), n / 10) for n in range(1, 9)], 1)[None]
        return fake_forecast(KIND_CONDITIONED, d, np.asarray(executed_plan, np.float32), frames)


class Store(unittest.TestCase):
    def test_rewrite_invalidates_joint_forecast(self):
        a = np.full((32, 8), 2.0, np.float32)
        post = ActionPostprocessor(PostprocessConfig(high=(1.0,) * 8))
        cmd = post.finalize(a, 0)
        st = PredictionStore()
        b = st.bind(cmd, fake_forecast(KIND_JOINT, 0, a))
        self.assertIsNone(b.forecast)
        self.assertIn("re-predict", b.invalid_reason)
        with self.assertRaises(ValueError):
            st.attach_forecast(0, fake_forecast(KIND_CONDITIONED, 0, a))       # wrong actions
        st.attach_forecast(0, fake_forecast(KIND_CONDITIONED, 0, cmd.actions))
        st.dispatch(0, 0)
        with self.assertRaises(ValueError):
            st.attach_forecast(0, fake_forecast(KIND_CONDITIONED, 0, cmd.actions))

    def test_identity_postprocess_keeps_forecast(self):
        a = np.zeros((32, 8), np.float32)
        cmd = ActionPostprocessor(PostprocessConfig()).finalize(a, 4)
        st = PredictionStore()
        self.assertIsNotNone(st.bind(cmd, fake_forecast(KIND_JOINT, 4, a)).forecast)
        with self.assertRaises(ValueError):
            st.bind(cmd, fake_forecast(KIND_JOINT, 0, a))                       # different decision

    def test_usable_only_inside_executed_prefix(self):
        a = np.zeros((32, 8), np.float32)
        st = PredictionStore()
        st.bind(ActionPostprocessor(PostprocessConfig()).finalize(a, 0), fake_forecast(KIND_JOINT, 0, a))
        for k in range(4):
            st.observe(k, k / 25)
            st.dispatch(k, k)
        st.observe(4, 4 / 25)
        self.assertIsNotNone(st.usable(0, 4))
        st.observe(16, 16 / 25)
        self.assertIsNone(st.usable(0, 16))          # rows 4..15 never executed
        with self.assertRaises(ValueError):
            st.dispatch(6, 5)                         # out of order / wrong time base
        st.interrupt("test")
        self.assertEqual(st.binding(0).status, "interrupted")
        self.assertEqual(st.binding(0).executed_prefix, 4)

    def test_expiry(self):
        a = np.zeros((32, 8), np.float32)
        st = PredictionStore()
        post = ActionPostprocessor(PostprocessConfig())
        st.bind(post.finalize(a, 0), fake_forecast(KIND_JOINT, 0, a))
        st.bind(post.finalize(a, 16), fake_forecast(KIND_JOINT, 16, a))
        self.assertEqual(st.expire_before(20), 0)    # forecast at 0 covers rows up to 32
        self.assertEqual(st.expire_before(40), 1)


class Controller(unittest.TestCase):
    def test_execute_16_and_latent_compare_on_decoded_frame(self):
        inf = FakeInference()
        ctl = ChunkedController(inf, ActionPostprocessor(PostprocessConfig()), "pick",
                                tracking_fn=lambda c, s: float(np.abs(c[0] - s[0])))
        ctl.begin("ep")
        for s in range(0, 33):
            ctl.act(obs(s, 0.4))
        self.assertEqual(inf.calls, [0, 16, 32])
        self.assertEqual(inf.predict_calls, [])
        # row 16 = latent 1 = frame 4 (value 0.4): compared before the replan; 16 tracked steps
        errs = ctl.last_chunk_errors
        self.assertEqual([e["step"] for e in errs if e["kind"] == "visual"], [32])
        self.assertEqual(sum(e["kind"] == "track" for e in errs), 16)
        self.assertEqual(len(ctl.decisions), 3)

    def test_visual_error_uses_frame_of_latent(self):
        inf = FakeInference()
        ctl = ChunkedController(inf, ActionPostprocessor(PostprocessConfig()), "pick",
                                config=ControllerConfig(execute_steps=32))
        ctl.begin("ep")
        for s in range(0, 17):
            ctl.act(obs(s, 0.4))
        self.assertEqual(ctl.tracker.visual, [0.0])
        for s in range(17, 33):
            ctl.act(obs(s, 0.4))
        vis = [e for e in ctl.last_chunk_errors if e["kind"] == "visual"]
        self.assertEqual([e["step"] for e in vis], [16, 32])
        self.assertAlmostEqual(vis[-1]["value"], 0.16, places=5)   # frame 8 = 0.8 vs 0.4 -> 0.16

    def test_rewritten_command_is_reforecast(self):
        inf = FakeInference(scale=3.0)
        ctl = ChunkedController(inf, ActionPostprocessor(PostprocessConfig(high=(1.0,) * 8)), "pick")
        ctl.begin("ep")
        row = ctl.act(obs(0, 0.4))
        np.testing.assert_array_equal(row, np.ones(8, np.float32))
        self.assertEqual(inf.predict_calls, [0])
        self.assertEqual(ctl.store.binding(0).forecast.kind, KIND_CONDITIONED)

    def test_unit_scale_stays_out_of_model_units(self):
        inf = FakeInference(scale=3.0)
        post = ActionPostprocessor(PostprocessConfig(high=(1.0,) * 8, unit_scale=(10.0,) * 8))
        ctl = ChunkedController(inf, post, "pick")
        ctl.begin("ep")
        np.testing.assert_array_equal(ctl.act(obs(0, 0.4)), np.full(8, 10.0, np.float32))
        np.testing.assert_array_equal(ctl.store.binding(0).forecast.actions, np.ones((32, 8)))
        # A pure unit conversion is not a rewrite: the joint forecast stays bound.
        inf = FakeInference()
        ctl = ChunkedController(inf, ActionPostprocessor(PostprocessConfig(unit_scale=(10.0,) * 8)), "pick")
        ctl.begin("ep")
        ctl.act(obs(0, 0.4))
        self.assertEqual(inf.predict_calls, [])
        self.assertEqual(ctl.store.binding(0).forecast.kind, KIND_JOINT)

    def test_begin_resets_history_and_store(self):
        hb = HistoryBuffer(WindowSpec(), robot_name=None)
        ctl = ChunkedController(FakeInference(), ActionPostprocessor(PostprocessConfig()), "pick", history=hb)
        ctl.begin("a")
        for s in range(5):
            ctl.act(obs(s, 0.1))
        ctl.end()
        ctl.begin("b")
        self.assertEqual(len(hb), 0)
        self.assertIsNone(ctl.store.active)
        ctl.act(obs(0, 0.1))
        self.assertEqual(len(hb), 1)


class StepEnv:
    def __init__(self, done_after=None):
        self.done_after, self.n = done_after, 0

    def reset(self):
        self.n = 0
        return 0

    def step(self, action):
        self.n += 1
        return self.n, self.done_after is not None and self.n >= self.done_after, dict(success=False)


class StepAdapter:
    def observation(self, live, step):
        assert live == step
        return obs(step, 0.4), None


class Runner(unittest.TestCase):
    """Logged history + planning; final feedback after done / budget; executed rows."""

    def run_episode(self, root, env, max_steps):
        hb = HistoryBuffer(WindowSpec(), robot_name=None)
        inf = FakeInference()
        lg = TrajectoryLogger(root, "ep", dict(checkpoint="k", action_stats_version="s",
                                               camera_transform="c", runtime="r"))
        ctl = ChunkedController(inf, ActionPostprocessor(PostprocessConfig()), "pick", history=hb, logger=lg)
        summary = EpisodeRunner(ctl, StepAdapter(), max_steps).run(env, "ep")
        lg.close()
        return summary, inf, hb, read_trajectory(os.path.join(root, "ep"))

    def test_done_mid_chunk(self):
        with tempfile.TemporaryDirectory() as root:
            summary, inf, hb, recs = self.run_episode(root, StepEnv(done_after=20), max_steps=100)
            self.assertEqual((summary["status"], summary["steps"]), ("terminated", 20))
            self.assertEqual(inf.calls, [0, 16])                  # final feedback never plans
            self.assertEqual(len(hb), 21)                        # history includes the last feedback
            self.assertEqual(sum(r["role"] == "history" for r in recs), 2)
            measured = [r for r in recs if r["role"] == "measured"]
            self.assertEqual((measured[-1]["step"], measured[-1]["executed_row"]), (20, 19))
            segs = segments_from_trajectory(os.path.join(root, "ep"), outcome="success",
                                            split="development", task="t")
            self.assertEqual([(s.decision_step, s.end_step) for s in segs], [(0, 15), (16, 19)])

    def test_budget_exhausted(self):
        with tempfile.TemporaryDirectory() as root:
            summary, inf, _, recs = self.run_episode(root, StepEnv(), max_steps=10)
            self.assertEqual((summary["status"], summary["steps"]), ("budget", 10))
            self.assertEqual(inf.calls, [0])
            measured = [r for r in recs if r["role"] == "measured"]
            self.assertEqual((measured[-1]["step"], measured[-1]["executed_row"]), (10, 9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
