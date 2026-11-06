"""PhysicalRSI adapter: single step owner, outcome classes, fixed-model comparison,
candidate revision/inheritance, history reset per episode.

Uses the real ``PhysicalRSI_core`` (ExperimentRuntime) when importable, else
the contract double in ``_physicalrsi_double`` (prints which one ran).

Run: python scripts/test_physicalrsi_adapter.py   (CPU)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _rsi_cpu_stubs  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _physicalrsi_double as D  # noqa: E402
import _rsi_harness as H  # noqa: E402
from thinker_wm.data_prepare.robodojo_core import FieldMap  # noqa: E402
from thinker_wm.data_prepare.temporal_windows import WindowSpec  # noqa: E402
from thinker_wm.models.inference_api import KIND_JOINT, Forecast, GenerationSettings, PolicyOutput, WAMInference, action_hash  # noqa: E402,E501
from rsi.core_adapter import InterfaceSpec, SimEnvironment, TaskSpec, ThinkerWAMSuite  # noqa: E402
from rsi.proposal_adapter import materialize_candidate  # noqa: E402
from rsi.version_manifest import CLOSURE, build_harness, verify_harness, write_wam_version  # noqa: E402
from skills.wam.core import ImageStore, WAMSkill  # noqa: E402
from thinker_wm.runtime.action_postprocess import ActionPostprocessor, PostprocessConfig  # noqa: E402
from thinker_wm.runtime.episode_runner import ChunkedController, ControllerConfig  # noqa: E402
from thinker_wm.runtime.history_buffer import HistoryBuffer  # noqa: E402
from thinker_wm.runtime.recovery_controller import STOP, Decision, RecoveryController  # noqa: E402
from thinker_wm.runtime.robodojo_adapter import ObservationBinding, RoboDojoObservationAdapter  # noqa: E402

try:
    from rsi.core_adapter import load_core
    CORE = load_core()
    from PhysicalRSI_core.embodiment import System2Trial
    from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
    REAL = True
except ImportError:
    CORE, System2Trial, REAL = D.CORE, D.System2Trial, False
print(f"[test_physicalrsi_adapter] core = {'PhysicalRSI_core (real)' if REAL else 'contract double'}")

FM = FieldMap.from_dict({"schema_version": 1, "control_hz": 25, "rgb": {"views": {"cam": "/c"},
                         "encoding": "uint8_thwc"}, "state": {"path": "/s"},
                         "action": {"path": "/a", "semantics": "delta_state"},
                         "layout": {"rows": [["cam"]], "row_hw": [[H.H_PIX, H.W_PIX]]}}, digest="fm-test")
ADAPTER = RoboDojoObservationAdapter(FM, ObservationBinding(rgb_keys={"cam": "cam"}, state_key="state"),
                                     model_hw=(H.H_PIX, H.W_PIX), resize_mode="pad", robot_name=None)
IFACE = InterfaceSpec(robot="tiny", camera_transform=ADAPTER.camera_transform, action_semantics="delta_state",
                      action_unit="rad", control_hz=25.0, execute_steps=16)


class FakeSim:
    def __init__(self, signal=True, limit=60):
        self.signal, self.limit, self.steps = signal, limit, 0

    def identity(self):
        return dict(sim="fake", signal=self.signal, limit=self.limit)

    def _live(self):
        img = np.full((H.H_PIX, H.W_PIX, 3), int(np.clip(self.state[0], 0.0, 1.0) * 255), np.uint8)
        return dict(cam=img, state=self.state.copy())

    def reset(self, case):
        if case["seed"] < 0:
            return None
        self.state, self.goal, self.t = np.zeros(8, np.float32), float(case.get("goal", 0.5)), 0
        return self._live()

    def step(self, a):
        self.steps += 1
        self.t += 1
        self.state = self.state + a
        ok = bool(self.state[0] >= self.goal)
        info = dict(success=ok) if self.signal else {}
        return self._live(), ok or self.t >= self.limit, info


class FakeInference:
    ratio, num_frames = 4, 8

    def __init__(self, step=0.05):
        self.step, self.calls = step, []

    def generate_action(self, observation, history, instruction, recovery=None):
        self.calls.append((observation.step, None if history is None else len(history)))
        a = np.full((32, 8), self.step, np.float32)
        d = observation.step
        fc = Forecast(KIND_JOINT, d, d / 25, a, action_hash(a), "s", "c", "k", {},
                      [d + 4 * n for n in range(1, 9)], [d + 16, d + 32])
        return PolicyOutput(a, fc, {})

    def predict_outcome(self, *a, **k):
        raise AssertionError("identity post-processing never needs re-forecasting")


class StopAt(RecoveryController):
    def __init__(self, step):
        super().__init__()
        self.at, self.n = step, 0

    def decide(self, f):
        self.n += 1
        return Decision(STOP, "test stop", dict(f.values)) if self.n > self.at else super().decide(f)


def make_harness(root: Path, hid: str, control: dict):
    files = {k: [] for k in CLOSURE}
    for k in CLOSURE:
        if k in ("configuration", "control"):
            continue
        p = root / k / "x.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{k} v1\n")
        files[k].append(f"{k}/x.txt")
    write_wam_version(root, checkpoint="ckpt", checkpoint_sha256="0" * 64, action_stats_version="stats",
                      history=dict(recent=5, sparse=8, stride_s=1.0), camera_transform=ADAPTER.camera_transform,
                      data_version="d1", model_config_sha256="1" * 64)
    files["configuration"].append("configuration/wam_version.json")
    (root / "control").mkdir(parents=True, exist_ok=True)
    (root / "control" / "controller.json").write_text(json.dumps(control))
    files["control"].append("control/controller.json")
    return build_harness(root, hid, files)


class Fixture:
    def __init__(self, tmp: Path, inference_factory=FakeInference, recovery_factory=RecoveryController,
                 signal=True, shared_controller=False):
        self.tmp, self.store = tmp, ImageStore(tmp / "images")
        self.inference_factory, self.recovery_factory = inference_factory, recovery_factory
        self.signal, self.shared = signal, shared_controller
        self.sims, self.histories, self.inferences, self._ctl = [], [], [], None

    def env(self, task):
        sim = FakeSim(signal=self.signal)
        self.sims.append(sim)
        return SimEnvironment(sim=sim, adapter=ADAPTER, store=self.store, interface=IFACE,
                              embodiment_name="tiny-sim", embodiment_revision="sim-rev-1", core=CORE)

    def skill(self, candidate, task):
        sha = verify_harness(candidate)
        control = json.loads((Path(candidate["root"]) / "control/controller.json").read_text())

        def controller(instruction):
            if self.shared and self._ctl is not None:
                return self._ctl
            hb = HistoryBuffer(WindowSpec(), robot_name=None)
            self.histories.append(hb)
            inf = self.inference_factory()
            self.inferences.append(inf)
            self._ctl = ChunkedController(inf, ActionPostprocessor(PostprocessConfig()), instruction,
                                          config=ControllerConfig(execute_steps=control["execute_steps"],
                                                                  compare_visual=False),
                                          history=hb, recovery=self.recovery_factory())
            return self._ctl

        obs_c, act_c, timing = IFACE.contracts(CORE)
        return WAMSkill(name="thinkerwam", revision=sha, controller_factory=controller, store=self.store,
                        camera_transform=ADAPTER.camera_transform, observation_contract=obs_c,
                        action_contract=act_c, timing=timing,
                        dependencies=(CORE.Dependency("ckpt", "0" * 64, "model"),), system1_cls=CORE.System1)

    def suite(self, frozen=None):
        tasks = {"lift": TaskSpec("lift the block", development_seeds=(1, 2), validation_pool=(10, 11, 12, 13),
                                  validation_episodes=2, test_seeds=(100, 101))}
        return ThinkerWAMSuite(name="tiny", tasks=tasks, env_factory=self.env, skill_factory=self.skill,
                               frozen_reference=frozen)


def execute(tmp: Path, run_id, suite, candidate, case, max_steps=40, system2=None):
    env, pol, ver = suite.environment(case["task"]), suite.policy(candidate, case["task"]), suite.verifier(case["task"])
    if REAL:
        rt = ExperimentRuntime(tmp / "runs")
        r = rt.run(run_id, task=case["task"], case=case, scope="test", environment=env, policy=pol, verifier=ver,
                   budget=Budget(max_steps=max_steps, seconds=600), system2=system2)
        trace = json.loads((rt.root / run_id / "trajectory.json").read_text())
        return dict(outcome=r["outcome"], termination=r.get("termination"), steps=r["steps"], trace=trace,
                    policy=json.loads((rt.root / run_id / "policy.json").read_text()) if r["outcome"] != "invalid"
                    else None)
    return D.run(task=case["task"], case=case, environment=env, policy=pol, verifier=ver, max_steps=max_steps,
                 system2=system2)


class Adapter(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.parent = make_harness(self.tmp / "h" / "parent", "parent", dict(execute_steps=16))

    def tearDown(self):
        self._t.cleanup()

    def case(self, seed, **kw):
        return dict(task="lift", seed=seed, split="development", instruction="lift the block", **kw)

    def test_single_owner_and_success(self):
        fx = Fixture(self.tmp)
        r = execute(self.tmp, "run-ok", fx.suite(), self.parent, self.case(1, goal=0.5))
        self.assertEqual(r["outcome"], "success")
        self.assertEqual(r["termination"], "environment")
        self.assertEqual(fx.sims[0].steps, r["steps"])                       # one env.step per runtime step
        self.assertEqual(r["steps"], 10)                                     # 10 * 0.05 = 0.5
        self.assertEqual([c[0] for c in fx.inferences[0].calls], [0])       # one chunk, 16 rows executed max
        self.assertTrue(r["policy"]["stopped"])
        for e in r["trace"][1:]:
            self.assertEqual(len(e["action"]["command"]), 8)
            self.assertIn("path", e["observation"]["rgb"])                  # observation by reference

    def test_replans_every_16_steps(self):
        fx = Fixture(self.tmp, inference_factory=lambda: FakeInference(step=0.01))
        r = execute(self.tmp, "run-long", fx.suite(), self.parent, self.case(2, goal=0.35), max_steps=40)
        self.assertEqual(r["outcome"], "success")
        self.assertEqual([c[0] for c in fx.inferences[0].calls], [0, 16, 32])

    def test_outcome_classes(self):
        r = execute(self.tmp, "run-notready", Fixture(self.tmp).suite(), self.parent, self.case(-1))
        self.assertEqual(r["outcome"], "invalid")
        r = execute(self.tmp, "run-nosignal", Fixture(self.tmp, signal=False).suite(), self.parent,
                    self.case(1, goal=0.5), max_steps=12)
        self.assertEqual(r["outcome"], "uncertain")
        fx = Fixture(self.tmp, recovery_factory=lambda: StopAt(3))
        r = execute(self.tmp, "run-stop", fx.suite(), self.parent, self.case(1, goal=5.0))
        self.assertEqual(r["outcome"], "failure")
        self.assertEqual(r["steps"], 4)
        r = execute(self.tmp, "run-budget", Fixture(self.tmp).suite(), self.parent, self.case(1, goal=50.0),
                    max_steps=8)
        self.assertEqual((r["outcome"], r["termination"]), ("failure", "step_budget"))

    def test_revision_binding(self):
        fx = Fixture(self.tmp)
        suite = fx.suite()
        sha = verify_harness(self.parent)
        trial = System2Trial(CORE.Revision("system2", "v1"), "parent", sha, "development")
        r = execute(self.tmp, "run-s2", suite, self.parent, self.case(1), system2=trial)
        self.assertEqual(r["outcome"], "success")
        bad = System2Trial(CORE.Revision("system2", "v1"), "parent", "f" * 64, "development")
        with self.assertRaises(ValueError):
            execute(self.tmp, "run-s2-bad", suite, self.parent, self.case(1), system2=bad)
        wrong = ThinkerWAMSuite(name="x", tasks=suite.tasks, env_factory=fx.env,
                                skill_factory=lambda c, t: type("S", (), {"revision": "0" * 64})())
        with self.assertRaises(ValueError):
            wrong.policy(self.parent, "lift")

    def test_candidate_inheritance_with_fixed_model(self):
        frozen = {k: self.parent["components"][k] for k in ("weights", "configuration")}
        fx = Fixture(self.tmp)
        suite = fx.suite(frozen=frozen)
        cand = materialize_candidate(self.parent, {"control/controller.json": {"execute_steps": 8}},
                                     new_id="cand1", out_root=self.tmp / "h")
        self.assertNotEqual(verify_harness(cand), verify_harness(self.parent))
        self.assertEqual(cand["components"]["weights"], self.parent["components"]["weights"])
        self.assertTrue(suite.admit(cand)["admitted"])
        lineage = json.loads((self.tmp / "h" / "cand1.proposal.json").read_text())
        self.assertEqual(lineage["parent_sha256"], verify_harness(self.parent))
        with self.assertRaises(ValueError):
            materialize_candidate(self.parent, {"weights/x.txt": "new weights"}, new_id="cand2",
                                  out_root=self.tmp / "h")
        rogue_root = self.tmp / "h" / "rogue"
        rogue = make_harness(rogue_root, "rogue", dict(execute_steps=16))
        (rogue_root / "weights" / "x.txt").write_text("other\n")
        rogue = build_harness(rogue_root, "rogue", {k: list(v) for k, v in rogue["components"].items()})
        with self.assertRaises(ValueError):
            suite.admit(rogue)
        r = execute(self.tmp, "run-cand", suite, cand, self.case(2, goal=0.6), max_steps=20)
        self.assertEqual(r["outcome"], "success")
        self.assertEqual([c[0] for c in fx.inferences[-1].calls], [0, 8])   # candidate control in effect

    def test_history_reset_per_episode(self):
        fx = Fixture(self.tmp, shared_controller=True)
        suite = fx.suite()
        execute(self.tmp, "run-h1", suite, self.parent, self.case(1, goal=0.5))
        execute(self.tmp, "run-h2", suite, self.parent, self.case(2, goal=0.5))
        calls = fx.inferences[0].calls
        firsts = [c for c in calls if c[0] == 0]
        self.assertEqual(len(firsts), 2)
        self.assertTrue(all(n == 1 for _, n in firsts))       # only the current observation after reset

    def test_terminal_feedback(self):
        fx = Fixture(self.tmp)
        r = execute(self.tmp, "run-fb-done", fx.suite(), self.parent, self.case(1, goal=0.5))
        self.assertEqual((r["termination"], r["steps"]), ("environment", 10))
        self.assertEqual(len(fx.histories[-1]), 11)               # observations 0..10, last = feedback of row 9
        self.assertEqual([c[0] for c in fx.inferences[-1].calls], [0])
        r = execute(self.tmp, "run-fb-budget", fx.suite(), self.parent, self.case(1, goal=50.0), max_steps=8)
        self.assertEqual((r["termination"], r["steps"]), ("step_budget", 8))
        self.assertEqual(len(fx.histories[-1]), 9)
        fx = Fixture(self.tmp, recovery_factory=lambda: StopAt(3))
        r = execute(self.tmp, "run-fb-stop", fx.suite(), self.parent, self.case(1, goal=5.0))
        self.assertEqual(r["steps"], 4)
        self.assertEqual(len(fx.histories[-1]), 4)                # a stop is not an executed row

    def test_core_calling_observe_final_is_not_duplicated(self):
        fx = Fixture(self.tmp)
        suite = fx.suite()
        env, skill = suite.environment("lift"), suite.policy(self.parent, "lift")
        seen = []
        hook = env.on_observation
        env.on_observation = lambda o: (seen.append(o), hook(o))

        class NewCorePolicy:
            def __getattr__(self, name):
                return getattr(skill, name)

            def end_episode(self, ctx):
                skill.observe_final(seen[-1], ctx)
                return skill.end_episode(ctx)

        D.run(task="lift", case=self.case(1, goal=0.5), environment=env, policy=NewCorePolicy(),
              verifier=suite.verifier("lift"), max_steps=40)
        self.assertEqual(len(fx.histories[-1]), 11)

    def test_case_splits(self):
        suite = Fixture(self.tmp).suite()
        dev = suite.development_cases(self.parent)["lift"]
        v1 = suite.validation_cases({"id": 1})["lift"]
        v2 = suite.validation_cases({"id": 2})["lift"]
        self.assertEqual(v1, suite.validation_cases({"id": 1})["lift"])
        seeds = lambda cs: {c["seed"] for c in cs}  # noqa: E731
        self.assertFalse(seeds(dev) & seeds(v1))
        self.assertFalse((seeds(v1) | seeds(v2)) & {100, 101})

    def test_real_wam_episode(self):
        m = H.Harness(H.config(history_enable=True))
        m.eval()
        fx = Fixture(self.tmp, inference_factory=lambda: WAMInference(
            m, checkpoint="ckpt", action_stats_version="stats", camera_transform=ADAPTER.camera_transform,
            settings=GenerationSettings(num_inference_steps=2, seed=1)))
        r = execute(self.tmp, "run-real", fx.suite(), self.parent, self.case(1, goal=1e9), max_steps=18)
        self.assertEqual(r["steps"], 18)
        self.assertEqual(r["outcome"], "failure")
        self.assertTrue(all(np.isfinite(e["action"]["command"]).all() for e in r["trace"][1:]))


class SuiteValidation(unittest.TestCase):
    def test_overlapping_seeds_rejected(self):
        with self.assertRaises(ValueError):
            ThinkerWAMSuite(name="x", tasks={"t": TaskSpec("x", (1,), (1, 2), 1)}, env_factory=None,
                            skill_factory=None)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main(verbosity=2)
