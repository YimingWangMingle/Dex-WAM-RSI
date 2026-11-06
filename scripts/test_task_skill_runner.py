"""CPU checks for review, the WAM commit/discard lifecycle and in-task skill hand-off.

Uses a scripted Agent, a fake executor and test backends. No robot, simulator,
LLM or model checkpoint is involved; this is not P1 acceptance.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.context import AgentContext, TaskState
from skills.contracts import (
    BackendProposal, ControlSkill, MeasuredObservation, SkillIdentity, SkillRegistry,
    SkillRequest,
)
from execution.review import BoundsChecker, ReviewDecision, ReviewRejected, no_checks, validate
from execution.runner import (
    CallSkill, ExecutionResult, JsonlEventLog, ReadExperience, Review, RunnerConfig, Stop,
    TaskSkillRunner, UpdateTaskState,
)
from skills.wam.backend import WAMControlBackend

EP = "ep-1"
CONTRACT = "test-joint-v1"


def obs(step, frame=0):
    """frame > 0: a new camera frame at the same control step (re-observation)."""
    name = f"obs-{step}" if not frame else f"obs-{step}.{frame}"
    return MeasuredObservation(EP, name, step, step / 25 + frame * 1e-3, f"bundle-{name}")


class FakeExecutor:
    """Executes rows one by one; may interrupt after ``interrupt_after`` rows."""
    def __init__(self, terminate_at=None):
        self.step, self.frame, self.terminate_at = 0, 0, terminate_at
        self.interrupt_after = None
        self.dispatched, self.cancels = [], 0

    def current_observation(self):
        return obs(self.step, self.frame)

    def execute(self, rows, *, action_contract):
        assert action_contract == CONTRACT
        done, terminated = [], False
        for row in rows:
            if self.interrupt_after is not None and len(done) == self.interrupt_after:
                self.interrupt_after = None
                return ExecutionResult(tuple(done), obs(self.step), False, "interrupted:deviation")
            self.dispatched.append(row)
            done.append(row)
            self.step, self.frame = self.step + 1, 0
            if self.terminate_at is not None and self.step >= self.terminate_at:
                terminated = True
                break
        return ExecutionResult(tuple(done), obs(self.step), terminated,
                               "terminated" if terminated else "completed",
                               measured=dict(steps=len(done)))

    def in_flight(self):
        return False

    def cancel(self):
        self.cancels += 1


class FakeWAM:
    """Explicit-commit model double that records what entered its memory."""
    def __init__(self):
        self.history, self.committed, self.discards, self.resets = [], [], [], 0

    def reset(self):
        self.history.clear()
        self.resets += 1

    def update_obs(self, o):
        assert o.source == "measured"
        self.history.append(o.observation_id)

    def get_action(self, request):
        return [(0.1 * i, 0.0) for i in range(1, 5)], f"pred-{request.observation.control_step}"

    def commit_action(self, rows):
        self.committed.append(rows)

    def discard_action(self, execution_started):
        self.discards.append(execution_started)

    def close(self):
        pass


class SimpleBackend:
    def __init__(self, rows):
        self.rows, self.resets, self.seen = rows, 0, []

    def reset(self, episode_id):
        self.resets += 1

    def propose(self, request):
        self.seen.append(request)
        return BackendProposal(self.rows)

    def observe(self, feedback):
        pass


class Scripted:
    def __init__(self, steps):
        self.steps, self.views = list(steps), []

    def next(self, view):
        self.views.append(view)
        if not self.steps:
            return Stop("script exhausted")
        step = self.steps.pop(0)
        return step(view) if callable(step) else step


def accept(n=None, reason="tracks the cup", subgoal="grasp cup"):
    return lambda v: Review(ReviewDecision(v.pending["proposal_id"], "accept", reason, subgoal, n))


def reject(reason="wrong side"):
    return lambda v: Review(ReviewDecision(v.pending["proposal_id"], "reject", reason, "grasp cup"))


class RunnerTests(unittest.TestCase):
    def build(self, steps, terminate_at=None, with_vla=False, **options):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wam_model = FakeWAM()
        self.direct = SimpleBackend(((0.5, 0.5), (0.6, 0.6)))
        self.vla = SimpleBackend(((0.2, 0.2),))
        registry = SkillRegistry()
        registry.register(ControlSkill(SkillIdentity("wam_control", "wam-adapter-v1", CONTRACT, 2,
                                                     "thinkerwam-ckpt-A"),
                                       WAMControlBackend(self.wam_model)))
        registry.register(ControlSkill(SkillIdentity("direct_control", "ctrl-v1", CONTRACT, 2),
                                       self.direct))
        if with_vla:
            registry.register(ControlSkill(SkillIdentity("vla_control", "vla-adapter-v1",
                                                            CONTRACT, 2, "vla-ckpt-B"), self.vla))
        self.executor = FakeExecutor(terminate_at)
        self.agent = Scripted(steps)
        self.state = TaskState("put the cup on the tray")
        self.log = Path(self.tmp.name) / "events.jsonl"
        self.runner = TaskSkillRunner(
            registry, self.executor, self.agent, AgentContext(self.state),
            sink=JsonlEventLog(self.log),
            row_checkers={"wam_control": BoundsChecker(upper=(0.25, None))},
            config=RunnerConfig(max_agent_steps=30), **options)
        return self.runner

    def test_per_skill_experience_is_opt_in_and_survives_handoff(self):
        class Workspace:
            def identity(self): return {'bundle_id': 'test-v1'}
            def instructions(self): return 'Historical evidence only'
            def read_text(self, path, limit): return {'ICL.md':'demonstration', 'HMR.md':'past slip'}[path][:limit]
        runner = self.build([
            CallSkill('direct_control', 'approach', 2), accept(1),
            CallSkill('vla_control', 'grasp', 1), accept(1),
            Stop('done')], with_vla=True, experience=Workspace(),
            skill_experience={'direct_control': {'icl':['ICL.md'], 'hmr':['HMR.md']}})
        runner.run(EP)
        context = self.direct.seen[0].experience
        self.assertEqual(context.icl, (('ICL.md','demonstration'),))
        self.assertEqual(context.hmr, (('HMR.md','past slip'),))
        self.assertIsNone(self.vla.seen[0].experience)
        self.assertEqual(self.events('proposal')[0]['skill_experience']['identity']['bundle_id'], 'test-v1')
        self.assertEqual(len(self.executor.dispatched), 2)

    def test_skill_experience_invalid_configuration_fails_before_motion(self):
        for config in ({'typo':{}}, {'direct_control':{'bad':[]}},
                       {'direct_control':{'icl':'ICL.md'}},
                       {'direct_control':{'hmr':['HMR.md']}}):
            with self.subTest(config=config), self.assertRaises((ValueError, LookupError)):
                self.build([], skill_experience=config)
            self.assertEqual(self.executor.dispatched, [])

    def test_missing_skill_memory_never_dispatches(self):
        class Workspace:
            def identity(self): return {'bundle_id':'test'}
            def instructions(self): return 'Historical evidence'
            def read_text(self, path, limit): raise ValueError('missing entry')
        runner=self.build([CallSkill('direct_control','approach',2)], experience=Workspace(),
                          skill_experience={'direct_control':{'hmr':['missing']}})
        with self.assertRaisesRegex(ValueError,'missing entry'): runner.run(EP)
        self.assertEqual(self.executor.dispatched, [])

    def events(self, kind=None):
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [r for r in rows if kind is None or r["event"] == kind]

    def test_handoff_partial_commit_and_four_records(self):
        runner = self.build([
            UpdateTaskState(fix_goal={"tray_xy": [0.4, 0.1]}, holding={"left": None}),
            CallSkill("wam_control", "reach cup", 4),
            accept(4),                                   # rows 2-3 exceed bound 0.25 -> invalid
            accept(3),                                   # prefix with violating row 3 -> invalid
            accept(2),                                   # executes 2 of 4 -> WAM discards
            CallSkill("vla_control", "grasp", 2),       # not configured -> explicit error
            CallSkill("direct_control", "grasp", 2),    # hand-off wam -> direct
            CallSkill("wam_control", "lift", 4),        # pending direct auto-rejected, hand-off
            accept(2),
            UpdateTaskState(progress="cup lifted"),
            Stop("subgoal done"),
        ])
        result = runner.run(EP)
        self.assertEqual(result["reason"], "agent_stop:subgoal done")
        c = result["counters"]
        self.assertEqual((c["proposals"], c["accepted"], c["invalid_reviews"], c["handoffs"]), (3, 2, 2, 2))
        # direct proposal was superseded by the switch: rejected with zero rows, robot not moved
        auto = [e for e in self.events("review") if e.get("automatic")]
        self.assertEqual(auto[0]["reason"], "superseded_by_skill_switch")
        # §5.4 six steps recorded on each hand-off, task state preserved
        for h in self.events("handoff")[1:]:
            self.assertEqual(h["steps"], ["stop_old_submission", "in_flight_resolved", "queue_cleared",
                                          "fresh_observation", "histories_reset_from_measured",
                                          "new_skill_ready"])
            self.assertEqual(h["task_state"]["fixed_goals"], {"tray_xy": [0.4, 0.1]})
        # WAM: partial chunk discarded; rebuilt from measured obs only after hand-off
        self.assertEqual(self.wam_model.discards[0], True)
        self.assertEqual(self.wam_model.committed, [])
        self.assertEqual(self.wam_model.history[0], "obs-2")      # reset dropped earlier context
        self.assertNotIn("pred", " ".join(self.wam_model.history))
        # proposed -> approved -> executed -> measured are separate records
        ex = self.events("execution")
        self.assertEqual(len(ex[0]["approved_actions"]), 2)
        self.assertEqual(ex[0]["executed_actions"], ex[0]["approved_actions"])
        self.assertEqual((ex[0]["before"]["control_step"], ex[0]["after"]["control_step"]), (0, 2))
        prop = self.events("proposal")[0]
        self.assertEqual(len(prop["proposed_actions"]), 4)
        self.assertEqual(prop["checkpoint"], "thinkerwam-ckpt-A")
        self.assertEqual(prop["violations"][2] != [], True)
        self.assertEqual(self.executor.step, 4)
        errors = [m for m in self.agent.views[-1].messages if m["kind"] == "error"]
        self.assertTrue(any("vla_control" in m["text"] for m in errors))
        self.assertEqual(self.events()[-1]["event"], "episode_end")

    def test_full_chunk_commit_then_native_termination(self):
        runner = self.build([CallSkill("wam_control", "reach", 2), accept(2),
                             CallSkill("wam_control", "reach", 2), accept(2)], terminate_at=3)
        self.wam_model.get_action = lambda r: ([(0.1, 0.0), (0.2, 0.0)], None)
        result = runner.run(EP)
        self.assertEqual(result["reason"], "native_termination")
        self.assertEqual(len(self.wam_model.committed), 1)       # first chunk fully executed
        self.assertEqual(self.wam_model.discards, [True])         # second cut by termination
        self.assertEqual(self.events("execution")[-1]["stop_reason"], "terminated")
        self.assertFalse(self.events("episode_end")[0]["pending_at_end"])

    def test_interrupted_execution_and_stale_review(self):
        runner = self.build([CallSkill("direct_control", "push", 2),
                             lambda v: self._interrupt(v), accept(2),
                             CallSkill("direct_control", "push", 2),
                             lambda v: self._move(v), accept(2),
                             reject(), Stop("done")])  # stale -> invalid; reject stays valid
        result = runner.run(EP)
        ex = self.events("execution")[0]
        self.assertEqual((ex["unexecuted_approved_steps"], ex["stop_reason"]), (1, "interrupted:deviation"))
        invalid = self.events("review_invalid")
        self.assertIn("stale", invalid[0]["error"])
        self.assertEqual(result["counters"]["rejected"], 1)

    def _interrupt(self, view):
        self.executor.interrupt_after = 1
        return UpdateTaskState(progress="watching for slip")

    def _move(self, view):
        self.executor.frame += 1   # newer camera frame at the same control step
        return UpdateTaskState(progress="re-observed")

    def test_vla_switch_history_and_pending_at_end(self):
        runner = self.build([CallSkill("wam_control", "reach", 4), accept(2),
                             CallSkill("vla_control", "grasp", 1), accept(1),
                             CallSkill("direct_control", "retreat", 2)], with_vla=True)
        result = runner.run(EP)
        self.assertEqual(result["activated_skills"], ["wam_control", "vla_control", "direct_control"])
        request = self.vla.seen[0]
        self.assertEqual([o.observation_id for o in request.history], ["obs-0"])
        vla_prop = [e for e in self.events("proposal") if e["skill"] == "vla_control"][0]
        self.assertEqual((vla_prop["checkpoint"], vla_prop["prediction_ref"]), ("vla-ckpt-B", None))
        end = self.events("episode_end")[0]
        self.assertTrue(end["pending_at_end"])
        self.assertEqual(self.events("review")[-1]["reason"], "episode_end")


class ReviewTests(unittest.TestCase):
    def proposal(self):
        identity = SkillIdentity("wam_control", "a", CONTRACT, 2, "ckpt")
        skill = ControlSkill(identity, SimpleBackend(((0.1, 0.1), (0.2, 0.2), (0.3, 0.3))))
        skill.reset(EP)
        return skill.propose(SkillRequest(obs(0), "reach", CONTRACT, 3))

    def test_overlong_wam_chunk_leaves_no_pending(self):
        model = FakeWAM()
        backend = WAMControlBackend(model)
        backend.reset(EP)
        with self.assertRaises(ValueError):
            backend.propose(SkillRequest(obs(0), "reach", CONTRACT, 2))
        self.assertIsNone(backend.pending)
        self.assertEqual(model.discards, [False])

    def test_rules(self):
        p = self.proposal()
        ok = no_checks(p, obs(0))
        self.assertEqual(len(validate(ReviewDecision(p.proposal_id, "accept", "r", "g", 2), p, obs(0), ok)
                             .approved_rows), 2)
        for decision, latest in [
                (ReviewDecision("other", "accept", "r", "g"), obs(0)),
                (ReviewDecision(p.proposal_id, "accept", " ", "g"), obs(0)),
                (ReviewDecision(p.proposal_id, "accept", "r", ""), obs(0)),
                (ReviewDecision(p.proposal_id, "accept", "r", "g", 4), obs(0)),
                (ReviewDecision(p.proposal_id, "accept", "r", "g", 0), obs(0)),
                (ReviewDecision(p.proposal_id, "modify", "r", "g"), obs(0)),
                (ReviewDecision(p.proposal_id, "accept", "r", "g"), obs(1))]:
            with self.assertRaises(ReviewRejected):
                validate(decision, p, latest, ok)
        bounded = BoundsChecker(max_delta=(0.15, None), reference=lambda o: (0.0, 0.0))(p, obs(0))
        self.assertEqual([bool(v) for v in bounded], [False, False, False])
        bounded = BoundsChecker(max_delta=(0.05, None), reference=lambda o: (0.0, 0.0))(p, obs(0))
        with self.assertRaises(ReviewRejected):
            validate(ReviewDecision(p.proposal_id, "accept", "r", "g", 1), p, obs(0), bounded)
        self.assertEqual(validate(ReviewDecision(p.proposal_id, "reject", "r", "g"), p, obs(0),
                                  bounded).approved_rows, ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
