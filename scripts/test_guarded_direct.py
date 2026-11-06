"""CPU evidence/guarded-stage integration; fake motion, never P0/P1 acceptance."""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skills.contracts import ControlSkill, SkillIdentity, SkillRequest
from skills.direct.guarded import DirectStage, GuardedDirectBackend
from execution.feedback import (LocalMetric, ObservationEvidence, OperationBlocked, OperationCheck,
                                    OperationReport, diagnose_operation)
from execution.runner import CallSkill, Measure, Stop
import test_task_skill_runner as fixtures
from test_task_skill_runner import CONTRACT, EP, accept, obs, reject


class GuardedTests(unittest.TestCase):
    build = fixtures.RunnerTests.build
    events = fixtures.RunnerTests.events
    def guarded(self, verdict="passed", steps=None):
        self.planned = []
        def stage(name, rows, check):
            def plan(request):
                self.planned.append((name, request.observation.control_step))
                return rows
            def verify(before, after):
                return (OperationCheck(check, verdict, "test-verifier/v1",
                                       (ObservationEvidence.from_observation(after),)),)
            return DirectStage(name, plan, verify, (check,))
        self.backend = GuardedDirectBackend({"grasp": (
            stage("approach", ((0.1, 0.0), (0.2, 0.0)), "at_target"),
            stage("lift", ((0.3, 0.0),), "object_lifted"))})
        runner = self.build(steps or [CallSkill("direct_control", "grasp", 3), accept(),
                                     CallSkill("direct_control", "grasp", 3), accept(), Stop("done")])
        self.skill = ControlSkill(SkillIdentity("direct_control", "guard/v1", CONTRACT, 2), self.backend)
        runner.registry._skills["direct_control"] = self.skill
        runner.operation_verifiers["direct_control"] = self.backend.verify_execution
        return runner

    def test_stage_planning_waits_for_actual_feedback(self):
        r = self.guarded(); r.run(EP)
        self.assertEqual(self.planned, [("approach", 0), ("lift", 2)])
        self.assertEqual(self.backend.index, 2)
        self.assertEqual(self.executor.step, 3)
        self.assertEqual([e["stage"] for e in self.events("proposal")], ["approach", "lift"])
        checks = self.events("operation_verification")
        self.assertEqual(checks[1]["operation"]["checks"][0]["evidence"][0]["control_step"], 3)
        self.assertTrue(any('"operation"' in m["text"] for m in self.agent.views[-1].messages
                            if m["kind"] == "execution"))

    def test_unknown_and_failed_stop_next_stage(self):
        for status in ("unknown", "failed"):
            with self.subTest(status=status):
                r = self.guarded(status, [CallSkill("direct_control", "grasp", 3), accept(), Stop("inspect")])
                r.run(EP)
                self.assertTrue(self.backend.blocked)
                self.assertEqual(self.backend.index, 0)
                with self.assertRaises(OperationBlocked):
                    self.skill.propose(SkillRequest(obs(2), "grasp", CONTRACT, 3))
                self.assertEqual(len(self.planned), 1)

    def test_failure_lock_survives_switch_away_and_back(self):
        for status in ('failed','unknown'):
            with self.subTest(status=status):
                r=self.guarded(status,[CallSkill('direct_control','grasp',3),accept(),
                    CallSkill('wam_control','transit',4),accept(2),
                    CallSkill('direct_control','grasp',3),Stop('inspect')])
                r.run(EP)
                self.assertEqual(self.planned,[('approach',0)])
                self.assertTrue(self.backend.blocked)
                self.assertEqual(self.executor.step,4)
                self.assertEqual(len(self.events('proposal_blocked')),1)

    def test_completed_stages_do_not_replay_after_handoff(self):
        r=self.guarded(steps=[CallSkill('direct_control','grasp',3),accept(),
            CallSkill('wam_control','transit',4),accept(2),
            CallSkill('direct_control','grasp',3),accept(),
            CallSkill('wam_control','transit',4),reject(),
            CallSkill('direct_control','grasp',3),Stop('done')])
        r.run(EP)
        self.assertEqual(self.planned,[('approach',0),('lift',4)])
        self.assertEqual(self.backend.index,2)
        self.assertEqual(len(self.events('proposal_blocked')),1)

    def test_switching_subgoals_cannot_erase_failure_and_new_episode_can_reset(self):
        r=self.guarded('failed',[CallSkill('direct_control','grasp',3),accept(),
            CallSkill('direct_control','recovery',3),reject(),
            CallSkill('direct_control','grasp',3),Stop('inspect')])
        self.backend.recipes['recovery']=self.backend.recipes['grasp']
        r.run(EP)
        self.assertEqual(len(self.planned),2)
        self.assertTrue(self.backend.blocked)
        self.skill.reset('new-episode')
        request=SkillRequest(replace(obs(0),episode_id='new-episode'),'grasp',CONTRACT,3)
        self.skill.propose(request)
        self.assertFalse(self.backend.blocked)
        self.assertEqual(len(self.planned),3)

    def test_partial_and_rejected_motion_do_not_advance(self):
        r = self.guarded(steps=[CallSkill("direct_control", "grasp", 3), accept(1), Stop("inspect")])
        r.run(EP)
        self.assertTrue(self.backend.blocked)
        self.assertEqual(self.executor.step, 1)
        self.assertEqual(self.events("operation_verification")[0]["operation"]["motion"], "unknown")
        r = self.guarded(steps=[CallSkill("direct_control", "grasp", 3), reject(), Stop("inspect")])
        r.run(EP)
        self.assertTrue(self.backend.blocked)
        self.assertEqual(self.executor.step, 0)

    def test_unwired_verifier_never_assumes_success(self):
        r = self.guarded(steps=[CallSkill("direct_control", "grasp", 3), accept(), Stop("inspect")])
        r.operation_verifiers.clear(); r.run(EP)
        self.assertTrue(self.backend.blocked)
        self.assertIsNone(self.events("operation_verification")[0]["operation"])

    def test_verifier_exception_preserves_execution_and_blocks(self):
        r = self.guarded(steps=[CallSkill("direct_control", "grasp", 3), accept(), Stop("inspect")])
        def broken(*args):
            raise RuntimeError("camera unavailable")
        r.operation_verifiers["direct_control"] = broken
        r.run(EP)
        self.assertEqual(self.executor.step, 2)
        self.assertEqual(len(self.events("execution")), 1)
        self.assertTrue(self.backend.blocked)
        self.assertIn("camera unavailable", self.events("operation_verification")[0]["error"])
        self.assertFalse(self.events("episode_end")[0]["pending_at_end"])

    def test_stale_report_is_unknown_not_a_verified_grasp(self):
        r = self.guarded(steps=[CallSkill("direct_control", "grasp", 3), accept(), Stop("inspect")])
        def stale(p, result):
            return OperationReport("approach", "passed", (OperationCheck(
                "at_target", "passed", "test/v1", (ObservationEvidence.from_observation(obs(0)),)),))
        r.operation_verifiers["direct_control"] = stale
        r.run(EP)
        self.assertTrue(self.backend.blocked)
        self.assertIsNotNone(self.events("operation_verification")[0]["error"])

    def test_feedback_delivery_error_does_not_zero_or_replay_motion(self):
        r = self.guarded(steps=[CallSkill("direct_control", "grasp", 3), accept()])
        def broken(feedback):
            raise OSError("backend unavailable")
        self.backend.observe = broken
        with self.assertRaises(OSError):
            r.run(EP)
        self.assertEqual(self.executor.step, 2)
        self.assertEqual(self.events("feedback_delivery_failed")[0]["executed_steps"], 2)
        self.assertEqual(self.events("execution")[0]["after"]["control_step"], 2)
        self.assertFalse(any(e.get("automatic") for e in self.events("review")))

    def test_measurement_is_read_only_and_visible_to_agent(self):
        r = self.guarded(steps=[Measure("clearance"), Stop("inspect")])
        def measure(o):
            return OperationReport("measure", checks=(OperationCheck("clear", "passed", "test/v1",
                (ObservationEvidence.from_observation(o),)),), metrics=(LocalMetric("gap", 5, "mm", 2),))
        r.measurement_tools["clearance"] = measure
        r.run(EP)
        self.assertEqual(self.executor.step, 0)
        self.assertEqual(len(self.events("measurement")), 1)
        self.assertEqual(self.agent.views[0].available_measurements, ("clearance",))
        self.assertTrue(any(m["kind"] == "measurement" for m in self.agent.views[-1].messages))

    def test_precondition_failure_prevents_planning(self):
        self.guarded(); self.skill.reset(EP)
        original = self.backend.recipes["grasp"][0]
        guarded = replace(original, precondition=lambda r: OperationReport("measure"),
                          required_preconditions=("geometry_reliable",))
        self.backend.recipes["grasp"] = (guarded,)
        with self.assertRaises(ValueError):
            self.skill.propose(SkillRequest(obs(0), "grasp", CONTRACT, 3))
        self.assertEqual(self.planned, [])
        self.assertIsNone(self.backend.pending)

    def test_precondition_refusal_returns_control_to_agent(self):
        r = self.guarded(steps=[CallSkill("direct_control", "grasp", 3), Stop("need another view")])
        original = self.backend.recipes["grasp"][0]
        self.backend.recipes["grasp"] = (replace(original,
            precondition=lambda r: OperationReport("measure"),
            required_preconditions=("geometry_reliable",)),)
        result = r.run(EP)
        self.assertEqual(result["reason"], "agent_stop:need another view")
        self.assertEqual(self.executor.step, 0)
        self.assertEqual(len(self.events("proposal_blocked")), 1)
        self.assertTrue(any(m["kind"] == "operation_blocked" for m in self.agent.views[-1].messages))

    def test_budget_does_not_truncate_stage(self):
        self.guarded(); self.skill.reset(EP)
        with self.assertRaises(ValueError):
            self.skill.propose(SkillRequest(obs(0), "grasp", CONTRACT, 1))
        self.assertIsNone(self.backend.pending)
        self.assertEqual(self.executor.step, 0)


class EvidenceTests(unittest.TestCase):
    def test_non_measured_and_foreign_evidence_rejected(self):
        with self.assertRaises(ValueError):
            OperationCheck("held", "passed", "test/v1")
        evidence = ObservationEvidence.from_observation(obs(1))
        for bad in (replace(evidence, episode_id="other"), replace(evidence, payload_ref="predicted")):
            report = OperationReport("lift", checks=(OperationCheck("held", "passed", "v1", (bad,)),))
            with self.assertRaises(ValueError):
                report.validate_observations(obs(0), obs(1))

    def test_missing_required_check_cannot_unlock(self):
        report = OperationReport("lift", "passed")
        self.assertFalse(report.permits_next_stage(("held",)))

    def test_failure_routes_are_hypotheses(self):
        report = OperationReport("lift", failure_kind="contact", failure_detail="possible slip")
        result = diagnose_operation(report)
        self.assertEqual(result["route"], "collect_verified_recovery")
        self.assertTrue(result["requires_independent_validation"])
        self.assertTrue(result["hypothesis"])

    def test_metric_requires_units_finite_and_tolerance(self):
        for value, unit, tolerance in [(float("nan"), "mm", 1), (1, "", 1), (1, "mm", -1)]:
            with self.assertRaises(ValueError):
                LocalMetric("gap", value, unit, tolerance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
