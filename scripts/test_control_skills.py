"""CPU contract checks only: no robot execution or real VLA model acceptance."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills.contracts import (
    BackendProposal, ControlSkill, ExecutionFeedback, MeasuredObservation,
    SKILL_NAMES, SkillIdentity, SkillRegistry, SkillRequest,
)


def observation(step=0, episode="e1", source="measured"):
    return MeasuredObservation(episode, f"obs-{step}", step, step / 25,
                               f"bundle-{episode}-{step}", source)


class Backend:
    def __init__(self):
        self.actions = ((0.1, 0.2), (0.3, 0.4))
        self.receipts = []

    def reset(self, episode_id):
        self.receipts.clear()

    def propose(self, request):
        return BackendProposal(self.actions)

    def observe(self, feedback):
        self.receipts.append(feedback)


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.identity = SkillIdentity("vla_control", "adapter-v1", "test-action-v1", 2, "ckpt-123")
        self.skill = ControlSkill(self.identity, self.backend)
        self.skill.reset("e1")
        self.request = SkillRequest(observation(), "pick object", "test-action-v1", 2)

    def test_three_names_and_explicit_missing_backend(self):
        registry = SkillRegistry()
        for name in SKILL_NAMES:
            with self.assertRaises(LookupError):
                registry.get(name)
            item = ControlSkill(SkillIdentity(name, "r1", "c1", 2, "ckpt"), Backend())
            registry.register(item)
            self.assertIs(registry.get(name), item)
            with self.assertRaises(ValueError):
                registry.register(item)

    def test_dynamic_names_and_duplicate_registration(self):
        registry=SkillRegistry()
        registry.register(ControlSkill(SkillIdentity('grasp_recovery','v1','c1',2),Backend()))
        self.assertEqual(registry.names(),('grasp_recovery',))
        for name in ('../escape','', 'bad name'):
            with self.assertRaises(ValueError):SkillIdentity(name,'v1','c1',2)

    def test_model_skills_need_checkpoint(self):
        with self.assertRaises(ValueError):
            SkillIdentity("vla_control", "r1", "c1", 2)
        with self.assertRaises(ValueError):
            SkillIdentity("wam_control", "r1", "c1", 2)

    def test_proposal_identity_and_no_prediction_required(self):
        proposal = self.skill.propose(self.request)
        self.assertEqual(proposal.identity, self.identity)
        self.assertEqual(proposal.request.observation.observation_id, "obs-0")
        self.assertIsNone(proposal.prediction_ref)
        self.assertFalse(self.backend.receipts)

    def test_cross_episode_or_contract_rejected(self):
        with self.assertRaises(RuntimeError):
            self.skill.propose(SkillRequest(observation(0, "e2"), "pick", "test-action-v1", 2))
        with self.assertRaises(ValueError):
            self.skill.propose(SkillRequest(observation(), "pick", "different-contract", 2))

    def test_invalid_actions_rejected(self):
        for actions in ((), ((float("nan"), 0),), ((1,),), ((1, 2),) * 3):
            self.backend.actions = actions
            with self.assertRaises(ValueError):
                self.skill.propose(self.request)

    def test_pending_blocks_replan_and_reset(self):
        self.skill.propose(self.request)
        with self.assertRaises(RuntimeError):
            self.skill.reset("e2")
        with self.assertRaises(RuntimeError):
            self.skill.propose(self.request)

    def test_partial_execution_records_actual_commands(self):
        proposal = self.skill.propose(self.request)
        actual = ((0.05, 0.1),)  # Reviewed/postprocessed prefix, not original command.
        self.skill.observe(ExecutionFeedback(proposal.proposal_id, actual, observation(1), False))
        self.assertEqual(self.backend.receipts[0].executed_actions, actual)
        with self.assertRaises(ValueError):
            self.skill.propose(self.request)
        self.skill.propose(SkillRequest(observation(1), "pick", "test-action-v1", 2))

    def test_wrong_feedback_rejected(self):
        proposal = self.skill.propose(self.request)
        for identity, after in (("wrong-id", observation(1)),
                                (proposal.proposal_id, observation(2)),
                                (proposal.proposal_id, observation(1, "e2"))):
            with self.assertRaises(ValueError):
                self.skill.observe(ExecutionFeedback(identity, ((0, 0),), after, False))
        self.assertFalse(self.backend.receipts)

    def test_rejection_has_zero_executed_rows(self):
        proposal = self.skill.propose(self.request)
        self.skill.observe(ExecutionFeedback(proposal.proposal_id, (), observation(), False))
        self.skill.reset("e2")

    def test_terminal_requires_new_reset(self):
        proposal = self.skill.propose(self.request)
        self.skill.observe(ExecutionFeedback(proposal.proposal_id, ((0, 0),), observation(1), True))
        with self.assertRaises(RuntimeError):
            self.skill.propose(self.request)

    def test_history_rejects_predicted_future_or_other_episode(self):
        with self.assertRaises(ValueError):
            observation(source="predicted")
        for history in ((observation(2),), (observation(0, "e2"),),
                        (observation(1), observation(0))):
            with self.assertRaises(ValueError):
                SkillRequest(observation(2), "pick", "test-action-v1", 2, history)
        SkillRequest(observation(2), "pick", "test-action-v1", 2, (observation(0), observation(1)))


if __name__ == "__main__":
    unittest.main()
