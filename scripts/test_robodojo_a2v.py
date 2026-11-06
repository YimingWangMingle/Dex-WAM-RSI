"""CPU interface tests; do not claim trained A2V accuracy or simulator success."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.models.inference_api import GenerationSettings, Forecast, action_hash
from environments.robodojo.runtime.robodojo_a2v import (
    A2VReviewMixin, eval_reviewed_episode, mode3_astra_input, save_forecast)


class Inference:
    num_frames, ratio = 12, 2
    settings = GenerationSettings(seed=12, return_depth=True)
    def generate_action(self, *args):
        return NS(actions=np.full((24,14), .1 + self.settings.seed/1000, np.float32))
    def predict_outcome(self, obs, history, instruction, plan):
        assert self.settings.decode_outputs
        return Forecast('action_conditioned', obs.step, obs.step/25, plan.copy(),
            action_hash(plan), 'stats', 'camera', 'checkpoint', {},
            [obs.step+2*i for i in range(1,13)], [obs.step+24],
            frames=torch.full((1,3,12,16,16), float(plan[0,0])))


class Base:
    def reset(self):
        self.episode_id = 'reset-episode'
        self.next_decision = 0
        self.observation = None


class Policy(A2VReviewMixin, Base):
    def __init__(self, root):
        self._review = None
        self.inf = Inference()
        self.observation = NS(step=0,state=torch.zeros(14))
        self.next_decision = 0
        self.horizon = NS(predict_steps=24)
        self.block_size, self.control_hz = 8, 25
        self.log_dir, self.episode_id = Path(root), 'episode'
        self.instruction = 'stack bowls'
        self.history = NS(last_step=0)
        self.continuity = self.async_replan = self.measured_velocity = None
        self.last_command = None
        self.dims = None
        self.events = []
        self._record = self.events.append


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Policy(self.tmp.name)
        module = ModuleType('XPolicyLab.utils.process_data')
        module.unpack_robot_state = lambda a,*args,**kw:a.copy()
        self.patcher = patch.dict(sys.modules, {'XPolicyLab.utils.process_data':module})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_preview_export_select_exact_prefix(self):
        r = self.p.preview_candidates({'count':2})
        self.assertEqual(self.p.next_decision,0)
        self.assertEqual(self.p.history.last_step,0)
        self.assertFalse(self.p.inf.settings.decode_outputs)
        self.assertNotEqual(r['candidates'][0]['action_hash'],r['candidates'][1]['action_hash'])
        media = r['candidates'][1]['forecast']
        self.assertEqual(len(media['frames']),12)
        self.assertTrue(Path(media['frames'][0]).is_file())
        plan = np.array(r['candidates'][1]['actions'], np.float32)
        out = self.p.select_candidate({'review_id':r['review_id'],'candidate_id':1})
        np.testing.assert_array_equal(out,plan[:8])
        self.assertEqual(self.p.next_decision,8)
        self.assertEqual(self.p.history.last_step,0)
        with self.assertRaises(RuntimeError): self.p.preview_candidates({})

    def test_discard_does_not_issue_or_update_history(self):
        r=self.p.preview_candidates({'predict':False})
        self.p.discard_candidates({'review_id':r['review_id']})
        self.assertEqual(self.p.next_decision,0)
        self.assertIsNone(self.p.last_command)
        with self.assertRaises(ValueError):
            self.p.select_candidate({'review_id':r['review_id'],'candidate_id':0})

    def test_returned_plan_mutation_cannot_change_execution(self):
        r=self.p.preview_candidates({'predict':False})
        expected=np.array(r['candidates'][0]['actions'],np.float32)[:8].copy()
        r['candidates'][0]['actions'][0][0]=99
        out=self.p.select_candidate({'review_id':r['review_id'],'candidate_id':0})
        np.testing.assert_array_equal(out,expected)

    def test_reset_invalidates_review(self):
        r=self.p.preview_candidates({'predict':False})
        self.p.reset()
        self.assertIsNone(self.p._review)
        with self.assertRaises(RuntimeError): self.p.select_candidate({'review_id':r['review_id'],'candidate_id':0})

    def test_invalid_plan_and_selection(self):
        for plan in (np.zeros((8,14)),np.full((24,14),np.nan),np.full((24,14),2.)):
            with self.assertRaises(ValueError):self.p.preview_candidates({'plans':[plan.tolist()]})
        self.assertEqual(self.p.next_decision,0)
        r=self.p.preview_candidates({'predict':False})
        for index in (-1,True,9):
            with self.assertRaises(ValueError):self.p.select_candidate({'review_id':r['review_id'],'candidate_id':index})
        with self.assertRaises(RuntimeError):self.p.preview_candidates({})
        with self.assertRaises(RuntimeError):self.p.get_action()

    def test_action_forecast_mismatch_fails_without_dispatch(self):
        original=Inference.predict_outcome
        def broken(*args):return replace(original(*args),action_hash='wrong')
        with patch.object(Inference,'predict_outcome',broken):
            with self.assertRaises(ValueError):self.p.preview_candidates({})
        self.assertIsNone(self.p._review)
        self.assertEqual(self.p.next_decision,0)

    def test_external_plan_unchanged_and_async_rejected(self):
        plan=np.full((24,14),.3,np.float32)
        r=self.p.preview_candidates({'plans':[plan.tolist()],'predict':False})
        self.assertEqual(r['candidates'][0]['action_hash'],action_hash(plan))
        self.p.discard_candidates({'review_id':r['review_id']})
        self.p.async_replan=NS()
        with self.assertRaises(ValueError):self.p.preview_candidates({})

    def test_filter_state_changes_only_for_selected_candidate(self):
        class Filter:
            calls=0
            def process(self,raw,*args):
                self.calls+=1
                return raw*.5,{}
        self.p.continuity=Filter()
        r=self.p.preview_candidates({'count':2,'predict':False})
        self.assertEqual(self.p.continuity.calls,0)
        self.p.select_candidate({'review_id':r['review_id'],'candidate_id':1})
        self.assertEqual(self.p.continuity.calls,1)

    def test_framework_select_and_reject(self):
        from test_framework_native import Env, Generic, Client
        policy=Generic(self.tmp.name);client=Client(policy)
        result=eval_reviewed_episode(Env(),client,lambda r:None,predict=False,root=Path(self.tmp.name)/'reject')
        self.assertEqual(result['executed_steps'],0)
        self.assertIn('discard_candidates',[n for n,p in client.calls])
        self.assertNotIn('framework_select',[n for n,p in client.calls])
        env=Env(stop=8)
        result=eval_reviewed_episode(env,client,lambda r:0,predict=False,root=Path(self.tmp.name)/'select')
        self.assertEqual(result['executed_steps'],8)

    def test_native_episode_loop_logs_terminal_row_and_stop(self):
        from test_framework_native import Env, Generic, Client
        policy=Generic(self.tmp.name);client=Client(policy);env=Env(stop=3)
        result=eval_reviewed_episode(env,client,lambda r:0,root=Path(self.tmp.name)/'run')
        self.assertEqual(result['executed_steps'],3)
        self.assertEqual(policy.measured,[0,1,2,3])
        self.assertEqual(client.calls[-1][0],'finish_episode')
        client.calls.clear();env=Env()
        result=eval_reviewed_episode(env,client,lambda r:None,root=Path(self.tmp.name)/'stop')
        self.assertEqual(result['executed_steps'],0)
        self.assertEqual(result['stop_reason'],'agent_stop:reviewer_stopped')
        self.assertNotIn('framework_select',[n for n,o in client.calls])

    def test_mode3_visual_review_hides_actions_and_marks_horizon(self):
        self.p.block_size = 16
        preview = self.p.preview_candidates({'count': 3, 'predict': True, 'reveal_actions': False})
        packet = mode3_astra_input(preview, images=['current.png'], history=[],
                                    instruction='stack bowls')
        self.assertEqual(packet['response'], 'candidate_id + execute_steps')
        self.assertEqual(packet['allowed_execute_steps'], [24,16,8,4])
        blob = json.dumps(packet)
        self.assertNotIn('actions', blob)
        self.assertNotIn('action_hash', blob)
        for candidate in packet['candidates']:
            self.assertEqual(len(candidate['frames']), 12)
            self.assertEqual([row['segment'] for row in candidate['frames']].count('prediction'), 12)
            self.assertEqual(candidate['frames'][7]['offset_steps'], 16)
            self.assertEqual(candidate['frames'][8]['segment'], 'prediction')
        chosen = packet['candidates'][2]['candidate_id']
        out = self.p.select_candidate({'review_id': packet['review_id'], 'candidate_id': chosen})
        self.assertEqual(out.shape, (16, 14))
        self.p.observation.step = self.p.next_decision
        leaked = self.p.preview_candidates({'count': 3, 'predict': True})
        with self.assertRaises(ValueError):
            mode3_astra_input(leaked, images=[], history=[], instruction='task')

    def test_reviewer_exception_discards_pending(self):
        from test_framework_native import Env, Generic, Client
        policy=Generic(self.tmp.name);client=Client(policy);env=Env()
        def reviewer(r):raise RuntimeError('review unavailable')
        with self.assertRaisesRegex(RuntimeError,'review unavailable'):
            eval_reviewed_episode(env,client,reviewer,predict=False,root=Path(self.tmp.name)/'error')
        self.assertIsNone(policy._review)
        self.assertEqual(env.steps,0)


if __name__=='__main__':unittest.main(verbosity=2)
