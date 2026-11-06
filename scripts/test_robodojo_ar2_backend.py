from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from types import SimpleNamespace as NS
import numpy as np
from thinker_wm.runtime.joint_continuity import JointContinuity
from skills.wam.thinker import ThinkerWAMBackend, load_robodojo_skill

class BackendTests(unittest.TestCase):
    def make(self, continuity=None):
        self.saved=[]
        self.measured={i:NS(step=i,state=np.zeros(14)) for i in range(17)}
        history=NS(last_step=None,reset=lambda episode:None)
        history.add=lambda obs:setattr(history,'last_step',obs.step)
        store=NS(load=lambda ref:self.measured[ref.control_step],
            since=lambda episode,last,end:[self.measured[i] for i in range(0 if last is None else last+1,end+1)],
            save_prediction=lambda ref, forecast:self.saved.append(forecast) or 'prediction')
        raw=np.ones((16,14),np.float32)*.2
        inf=NS(num_frames=8,ratio=2,generate_action=lambda *args:NS(actions=raw,forecast='raw_forecast'))
        profile=NS(actions_per_block=8,training_actions=24,action_dim=14,control_hz=25,action_contract='joint14')
        backend=ThinkerWAMBackend(inf,history,store,profile,execute_steps=8,continuity=continuity)
        backend.reset('episode')
        return backend

    def ref(self,step):return NS(episode_id='episode',control_step=step)
    def request(self,step=0,limit=8):return NS(observation=self.ref(step),action_contract='joint14',subgoal='stack',max_actions=limit)

    def test_prefix_only_measured_history_and_partial_feedback(self):
        b=self.make()
        result=b.propose(self.request(limit=3))
        self.assertEqual(len(result.actions),3)
        self.assertEqual(b.history.last_step,0)
        b.observe(NS(final_observation=self.ref(2),executed_actions=result.actions[:2]))
        self.assertEqual(b.history.last_step,2)
        self.assertEqual(b.prediction_valid_steps,2)
        self.assertEqual(len(b.propose(self.request(2)).actions),8)

    def test_modified_commands_invalidate_raw_forecast(self):
        c=JointContinuity(control_hz=25,max_velocity_rad_s=[1]*12,max_acceleration_rad_s2=[2]*12)
        b=self.make(c)
        result=b.propose(self.request())
        self.assertIsNone(result.prediction_ref)
        self.assertEqual(self.saved,[])
        b.observe(NS(final_observation=self.ref(8),executed_actions=result.actions))
        self.assertEqual(b.prediction_valid_steps,0)
        self.assertIsNone(b.last_prediction)
        b.propose(self.request(8))
        self.assertEqual(b.last_diagnostics['overlap_rows'],4)

    def test_subgoal_change_discards_previous_task_tail(self):
        c=JointContinuity(control_hz=25,max_velocity_rad_s=[1]*12,max_acceleration_rad_s2=[2]*12)
        b=self.make(c)
        result=b.propose(self.request())
        b.observe(NS(final_observation=self.ref(8),executed_actions=result.actions))
        request=self.request(8)
        request.subgoal='return home'
        b.propose(request)
        self.assertEqual(b.last_diagnostics['overlap_rows'],0)

    def test_adapter_requires_explicit_identity_before_loading(self):
        with self.assertRaises(ValueError):load_robodojo_skill('missing',None,model_adapter=lambda m,p:m)
        with self.assertRaises(ValueError):load_robodojo_skill('missing',None,adapter_revision='rev')

if __name__=='__main__':unittest.main(verbosity=2)
