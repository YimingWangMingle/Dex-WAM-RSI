import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.robodojo.runtime import robodojo_deepseek_live as p

class Contracts(unittest.TestCase):
    def test_invalid_decisions(self):
        for name, args in [("continue_wam", dict(chunks=True, reason="x")),
                           ("continue_wam", dict(chunks=5, reason="x")),
                           ("adjust_ik", dict(arm="left", delta_xyz_m=[float('nan'),0,0], gripper="keep", reason="x")),
                           ("adjust_ik", dict(arm="left", delta_xyz_m=[.03,.03,0], gripper="keep", reason="x")),
                           ("adjust_ik", dict(arm="left", delta_xyz_m=[0,0,0], gripper=True, reason="x")),
                           ("stop", dict(reason="x", success=True))]:
            with self.subTest(name=name,args=args), self.assertRaises(ValueError):
                p.validate_decision(name,args)

    def test_native_handoff_history_and_counts(self):
        decisions = iter([("continue_wam", dict(chunks=1, reason="start")),
                          ("adjust_ik", dict(arm="left", delta_xyz_m=[0,0,.01], gripper="keep", reason="correct")),
                          ("continue_wam", dict(chunks=1, reason="resume"))])
        state = {a+"_arm_joint_state": [0.]*6 for a in ('left','right')}
        state.update({a+"_ee_joint_state": [1.] for a in ('left','right')})
        class FakeClient:
            last_metadata = {}
            def complete(self, messages, tools):
                assert 'hidden_object_pose' not in str(messages)
                name,args = next(decisions)
                return {"choices":[{"message":{"tool_calls":[{"function":{"name":name,"arguments":json.dumps(args)}}]}}]}
        class RPC:
            def __init__(self): self.observations=[]; self.segment=-1
            def call(self,func_name,**kw):
                if func_name=='reset': self.segment+=1
                if func_name=='update_obs': self.observations.append((self.segment,kw['obs']['rsi_control_step']))
                if func_name=='get_action': return [dict(state) for _ in range(8)]
        class Env:
            env_seeds=[0]; success=[False]
            def __init__(self):
                self.count=0
                robots=[SimpleNamespace(arm_name=a+'_arm',type='target',gripper_scale=[-.01,.044],gripper_move={'sign':1}) for a in ('left','right')]
                self.robot_manager=SimpleNamespace(robot_list=robots,
                    get_real_endpose=lambda *a,**k: {0:np.array([0.,0.,1.,1.,0.,0.,0.])},
                    solve_ik=lambda **k: {'status':'Success','joint_value':[0.]*6})
            def get_obs(self): return dict(state=state,vision={c:{'color':np.zeros((4,4,3),np.uint8)} for c in p.CAMERAS}, hidden_object_pose='NEVER SEND')
            def take_action(self, action): self.count+=1
            def validate_action_dict(self, action): pass
            def is_episode_end(self):
                if self.count>=24: self.success[0]=True
                return self.count>=24
        with tempfile.TemporaryDirectory() as d:
            p.CLIENT=FakeClient(); p.ROOT=d; p.EXPERIENCE='candidate'
            rpc=RPC(); env=Env()
            p.eval_one_episode(env,rpc)
            summary=json.loads((Path(d)/'episode-summary.json').read_text())
            self.assertEqual(summary['counts'],dict(wam_actions=16,ik_actions=8,ik_rejections=0))
            self.assertEqual(rpc.observations,[(s,i) for s in range(2) for i in range(9)])
            self.assertTrue(summary['native_success'])

if __name__=='__main__': unittest.main()
