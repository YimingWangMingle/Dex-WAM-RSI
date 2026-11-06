"""Pose search, rejection, geometry freshness and live execution guard tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.robodojo.runtime.robodojo_pose_control import (orientation_candidates, quaternion_angle,
    RejectionLedger, tracking_error, validate_transfer_points)
from environments.robodojo.runtime.robodojo_motion_plan import NativeMotionPlanner, planner_diagnostics
from environments.robodojo.runtime.robodojo_local_geometry import LocalGeometry


class PoseTests(unittest.TestCase):
    def test_candidates_and_quaternion_sign(self):
        qs=list(orientation_candidates([.5,-.5,.5,.5],30))
        self.assertEqual(len(qs),9)
        for q in qs:
            self.assertAlmostEqual(np.linalg.norm(q),1)
            self.assertLessEqual(quaternion_angle(q,qs[0]),np.deg2rad(30)+1e-8)
        self.assertAlmostEqual(quaternion_angle(qs[0],-qs[0]),0)
        for bad in (float('nan'),46,-1,True):
            with self.assertRaises(ValueError): list(orientation_candidates([1,0,0,0],bad))

    def test_repetition_uses_state_and_equivalent_quaternion(self):
        ledger=RejectionLedger()
        k=ledger.key('left',[0,0,1],[1,0,0,0],[0,1],'s',30)
        ledger.reject(k); self.assertFalse(ledger.blocked(k)); ledger.reject(k)
        self.assertTrue(ledger.blocked(ledger.key('left',[0,0,1],[-1,0,0,0],[0,1],'s',30)))
        self.assertFalse(ledger.blocked(ledger.key('left',[0,0,1],[1,0,0,0],[.01,1],'s',30)))

    def test_transfer_and_tracking_validation(self):
        valid=[[0,0,1],[0,0,.95],[0,0,1.02],[.1,0,1.02]]
        validate_transfer_points(valid)
        with self.assertRaises(ValueError): validate_transfer_points(valid[:3])
        with self.assertRaises(ValueError): validate_transfer_points([[0,0,1]]*4)
        self.assertEqual(tracking_error([float('nan')],[0]),float('inf'))

    def test_diagnostics_restored_even_on_exception(self):
        class Solver:
            def _get_result(self,*args): return None
        solver=Solver(); original=solver._get_result
        with self.assertRaises(RuntimeError):
            with planner_diagnostics(NS(motion_planner=NS(ik_solver=solver))):
                raise RuntimeError('test')
        self.assertEqual(solver._get_result,original)

    def test_geometry_point_and_freshness(self):
        from tools.perception.geometry import CameraFrame
        image=np.zeros((100,100,3),np.uint8); k=np.array([[100,0,50],[0,100,50],[0,0,1]])
        t1=np.eye(4); t2=np.eye(4); t2[0,3]=.2
        frames={n:CameraFrame(image,k,t,0,'test',True,(100,100)) for n,t in [('a',t1),('b',t2)]}
        tool=LocalGeometry(None)
        with patch.object(tool,'camera',side_effect=lambda n,*a:frames[n]):
            result=tool.measure({},0,'a',[50,50],'b',[30,50],'rim')
            np.testing.assert_allclose(result['point_world_m'],[0,0,1],atol=1e-8)
            tool.current(result['measurement_id'],0)
            with self.assertRaises(ValueError): tool.current(result['measurement_id'],1)
            with self.assertRaises(ValueError): tool.measure({},0,'a',[50,50],'b',[30,80],'rim')

    def test_grasp_following_requires_object_motion(self):
        tool=LocalGeometry(None)
        tool.points={'b':dict(label='rim',control_step=0,point_world_m=[0,0,1]),
                     'a':dict(label='rim',control_step=1,point_world_m=[0,0,1])}
        self.assertFalse(tool.check_following('b','a',[0,0,1],[0,0,1.02],1)['follows_lift'])
        tool.points['a']['point_world_m']=[0,0,1.02]
        self.assertTrue(tool.check_following('b','a',[0,0,1],[0,0,1.02],1)['follows_lift'])


class PilotTests(unittest.TestCase):
    def run_pilot(self, requests, *, rejected=False, lag=False, holding=False):
        from environments.robodojo.runtime import robodojo_deepseek_ik as pilot
        state={a+'_arm_joint_state':np.zeros(6) for a in ('left','right')}
        state.update({a+'_ee_joint_state':np.ones(1) for a in ('left','right')})
        active=[False]
        simulated_orientation=[[1,0,0,0]]
        class Planner:
            active_joints_name=[f'joint{i}' for i in range(1,7)]
            def plan_path(self,start,pose,base):
                if active[0] and (rejected(pose) if callable(rejected) else rejected): return {'status':'Fail'}
                simulated_orientation[0]=list(pose[3:])
                goal=np.zeros(6); goal[:3]=np.asarray(pose[:3])-[0,0,1]
                return dict(status='Success',position=[start,goal])
        native=Planner()
        robots=[NS(type='target',arm_name=a+'_arm',robot_name=a,gripper_bias=.145,
                   arm_joints_name=native.active_joints_name,arm_joint_indices=list(range(6)),
                   entity_origin_pose=[0,0,0,1,0,0,0]) for a in ('left','right')]
        asset=NS(data=NS(soft_joint_pos_limits=np.array([[[-3,3]]*6]),joint_vel_limits=np.array([[5]*6])))
        def flange(robot,*a,**kw): return {0:np.r_[state[robot.robot_name+'_arm_joint_state'][:3]+[0,0,1],simulated_orientation[0]]}
        class Env:
            env_seeds=[1]; success=[False]; count=0
            obs_manager=NS(collect_freq=25)
            robot_manager=NS(robot_list=robots,robot_key=[asset,asset],planner={'left':native,'right':native},
                             get_real_endpose=flange,get_end_effector_real_val=lambda *a:{0:[.044,.044]})
            def get_obs(self): return dict(state=state,vision={c:{'color':np.zeros((4,4,3),np.uint8)} for c in pilot.CAMERAS})
            def take_action(self,action):
                self.count+=1
                state.update({k:np.asarray(v).copy() for k,v in action.items()})
                if active[0] and lag: state['left_arm_joint_state'][0]+=(.3 if lag is True else lag)
            def validate_action_dict(self,action): pass
            def is_episode_end(self): return False
        class Client:
            last_metadata={}; index=0
            def complete(self,*args):
                active[0]=True
                if holding: state['left_ee_joint_state'][0]=0
                if self.index<len(requests): name,data=requests[self.index]
                else: name,data='stop',dict(reason='done')
                self.index+=1
                return {'choices':[{'message':{'tool_calls':[{'function':{'name':name,'arguments':json.dumps(data)}}]}}]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.dict(os.environ,{'RSI_CONTACT_SCENE':'0'}), \
             patch.object(pilot,'CLIENT',Client()), patch.object(pilot,'ROOT',directory), \
             patch.object(NativeMotionPlanner,'robot_spheres_world',return_value=[[0,0,0,.1]]), \
             patch.object(NativeMotionPlanner,'update_other_arm_scene',return_value={'test_double':True}):
            env=Env(); pilot.eval_one_episode(env,None)
            return [json.loads(s) for s in (Path(directory)/'events.jsonl').read_text().splitlines()],env.count

    def test_two_rejections_block_third_without_motion(self):
        req=('move_tcp',dict(arm='left',position=[.16,0,1],orientation='keep',reason='test'))
        rows,count=self.run_pilot([req]*3,rejected=True)
        outcomes=[r for r in rows if r['kind']=='motion_result' and r['control_step']==4]
        self.assertEqual(outcomes[-1]['status'],'repeat_blocked')
        self.assertEqual(count,4)

    def test_tracking_divergence_stops_remaining_commands(self):
        req=('move_tcp',dict(arm='left',position=[.145,0,1.3],orientation='keep',reason='test'))
        rows,count=self.run_pilot([req],lag=True)
        outcome=[r for r in rows if r['kind']=='motion_result'][-1]
        self.assertEqual(outcome['status'],'execution_tracking_diverged')
        self.assertEqual(outcome['executed_steps'],1)
        self.assertGreater(outcome['planned_steps'],1)

    def test_nonconverging_servo_has_finite_hold_budget(self):
        req=('move_tcp',dict(arm='left',position=[.145,0,1.3],orientation='keep',reason='test'))
        rows,count=self.run_pilot([req],lag=.01)
        outcome=[r for r in rows if r['kind']=='motion_result'][-1]
        self.assertEqual(outcome['status'],'execution_settle_timeout')
        self.assertEqual(outcome['waypoint_hold_steps'],25)
        self.assertEqual(outcome['executed_steps'],26)

    def test_down_search_reports_selected_pose_and_zero_failed_execution(self):
        req=('move_tcp',dict(arm='left',position=[.16,0,1],orientation='down',reason='test'))
        rows,count=self.run_pilot([req],rejected=lambda pose:np.allclose(pose[3:],[.5,-.5,.5,.5]))
        result=[r for r in rows if r['kind']=='motion_result'][-1]
        self.assertEqual(result['status'],'executed')
        self.assertEqual(len(result['orientation_trials']),2)
        self.assertEqual(result['orientation_trials'][0]['executed_steps'],0)
        self.assertFalse(np.allclose(result['selected_quaternion_wxyz'],[.5,-.5,.5,.5]))

    def test_gripper_cannot_close_without_preview(self):
        rows,count=self.run_pilot([('grip',dict(arm='left',opening=0,reason='test'))])
        self.assertEqual(count,4)
        self.assertTrue(any(r['kind']=='tool_rejected' and 'preview_transfer_required' in r['reason'] for r in rows))

    def test_preview_pass_does_not_execute_or_waive_fresh_measurement(self):
        request=('preview_transfer',dict(arm='left',points=[[.145,0,1.02],[.145,0,1],[.145,0,1.02],[.20,0,1.02]],
            quaternion_wxyz=[1,0,0,0],max_tilt_deg=0,reason='test'))
        with patch.object(NativeMotionPlanner,'check_orientation_path',return_value=None):
            rows,count=self.run_pilot([request,('grip',dict(arm='left',opening=0,reason='test'))])
        self.assertEqual(count,4)
        result=[r for r in rows if r['kind']=='tool_result'][0]
        self.assertEqual(result['status'],'kinematic_preview_passed')
        self.assertEqual(result['executed_steps'],0)
        self.assertTrue(any(r['kind']=='tool_rejected' and 'missing_or_stale_point' in r['reason'] for r in rows))

    def test_closed_gripper_cannot_search_tilts(self):
        req=('move_tcp',dict(arm='left',position=[.16,0,1],orientation='down',reason='test'))
        rows,count=self.run_pilot([req],holding=True)
        self.assertEqual(count,4)
        self.assertTrue(any(r['kind']=='tool_rejected' and ('open_gripper' in r['reason'] or 'possibly_holding' in r['reason']) for r in rows))


if __name__=='__main__': unittest.main()
