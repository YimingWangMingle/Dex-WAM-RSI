"""Perception gates, conservative scene admission, and independent lift checks."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from environments.robodojo.runtime.robodojo_bowl_perception import BowlPerception, verify_lift
from environments.robodojo.runtime.robodojo_contact_scene import ObservedBowlScene
from environments.robodojo.runtime.robodojo_bowl_vision import extract_rim
from scripts.test_robodojo_pose_control import PilotTests
from tools.perception.geometry import CameraFrame
from environments.robodojo.runtime.robodojo_local_geometry import qualified_rgb_intrinsics
from environments.robodojo.runtime.robodojo_observation_pose import observation_pose, head_approach_pose
from environments.robodojo.runtime.robodojo_run_limits import read_limits, exceeded


def estimate(x=0., z=.85):
    return dict(status='accepted',bowl_id='bowl_1',episode_id='test',observation_id=1,control_step=1,identity_continuous=True,center_xyz_m=[x,0,z],radius_m=.06,
                uncertainty=dict(center_bound_m=.001,radius_bound_m=.001))


class PerceptionTests(unittest.TestCase):
    def test_unlimited_budgets_do_not_stop_and_explicit_budgets_still_work(self):
        self.assertIsNone(exceeded(100000,1e9,read_limits({})))
        self.assertEqual(exceeded(10,1,read_limits({'RSI_MAX_AGENT_CALLS':'10'})), 'agent_call_budget')
        self.assertEqual(exceeded(0,60,read_limits({'RSI_MAX_EPISODE_WALL_SECONDS':'60'})), 'episode_wall_time_budget')
        for config in [{'RSI_MAX_AGENT_CALLS':'-1'}, {'RSI_MAX_EPISODE_WALL_SECONDS':'nan'}]:
            with self.assertRaises(ValueError):read_limits(config)

    def test_head_approach_raises_then_moves_without_contact_authorization(self):
        bounds=[[-.1,-.1,.755],[.1,.1,.935]]
        q=[1,0,0,0]
        first=head_approach_pose(bounds,[.4,0,.96],q)
        self.assertEqual(first['stage'],'raise_before_horizontal_approach')
        self.assertAlmostEqual(first['tcp_world_m'][0],.4)
        self.assertGreaterEqual(first['tcp_world_m'][2],1.015-1e-9)
        second=head_approach_pose(bounds,first['tcp_world_m'],q)
        self.assertEqual(second['stage'],'high_approach')
        self.assertAlmostEqual(np.linalg.norm(np.array(second['tcp_world_m'])-first['tcp_world_m']),.30)
        self.assertFalse(second['contact_authorized'])
        self.assertEqual(second['quaternion_wxyz'],q)
        with self.assertRaises(ValueError):head_approach_pose([[0,0,0],[1,1,1.5]],[0,0,1],q)

    def test_renderer_intrinsics_are_signature_bound(self):
        k=np.array([[288.13253784,0,320],[0,336.46432495,240],[0,0,1]])
        corrected,revision=qualified_rgb_intrinsics('cam_head',k,(480,640),tiled_capture=True)
        self.assertEqual(corrected[0,0],corrected[1,1])
        self.assertNotEqual(k[0,0],k[1,1])  # original SDK matrix is unchanged
        self.assertIn('renderer-qualified',revision)
        for shape,tiled in [((240,320),True),((480,640),False)]:
            with self.assertRaises(ValueError):qualified_rgb_intrinsics('cam_head',k,shape,tiled_capture=tiled)

    def test_one_localized_bowl_does_not_waive_other_obstacles(self):
        scene=ObservedBowlScene()
        rows=[dict(bowl_id=str(i),status='unknown',obstacle_bounds_m=[[i*.2-.08,-.08,.755],[i*.2+.08,.08,.935]],
                   obstacle_observation_step=5) for i in range(3)]
        rows[0].update(status='localized',geometry=estimate())
        scene.update_visual_instances(rows,5)
        scene.require_target('0',5)
        self.assertEqual(len(scene.boxes()),3)
        with self.assertRaises(ValueError):scene.require_target('1',5)
        with self.assertRaises(ValueError):scene.require_target('0',6)
        rows[1]['status']='stale'
        scene.update_visual_instances(rows,6)
        scene.require_target('0',6)
        self.assertTrue(scene.bowls['1']['obstacle_evidence_stale'])
        with self.assertRaises(ValueError):scene.require_target('1',6)

    def test_lost_head_detection_retains_id_without_geometry(self):
        class Detector:
            provenance={}
            rows=[dict(box_xyxy=b,score=.8) for b in [[5,40,25,60],[40,40,60,60],[75,40,95,60]]]
            def detect(self,rgb):return self.rows
        detector=Detector();p=BowlPerception('',detector=detector)
        t=np.diag([1.,-1.,-1.,1.]);t[2,3]=1.3
        frame=CameraFrame(np.zeros((100,100,3),np.uint8),np.array([[100,0,50],[0,100,50],[0,0,1]]),t,0,'test',True,(100,100))
        with patch('environments.robodojo.runtime.robodojo_bowl_perception.extract_rim',return_value=dict(status='unknown',reason='occluded')):
            p.observe({'cam_head':frame},1)
            detector.rows=[detector.rows[0],detector.rows[2]]
            r=p.observe({'cam_head':frame},2)
            detector.rows=[dict(box_xyxy=b,score=.8) for b in [[5,40,25,60],[40,40,60,60],[75,40,95,60]]]
            first_return=p.observe({'cam_head':frame},3)
            self.assertEqual(p.latest['bowl_2']['status'],'stale')
            p.observe({'cam_head':frame},4)
            self.assertEqual(p.latest['bowl_2']['status'],'unknown')
            self.assertNotIn('bowl_2',p.lost_ids)
            detector.rows=[detector.rows[0],detector.rows[2]]
            p.observe({'cam_head':frame},5)
        self.assertEqual(set(p.latest),{'bowl_1','bowl_2','bowl_3'})
        self.assertEqual(p.latest['bowl_2']['status'],'stale')
        self.assertEqual(p.latest['bowl_1']['status'],'unknown')
        with self.assertRaises(ValueError):p.current('bowl_2',2)

    def test_view_proposal_uses_mount_and_keeps_contact_unauthorized(self):
        camera=np.eye(4);camera[:3,3]=[0,0,1.1]
        result=observation_pose([[-.1,-.1,.755],[.1,.1,.935]],camera,[0,0,1,1,0,0,0],.145,'front')
        self.assertGreaterEqual(result['tcp_world_m'][2],.95)
        self.assertFalse(result['contact_authorized'])
        self.assertAlmostEqual(np.linalg.norm(result['quaternion_wxyz']),1)

    def test_robot_motion_alone_cannot_validate_grasp(self):
        before=estimate()
        stationary=copy.deepcopy(before);stationary.update(observation_id=2,control_step=2)
        self.assertFalse(verify_lift(before,stationary,[0,0,.85],[0,0,.87])['follows_lift'])
        after=estimate(z=.87);after.update(observation_id=2,control_step=2)
        self.assertTrue(verify_lift(before,after,[0,0,.85],[0,0,.87])['follows_lift'])
        wrong=copy.deepcopy(after);wrong['bowl_id']='bowl_2'
        with self.assertRaises(ValueError):verify_lift(before,wrong,[0,0,.85],[0,0,.87])
        with self.assertRaises(ValueError):verify_lift(before,before,[0,0,.85],[0,0,.87])
        after['radius_m']=.08
        self.assertFalse(verify_lift(before,after,[0,0,.85],[0,0,.87])['follows_lift'])

    def test_scene_requires_atomic_three_object_geometry(self):
        scene=ObservedBowlScene()
        good={str(i):estimate(x=(i-1)*.2) for i in range(3)}
        scene.register_visual_rims(good,4)
        previous=copy.deepcopy(scene.bowls)
        bad=copy.deepcopy(good); bad['1']['status']='unknown'
        with self.assertRaises(ValueError):scene.register_visual_rims(bad,5)
        self.assertEqual(previous,scene.bowls)
        self.assertFalse(scene.bowls['0']['bottom_is_observed'])
        self.assertGreater(scene.bowls['0']['margin_m'],.005)
        with self.assertRaises(ValueError):scene.register_visual_rims({'0':good['0']},6)
        with self.assertRaises(ValueError):scene.register_visual_rims({str(i):estimate() for i in range(3)},6)

    def test_geometry_has_to_be_current_and_localized(self):
        perception=BowlPerception('',detector=object())
        perception.latest={'bowl_1':dict(status='localized',geometry=estimate())};perception.step=3
        perception.current('bowl_1',3)
        with self.assertRaises(ValueError):perception.current('bowl_1',4)
        perception.latest['bowl_1']['status']='unknown'
        with self.assertRaises(ValueError):perception.current('bowl_1',3)

    def test_partial_perception_log_and_no_contact(self):
        # Exercise the actual pilot dispatch, logging, and rejected grasp path.
        # No model download/GPU/simulator required for this contract test.
        image=np.zeros((4,4,3),np.uint8)
        frame=CameraFrame(image,np.array([[10,0,2],[0,10,2],[0,0,1]]),np.eye(4),0,'test',True,(4,4))
        class Perception:
            latest={}
            def __init__(self,*a):pass
            def observe(self,frames,step):
                return dict(status='partial',control_step=step,seconds=.01,instances=[])
        requests=[('observe_bowls',dict(reason='test')),
                  ('preview_bowl_grasp',dict(arm='left',source='bowl_1',destination='bowl_2',reason='test')),
                  ('grip',dict(arm='left',opening=0,reason='test')),
                  ('measure_point',dict(camera='cam_head',camera2='cam_left_wrist',uv=[1,1],uv2=[1,1],label='bowl',reason='forbidden legacy path'))]
        with patch.dict(os.environ,{'RSI_BOWL_PERCEPTION_MODEL':'test'}), \
             patch('environments.robodojo.runtime.robodojo_bowl_perception.BowlPerception',Perception), \
             patch('environments.robodojo.runtime.robodojo_local_geometry.LocalGeometry.camera',return_value=frame):
            rows,count=PilotTests().run_pilot(requests)
        self.assertEqual(count,4)  # calibration only, zero requested contact steps
        self.assertTrue(any(r['kind']=='perception_result' and r['status']=='partial' for r in rows))
        rejected=[r for r in rows if r['kind']=='tool_rejected']
        self.assertEqual(len(rejected),3)
        self.assertTrue(any(r['reason']=='invalid tool fields' for r in rejected))

    def test_visible_rim_and_clipped_wrist(self):
        import cv2
        rgb=np.zeros((300,400,3),np.uint8);rgb[:]=[120,55,20]
        cv2.ellipse(rgb,(200,150),(65,40),15,0,360,(240,245,245),-1)
        r=extract_rim(rgb,[120,95,280,205])
        self.assertEqual(r['status'],'rim_observed')
        self.assertGreater(r['coverage_deg'],340)
        cropped=rgb[145:,:,:]
        r=extract_rim(cropped,[120,0,280,60])
        self.assertEqual(r['status'],'unknown')


if __name__=='__main__':unittest.main()
