import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from environments.robodojo.runtime.robodojo_contact_scene import fit_bowl, ObservedBowlScene, check_fingers
from environments.robodojo.runtime.robodojo_local_geometry import LocalGeometry
from environments.robodojo.runtime.robodojo_motion_plan import PlanRejected
from environments.robodojo.runtime.robodojo_episode_memory import EpisodeMemory


class ContactTests(unittest.TestCase):
    def points(self):
        a=np.arange(4)*np.pi/2
        return np.column_stack([.06*np.cos(a),.06*np.sin(a),np.full(4,.85)])

    def test_bowl_fit_and_conservative_bounds(self):
        r=fit_bowl(self.points(),[0,.03,.78])
        self.assertAlmostEqual(r['radius_m'],.06)
        self.assertLess(r['bounds'][0][2],.78)
        self.assertGreater(r['bounds'][1][0],.06)

    def test_distributed_points_required(self):
        p=self.points(); p[:,1]=0
        with self.assertRaises(ValueError): fit_bowl(p,[0,0,.78])
        p=self.points(); p[0,2]+=.02
        with self.assertRaises(ValueError): fit_bowl(p,[0,0,.78])

    def test_scene_requires_three_fresh_distinct_bowls(self):
        s=ObservedBowlScene(); g=LocalGeometry(None)
        for i,p in enumerate([*self.points(),[0,0,.78]]):
            g.points[str(i)]=dict(point_world_m=p,control_step=4)
        s.register('one',['0','1','2','3'],'4',g,4)
        with self.assertRaises(ValueError): s.require_complete()
        with self.assertRaises(ValueError): s.register('two',['0','1','2','3'],'4',g,5)
        s.invalidate('one')
        with self.assertRaises(ValueError): s.boxes()

    def test_contact_exemption_is_local_and_target_only(self):
        box=np.array([[-.06,-.06,.78],[.06,.06,.85]])
        spheres=[[.06,0,.85,.004]]
        c=dict(label='target',point=np.array([.06,0,.85]))
        with self.assertRaises(PlanRejected): check_fingers(spheres,{'target':box})
        check_fingers(spheres,{'target':box},contact=c)
        with self.assertRaises(PlanRejected): check_fingers(spheres,{'other':box},contact=c)
        with self.assertRaises(PlanRejected): check_fingers([[0,0,.82,.004]],{'target':box},contact=c)

    def test_refresh_requires_matching_distributed_features(self):
        s=ObservedBowlScene(); g=LocalGeometry(None)
        for i,p in enumerate([*self.points(),[0,0,.78]]):
            g.points[str(i)]=dict(label=str(i),point_world_m=p,control_step=4)
        s.register('one',['0','1','2','3'],'4',g,4)
        before=np.array(s.bowls['one']['bounds'])
        for i,p in enumerate(self.points()[:3]):
            g.points['new'+str(i)]=dict(label=str(i),point_world_m=p+[.1,0,.03],control_step=5)
        s.invalidate('one')
        s.refresh('one',['0','1','2'],['new0','new1','new2'],g,5)
        np.testing.assert_allclose(s.boxes()['one'],before+[.1,0,.03])
        with self.assertRaises(ValueError): s.refresh('one',['0','1','2'],['new0','new1','new2'],g,5)

    def test_table_and_other_arm_never_exempt(self):
        s=[[0,0,1,.005]]; b=[[-.1,-.1,.9],[.1,.1,1.1]]
        with self.assertRaises(PlanRejected): check_fingers(s,{},table_boxes=[b])
        with self.assertRaises(PlanRejected): check_fingers(s,{},other_arm=[[0,0,1,.01]])
        check_fingers(s,{},other_arm=[[1,0,1,.01]])

    def test_memory_keeps_evidence_without_auto_promoting(self):
        with tempfile.TemporaryDirectory() as d:
            m=EpisodeMemory(d,lambda x:x.tolist())
            m.add(2,'move_to_point',{'measurement_id':'p'},dict(status='rejected',reason='unknown',executed_steps=0))
            m.add(3,'check_grasp_motion',{},dict(status='checked',follows_lift=True,executed_steps=0))
            m.save(native_success=False)
            r=json.loads((Path(d)/'experience-candidate.json').read_text())
            self.assertEqual(r['status'],'candidate_not_validated')
            self.assertFalse(r['native_success'])
            self.assertEqual(len(m.context()['earlier_evidence']),2)
            self.assertIn('step 2',(Path(d)/'experience-live.md').read_text())


if __name__=='__main__': unittest.main()
