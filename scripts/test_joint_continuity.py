from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import numpy as np
from thinker_wm.runtime.joint_continuity import JointContinuity, PlanningHorizon, JOINTS

class ContinuityTests(unittest.TestCase):
    def processor(self, **kwargs):
        return JointContinuity(control_hz=25, max_velocity_rad_s=[1.]*12,
            max_acceleration_rad_s2=[2.]*12, **kwargs)

    def test_limits_and_gripper_timing(self):
        p=self.processor()
        raw=np.ones((16,14))*10
        raw[:,[6,13]]=np.arange(16)[:,None]%2
        plan,d=p.process(raw,np.zeros(14),0,8)
        velocity=np.diff(np.vstack([np.zeros(12),plan[:,JOINTS]]),axis=0)*25
        acceleration=np.diff(np.vstack([np.zeros(12),velocity]),axis=0)*25
        self.assertLessEqual(np.abs(velocity).max(),1.00001)
        self.assertLessEqual(np.abs(acceleration).max(),2.0001)
        np.testing.assert_array_equal(plan[:,[6,13]],raw[:,[6,13]])
        self.assertFalse(d['forecast_matches_commands'])
        self.assertEqual(len(d['retained_tail']),8)

    def test_overlap_uses_actual_time_and_stale_is_discarded(self):
        p=self.processor(ema_alpha=1,bridge_steps=0)
        raw=np.arange(16)[:,None]*np.ones((1,14))*.001
        old,_=p.process(raw,np.zeros(14),0,8)
        new=np.zeros((16,14))
        _,d=p.process(new,old[3],4,8)
        np.testing.assert_allclose(np.array(d['fused'])[0,JOINTS],old[4,JOINTS]*.5,atol=1e-8)
        self.assertTrue(d['overlap_used'])
        _,d=p.process(new,np.ones(14),8,8)
        self.assertTrue(d['stale_tail'])
        self.assertFalse(d['overlap_used'])
        p.reset()
        _,d=p.process(new,np.zeros(14),0,8)
        self.assertEqual(d['overlap_rows'],0)

    def test_invalid_horizons_and_nonfinite(self):
        for blocks,execute in [(0,8),(4,8),(2,17),(2,0)]:
            with self.assertRaises(ValueError):PlanningHorizon(blocks,execute)
        p=self.processor()
        with self.assertRaises(ValueError):p.process(np.full((16,14),np.nan),np.zeros(14),0,8)
        p.process(np.zeros((16,14)),np.zeros(14),0,8)
        with self.assertRaises(ValueError):p.process(np.zeros((16,14)),np.zeros(14),0,8)

if __name__=='__main__':unittest.main(verbosity=2)
