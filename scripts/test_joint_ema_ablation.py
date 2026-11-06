from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import numpy as np
from thinker_wm.runtime.joint_continuity import JointEMA, build_joint_postprocessor, JOINTS

class EMAAblationTests(unittest.TestCase):
    def test_ema_has_no_bridge_limits_or_gripper_delay(self):
        raw=np.zeros((8,14),np.float32)
        raw[0,JOINTS]=2
        raw[1:,JOINTS]=4
        raw[:,[6,13]]=np.arange(8)[:,None]%2
        result,d=JointEMA(ema_alpha=.5).process(raw,np.zeros(14),0,8)
        np.testing.assert_array_equal(result[0,JOINTS],2)
        np.testing.assert_array_equal(result[1,JOINTS],3)
        np.testing.assert_array_equal(result[2,JOINTS],3.5)
        np.testing.assert_array_equal(result[:,[6,13]],raw[:,[6,13]])
        self.assertFalse(d['overlap_used'])
        self.assertEqual(d['joint_limit_clipped_elements'],0)

    def test_no_previous_chunk_state_and_identity_mode(self):
        p=JointEMA()
        p.process(np.ones((8,14)),np.zeros(14),0,8)
        raw=np.zeros((8,14),np.float32)
        out,_=p.process(raw,np.ones(14),8,8)
        np.testing.assert_array_equal(out,raw)
        raw[:,:6]=np.arange(8)[:,None]
        out,_=JointEMA(ema_alpha=1).process(raw,np.zeros(14),0,8)
        np.testing.assert_array_equal(out,raw)

    def test_selection_and_invalid_configuration(self):
        self.assertIsNone(build_joint_postprocessor(None,control_hz=25))
        self.assertIsInstance(build_joint_postprocessor({'mode':'ema','ema_alpha':.5},control_hz=25),JointEMA)
        with self.assertRaises(ValueError):build_joint_postprocessor({'mode':'unknown'},control_hz=25)
        for alpha in [0,-1,1.1,float('nan')]:
            with self.assertRaises(ValueError):JointEMA(ema_alpha=alpha)

if __name__=='__main__':unittest.main(verbosity=2)
