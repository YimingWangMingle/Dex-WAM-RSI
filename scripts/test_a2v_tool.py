"""Real A2V RPC and shared inference helper with deterministic CPU model fixture."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from scripts import test_robodojo_a2v as base

class Tests(unittest.TestCase):
    setUp=base.Tests.setUp
    def test_independent_prediction_preserves_pending_review(self):
        pending=self.p.preview_candidates({'count':1,'predict':False})
        before=self.p._review
        plan=pending['candidates'][0]['actions']
        result=self.p.framework_predict(dict(decision_step=0,plans=[plan],candidate_ids=[5]))
        self.assertIs(self.p._review,before)
        self.assertEqual(self.p.next_decision,0)
        self.assertEqual(self.p.history.last_step,0)
        self.assertIsNone(self.p.last_command)
        self.assertEqual(result['candidates'][0]['action_hash'],pending['candidates'][0]['action_hash'])
        self.assertTrue(Path(result['candidates'][0]['forecast']['frames'][0]).exists())
        np.testing.assert_array_equal(self.p.select_candidate({'review_id':pending['review_id'],'candidate_id':0}),np.array(plan,np.float32)[:8])
    def test_stale_and_short_plans_rejected(self):
        for step,length in ((1,24),(0,8)):
            with self.assertRaises(ValueError):
                self.p.framework_predict(dict(decision_step=step,plans=[np.zeros((length,14)).tolist()],candidate_ids=[0]))
        self.assertEqual(self.p.next_decision,0)

if __name__=='__main__':unittest.main()
