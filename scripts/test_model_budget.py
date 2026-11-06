"""Shared model-decision budget regression tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from agents.budget import ModelBudget, ModelBudgetExhausted

class Client:
    def __init__(self): self.calls = 0
    def complete(self): self.calls += 1; return self.calls

class Tests(unittest.TestCase):
    def test_shared_roles_and_no_171st_dispatch(self):
        b=ModelBudget(170); c=Client()
        supervisor=b.client(c,'supervisor'); direct=b.client(c,'direct')
        review=b.callback(lambda: None,'wam_review')
        for _ in range(56): supervisor.complete(); direct.complete(); review()
        supervisor.complete(); direct.complete()
        with self.assertRaises(ModelBudgetExhausted): review()
        with self.assertRaises(ModelBudgetExhausted): direct.complete()
        self.assertEqual(b.used,170); self.assertEqual(c.calls,114)
    def test_nested_client_not_double_charged_but_retry_is(self):
        b=ModelBudget(2); c=Client(); metered=b.client(c,'retry')
        review=b.callback(lambda: (metered.complete(),metered.complete()),'review')
        review(); self.assertEqual(b.used,2); self.assertEqual(c.calls,2)
        with self.assertRaises(ModelBudgetExhausted): review()
    def test_failure_is_charged_and_new_episode_is_fresh(self):
        b=ModelBudget(1)
        def fail(): raise RuntimeError('transport')
        with self.assertRaises(RuntimeError): b.callback(fail,'review')()
        with self.assertRaises(ModelBudgetExhausted): b.callback(fail,'review')()
        self.assertEqual(ModelBudget(1).used,0)
    def test_invalid_limits(self):
        for x in (0,-1,True,1.5):
            with self.assertRaises(ValueError): ModelBudget(x)

if __name__=='__main__': unittest.main()
