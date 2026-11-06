"""Recovery regression: no movement before fresh review; actual edited-plan binding."""
import test_framework_native as fixtures
from test_framework_native import Generic, Client, Env
import json
from environments.robodojo.action_revision import revised_plans
import unittest


def choice(operation, candidate_id=None, execute_steps=0, edits=None):
    return dict(operation=operation, candidate_id=candidate_id,
                execute_steps=execute_steps, edits=edits or [], reason='test recovery')


class Tests(unittest.TestCase):
    setUp = fixtures.Tests.setUp
    def run_episode(self,env,**kw):
        return fixtures.Tests.run_episode(self,env,review_protocol="experiment-v3",**kw)
    events = fixtures.Tests.events

    def test_rejection_resamples_different_actions_then_executes_seven(self):
        env=Env(stop=7);client=Client(Generic(self.root));packets=[]
        def review(p):
            packets.append(p)
            self.assertEqual(env.steps,0)
            return choice('resample') if len(packets)==1 else choice('execute',0,7)
        result=self.run_episode(env,client=client,mode3=False,predict=False,candidate_count=3,reviewer=review)
        self.assertEqual(result['executed_steps'],7)
        self.assertEqual(json.loads((self.root/'run/episode-summary.json').read_text())['model_budget']['used'],2)
        self.assertNotEqual(packets[0]['candidates'][0]['actions'],packets[1]['candidates'][0]['actions'])
        self.assertEqual(packets[1]['recovery_feedback']['operation'],'resample')
        self.assertEqual(sum(n=='framework_select' for n,_ in client.calls),1)

    def test_edit_repredicts_then_requires_review(self):
        env=Env(stop=7);client=Client(Generic(self.root));packets=[]
        def review(p):
            packets.append(p)
            self.assertEqual(env.steps,0)
            if len(packets)==1:
                return choice('edit',0,edits=[dict(start=0,end=7,channel=6,value=.9)])
            old,new=packets[0]['candidates'][0],p['candidates'][0]
            self.assertNotEqual(old['action_hash'],new['action_hash'])
            self.assertEqual(new['action_hash'],new['forecast']['action_hash'])
            self.assertNotEqual(old['forecast']['frames'],new['forecast']['frames'])
            return choice('execute',0,7)
        result=self.run_episode(env,client=client,mode3=False,predict=True,candidate_count=3,reviewer=review)
        self.assertEqual(result['executed_steps'],7)
        self.assertAlmostEqual(env.actions[0][6],.9,places=6)
        self.assertEqual(json.loads((self.root/'run/episode-summary.json').read_text())['counters']['rejected'],1)

    def test_resample_is_bounded_by_global_model_budget(self):
        env=Env();client=Client(Generic(self.root))
        result=self.run_episode(env,client=client,mode3=False,predict=False,candidate_count=3,
            reviewer=lambda _:choice('resample'),max_model_calls=3)
        self.assertEqual(result['model_budget']['used'],3)
        self.assertEqual(env.steps,0)
        self.assertEqual(result['stop_reason'],'model_budget_exhausted')

    def test_explicit_stop_and_invalid_edits_move_nothing(self):
        env=Env();client=Client(Generic(self.root))
        result=self.run_episode(env,client=client,mode3=False,predict=False,candidate_count=3,reviewer=lambda _:choice('stop'))
        self.assertEqual(env.steps,0)
        self.assertIn('reviewer_stop:',result['stop_reason'])
        c=[dict(candidate_id=0,actions=[[0.]*14]*24)]
        for edit in [dict(start=0,end=25,channel=0,value=1),dict(start=0,end=1,channel=6,value=2),dict(start=0,end=1,channel=0,value=float('nan'))]:
            with self.assertRaises(ValueError):revised_plans(c,0,[edit])

if __name__=='__main__':unittest.main()
