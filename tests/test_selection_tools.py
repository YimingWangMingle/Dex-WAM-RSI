"""CPU contract tests: no claim of world-model accuracy."""
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from examples.modular.fixture_driver import Environment, options
from execution.episode import run_episode
from rsi.selection_policy import SelectionPolicy
from tools.contracts import TaskTool, ToolLimits, ToolSession, ToolRequest, ToolBudgetExceeded
from execution.runner import CallSkill, CallTool, Review
from execution.review import ReviewDecision
from skills.contracts import ActionCandidate, MeasuredObservation


def predictor(request):
    return {'predictions':[{'candidate_id':c.candidate_id,'forecast':{'fixture_x':c.actions[-1][0]}}
                           for c in request.candidates]}

class Agent:
    def __init__(self,skip=False,stale=False):self.skip=skip;self.stale=stale
    def next(self,view):
        if not view.pending:return CallSkill(view.available_skills[0],'move',1)
        p=view.pending
        if not self.skip and not p['tool_predictions']:
            return CallTool('a2v',dict(proposal_id=p['proposal_id'],
                observation_id='stale' if self.stale else view.latest_observation.observation_id,candidate_ids=[0]))
        return Review(ReviewDecision(p['proposal_id'],'accept','fixture review','move',1))

class Tests(unittest.TestCase):
    def run_episode(self,*,automatic=False,skip=False,stale=False,limit=32):
        with tempfile.TemporaryDirectory() as tmp:
            env=Environment();env.reset({'instruction':'move'})
            opts=options(.6);opts['agent_factory']=lambda load,budget:Agent(skip,stale)
            result=run_episode(env,root=Path(tmp)/'episode',max_steps=2,**opts,
                task_tools={'a2v':TaskTool('a2v','fixture',predictor)},tool_limits=ToolLimits(max_calls=limit),
                required_prediction_skills=('direct_control',),
                selection_policy=SelectionPolicy(('direct_control',),('a2v',),('a2v',) if automatic else (), 'agent'))
            events=[json.loads(x) for x in (Path(tmp)/'episode/framework-events.jsonl').read_text().splitlines()]
            return result,events
    def test_automatic_and_on_demand(self):
        for automatic in (False,True):
            result,events=self.run_episode(automatic=automatic)
            self.assertEqual(result['executed_steps'],2)
            self.assertEqual(result['framework']['tool_usage']['calls'],2)
            self.assertEqual([e['automatic'] for e in events if e['event']=='tool_call'],[automatic]*2)
    def test_missing_prediction_never_moves(self):
        result,events=self.run_episode(skip=True)
        self.assertEqual(result['executed_steps'],0)
        self.assertTrue(any(e['event']=='review_invalid' for e in events))
    def test_stale_request_never_moves(self):
        result,_=self.run_episode(stale=True)
        self.assertEqual(result['executed_steps'],0)
        self.assertEqual(result['framework']['tool_usage']['calls'],0)
    def test_budget_stops_before_second_motion(self):
        result,_=self.run_episode(limit=1)
        self.assertEqual(result['executed_steps'],1)
        self.assertEqual(result['stop_reason'],'tool_call_budget_exhausted')
    def test_policy_cannot_contain_host_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'policy.json';p.write_text(json.dumps(dict(skills=['direct_control'],max_calls=999)))
            with self.assertRaises(ValueError):SelectionPolicy.load(p)
    def test_failed_callback_is_charged(self):
        def fail(req):raise RuntimeError('inference failed')
        session=ToolSession({'a2v':TaskTool('a2v','fixture',fail)},limits=ToolLimits(max_calls=1))
        request=ToolRequest(MeasuredObservation('ep','obs',0,0.,'payload'),'proposal','contract',(ActionCandidate(0,((.1,),)),))
        with self.assertRaises(RuntimeError):session.invoke('a2v',request)
        self.assertEqual(session.calls,1)
        self.assertEqual(session.receipts,[])
        with self.assertRaises(ToolBudgetExceeded):session.invoke('a2v',request)
    def test_receipt_bound_to_tool_candidate_and_observation(self):
        from types import SimpleNamespace as NS
        obs=MeasuredObservation('ep','obs',0,0.,'payload')
        request=ToolRequest(obs,'proposal','contract',(ActionCandidate(7,((.1,),)),))
        session=ToolSession({'other':TaskTool('other','fixture',predictor)})
        session.invoke('other',request)
        proposal=NS(proposal_id='proposal',identity=NS(action_contract='contract'),actions=((.1,),))
        self.assertIsNone(session.matching(proposal,obs,((.1,),),7))
        self.assertTrue(session.matching(proposal,obs,((.1,),),7,tool='other'))
        self.assertIsNone(session.matching(proposal,obs,((.1,),),8,tool='other'))
        self.assertIsNone(session.matching(proposal,replace(obs,observation_id='next'),((.1,),),7,tool='other'))

if __name__=='__main__':unittest.main()
