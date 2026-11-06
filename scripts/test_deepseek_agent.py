"""Offline command-boundary regression; no API or robot calls."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.deepseek import DeepSeekAgent, DeepSeekError
from skills.contracts import MeasuredObservation
from execution.runner import AgentView, CallSkill, ReadExperience


class Client:
    last_metadata = {}
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments
    def complete(self, messages, tools):
        self.messages, self.tools = messages, tools
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": self.name, "arguments": json.dumps(self.arguments)}}]}}]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.obs = MeasuredObservation('e', 'o', 0, 0, '/not-a-live-bundle')
        self.view = AgentView([], self.obs, None, None, ('wam_control',), 0)
    def invoke(self, name, args, *, loader=None):
        self.client = Client(name, args)
        loader = loader or (lambda _: dict(observation_id='o', state=[0]*14,
            images=[dict(camera='head', data_url='data:image/png;base64,fixture')],
            success=True, score=100))
        return DeepSeekAgent(self.client, loader, instruction='Stack the three bowls together.').next(self.view)
    def test_pending_can_request_prediction_but_not_new_skill(self):
        from execution.runner import CallTool
        self.view=AgentView([],self.obs,'wam_control',{'proposal_id':'p'},('wam_control',),1,available_tools=('a2v',))
        command=self.invoke('call_tool',dict(tool='a2v',proposal_id='p',observation_id='o',candidate_ids=[0],reason='compare consequences'))
        self.assertIsInstance(command,CallTool)
        self.assertEqual(command.arguments['candidate_ids'],[0])
        with self.assertRaises(DeepSeekError):
            self.invoke('call_tool',dict(tool='unknown',proposal_id='p',observation_id='o',candidate_ids=[0],reason='x'))

    def test_policy_hides_experience(self):
        from dataclasses import replace
        self.view=replace(self.view,allow_experience=False)
        with self.assertRaises(DeepSeekError):
            self.invoke('read_experience',dict(op='list',path='',reason='x'))

    def test_prediction_loader_labels_unmeasured_images(self):
        view=AgentView([],self.obs,'wam_control',{'proposal_id':'p','tool_predictions':[{'receipt_id':'receipt'}]},('wam_control',),1)
        client=Client('stop',{'reason':'uncertain'})
        loaded=[]
        def predictions(receipts):
            loaded.extend(receipts)
            return [dict(label='candidate 0/frame 1',data_url='data:image/png;base64,fixture')]
        agent=DeepSeekAgent(client,lambda _:dict(observation_id='o',images=[dict(camera='head',data_url='data:image/png;base64,fixture')]),
            instruction='move',prediction_loader=predictions)
        agent.next(view)
        self.assertEqual(loaded,[{'receipt_id':'receipt'}])
        self.assertIn('not measured',client.messages[1]['content'][-2]['text'])

    def test_wam_language_is_original(self):
        result = self.invoke('call_backend', dict(backend='wam_control', subgoal='lift slightly',
                                                max_actions=8, reason='observed a misalignment'))
        self.assertIsInstance(result, CallSkill)
        self.assertEqual(result.subgoal, 'Stack the three bowls together.')
    def test_unregistered_direct_rejected(self):
        with self.assertRaises(DeepSeekError):
            self.invoke('call_backend', dict(backend='direct_control', subgoal='move', max_actions=1, reason='x'))
    def test_unknown_tool_rejected(self):
        with self.assertRaises(DeepSeekError): self.invoke('execute_ik', dict(reason='x'))
    def test_boolean_prefix_rejected(self):
        with self.assertRaises(DeepSeekError):
            self.invoke('call_backend', dict(backend='wam_control', subgoal='move', max_actions=True, reason='x'))
    def test_scoring_not_sent(self):
        self.invoke('read_experience', dict(op='list', path='', reason='inspect task notes'))
        sent = json.loads(self.client.messages[1]['content'][0]['text'])
        self.assertNotIn('score', sent)
        self.assertNotIn('success', sent)
        self.assertEqual(self.client.messages[1]['content'][-1]['type'], 'image_url')
    def test_stale_image_rejected_before_api(self):
        with self.assertRaises(ValueError):
            self.invoke('stop', dict(reason='x'), loader=lambda _: dict(observation_id='other'))
    def test_pending_disables_new_motion(self):
        self.view = AgentView([], self.obs, 'wam_control', {'proposal_id':'p'}, ('wam_control',), 1)
        with self.assertRaises(DeepSeekError):
            self.invoke('call_backend', dict(backend='wam_control', subgoal='move', max_actions=1, reason='x'))
    def test_stale_review_rejected(self):
        self.view = AgentView([], self.obs, 'wam_control', {'proposal_id':'p'}, ('wam_control',), 1)
        with self.assertRaises(DeepSeekError):
            self.invoke('review', dict(proposal_id='other', decision='accept', approved_steps=1, reason='x'))
    def test_review_selects_candidate(self):
        self.view = AgentView([],self.obs,'wam_control',dict(proposal_id='p',candidates=[dict(candidate_id=2)]),('wam_control',),1)
        result=self.invoke('review',dict(proposal_id='p',decision='accept',approved_steps=4,reason='visual evidence',candidate_id=2))
        self.assertEqual(result.decision.candidate_id,2)
        with self.assertRaises(DeepSeekError):
            self.invoke('review',dict(proposal_id='p',decision='accept',approved_steps=4,reason='evidence',candidate_id=True))

    def test_experience_command(self):
        self.assertIsInstance(self.invoke('read_experience', dict(op='read_text',path='task.md',reason='x')), ReadExperience)


if __name__ == '__main__': unittest.main()
