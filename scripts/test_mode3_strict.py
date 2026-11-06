"""CPU tests: strict contract, predicted depth, actual image payloads, fail-closed bridge."""
import copy
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_robodojo_a2v import Tests as BaseTests, Inference, Policy
from environments.robodojo.runtime.robodojo_a2v import Mode3ReviewMixin, mode3_astra_input
from environments.robodojo.runtime.robodojo_mode3 import Mode3FileReviewer, mode3_astra_messages


class StrictPolicy(Mode3ReviewMixin, Policy):
    def __init__(self, root):
        Policy.__init__(self, root)
        self.block_size = 16


class Mode3Tests(unittest.TestCase):
    def setUp(self):
        BaseTests.setUp(self)
        self.p = StrictPolicy(self.tmp.name)

    # Only new tests: generic tests are run independently, using their 8-step fixture.
    def packet(self):
        preview = self.p.preview_candidates({'count':3,'predict':True,'reveal_actions':False})
        image = preview['candidates'][0]['forecast']['frames'][0]
        return preview, mode3_astra_input(preview,images=[image],history=[],instruction='stack bowls')

    def test_strict_requests_before_inference(self):
        for payload in ({}, {'count':1,'predict':True,'reveal_actions':False},
                        {'count':3,'predict':False,'reveal_actions':False},
                        {'count':3,'predict':True},
                        {'count':3,'predict':True,'reveal_actions':True}):
            with patch.object(Inference,'generate_action') as generate:
                with self.assertRaises(ValueError):self.p.preview_candidates(payload)
                generate.assert_not_called()
        self.p.control_hz=50
        with self.assertRaises(ValueError):self.packet()

    def test_reject_wrong_timing_ids_and_history(self):
        preview,packet=self.packet()
        current=packet['current_rgb']
        for field,value in [('control_hz',50),('predict_steps',16)]:
            broken=copy.deepcopy(preview);broken[field]=value
            with self.assertRaises(ValueError):mode3_astra_input(broken,images=current,history=[],instruction='stack bowls')
        for mutate in ('offset','duplicate','bool_id'):
            broken=copy.deepcopy(preview)
            if mutate=='offset':broken['candidates'][0]['forecast']['frame_steps']=list(range(1,24,2))
            if mutate=='duplicate':broken['candidates'][1]['candidate_id']=0
            if mutate=='bool_id':broken['candidates'][0]['candidate_id']=False
            with self.assertRaises(ValueError):mode3_astra_input(broken,images=current,history=[],instruction='stack bowls')
        with self.assertRaises(ValueError):mode3_astra_input(preview,images=current,history=[{'actions':[1]}],instruction='stack bowls')
        self.assertNotIn('action_hash',json.dumps(preview))
        self.assertEqual([f['offset_s'] for f in packet['candidates'][0]['frames']], [i/25 for i in range(2,25,2)])

    def test_same_a2v_seed_final_action_and_depth(self):
        from dataclasses import replace
        original=Inference.predict_outcome
        joint_seeds=[];a2v_seeds=[]
        gen=Inference.generate_action
        def generate(obj,*args):
            joint_seeds.append(obj.settings.seed);return gen(obj,*args)
        def forecast(obj,*args):
            a2v_seeds.append(obj.settings.seed)
            f=original(obj,*args)
            return replace(f,depth=torch.full((1,3,12,16,16),0.5))
        with patch.object(Inference,'generate_action',generate),patch.object(Inference,'predict_outcome',forecast):
            preview,packet=self.packet()
        self.assertEqual(joint_seeds,[12,13,14]);self.assertEqual(a2v_seeds,[12,12,12])
        self.assertTrue(packet['depth_available'])
        messages=mode3_astra_messages(packet)
        parts=messages[1]['content']
        self.assertEqual(sum(p['type']=='image_url' for p in parts),73)
        self.assertNotIn('action_hash',json.dumps(messages))
        self.assertNotIn(self.tmp.name,json.dumps(messages))
        self.assertTrue(all(p['image_url']['url'].startswith('data:image/png;base64,') for p in parts if p['type']=='image_url'))
        from PIL import Image
        self.assertEqual(np.asarray(Image.open(packet['candidates'][0]['frames'][0]['depth_path']))[0,0],128)
        expected=self.p._review[1][2][0][:16].copy()
        actual=self.p.select_candidate({'review_id':preview['review_id'],'candidate_id':2,'execute_steps':16})
        np.testing.assert_array_equal(actual,expected)
        with self.assertRaises(RuntimeError):self.p.select_candidate({'review_id':preview['review_id'],'candidate_id':2,'execute_steps':16})

    def test_adaptive_lengths_and_invalid_selection(self):
        for count in (24,16,8,4):
            self.p = StrictPolicy(self.tmp.name)
            preview,packet=self.packet()
            for bad in (0,3,25,True):
                with self.assertRaises(ValueError):
                    self.p.select_candidate({'review_id':preview['review_id'],'candidate_id':0,'execute_steps':bad})
            expected=self.p._review[1][0][0][:count].copy()
            out=self.p.select_candidate({'review_id':preview['review_id'],'candidate_id':0,'execute_steps':count})
            np.testing.assert_array_equal(out,expected)
            self.assertEqual(self.p.next_decision,count)
        from environments.robodojo.runtime.robodojo_mode3 import validate_choice
        self.assertIsNone(validate_choice({'candidate_id':None,'execute_steps':0},[24,16,8,4]))
        with self.assertRaises(ValueError):validate_choice({'candidate_id':0,'execute_steps':24},[8,4])

    def test_framework_invalid_response_discards(self):
        from test_framework_native import Strict, Env, Client
        from environments.robodojo.framework import run_framework_episode
        policy=Strict(self.tmp.name);env=Env()
        with self.assertRaises(ValueError):
            run_framework_episode(env,root=Path(self.tmp.name)/'invalid',client=Client(policy),
                reviewer=lambda packet:True,max_steps=8)
        self.assertIsNone(policy._review);self.assertEqual(policy.next_decision,0)
        self.assertEqual(env.steps,0)

    def test_file_bridge_delivers_pixels_and_selects(self):
        _,packet=self.packet()
        source_image=packet['current_rgb'][0]
        packet['framework_context']=[dict(text=str(Path(self.tmp.name)/'private-events.jsonl'))]
        packet['review_feedback']=dict(error='shorten the prefix',robot_moved=False)
        root=Path(self.tmp.name)/'bridge'
        reviewer=Mode3FileReviewer(root,timeout_s=5)
        errors=[]
        def respond():
            try:
                deadline=time.monotonic()+4
                pending=root/'pending.json'
                while not pending.exists():
                    if time.monotonic()>deadline:raise TimeoutError()
                    time.sleep(.01)
                request=json.loads(pending.read_text());directory=Path(request['directory'])
                self.assertTrue((directory/'messages.json').exists())
                public=json.loads((directory/'request.json').read_text())
                self.assertNotIn(source_image,json.dumps(public))
                self.assertNotIn('private-events.jsonl',json.dumps(public))
                self.assertNotIn('action_hash',json.dumps(public))
                self.assertTrue(Path(public['current_rgb'][0]).is_relative_to(directory/'media'))
                self.assertTrue(Path(public['candidates'][0]['frames'][0]['path']).is_file())
                self.assertFalse(list(directory.rglob('forecast.json')))
                self.assertIn('shorten the prefix',(directory/'messages.json').read_text())
                tmp=directory/'response.tmp'
                tmp.write_text(json.dumps({'request_id':request['request_id'],'candidate_id':1,'execute_steps':24}))
                tmp.replace(directory/'response.json')
            except Exception as e:errors.append(e)
        thread=threading.Thread(target=respond);thread.start()
        try:self.assertEqual(reviewer(packet),{'candidate_id':1,'execute_steps':24})
        finally:thread.join()
        self.assertFalse(errors);self.assertFalse((root/'pending.json').exists())

    def test_episode_uses_measured_history_and_exact_prefix(self):
        from test_framework_native import Env, Strict, Client
        from environments.robodojo.runtime.robodojo_mode3 import eval_mode3_episode
        env=Env(stop=32); client=Client(Strict(Path(self.tmp.name)))
        packets=[]
        def review(packet):
            packets.append(packet)
            mode3_astra_messages(packet)
            return {'candidate_id':1,'execute_steps':packet['allowed_execute_steps'][0]}
        root=Path(self.tmp.name)/'episode'
        result=eval_mode3_episode(env,client,root=root,reviewer=review,max_steps=32)
        self.assertEqual(result['executed_steps'],32)
        summary=json.loads((root/'episode-summary.json').read_text())
        self.assertEqual(summary['framework'],'TaskSkillRunner/v1')
        self.assertEqual(summary['counters']['accepted'],2)
        self.assertEqual([len(p['history']) for p in packets],[0,1])
        self.assertEqual(packets[1]['decision_step'],24)
        self.assertEqual(packets[1]['allowed_execute_steps'],[8,4])
        self.assertEqual(client.p.measured,list(range(33)))
        self.assertEqual(sum(n=='framework_select' for n,p in client.calls),2)
        self.assertEqual(client.calls[-1][0],'finish_episode')

    def test_missing_media_and_timeout_fail_closed(self):
        _,packet=self.packet()
        root=Path(self.tmp.name)/'bridge'
        with self.assertRaises(TimeoutError):Mode3FileReviewer(root,timeout_s=.01)(packet)
        self.assertFalse((root/'pending.json').exists())
        packet['current_rgb']=['/nonexistent/mode3.png']
        with self.assertRaises(FileNotFoundError):Mode3FileReviewer(root)(packet)
        self.assertFalse((root/'pending.json').exists())


if __name__=='__main__':
    unittest.main(verbosity=2, defaultTest='Mode3Tests')
