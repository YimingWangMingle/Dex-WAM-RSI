import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from environments.robodojo.experiments.layout3 import group_environment,experiment_environment,run_experiment
from environments.robodojo.experiments.review_protocol import reviewer_payload

class Tests(unittest.TestCase):
    def test_groups_and_environment_restored_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(group_environment('S83',tmp)['RSI_PI05_POOL_SIZE'],'8')
            self.assertEqual(group_environment('D6M',tmp)['RSI_A2V_METRIC_DEPTH'],'1')
            self.assertEqual(group_environment('V1',tmp)['RSI_A2V_DEPTH'],'1')
            self.assertEqual(group_environment('R6',tmp)['RSI_A2V_DEPTH'],'0')
            with patch.dict(os.environ,{'RSI_A2V_DEFER':'external'}):
                with self.assertRaises(RuntimeError):
                    with experiment_environment('ADAPT',tmp):
                        self.assertEqual(os.environ['RSI_A2V_DEFER'],'1')
                        raise RuntimeError('fixture')
                self.assertEqual(os.environ['RSI_A2V_DEFER'],'external')
    def test_existing_output_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):run_experiment(object(),group='R6',root=tmp)
    def test_public_entry_uses_frozen_profile_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen=[]
            def fake(env,client,**kwargs):
                seen.append(dict(os.environ));return {'executed_steps':0}
            with patch('environments.robodojo.experiments.layout3.bridge_episode',fake):
                result=run_experiment(object(),group='ADAPT',root=Path(tmp)/'new',exchange=lambda p:None)
            self.assertEqual(result['executed_steps'],0)
            self.assertEqual(seen[0]['RSI_A2V_DEFER'],'1')
            self.assertEqual(seen[0]['RSI_MATRIX_MODE'],'pi05_a2v')

    def test_review_export_verifies_content(self):
        import hashlib,json
        from scripts.prepare_layout3_review import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'frame.png';image.write_bytes(b'fixture image')
            packet=dict(images=[dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest())],
                        candidates=[dict(candidate_id=0)],allowed_execute_steps=[7])
            request=root/'request.json';request.write_text(json.dumps(packet))
            prepare(request,root/'review','R6')
            self.assertEqual(json.loads((root/'review/review-settings.json').read_text())['max_calls'],60)
            image.write_bytes(b'changed')
            with self.assertRaises(ValueError):prepare(request,root/'bad-review','R6')
            self.assertFalse((root/'bad-review').exists())

    def test_optional_schema_and_packet_not_mutated(self):
        packet=dict(images=[dict(path='frame',sha256='hash',kind='measured_current')],
                    candidates=[dict(candidate_id=0)],allowed_execute_steps=[1,7,24],prediction_request_allowed=True)
        schema,prompt,attachments=reviewer_payload(packet,'ADAPT')
        self.assertIn('request_prediction',schema['properties']['operation']['enum'])
        self.assertNotIn('attachment_order',packet)
        self.assertIn('initial_measured_state',prompt)
        self.assertEqual(attachments[0]['path'],'frame')

if __name__=='__main__':unittest.main()
