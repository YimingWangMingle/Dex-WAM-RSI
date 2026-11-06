"""Audited PI0.5 wire contract and shared execution path; no trained model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import io,json,tempfile,threading,unittest
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from skills.pi05.backend import PI05Client,PI05Error,CONFIG,CONTRACT
from environments.robodojo.framework import run_framework_episode
from test_framework_native import Env,Direct,Client,Generic

IDENTITY=dict(config=CONFIG,checkpoint='/fixture/pi05',action_horizon=50,output_dim=14,internal_dim=32,
    normalization='quantile_q01_q99',internal_arm_actions='relative_to_current_state',
    output_actions='absolute_joint14_continuous_grippers')

class PiEnv(Env):
    def get_obs(self):
        value=super().get_obs();value['vision']={k:{'color':np.full((8,9,3),self.steps,np.uint8)}
            for k in ('cam_head','cam_left_wrist','cam_right_wrist')};return value

class PolicyClient:
    def __init__(self):self.requests=[]
    def describe(self):return dict(IDENTITY)
    def infer(self,**kw):
        self.requests.append(kw)
        a=np.full((50,14),.1+kw['seed']*.001,np.float32)
        a[:,6]=1.1;a[:,13]=-.1
        return a

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.pi=PolicyClient()
        module=ModuleType('XPolicyLab.utils.process_data');module.unpack_robot_state=lambda a,*args,**kw:a.copy()
        patcher=patch.dict(sys.modules,{'XPolicyLab.utils.process_data':module});patcher.start();self.addCleanup(patcher.stop)
    def run_episode(self,env,**kw):
        return run_framework_episode(env,root=self.root/'run',pi05_client=self.pi,max_steps=32,**kw)
    def events(self):return [json.loads(line) for line in (self.root/'run/framework-events.jsonl').read_text().splitlines()]
    def review(self,p):return dict(candidate_id=2,execute_steps=p['allowed_execute_steps'][0])
    def world(self):
        policy=Generic(self.root);policy.profile.action_contract=CONTRACT
        policy.inf.generate_action=lambda *a:(_ for _ in ()).throw(AssertionError('must not generate WAM actions'))
        return policy,Client(policy)
    def test_audited_wam_contract_matches_pi05(self):
        # Identity emitted by the audited native Joint24 service; not a fixture alias.
        self.assertEqual(CONTRACT, 'robodojo-joint14/v1:7322c3cfe3c806534bba76cf78e7e9661209805dffbe5cc7df2a5592a8549a34')

    def test_pi05_baseline_shared_executor_exact_prefix(self):
        env=PiEnv(stop=27);r=self.run_episode(env)
        self.assertEqual(r['executed_steps'],27)
        self.assertEqual([x['seed'] for x in self.pi.requests],[0,24])
        self.assertTrue(all(a[6]==1 and a[13]==0 for a in env.actions))
        self.assertEqual([e['executed_steps'] for e in self.events() if e['event']=='pi05_feedback'],[24,3])
        self.assertEqual([e['skill'] for e in self.events() if e['event']=='execution'],['vla_control']*2)
    def test_three_candidates_review_is_exact(self):
        env=PiEnv(stop=8)
        self.run_episode(env,pi05_candidates=3,pi05_seed=10,vla_reviewer=lambda p:dict(candidate_id=2,execute_steps=8))
        self.assertEqual([r['seed'] for r in self.pi.requests],[10,11,12])
        self.assertAlmostEqual(env.actions[0][0],.112,places=6)
    def test_reject_proposal_has_zero_motion(self):
        env=PiEnv();self.run_episode(env,pi05_candidates=3,vla_reviewer=lambda p:None)
        self.assertEqual(env.steps,0)
        self.assertEqual([e['executed_steps'] for e in self.events() if e['event']=='pi05_feedback'],[0])
    def test_explicit_reviewer_required(self):
        env=PiEnv()
        with self.assertRaises(ValueError):self.run_episode(env,pi05_candidates=3)
        self.assertEqual(self.pi.requests,[])
    def test_a2v_uses_pi05_actions_and_partial_feedback(self):
        policy,client=self.world();env=PiEnv(stop=3);packets=[]
        def review(p):packets.append(p);return self.review(p)
        self.run_episode(env,pi05_candidates=3,pi05_a2v_client=client,vla_reviewer=review)
        self.assertEqual(policy.measured,[0,1,2,3])
        self.assertTrue(packets[0]['a2v_enabled'])
        self.assertEqual(len(packets[0]['candidates'][2]['forecast']['frames']),12)
        self.assertEqual(policy.next_decision,3);self.assertIsNone(policy.last_command)
        self.assertEqual([len(p['executed_actions']) for n,p in client.calls if n=='framework_feedback'],[3])
    def test_a2v_rejection_discards_preview(self):
        policy,client=self.world();env=PiEnv()
        self.run_episode(env,pi05_candidates=3,pi05_a2v_client=client,vla_reviewer=lambda p:None)
        self.assertEqual(env.steps,0);self.assertIsNone(policy._review)
        self.assertNotIn('framework_select',[n for n,p in client.calls])
    def test_direct_vla_direct_share_measured_state(self):
        env=PiEnv(stop=32);direct=Direct()
        self.run_episode(env,direct_policy=direct,selector=lambda v:'vla_control' if 4<=v.latest_observation.control_step<28 else 'direct_control')
        self.assertEqual([e['skill'] for e in self.events() if e['event']=='execution'],['direct_control','vla_control','direct_control'])
        self.assertEqual(self.pi.requests[0]['seed'],4)
        self.assertTrue(np.allclose(self.pi.requests[0]['state'],.2))
    def test_selected_candidate_bounds_prevent_dispatch(self):
        from execution.review import BoundsChecker
        env=PiEnv();self.run_episode(env,row_checkers={'vla_control':BoundsChecker(upper=[0.05]*14)})
        self.assertEqual(env.steps,0)
    def test_invalid_vla_prefix_is_reviewed_again_without_new_inference(self):
        env=PiEnv(stop=4);packets=[]
        def review(packet):
            packets.append(packet)
            return dict(candidate_id=0,execute_steps=16 if len(packets)==1 else 4)
        checker=lambda proposal,obs:tuple(() if i<4 else ('blocked',) for i in range(len(proposal.actions)))
        self.run_episode(env,vla_review_mode='action_only',vla_reviewer=review,
                         row_checkers={'vla_control':checker})
        self.assertEqual(env.steps,4);self.assertEqual(len(self.pi.requests),1)
        self.assertEqual(len(packets),2);self.assertFalse(packets[1]['review_feedback']['robot_moved'])

    def test_failure_never_falls_back(self):
        self.pi.infer=lambda **kw:(_ for _ in ()).throw(PI05Error('unavailable'))
        env=PiEnv()
        with self.assertRaises(PI05Error):self.run_episode(env)
        self.assertEqual(env.steps,0)
    def test_foreign_a2v_contract_rejected(self):
        policy,client=self.world();policy.profile.action_contract='wrong'
        with self.assertRaises(ValueError):self.run_episode(PiEnv(),pi05_a2v_client=client,vla_reviewer=self.review)
        self.assertEqual(self.pi.requests,[])

    def test_wam_action_only_no_prediction_exact_prefix(self):
        policy=Generic(self.root);client=Client(policy);packets=[]
        policy.inf.predict_outcome=lambda *a,**k:(_ for _ in ()).throw(AssertionError('prediction forbidden'))
        def review(packet):
            packets.append(packet)
            return dict(candidate_id=2,execute_steps=8)
        env=PiEnv(stop=8)
        run_framework_episode(env,root=self.root/'run',client=client,reviewer=review,
            wam_review_mode='action_only',candidate_count=3,max_steps=32)
        self.assertEqual(env.steps,8)
        self.assertEqual(packets[0]['review_mode'],'action_only')
        self.assertEqual(packets[0]['allowed_execute_steps'],[24,16,8,4])
        self.assertTrue(all('forecast' not in c for c in packets[0]['candidates']))
        np.testing.assert_allclose(env.actions,packets[0]['candidates'][2]['actions'][:8])
        request=next(p for n,p in client.calls if n=='framework_preview')
        self.assertFalse(request['predict']);self.assertFalse(request['mode3'])
    def test_wam_action_only_reject_no_motion(self):
        policy=Generic(self.root);client=Client(policy);env=PiEnv()
        run_framework_episode(env,root=self.root/'run',client=client,reviewer=lambda p:None,
            wam_review_mode='action_only',max_steps=32)
        self.assertEqual(env.steps,0);self.assertIsNone(policy._review)
    def test_vla_action_only_packet_and_prefix(self):
        packets=[]
        def review(p):
            packets.append(p);return dict(candidate_id=1,execute_steps=4)
        env=PiEnv(stop=4)
        self.run_episode(env,vla_review_mode='action_only',pi05_candidates=3,vla_reviewer=review)
        self.assertEqual(env.steps,4);self.assertAlmostEqual(env.actions[0][0],.101,places=6)
        self.assertEqual(packets[0]['review_mode'],'action_only')
        self.assertTrue(all('forecast' not in c for c in packets[0]['candidates']))
    def test_vla_memory_reaches_review_without_changing_model_instruction(self):
        from experience.packages import build_package, write_catalog
        src=self.root/'memory';src.mkdir();(src/'HMR.md').write_text('past slip: check grip')
        pkg=build_package(src,self.root/'packages'/'bowls','bowls')
        catalog=self.root/'catalog.json'
        write_catalog(catalog,'fixture','packages',[pkg],{'bowls':['HMR.md']},
            {'stack_bowls':dict(bundle='bowls',kind='original_preparation',
                               base_task='stack_bowls',reason='fixture')})
        packets=[]
        def review(packet):
            packets.append(packet);return dict(candidate_id=0,execute_steps=4)
        env=PiEnv(stop=4)
        self.run_episode(env,vla_review_mode='action_only',vla_reviewer=review,
            experience_catalog=catalog,skill_experience={'vla_control':{'hmr':['HMR.md']}})
        self.assertEqual(env.steps,4)
        self.assertEqual(packets[0]['skill_experience']['hmr'][0]['text'],'past slip: check grip')
        self.assertEqual(packets[0]['skill_experience']['icl'],[])
        self.assertEqual(self.pi.requests[0]['instruction'],'stack bowls')
        self.assertNotIn('experience',self.pi.requests[0])

    def test_action_only_requires_reviewer_and_forbids_a2v(self):
        with self.assertRaises(ValueError):self.run_episode(PiEnv(),vla_review_mode='action_only')
        policy,client=self.world()
        with self.assertRaises(ValueError):self.run_episode(PiEnv(),vla_review_mode='action_only',pi05_a2v_client=client,vla_reviewer=self.review)
        with self.assertRaises(ValueError):run_framework_episode(PiEnv(),root=self.root/'run',client=client,wam_review_mode='action_only')
        self.assertEqual(self.pi.requests,[])
    def test_action_only_respects_supervisor_prefix(self):
        from execution.runner import CallSkill
        policy=Generic(self.root);packets=[]
        def review(p):
            packets.append(p);return dict(candidate_id=0,execute_steps=4)
        run_framework_episode(PiEnv(stop=4),root=self.root/'run',client=Client(policy),
            reviewer=review,wam_review_mode='action_only',
            selector=lambda v:CallSkill('wam_control','stack bowls',4),max_steps=32)
        self.assertEqual(packets[0]['allowed_execute_steps'],[4])

    def test_wam_vla_action_review_handoff(self):
        from execution.runner import CallSkill
        policy=Generic(self.root);policy.profile.action_contract=CONTRACT
        seen=[]
        def review(packet):
            seen.append(packet['actor'])
            return dict(candidate_id=0,execute_steps=4)
        env=PiEnv(stop=12)
        self.run_episode(env,client=Client(policy),wam_review_mode='action_only',
            reviewer=review,vla_review_mode='action_only',vla_reviewer=review,
            selector=lambda v:CallSkill('vla_control' if v.latest_observation.control_step==4 else 'wam_control','stack bowls',4))
        self.assertEqual(seen,['wam','pi05','wam'])
        self.assertEqual([e['skill'] for e in self.events() if e['event']=='execution'],
            ['wam_control','vla_control','wam_control'])
        self.assertEqual(env.steps,12)
    def test_unknown_skill_is_reported_without_attribute_error(self):
        from execution.runner import CallSkill,Stop
        calls=iter([CallSkill('wam_control','stack bowls',4),Stop('done')])
        result=self.run_episode(PiEnv(),selector=lambda v:next(calls))
        self.assertEqual(result['executed_steps'],0)

class WireTests(unittest.TestCase):
    def setUp(self):
        self.received=[];self.identity=dict(IDENTITY);self.bad_seed=False;self.bad_shape=False;self.status=200
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(owner.identity).encode())
            def do_POST(self):
                owner.received.append(self.path)
                with np.load(io.BytesIO(self.rfile.read(int(self.headers['Content-Length']))),allow_pickle=False) as z:
                    owner.request={k:z[k].copy() for k in z.files};seed=int(z['seed'])
                b=io.BytesIO();np.savez(b,actions=np.ones((49 if owner.bad_shape else 50,14),np.float32),seed=seed+int(owner.bad_seed))
                self.send_response(owner.status);self.end_headers();self.wfile.write(b.getvalue())
            def log_message(self,*a):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.client=PI05Client(f'http://127.0.0.1:{self.server.server_port}',checkpoint='/fixture/pi05')
        self.args=dict(state=np.arange(14,dtype=np.float32),images={k:np.zeros((4,5,3),np.uint8) for k in ('cam_head','cam_left_wrist','cam_right_wrist')},instruction='stack bowls',seed=123)
    def test_existing_npz_protocol_camera_mapping_and_no_extra_conversion(self):
        out=self.client.infer(**self.args)
        self.assertEqual(out.shape,(50,14));self.assertTrue(np.all(out==1))
        self.assertEqual(self.request['cam_high'].shape,(3,4,5))
        np.testing.assert_array_equal(self.request['state'],self.args['state'])
        self.assertEqual(self.received,['/infer'])
    def test_wrong_identity_fails_before_inference(self):
        self.identity['output_actions']='delta_joint14'
        with self.assertRaises(PI05Error):self.client.infer(**self.args)
        self.assertEqual(self.received,[])
    def test_wrong_seed_and_shape_rejected(self):
        self.bad_seed=True
        with self.assertRaises(PI05Error):self.client.infer(**self.args)
        self.bad_seed=False;self.bad_shape=True
        with self.assertRaises(PI05Error):self.client.infer(**self.args)
    def test_http_error_not_retried(self):
        self.status=500
        with self.assertRaises(PI05Error):self.client.infer(**self.args)
        self.assertEqual(self.received,['/infer'])
    def test_missing_camera_no_request(self):
        del self.args['images']['cam_right_wrist']
        with self.assertRaises(KeyError):self.client.infer(**self.args)
        self.assertEqual(self.received,[])

if __name__=='__main__':unittest.main()
