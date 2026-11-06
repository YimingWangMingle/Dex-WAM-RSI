"""Full unified control path with native-shaped observations and real A2V logic."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS, ModuleType
import numpy as np
import torch
from test_robodojo_a2v import Policy, Inference
from environments.robodojo.runtime.robodojo_a2v import Mode3ReviewMixin
from thinker_wm.runtime.robodojo_policy import RoboDojoPolicy
from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect.types import Action,ActionChunk,ActionSpace
from environments.robodojo.framework import run_framework_episode, flat
from environments.robodojo.training import NATIVE_ACTION_CONTRACT


class Env:
    task_name='stack_bowls'
    def __init__(self,stop=32,fail_at=None):
        self.steps=0;self.stop=stop;self.fail_at=fail_at
        self.success=[False];self.end_flag=[False];self.actions=[]
        self.state=dict(ActionSpace().decode([.0]*6+[.5]+[.0]*6+[.5]).data)
    def get_obs(self):
        return dict(instruction='stack bowls',state=self.state,
                    vision={'head':{'color':np.full((8,8,3),self.steps,np.uint8)}},
                    additional_info=dict(frequency=25))
    def take_action(self,action):
        if self.fail_at==self.steps:raise RuntimeError('ambiguous native dispatch')
        self.actions.append(flat(action));self.state=action;self.steps+=1
        self.end_flag[0]=self.steps>=self.stop
    def is_episode_end(self):return self.steps>=self.stop


class Strict(Mode3ReviewMixin,Policy):
    def __init__(self,root):
        Policy.__init__(self,root)
        self.profile=NS(action_contract=NATIVE_ACTION_CONTRACT,action_dim=14)
        self.inf.checkpoint='fixture-only'
        self.block_size=16
        self.measured=[]
    def update_obs(self,raw):
        step=raw['rsi_control_step']
        expected=0 if self.observation is None else self.observation.step+1
        if step!=expected or step>self.next_decision:raise ValueError('bad measured row')
        self.observation=NS(step=step,state=torch.tensor(flat(raw['state'])))
        self.history=NS(last_step=step)
        self.measured.append(step)
    def finish_episode(self):pass


class Generic(Policy):
    def __init__(self,root):
        Policy.__init__(self,root)
        self.profile=NS(action_contract=NATIVE_ACTION_CONTRACT,action_dim=14)
        self.inf.checkpoint='fixture-only'
        self.measured=[]
    update_obs=Strict.update_obs
    def finish_episode(self):pass


class Client:
    def __init__(self,p):self.p=p;self.calls=[]
    def call(self,name,payload=None):
        self.calls.append((name,payload))
        return getattr(self.p,name)() if payload is None else getattr(self.p,name)(payload)


class Direct:
    def __init__(self):self.feedback=[];self.resets=0
    def reset(self):self.resets+=1
    def act(self,obs):
        return ActionChunk([Action(dict(ActionSpace().decode([.2]*14).data)) for _ in range(4)],reasoning='measured direct plan')
    def confirm_executed(self,n):self.feedback.append(n)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        module=ModuleType('XPolicyLab.utils.process_data')
        module.unpack_robot_state=lambda a,*args,**kwargs:a.copy()
        patcher=patch.dict(sys.modules,{'XPolicyLab.utils.process_data':module})
        patcher.start();self.addCleanup(patcher.stop)
        self.p=Strict(self.root);self.client=Client(self.p)
    def test_agent_calls_independent_a2v_then_reviews_wam(self):
        from rsi.selection_policy import SelectionPolicy
        from execution.runner import CallSkill, CallTool, Review
        from execution.review import ReviewDecision
        actor=Client(Generic(self.root/'actor'))
        predictor=Client(Generic(self.root/'predictor'))
        def selector(view):
            if not view.pending:return CallSkill('wam_control','stack bowls',24)
            p=view.pending
            if not p['tool_predictions']:
                return CallTool('a2v',dict(proposal_id=p['proposal_id'],
                    observation_id=view.latest_observation.observation_id,candidate_ids=[0]))
            return Review(ReviewDecision(p['proposal_id'],'accept','forecast inspected','stack bowls',8,0))
        env=Env(stop=16)
        result=self.run_episode(env,client=actor,mode3=False,predict=False,selector=selector,
            a2v_tool_client=predictor,selection_policy=SelectionPolicy(('wam_control',),('a2v',),decision_mode='agent'),
            required_prediction_skills=('wam_control',))
        self.assertEqual(result['executed_steps'],16)
        self.assertEqual(sum(n=='framework_predict' for n,_ in predictor.calls),2)
        self.assertFalse(any(n=='framework_select' for n,_ in predictor.calls))
        self.assertEqual(sum(n=='framework_select' for n,_ in actor.calls),2)
        self.assertEqual([e['tool'] for e in self.events() if e['event']=='tool_result'],['a2v','a2v'])

    def test_experience_measurement_guarded_direct_and_wam_share_one_episode(self):
        from experience.packages import build_package, write_catalog
        from skills.direct.guarded import DirectStage, GuardedDirectBackend
        from execution.feedback import OperationCheck, ObservationEvidence, OperationReport
        from tools.measurement import MeasurementTool, MeasurementResult
        from execution.runner import CallSkill, Measure
        src=self.root/'memory';src.mkdir()
        (src/'ICL.md').write_text('approach from above')
        (src/'HMR.md').write_text('past slip: remeasure before lifting')
        pkg=build_package(src,self.root/'packages'/'stack_bowls','stack_bowls')
        catalog=self.root/'catalog.json'
        write_catalog(catalog,'fixture','packages',[pkg],
            {'stack_bowls':['ICL.md','HMR.md']},
            {'stack_bowls':dict(bundle='stack_bowls',kind='original_preparation',
                               base_task='stack_bowls',reason='integration fixture')})
        def factory(load):
            def plan(request):
                return [flat(load(request.observation)['state'])]*4
            def verify(before,after):
                self.assertEqual(load(after)['step'],4)
                return (OperationCheck('held','passed','fixture/v1',
                    (ObservationEvidence.from_observation(after),)),)
            return GuardedDirectBackend({'approach':(DirectStage('approach',plan,verify,('held',)),)})
        sequence=iter([Measure('probe',{}),CallSkill('direct_control','approach',4),
                       CallSkill('wam_control','stack bowls',4)])
        packets=[]
        def reviewer(packet):
            packets.append(packet)
            return dict(candidate_id=0,execute_steps=4)
        probe=MeasurementTool('probe','v1','fixture',{},
            lambda ref,args:MeasurementResult(OperationReport('probe'),dict(marker='measured_clearance')))
        self.run_episode(Env(stop=8),direct_backend_factory=factory,client=self.client,
            reviewer=reviewer,selector=lambda _:next(sequence),experience_catalog=catalog,
            measurement_tools={'probe':probe},
            skill_experience={'wam_control':{'icl':['ICL.md'],'hmr':['HMR.md']}})
        self.assertEqual(packets[0]['skill_experience']['hmr'][0]['text'], 'past slip: remeasure before lifting')
        context=packets[0]['framework_context'];text=json.dumps(context)
        for evidence in ('past slip','approach from above','measured_clearance','held'):
            self.assertIn(evidence,text)
        self.assertNotIn('actions',json.dumps(packets))
        self.assertNotIn('native_success',text)
        self.assertEqual(self.p.measured,list(range(9)))
        events=self.events()
        self.assertEqual(sum(e['event']=='handoff' and e['from_skill'] is not None for e in events),1)
        reports=[e for e in events if e['event']=='operation_verification']
        self.assertEqual(reports[0]['operation']['checks'][0]['verdict'],'passed')

    def test_wam_identity_must_match_native_contract_before_proposal(self):
        for field,value in [('action_contract','different-joint-order'),('action_dim',7),('control_hz',20)]:
            with self.subTest(field=field):
                original=self.client.call
                def call(name,payload=None):
                    result=original(name,payload)
                    if name=='framework_identity':result[field]=value
                    return result
                env=Env()
                with patch.object(self.client,'call',side_effect=call), self.assertRaisesRegex(ValueError,'contract mismatch'):
                    run_framework_episode(env,root=self.root/field,client=self.client,reviewer=self.reviewer)
                self.assertEqual(env.steps,0)
        self.assertTrue(all(n=='framework_identity' for n,_ in self.client.calls))

    def test_generic_wam_requires_reviewer_except_explicit_single_no_prediction(self):
        for predict,count in [(True,3),(True,1),(False,3)]:
            env=Env();client=Client(Generic(self.root))
            with self.subTest(predict=predict,count=count), self.assertRaisesRegex(ValueError,'explicit reviewer'):
                run_framework_episode(env,root=self.root/f'bad-{predict}-{count}',client=client,
                                      mode3=False,predict=predict,candidate_count=count)
            self.assertEqual(env.steps,0);self.assertEqual(client.calls,[])
        env=Env(stop=4)
        self.run_episode(env,client=Client(Generic(self.root)),mode3=False,predict=False,candidate_count=1)
        self.assertEqual(env.steps,4)

    def test_visual_invalid_prefix_can_be_shortened_without_reproposal_or_log_leak(self):
        env=Env(stop=4);packets=[]
        def checker(proposal,observation):
            return tuple(() if i<4 else ('secret-bound-value: 0.1234567',) for i in range(len(proposal.actions)))
        def reviewer(packet):
            packets.append(packet)
            live=self.events()
            self.assertFalse(any(e['event']=='proposal' for e in live))
            self.assertNotIn('proposed_actions',json.dumps(live))
            self.assertNotIn('secret-bound-value',json.dumps(live))
            self.assertEqual(env.steps,0)
            if len(packets)==2:
                self.assertFalse(packet['review_feedback']['robot_moved'])
                self.assertIn('shorten',packet['review_feedback']['error'])
            return dict(candidate_id=0,execute_steps=16 if len(packets)==1 else 4)
        result=self.run_episode(env,client=self.client,reviewer=reviewer,row_checkers={'wam_control':checker})
        self.assertEqual(env.steps,4)
        self.assertEqual(result['framework']['counters']['invalid_reviews'],1)
        self.assertEqual(sum(n=='framework_preview' for n,_ in self.client.calls),1)
        self.assertEqual(sum(n=='framework_select' for n,_ in self.client.calls),1)
        audit=[e for e in self.events() if e['event']=='proposal']
        self.assertEqual(len(audit),1);self.assertTrue(audit[0]['deferred_until_episode_end'])
        self.assertEqual(len(audit[0]['candidates']),3)
        self.assertEqual(len(audit[0]['candidates'][0]['actions']),24)

    def test_default_visual_bridge_is_separate_and_archived_after_review(self):
        roots=[]
        class Reviewer:
            def __init__(inner,root):inner.root=Path(root);roots.append(inner.root)
            def __call__(inner,packet):
                self.assertFalse(inner.root.is_relative_to(self.root/'run'))
                self.assertFalse((self.root/'run'/'review-artifacts').exists())
                inner.root.mkdir(parents=True,exist_ok=True)
                (inner.root/'response.json').write_text('{"candidate_id":0,"execute_steps":4}')
                return dict(candidate_id=0,execute_steps=4)
        import importlib
        review_module = importlib.import_module('environments.robodojo.runtime.robodojo_mode3')
        with patch.object(review_module,'Mode3FileReviewer',Reviewer):
            self.run_episode(Env(stop=4),client=self.client)
        self.assertEqual(roots,[self.root/'run-review'])
        self.assertEqual((self.root/'run'/'review-artifacts'/'response.json').read_bytes(),
                         (roots[0]/'response.json').read_bytes())

    def test_generic_wam_can_shorten_invalid_full_prefix(self):
        env=Env(stop=4);client=Client(Generic(self.root));calls=[]
        def reviewer(packet):
            calls.append(packet)
            return 0 if len(calls)==1 else dict(candidate_id=0,execute_steps=4)
        checker=lambda proposal,obs:tuple(() if i<4 else ('blocked',) for i in range(len(proposal.actions)))
        self.run_episode(env,client=client,mode3=False,predict=True,candidate_count=3,
                         reviewer=reviewer,row_checkers={'wam_control':checker})
        self.assertEqual(env.steps,4);self.assertEqual(len(calls),2)
        self.assertIsNotNone(calls[1]['review_feedback'])
        self.assertEqual(sum(n=='framework_preview' for n,_ in client.calls),1)
        self.assertEqual(sum(n=='framework_select' for n,_ in client.calls),1)

    def test_invalid_review_attempts_are_bounded_without_motion(self):
        env=Env();calls=[]
        def reviewer(packet):calls.append(packet);return dict(candidate_id=0,execute_steps=16)
        checker=lambda proposal,observation:tuple(('blocked',) for _ in proposal.actions)
        result=self.run_episode(env,client=self.client,reviewer=reviewer,max_review_attempts=2,
                               row_checkers={'wam_control':checker})
        self.assertEqual(len(calls),2);self.assertEqual(env.steps,0)
        self.assertIn('review_attempt_budget_exhausted',result['stop_reason'])
        self.assertNotIn('framework_select',[n for n,_ in self.client.calls])
        self.assertIsNone(self.p._review)

    def test_visual_pending_view_redacts_action_sidecars_and_numeric_violations(self):
        from execution.runner import TaskSkillRunner
        original=TaskSkillRunner._pending_view
        views=[]
        def capture(runner):
            view=original(runner)
            if view:views.append(view)
            return view
        checker=lambda proposal,observation:tuple(('joint value 12345',) for _ in proposal.actions)
        with patch.object(TaskSkillRunner,'_pending_view',capture):
            self.run_episode(Env(),client=self.client,reviewer=lambda p:None,row_checkers={'wam_control':checker})
        self.assertTrue(views)
        for view in views:
            self.assertIsNone(view['prediction_ref']);self.assertIsNone(view['actions'])
            self.assertTrue(all(c['prediction_ref'] is None and c['actions'] is None for c in view['candidates']))
            self.assertNotIn('12345',json.dumps(view))

    def test_guarded_factory_conflicts_rejected_without_motion(self):
        env=Env()
        with self.assertRaisesRegex(ValueError,'not both'):
            self.run_episode(env,direct_policy=Direct(),direct_backend_factory=lambda _:None)
        self.assertEqual(env.steps,0)

    def test_new_skill_registers_and_hands_off_without_core_name_changes(self):
        from skills.contracts import ControlSkill, SkillIdentity, BackendProposal
        from environments.robodojo.training import NATIVE_ACTION_CONTRACT
        from execution.review import ReviewDecision
        feedback=[];views=[]
        class Recovery:
            def reset(self,episode):pass
            def propose(self,request):return BackendProposal(tuple((.3,)*14 for _ in range(4)))
            def observe(self,value):feedback.append(value)
        def factory(load):
            return ControlSkill(SkillIdentity('recovery_control','recovery/v1',NATIVE_ACTION_CONTRACT,14),Recovery())
        def selector(view):
            views.append(view)
            return 'direct_control' if view.latest_observation.control_step<4 else 'recovery_control'
        def review(proposal,view):
            return ReviewDecision(proposal['proposal_id'],'accept','measured recovery',proposal['subgoal'],4)
        result=self.run_episode(Env(stop=8),direct_policy=Direct(),selector=selector,
            skill_factories=[factory],skill_reviewers={'recovery_control':review})
        self.assertEqual(result['executed_steps'],8)
        self.assertIn('recovery_control',views[0].available_skills)
        self.assertEqual(len(feedback),1)
        self.assertEqual(feedback[0].final_observation.control_step,8)
        self.assertEqual(result['framework']['counters']['handoffs'],1)

    def test_extension_requires_reviewer_before_any_movement(self):
        from skills.contracts import ControlSkill, SkillIdentity
        from environments.robodojo.training import NATIVE_ACTION_CONTRACT
        env=Env()
        factory=lambda load:ControlSkill(SkillIdentity('new_skill','v1',NATIVE_ACTION_CONTRACT,14),object())
        with self.assertRaisesRegex(ValueError,'review callback'):
            self.run_episode(env,skill_factories=[factory])
        self.assertEqual(env.steps,0)

    def test_executor_does_not_import_model_stack(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ('models', 'thinker_wm.models') or name.startswith(('models.', 'thinker_wm.models.')):
                raise ModuleNotFoundError('simulator cannot resolve the model package')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            result = self.run_episode(Env(stop=4),direct_policy=Direct())
        self.assertEqual(result['executed_steps'],4)
        self.assertEqual(len([e for e in self.events() if e['event']=='native_dispatch']),4)

    def run_episode(self,env,**kw):
        return run_framework_episode(env,root=self.root/'run',max_steps=32,**kw)
    def events(self):return [json.loads(s) for s in (self.root/'run/framework-events.jsonl').read_text().splitlines()]
    def reviewer(self,packet):return dict(candidate_id=2,execute_steps=packet['allowed_execute_steps'][0])
    def test_supervisor_prefix_limits_visual_review_options(self):
        from execution.runner import CallSkill
        packets=[]
        def review(packet):
            packets.append(packet)
            return dict(candidate_id=0,execute_steps=packet['allowed_execute_steps'][0])
        self.run_episode(Env(stop=8),client=self.client,reviewer=review,
            selector=lambda _:CallSkill('wam_control','stack bowls',8))
        self.assertEqual(packets[0]['allowed_execute_steps'],[8,4])

    def test_generic_reviewed_candidate_prefix(self):
        policy=Generic(self.root);client=Client(policy);env=Env(stop=10)
        self.run_episode(env,client=client,mode3=False,predict=False,candidate_count=2,reviewer=lambda p:1)
        self.assertEqual(policy.measured,list(range(11)))
        reviews=[e for e in self.events() if e['event']=='review']
        self.assertEqual([e['candidate_id'] for e in reviews],[1,1])
        self.assertEqual([e['approved_steps'] for e in reviews],[8,8])
    def test_trajectory_enters_data_rsi_as_measured_consequence_only(self):
        from data.recovery import segments_from_trajectory
        env=Env(stop=3);self.run_episode(env,direct_policy=Direct())
        trajectory=next((self.root/'run/trajectories').iterdir())
        rows=segments_from_trajectory(trajectory,outcome='success',split='development',task='stack_bowls')
        self.assertEqual([(r.decision_step,r.end_step) for r in rows],[(0,2)])
        self.assertTrue(rows[0].consequence_supervision)
        self.assertFalse(rows[0].action_supervision)
        from data.training_manifest import build_manifest
        manifest=build_manifest([],rows,strategy='balanced',rollout_budget=1.,parent_checkpoint='fixture',data_version='fixture')
        self.assertEqual(len(manifest['rows']),1)
        self.assertEqual(manifest['rows'][0]['mode'],'action_conditioned_world')
        import execution.trajectory as tl
        measured=[r for r in tl.read_trajectory(trajectory) if r['role']=='measured']
        self.assertEqual(len(measured),4)
        self.assertTrue(all((trajectory/r['attachment']).is_file() for r in measured))
        with self.assertRaises(ValueError):
            segments_from_trajectory(trajectory,outcome='success',split='test',task='stack_bowls')
    def test_supervisor_measures_without_motion_then_calls_direct(self):
        from tools.measurement import MeasurementTool, MeasurementResult
        from execution.feedback import OperationReport
        commands=iter([('measure',dict(tool='probe',arguments={},reason='inspect first')),
            ('call_backend',dict(backend='direct_control',subgoal='stack bowls',max_actions=4,reason='measured')),
            ('stop',dict(reason='done'))])
        class Supervisor:
            def complete(_,messages,tools):
                name,args=next(commands)
                return {'choices':[{'message':{'tool_calls':[{'function':dict(name=name,arguments=json.dumps(args))}]}}]}
        tool=MeasurementTool('probe','v1','test',{},lambda obs,args:MeasurementResult(OperationReport('measurement'),dict(step=obs.control_step)))
        env=Env();self.run_episode(env,direct_policy=Direct(),supervisor_client=Supervisor(),measurement_tools={'probe':tool})
        self.assertEqual(env.steps,4)
        self.assertEqual(sum(e['event']=='execution' for e in self.events()),1)
    def test_event_interrupt_reports_one_exact_measured_row(self):
        from environments.robodojo.runtime.robodojo_mode3_events import FrameworkEventGate
        self.p.candidate_auto_check=lambda req:dict(plan=dict(clear=True),measured=dict(clear=True))
        self.p.event_status=lambda:dict(clear=False,tracking_error_rad=.3)
        env=Env(stop=3)
        self.run_episode(env,client=self.client,reviewer=self.reviewer,event_gate=FrameworkEventGate())
        feedback=[p for n,p in self.client.calls if n=='framework_feedback']
        self.assertEqual([len(p['executed_actions']) for p in feedback],[1,1,1])
        self.assertEqual(self.p.measured,list(range(4)))
        self.assertEqual(sum(e['event']=='simulation_event_interrupt' for e in self.events()),2)
    def test_native_geometry_snapshot_is_used_by_framework_measurement(self):
        from tools.perception.geometry import CameraFrame
        from execution.runner import Measure, Stop
        frame=CameraFrame(np.zeros((8,8,3),np.uint8),np.array([[10,0,4],[0,10,4],[0,0,1]]),
            np.eye(4),0.,'fixture-calibration',True,(8,8))
        sequence=iter([Measure('rgb_ray',dict(camera='head',uv=[4,4])),Measure('perceive_bowls',{}),Stop('measured')])
        perception=NS(observe=lambda frames,step:dict(status='unknown',control_step=step,views=list(frames)))
        with patch('environments.robodojo.runtime.robodojo_local_geometry.LocalGeometry.camera',return_value=frame):
            env=Env();self.run_episode(env,direct_policy=Direct(),enable_geometry=True,bowl_perception=perception,selector=lambda _:next(sequence))
        self.assertEqual(env.steps,0)
        self.assertTrue(any(e['event']=='measurement' for e in self.events()))

    def test_mode3_uses_runner_review_and_exact_selected_prefix(self):
        env=Env();packets=[]
        def review(p):packets.append(p);return self.reviewer(p)
        result=self.run_episode(env,client=self.client,reviewer=review)
        self.assertEqual(result['executed_steps'],32)
        events=self.events()
        self.assertEqual([x['candidate_id'] for x in events if x['event']=='review' and x['decision']=='accept'],[2,2])
        self.assertEqual([x['approved_steps'] for x in events if x['event']=='review' and x['decision']=='accept'],[24,8])
        self.assertTrue(all(abs(row[0]-.114)<1e-6 for row in env.actions))
        self.assertEqual(self.p.measured,list(range(33)))
        self.assertEqual([len(p['history']) for p in packets],[0,1])
        self.assertNotIn('actions',json.dumps(packets))
    def test_shared_budget_stops_before_second_review(self):
        env=Env(); packets=[]
        def review(packet):
            packets.append(packet)
            return self.reviewer(packet)
        result=self.run_episode(env,client=self.client,reviewer=review,max_model_calls=1)
        self.assertEqual(result['stop_reason'],'model_budget_exhausted')
        self.assertEqual(len(packets),1)
        self.assertEqual(env.steps,24)
        self.assertEqual(result['model_budget']['used'],1)

    def test_supervisor_and_direct_share_budget_and_restore_client(self):
        calls=[]
        class Supervisor:
            def complete(_,messages,tools):
                calls.append('supervisor')
                args=dict(backend='direct_control',subgoal='stack bowls',max_actions=4,reason='test')
                return {'choices':[{'message':{'tool_calls':[{'function':dict(name='call_backend',arguments=json.dumps(args))}]}}]}
        class Controller:
            def complete(_): calls.append('direct')
        class MeteredDirect(Direct):
            def __init__(self):
                super().__init__()
                self._client=Controller()
            def act(self,obs):
                self._client.complete()
                return super().act(obs)
        policy=MeteredDirect(); original=policy._client; env=Env()
        result=self.run_episode(env,direct_policy=policy,supervisor_client=Supervisor(),max_model_calls=2)
        self.assertEqual(calls,['supervisor','direct'])
        self.assertEqual(env.steps,4)
        self.assertEqual(result['stop_reason'],'model_budget_exhausted')
        self.assertIs(policy._client,original)
        summary=json.loads((self.root/'run/episode-summary.json').read_text())
        self.assertEqual(summary['model_budget']['by_role'],dict(supervisor=1,direct=1))

    def test_rejection_has_no_selection_and_no_movement(self):
        env=Env();self.run_episode(env,client=self.client,reviewer=lambda _:None)
        self.assertEqual(env.steps,0)
        self.assertNotIn('framework_select',[x[0] for x in self.client.calls])
        self.assertIsNone(self.p._review)
    def test_partial_native_termination_sends_exact_feedback(self):
        env=Env(stop=3);self.run_episode(env,client=self.client,reviewer=self.reviewer)
        feedback=[p for n,p in self.client.calls if n=='framework_feedback']
        self.assertEqual(len(feedback[0]['executed_actions']),3)
        self.assertEqual(self.p.next_decision,3)
    def test_direct_uses_same_runner_executor_and_events(self):
        env=Env(stop=6);policy=Direct()
        result=self.run_episode(env,direct_policy=policy)
        self.assertEqual(result['executed_steps'],6)
        self.assertEqual(policy.feedback,[4,2])
        self.assertEqual(sum(x['event']=='execution' for x in self.events()),2)
        self.assertEqual(policy.resets,1)
    def test_direct_wam_direct_handoff_preserves_measured_history(self):
        env=Env(stop=32);policy=Direct()
        def select(view):
            step=view.latest_observation.control_step
            return 'direct_control' if step<4 or step>=28 else 'wam_control'
        result=self.run_episode(env,direct_policy=policy,client=self.client,
                                reviewer=self.reviewer,selector=select)
        self.assertEqual(result['framework']['counters']['handoffs'],2)
        self.assertEqual(policy.feedback,[4,4])
        self.assertEqual(self.p.measured,list(range(29)))
        self.assertEqual(result['framework']['activated_skills'],['direct_control','wam_control'])
    def test_bad_selected_row_is_rejected_before_movement(self):
        original=self.client.call
        def call(n,p=None):
            result=original(n,p)
            if n=='framework_select':result[0][0]+=1
            return result
        self.client.call=call
        env=Env()
        with self.assertRaisesRegex(ValueError,'differs'):self.run_episode(env,client=self.client,reviewer=self.reviewer)
        self.assertEqual(env.steps,0)
    def test_invalid_review_does_not_dispatch(self):
        env=Env()
        with self.assertRaises(ValueError):self.run_episode(env,client=self.client,reviewer=lambda _:dict(candidate_id=True,execute_steps=24))
        self.assertEqual(env.steps,0)
    def test_single_executor_fault_is_not_replayed(self):
        env=Env(fail_at=2)
        with self.assertRaises(RuntimeError):self.run_episode(env,client=self.client,reviewer=self.reviewer)
        self.assertEqual(env.steps,2)
        self.assertEqual(sum(n=='framework_select' for n,_ in self.client.calls),1)
    def test_direct_stop_does_not_send_hold_action(self):
        class Stopped(Direct):
            def act(self,obs):return ActionChunk([Action(dict(ActionSpace().decode([.2]*14).data),dict(request_stop=True))])
        env=Env();self.run_episode(env,direct_policy=Stopped());self.assertEqual(env.steps,0)
    def test_selected_candidate_bounds_are_checked(self):
        env=Env()
        checker=lambda proposal,obs:tuple(('blocked',) for row in proposal.actions)
        self.run_episode(env,client=self.client,reviewer=self.reviewer,row_checkers={'wam_control':checker})
        self.assertEqual(env.steps,0)
        self.assertNotIn('framework_select',[n for n,_ in self.client.calls])
    def test_missing_measured_state_is_not_zero_filled(self):
        class Missing(Env):
            def get_obs(self):
                raw=super().get_obs();raw['state']={};return raw
        with self.assertRaises(KeyError):self.run_episode(Missing(),direct_policy=Direct())
    def test_old_mode3_entry_delegates_to_framework(self):
        from environments.robodojo.runtime.robodojo_mode3 import eval_mode3_episode
        with patch('environments.robodojo.framework.run_framework_episode',return_value='called') as fn:
            self.assertEqual(eval_mode3_episode(object(),object(),root='x'), 'called')
            fn.assert_called_once()

if __name__=='__main__':unittest.main(verbosity=2)
