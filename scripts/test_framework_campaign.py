"""Actual PhysicalRSI core + unified framework, with a fixture simulator/policy.

Missing upstream is an error, never a substitute core. No trained-model claim.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from PhysicalRSI_core.contracts import Context, Cancelled, ReconciliationRequired
from rsi.framework_campaign import FrameworkSuite, FailureEdits, build_campaign, ENTRY
from rsi.core_adapter import TaskSpec
from rsi.version_manifest import CLOSURE, build_harness, write_wam_version, file_digest
from environments.robodojo.framework import flat
from test_framework_native import Env, Direct
from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect.types import Action, ActionChunk, ActionSpace


class FixedPolicy(Direct):
    def __init__(self,value):
        super().__init__();self.value=value
    def act(self,obs):
        return ActionChunk([Action(dict(ActionSpace().decode([self.value]*6+[.5]+[0]*6+[.5]).data))
                            for _ in range(2)],reasoning='fixture motion')


def code(value):
    return ('from test_framework_campaign import FixedPolicy\n'
            'def configure(root, case):\n'
            f'    return dict(direct_policy=FixedPolicy({value!r}))\n')


class NativeEnv(Env):
    def __init__(self):super().__init__(stop=2)
    def take_action(self,action):
        super().take_action(action)
        self.success[0] = bool(self.end_flag[0] and flat(action)[0]>=.15)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.resets=[];self.closed=[]
        root=self.root/'initial';files={k:[] for k in CLOSURE}
        for k in CLOSURE:
            p=root/k/'identity.txt';p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(k);files[k].append(str(p.relative_to(root)))
        write_wam_version(root,checkpoint='fixture-fixed',checkpoint_sha256='0'*64,
            action_stats_version='native',history={},camera_transform='fixture',
            data_version='fixture',model_config_sha256='1'*64)
        files['configuration'].append('configuration/wam_version.json')
        (root/ENTRY).write_text(code(0));files['control'].append(ENTRY)
        self.initial=build_harness(root,'parent',files)
        self.suite=FrameworkSuite(name='test',tasks={'stack_bowls':TaskSpec('stack bowls',[1],[2,3],2,[4])},
            initial=self.initial,open_env=self.open,close_env=lambda e:self.closed.append(e.steps),
            resources=lambda env,case:dict(action_spec=NS(control_hz=25,low=[-3]*14,high=[3]*14,max_step=[.5]*14)),
            driver_revision='2'*64,work_root=self.root/'native',max_steps=2)

    def open(self,task,case):
        self.resets.append(case['seed']);return NativeEnv()

    def campaign(self,value=.2):
        def repair(parent,evaluations):
            self.assertTrue(all(e['trial']['case']['split']=='development' for e in evaluations))
            self.assertTrue(any(e['outcome']=='failure' for e in evaluations))
            return [dict(id='child',reason='fixture correction from failure',edits={ENTRY:code(value)})]
        return build_campaign(root=self.root/'outer',suite=self.suite,
            repair=FailureEdits(repair,revision='fixture-repair-v1'),max_rounds=2,max_candidates=1,
            max_trials=6,seconds_per_episode=60,
            profile=dict(tasks={'stack_bowls':dict(weight=1,episodes=2,score_range=[0,1],maximum_regression=0)},
                         minimum_gain=0,tie_tolerance=1e-10))

    def test_actual_core_inherits_then_retains_and_resume_does_not_move(self):
        campaign=self.campaign();result=campaign.run(self.initial)
        self.assertEqual([r['outcome'] for r in result['rounds']],['inherited','retained'])
        self.assertEqual(result['current']['harness']['id'],'child')
        self.assertEqual(len(self.resets),6);self.assertEqual(self.closed,[2]*6)
        self.assertEqual(result['current']['harness']['components']['weights'],self.initial['components']['weights'])
        summaries=list((self.root/'outer').rglob('artifacts/episode-summary.json'))
        self.assertEqual(len(summaries),6)
        self.assertTrue(all(json.loads(p.read_text())['executed_steps']==2 for p in summaries))
        self.assertEqual(campaign.run(self.initial),result)
        self.assertEqual(len(self.resets),6)

    def test_no_gain_keeps_parent(self):
        campaign=self.campaign(.01)
        campaign.max_rounds=1
        result=campaign.run(self.initial)
        self.assertEqual(result['rounds'][0]['outcome'],'retained')
        self.assertEqual(result['current']['harness']['id'],'parent')

    def test_heldout_case_rejected_before_reset(self):
        trial=self.suite.trial(self.initial,'stack_bowls')
        with self.assertRaises(ValueError):
            trial.environment.reset(dict(task='stack_bowls',seed=4,split='test',instruction='stack bowls'),Context('test'))
        self.assertEqual(self.resets,[])

    def test_deadline_cancels_native_rows_without_replay(self):
        from environments.robodojo.framework import run_framework_episode
        env=NativeEnv()
        def check():
            if env.steps>=1:raise Cancelled('deadline')
        with self.assertRaises(Cancelled):
            run_framework_episode(env,root=self.root/'cancel',direct_policy=FixedPolicy(.2),
                max_steps=2,execution_check=check)
        self.assertEqual(env.steps,1)
        self.assertEqual(json.loads((self.root/'cancel/episode-summary.json').read_text())['executed_steps'],1)

    def test_unknown_native_result_never_promotes(self):
        from rsi.framework_campaign import _NativeVerdict
        verdict=_NativeVerdict().verify({},[dict(observation=dict(native_success=None))])
        self.assertEqual(verdict['outcome'],'uncertain')

    def test_candidate_cannot_override_constraints(self):
        (Path(self.initial['root'])/ENTRY).write_text(
            "def configure(root, case): return dict(action_spec=None)\n")
        self.initial['components']['control'][ENTRY]=file_digest(Path(self.initial['root'])/ENTRY)
        trial=self.suite.trial(self.initial,'stack_bowls');context=Context('override')
        trial.environment.reset(dict(task='stack_bowls',seed=1,split='development',instruction='stack bowls'),context)
        try:
            with self.assertRaisesRegex(ValueError,'override'):
                trial.environment.step(trial.policy.act({},context),context)
            self.assertEqual(trial.environment.env.steps,0)
        finally:trial.policy.end_episode(context)

    def test_candidate_paths_rejected_before_writing(self):
        from rsi.proposal_adapter import materialize_candidate
        for cid,edits in [('../escape',{ENTRY:code(.2)}),('child',{'../escape':'bad'})]:
            with self.assertRaises(ValueError):
                materialize_candidate(self.initial,edits,new_id=cid,out_root=self.root/'children')
        self.assertFalse((self.root/'escape').exists())

    def test_model_repair_uses_shared_client_and_limits_edit_scope(self):
        from rsi.framework_campaign import ModelFailureRepair
        row=dict(id='model-child',reason='fixture repair',edits={ENTRY:code(.2)})
        messages=[]
        class Client:
            def complete(self,history,tools):
                messages.extend(history)
                return {'choices':[{'message':{'tool_calls':[{'function':dict(
                    name='propose_repairs',arguments=json.dumps(dict(candidates=[row])))}]}}]}
        repair=ModelFailureRepair(Client(),files=[ENTRY])
        evaluation=dict(trial=dict(case=dict(split='development')),outcome='failure',
                        verdict=dict(reason='missed'),trajectory=[])
        self.assertEqual(repair(self.initial,[evaluation]),[row])
        self.assertIn('missed',messages[-1]['content'])
        row['edits']={'weights/identity.txt':'not allowed'}
        with self.assertRaises(ValueError):repair(self.initial,[evaluation])

    def test_fixed_component_change_rejected(self):
        from rsi.proposal_adapter import materialize_candidate
        with self.assertRaises(ValueError):
            materialize_candidate(self.initial,{'weights/identity.txt':'changed'},new_id='bad',out_root=self.root/'children')

    def test_interrupted_motion_cannot_resume_as_fresh_trial(self):
        campaign=self.campaign()
        with patch.object(NativeEnv,'take_action',side_effect=RuntimeError('uncertain dispatch')):
            with self.assertRaisesRegex(RuntimeError,'uncertain dispatch'):campaign.run(self.initial)
        resets=len(self.resets)
        with self.assertRaises((ReconciliationRequired,ValueError)):
            campaign.run(self.initial)
        self.assertEqual(len(self.resets),resets)


if __name__=='__main__':unittest.main(verbosity=2)
