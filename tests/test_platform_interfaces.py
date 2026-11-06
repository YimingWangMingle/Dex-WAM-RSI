import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from execution.episode import run_episode, MeasuredSession, EnvironmentExecutor
from environments import ActionContract
from examples.modular.fixture_driver import Environment, options
from environments.robodojo.environment import RoboDojoEnvironment

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def env(self):
        e=Environment();e.reset({'instruction':'move'});return e
    def test_generic_episode_and_budget(self):
        env=self.env();run_episode(env,root=self.root/'run',max_steps=1,**options(.6))
        result=json.loads((self.root/'run/episode-summary.json').read_text())
        self.assertEqual(result['executed_steps'],1)
        self.assertEqual(result['stop_reason'],'agent_stop:action_budget_exhausted')
        self.assertEqual(result['model_budget']['used'],0)
    def test_bounds_rejected_without_motion(self):
        env=self.env();run_episode(env,root=self.root/'run',max_steps=1,**options(2))
        self.assertEqual(env.step,0)
    def test_tampered_image_rejected(self):
        session=MeasuredSession(self.env(),self.root,2)
        ref=session.records[0];doc=session.load(ref)
        Path(doc['images']['head']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'image modified'):session.load(ref)
    def test_observation_failure_records_executed_motion_and_blocks_replay(self):
        env=self.env();session=MeasuredSession(env,self.root,2)
        executor=EnvironmentExecutor(session,lambda e:None)
        env.observe=lambda: (_ for _ in ()).throw(RuntimeError('capture failure'))
        with self.assertRaisesRegex(RuntimeError,'capture failure'):
            executor.execute(((.2,),),action_contract=env.contract.name)
        self.assertEqual(session.executed_steps,1)
        with self.assertRaisesRegex(RuntimeError,'unavailable'):
            executor.execute(((.2,),),action_contract=env.contract.name)
        self.assertEqual(env.step,1)
    def test_frozen_hmr_reaches_generic_skill_without_changing_source(self):
        from experience.packages import build_package, write_catalog
        from skills.contracts import BackendProposal, ControlSkill, SkillIdentity
        src=self.root/'memory';src.mkdir();(src/'HMR.md').write_text('past evidence')
        package=build_package(src,self.root/'packages/move','move')
        catalog=self.root/'catalog.json'
        write_catalog(catalog,'fixture','packages',[package],{'move':['HMR.md']},
            {'move':dict(bundle='move',kind='original_preparation',base_task='move',reason='fixture')})
        seen=[]
        class Backend:
            def reset(self,ep):pass
            def propose(self,request):
                seen.append(request.experience.to_dict())
                return BackendProposal(((.2,),))
            def observe(self,feedback):pass
        env=self.env();config=options(.2)
        config['skill_factories']=[lambda load,budget:ControlSkill(
            SkillIdentity('direct_control','fixture',env.contract.name,1),Backend())]
        run_episode(env,root=self.root/'run',max_steps=1,experience_catalog=catalog,
            experience_task='move',skill_experience={'direct_control':{'hmr':['HMR.md']}},**config)
        self.assertEqual(seen[0]['hmr'][0]['text'],'past evidence')
        self.assertEqual((src/'HMR.md').read_text(),'past evidence')

    def test_contract_rejects_bad_dimensions(self):
        with self.assertRaises(ValueError):
            ActionContract('bad',('x',),('m',),25,'absolute','world',(),(1,))
    def test_model_budget_shared_and_stops_before_second_call(self):
        config=options(.2);original=config['agent_factory']
        def factory(load,budget):
            agent=original(load,budget);old=agent.next
            agent.next=budget.callback(old,'controller');return agent
        config['agent_factory']=factory
        env=self.env();result=run_episode(env,root=self.root/'run',max_model_calls=1,**config)
        self.assertEqual(result['stop_reason'],'model_budget_exhausted')
        self.assertEqual(env.step,0)
    def test_robodojo_binding_uses_generic_runner(self):
        contract=ActionContract('test-joint14/v1',tuple('q'+str(i) for i in range(14)),
            ('rad',)*14,25,'absolute','joint',(-1,)*14,(1,)*14)
        class Dojo:
            def __init__(self):self.n=0;self.success=[False];self.end_flag=[False]
            def get_obs(self):return dict(instruction='move',additional_info={'frequency':25},
                state=dict(left_arm_joint_state=[0]*6,left_ee_joint_state=[0],right_arm_joint_state=[0]*6,right_ee_joint_state=[0]),
                vision={'head':{'color':np.zeros((8,8,3),np.uint8)}})
            def take_action(self,row):
                assert set(row)=={'left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state'}
                assert np.asarray(row['left_arm_joint_state']).shape==(6,)
                self.n+=1;self.success=[True];self.end_flag=[True]
            def is_episode_end(self):return self.end_flag[0]
        from skills.contracts import BackendProposal, ControlSkill, SkillIdentity
        from examples.modular.fixture_driver import Agent
        class Backend:
            def reset(self,ep):pass
            def propose(self,r):return BackendProposal(((.2,)*14,))
            def observe(self,f):pass
        env=RoboDojoEnvironment(Dojo(),contract=contract,reset=lambda e,c:None,
                               close=lambda e:None,revision='fixture')
        env.reset({})
        run_episode(env,root=self.root/'dojo',skill_factories=[lambda load,budget:ControlSkill(
            SkillIdentity('direct_control','fixture',contract.name,14),Backend())],
            agent_factory=lambda load,budget:Agent())
        summary=json.loads((self.root/'dojo/episode-summary.json').read_text())
        self.assertTrue(summary['native_success']);self.assertEqual(summary['executed_steps'],1)

    def test_only_robodojo_is_builtin(self):
        from environments.registry import available
        self.assertEqual(available(),('robodojo',))

if __name__=='__main__':unittest.main()
