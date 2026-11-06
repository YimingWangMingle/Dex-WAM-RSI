"""Run actual PhysicalRSI inheritance on a tiny fixture, with no GPU/model.

This exercises the production campaign implementation, not a second RSI loop.
The repair provider is deterministic; it is not evidence of learned improvement.
"""
import argparse
import json
from pathlib import Path
from rsi.framework_campaign import FrameworkSuite, FailureEdits, build_campaign, ENTRY
from rsi.core_adapter import TaskSpec
from rsi.version_manifest import CLOSURE, build_harness, write_backend_version
from execution.episode import run_episode
from examples.modular.fixture_driver import Environment, Backend, CONTRACT
from skills.contracts import ControlSkill, SkillIdentity
from execution.runner import CallSkill, CallTool, Review
from execution.review import ReviewDecision
from tools.contracts import TaskTool, ToolLimits

POLICY='skill_selection/policy.json'

class Agent:
    def next(self,view):
        if not view.pending:return CallSkill(view.available_skills[0],'move',1)
        p=view.pending
        if not p['tool_predictions']:
            return CallTool('a2v',dict(proposal_id=p['proposal_id'],
                observation_id=view.latest_observation.observation_id,candidate_ids=[0]))
        return Review(ReviewDecision(p['proposal_id'],'accept','fixture consequence checked','move',1))

def predict(request):
    return dict(predictions=[dict(candidate_id=c.candidate_id,forecast={'fixture_x':c.actions[-1][0]}) for c in request.candidates])

def resources(env,case):
    return dict(action_spec=env.contract,
        skill_factories=[lambda load,budget:ControlSkill(SkillIdentity('small','v1',CONTRACT.name,1),Backend(.1)),
                         lambda load,budget:ControlSkill(SkillIdentity('large','v1',CONTRACT.name,1),Backend(.6))],
        agent_factory=lambda load,budget:Agent(),task_tools={'a2v':TaskTool('a2v','fixture-v1',predict)},
        tool_limits=ToolLimits(max_calls=2),required_prediction_skills=('small','large'))

def policy(skill,automatic):
    return json.dumps(dict(skills=[skill],tools=['a2v'],automatic_tools=['a2v'] if automatic else [],decision_mode='agent'))



def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    parent=root/'parent';files={k:[] for k in CLOSURE}
    for kind in CLOSURE:
        path=parent/kind/'identity.txt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('fixture-'+kind);files[kind].append(str(path.relative_to(parent)))
    (parent/ENTRY).write_text('def configure(root,case): return {}\n');files['control'].append(ENTRY)
    (parent/POLICY).write_text(policy('small',False));files['skill_selection'].append(POLICY)
    metadata=write_backend_version(parent,backend_revision='fixture/v1',
        action_contract='fixture-position/v1',control_hz=10,runtime_dependencies={'driver':'synthetic/v1'})
    files['configuration'].append(str(metadata.relative_to(parent)))
    initial=build_harness(parent,'parent',files)
    def open_env(task,case):
        env=Environment();env.reset(case);return env
    suite=FrameworkSuite(name='portable-fixture',
        tasks={'move':TaskSpec('move',[1],[2,3],2,[4])},initial=initial,
        open_env=open_env,resources=resources,
        close_env=lambda env:env.close(),driver_revision='2'*64,
        work_root=root/'episodes',max_steps=2,episode_runner=run_episode,selection_policy_file=POLICY)
    def repair(parent,evaluations):
        assert all(e['trial']['case']['split']=='development' for e in evaluations)
        return [dict(id='child',reason='fixture correction from development failure',
                     edits={POLICY:policy('large',True)})]
    campaign=build_campaign(root=root/'rsi',suite=suite,
        repair=FailureEdits(repair,revision='fixture-repair-v1'),max_rounds=2,max_candidates=1,
        max_trials=6,seconds_per_episode=60,
        profile=dict(tasks={'move':dict(weight=1,episodes=2,score_range=[0,1],maximum_regression=0)},
                     minimum_gain=0,tie_tolerance=1e-10))
    result=campaign.run(initial)
    (root/'result.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    print(json.dumps(dict(status=result['status'],rounds=[r['outcome'] for r in result['rounds']],
                         selected=result['current']['harness']['id']),indent=2))
