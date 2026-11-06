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
from examples.modular.fixture_driver import Environment


def candidate_code(target):
    return ('from examples.modular.fixture_driver import options\n'
            f'def configure(root,case): return options({target!r})\n')


def run(root, target=.6):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    parent=root/'parent';files={k:[] for k in CLOSURE}
    for kind in CLOSURE:
        path=parent/kind/'identity.txt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('fixture-'+kind);files[kind].append(str(path.relative_to(parent)))
    (parent/ENTRY).write_text(candidate_code(.1));files['control'].append(ENTRY)
    metadata=write_backend_version(parent,backend_revision='fixture/v1',
        action_contract='fixture-position/v1',control_hz=10,runtime_dependencies={'driver':'synthetic/v1'})
    files['configuration'].append(str(metadata.relative_to(parent)))
    initial=build_harness(parent,'parent',files)
    def open_env(task,case):
        env=Environment();env.reset(case);return env
    suite=FrameworkSuite(name='portable-fixture',
        tasks={'move':TaskSpec('move',[1],[2,3],2,[4])},initial=initial,
        open_env=open_env,resources=lambda env,case:dict(action_spec=env.contract),
        close_env=lambda env:env.close(),driver_revision='2'*64,
        work_root=root/'episodes',max_steps=2,episode_runner=run_episode)
    def repair(parent,evaluations):
        assert all(e['trial']['case']['split']=='development' for e in evaluations)
        return [dict(id='child',reason='fixture correction from development failure',
                     edits={ENTRY:candidate_code(target)})]
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
