"""Runnable HMR/ICL plumbing demo; fixed actions, no learned performance claim."""
import argparse
import json
from pathlib import Path

from examples.modular.fixture_driver import Agent, Backend, CONTRACT, Environment
from skills.contracts import ControlSkill, SkillIdentity
from execution.episode import run_episode
from experience.packages import build_package, write_catalog


def run(output, mode):
    if mode not in ('none', 'hmr', 'icl', 'both'):
        raise ValueError('mode must be none/hmr/icl/both')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    channels = {'none': (), 'hmr': ('hmr',), 'icl': ('icl',), 'both': ('icl', 'hmr')}[mode]
    kwargs = {}
    if channels:
        source = output / 'source'
        source.mkdir()
        for channel in channels:
            (source / (channel.upper()+'.md')).write_text(
                f'{channel.upper()} teaching fixture only; not real robot evidence.\n')
        package = build_package(source, output/'packages'/'move', 'move',
                                metadata={'scope': 'teaching-fixture', 'channels': list(channels)})
        catalog = output/'catalog.json'
        write_catalog(catalog, 'fixture-'+mode, 'packages', [package],
            {'move': [c.upper()+'.md' for c in channels]},
            {'move': dict(bundle='move', kind='original_preparation', base_task='move',
                          reason='teaching fixture: experience plumbing only')})
        kwargs = dict(experience_catalog=catalog, experience_task='move',
            skill_experience={'direct_control': {c: [c.upper()+'.md'] for c in channels}})
    consumed = []

    class RecordingBackend(Backend):
        def propose(self, request):
            consumed.append(request.experience.to_dict() if request.experience else None)
            # Deliberately fixed: demonstrates delivery, not causal performance benefit.
            return super().propose(request)

    env = Environment()
    try:
        env.reset({'instruction': 'move to target'})
        result = run_episode(env, root=output/'episode', max_steps=2, max_model_calls=4,
            skill_factories=[lambda load, budget: ControlSkill(
                SkillIdentity('direct_control', 'experience-fixture/v1', CONTRACT.name, 1),
                RecordingBackend(.6))],
            agent_factory=lambda load, budget: Agent(), **kwargs)
    finally:
        env.close()
    (output/'consumed-experience.json').write_text(json.dumps(consumed, indent=2))
    summary = dict(mode=mode, scope='CPU experience delivery only; no model or performance claim',
                   expected_channels=list(channels), proposals=len(consumed),
                   executed_steps=result['executed_steps'], output=str(output))
    (output/'demo-summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('none', 'hmr', 'icl', 'both'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.mode), indent=2))
