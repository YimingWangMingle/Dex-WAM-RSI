"""Explicit entry points for fixed episodes and recursive improvement campaigns."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode',required=True)
    episode = commands.add_parser('episode',help='run one fixed version')
    episode.add_argument('--driver',type=Path,required=True)
    episode.add_argument('--config',type=Path,required=True)
    episode.add_argument('--output',type=Path,required=True)
    commands.add_parser('campaign',help='run bounded propose/validate/inherit rounds',add_help=False)
    commands.add_parser('platforms',help='list built-in lazy environment bindings')
    args, extra = parser.parse_known_args()
    if args.mode == 'campaign':
        sys.argv = ['run_framework_campaign.py',*extra]
        runpy.run_module('scripts.run_framework_campaign',run_name='__main__')
        return
    if extra: parser.error('unexpected arguments: '+' '.join(extra))
    if args.mode == 'platforms':
        from environments.registry import available
        print('\n'.join(available())); return
    from execution.episode import run_episode
    config = json.loads(args.config.read_text())
    driver = runpy.run_path(str(args.driver.resolve()),run_name='rsi_episode_driver')
    # Trusted driver owns simulator creation. Config cannot execute arbitrary imports.
    env = driver['create_environment'](config['environment'])
    try:
        env.reset(config['case'])
        options = driver['build_episode'](env,config.get('controller',{}))
        if set(options) & {'root','max_steps','max_model_calls','action_spec'}:
            raise ValueError('driver options conflict with fixed episode configuration')
        result = run_episode(env,root=args.output,action_spec=env.contract,
            max_steps=config['max_steps'],max_model_calls=config['max_model_calls'],**options)
        print(json.dumps(result,indent=2,default=asdict))
    finally:
        env.close()

if __name__ == '__main__': main()
