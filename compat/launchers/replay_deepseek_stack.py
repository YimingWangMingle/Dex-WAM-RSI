"""Real API + recorded RGB + experience reads. NEVER dispatches robot actions.

This is an observation/tool-call integration check, not a closed-loop trial.
The image bundle must be explicitly prepared from a single recorded instant.
"""
import argparse
import base64
from dataclasses import asdict
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skills.contracts import MeasuredObservation
from agents.deepseek import DeepSeekAgent, DeepSeekClient
from execution.runner import AgentView, JsonlEventLog, ReadExperience


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--experience', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    bundle_path = Path(args.bundle)
    bundle = json.loads(bundle_path.read_text())
    images = []
    image_ids = []
    for item in bundle['images']:
        path = bundle_path.parent / item['file']
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item['sha256']:
            raise ValueError('recorded image changed')
        images.append(dict(camera=item['camera'], data_url='data:image/png;base64,' + base64.b64encode(data).decode()))
        image_ids.append(dict(camera=item['camera'], sha256=digest))
    notes = Path(args.experience).read_text()
    obsid = hashlib.sha256(json.dumps(image_ids, sort_keys=True).encode()).hexdigest()
    # Replay has no synchronized native state/control-step binding. Do not invent one.
    obs = MeasuredObservation('recorded-stack-observation', obsid, 0, 0, str(bundle_path.resolve()))
    public = dict(observation_id=obsid, images=images, state=None)
    key = os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('DeepSeek credential (not stored): ')
    client = DeepSeekClient(key)
    events = JsonlEventLog(out / 'agent.jsonl')
    agent = DeepSeekAgent(client, lambda _: public,
        instruction='Stack the three bowls together.', sink=events)
    context = [dict(role='system', kind='pinned', text=(
        'Recorded single-instant observation inspection. No synchronized robot state is available. '
        'Experience workspace contains task.md. Read it before deciding. '
        'WAM can be proposed, but no commands will execute in this replay. '
        'Direct/IK and metric measurement are NOT registered. Do not use filenames to infer outcome.'))]
    decisions = []
    for turn in range(4):
        view = AgentView(context, obs, None, None, ('wam_control',), turn)
        command = agent.next(view)
        decisions.append(dict(type=type(command).__name__, value=asdict(command)))
        context.append(dict(role='agent', kind=type(command).__name__, text=json.dumps(asdict(command))))
        if isinstance(command, ReadExperience):
            if command.op == 'list': result = dict(files=['task.md'])
            elif command.path == 'task.md': result = dict(path='task.md', text=notes[:12000])
            else: result = dict(error='unknown experience path')
            events(dict(event='experience_result', operation=command.op, path=command.path,
                        returned_task_notes=command.op == 'read_text' and command.path == 'task.md'))
            context.append(dict(role='tool', kind='experience', text=json.dumps(result, ensure_ascii=False)))
        else:
            break
    summary = dict(scope='real DeepSeek API / recorded RGB / no robot execution',
        model=client.model, images=image_ids, observation_id=obsid, robot_actions_executed=0,
        synchronized_robot_state=False, decisions=decisions,
        native_task_success=None, ik_available=False, live_wam_called=False,
        experience_sha256=hashlib.sha256(notes.encode()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), decisions=len(decisions), robot_actions_executed=0)))


if __name__ == '__main__': main()
