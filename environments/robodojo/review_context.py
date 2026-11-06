"""Explicit episode-local visual evidence for isolated WAM reviewers.

No dialogue memory, full trajectories, scores or native success flags are exposed.
W0 and W1 share this schema; only W1 includes action-bound forecasts.
"""
import copy
import hashlib
import json
from pathlib import Path


def build_review_packet(packet: dict, root) -> dict:
    root = Path(root).resolve()
    step = packet['decision_step']
    files = list((root / 'observations').glob(f'*/{step:06d}/observation.json'))
    if len(files) != 1:
        raise ValueError('exactly one episode-local observation required')
    current_path = files[0]
    episode_id = current_path.parent.parent.name
    current = json.loads(current_path.read_text())
    if current['step'] != step or current['instruction'] != packet['instruction']:
        raise ValueError('observation and request mismatch')
    events_path = root / 'framework-events.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    events = [e for e in events if e.get('episode_id') == episode_id]
    executions = [e for e in events if e.get('event') == 'execution'
                  and e['after']['control_step'] <= step]
    selected = executions[-2:]
    history = []
    feedback = []
    for execution in selected:
        previous_path = Path(execution['before']['payload_ref']).resolve()
        if previous_path.parent.parent != current_path.parent.parent.resolve():
            raise ValueError('cross-episode history forbidden')
        previous = json.loads(previous_path.read_text())
        if previous['step'] >= step or (history and previous['step'] <= history[-1]['step']):
            raise ValueError('history must be ordered and precede current observation')
        history.append(previous)
        reviews = [e for e in events if e.get('event') == 'review'
                   and e.get('proposal_id') == execution['proposal_id'] and e.get('decision') == 'accept']
        accepted = reviews[-1] if reviews else {}
        feedback.append(dict(candidate_id=accepted.get('candidate_id'),
                        approved_steps=len(execution['approved_actions']),
                        executed_steps=len(execution['executed_actions']),
                        before_step=previous['step'], after_step=execution['after']['control_step'],
                        stop_reason=execution['stop_reason']))
    if step != 0 and (not selected or selected[-1]['after']['control_step'] != step):
        raise ValueError('missing execution evidence for noninitial observation')
    result = {k:copy.deepcopy(packet[k]) for k in
              ('instruction', 'decision_step', 'candidates', 'allowed_execute_steps',
               'review_feedback', 'skill_experience', 'recovery_feedback', 'review_operations') if k in packet}
    initial = json.loads((current_path.parent.parent/'000000'/'observation.json').read_text())
    result['initial_measured_state'] = copy.deepcopy(initial['state'])
    result.update(schema='wam-review/episode-local-v2', episode_id=episode_id,
                  measured_state=current['state'], execution_feedback=feedback,
                  images=[])
    def measured(doc, kind):
        for camera, path in doc['images'].items():
            content = Path(path).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != doc['image_hashes'][path]:
                raise ValueError('measured image changed')
            result['images'].append(dict(kind=kind, camera=camera, control_step=doc['step'],
                                        path=path, sha256=digest))
    for previous in history:
        measured(previous, 'measured_history')
    measured(current, 'measured_current')
    forecasts = [c.get('forecast') is not None for c in result['candidates']]
    if any(forecasts) and not all(forecasts):
        raise ValueError('all candidates need predictions in W1')
    for candidate in result['candidates']:
        forecast = candidate.get('forecast')
        if forecast is None:
            continue
        if not candidate.get('action_hash') or forecast.get('action_hash') != candidate['action_hash']:
            raise ValueError('forecast must match candidate actions')
        if len(forecast.get('frames', [])) != 12 or forecast.get('frame_steps') != list(range(step+2,step+25,2)):
            raise ValueError('twelve aligned predicted frames required')
        import os
        frame_count=int(os.environ.get('RSI_A2V_RGB_FRAMES','6'))
        if frame_count not in (6,12):raise ValueError('RGB frames must be 6 or 12')
        offsets = (2, 6, 10, 16, 20, 24) if frame_count==6 else tuple(range(2,25,2))
        include_depth=os.environ.get('RSI_A2V_DEPTH','0')=='1'
        depth=forecast.get('depth_frames')
        if include_depth and (not isinstance(depth,list) or len(depth)!=12):
            raise ValueError('requested depth missing; RGB fallback forbidden')
        for offset in offsets:
            path=forecast['frames'][offset//2-1]
            result['images'].append(dict(kind='predicted_RGB',candidate_id=candidate['candidate_id'],
                control_step=step+offset,offset_steps=offset,path=path,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
            if include_depth:
                depth_path=depth[offset//2-1]
                if os.environ.get('RSI_A2V_METRIC_DEPTH')=='1':
                    from environments.robodojo.metric_depth import render
                    depth_path=render(depth_path)
                result['images'].append(dict(kind='predicted_depth',candidate_id=candidate['candidate_id'],
                    control_step=step+offset,offset_steps=offset,path=depth_path,
                    sha256=hashlib.sha256(Path(depth_path).read_bytes()).hexdigest()))
        if include_depth:
            result['depth_description']=('Predicted depth reconstructed with training log codec (0.01-2.0 m), fixed meter legend; magenta marks padding or zero-code. Model prediction, not a measurement.' if os.environ.get('RSI_A2V_METRIC_DEPTH')=='1' else 'Model-predicted normalized grayscale, bright=near and dark=far/invalid; not measured or metric depth.')
        candidate['forecast'] = dict(action_hash=forecast['action_hash'],
            generated_frame_steps=forecast['frame_steps'], reviewed_offsets=list(offsets))
    return result
