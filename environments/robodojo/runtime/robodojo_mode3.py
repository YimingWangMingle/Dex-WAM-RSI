"""Strict mode-three visual review over the same file bridge as mode two.

No external API call is implicit. request.json exposes only observed RGB/history,
original task text and predicted RGB/depth frames. messages.json contains actual
inline image payloads for a multimodal reviewer adapter, not text-only paths.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import time
from uuid import uuid4


REVIEW_PROMPT = (
    'Select a WAM candidate (0, 1, 2) and how many of its actions to execute. '
    'Prefer the longest supported prefix: 24, then 16, then 8, then 4 steps, '
    'provided its predicted progress and risks are acceptable. This reduces repeated '
    'review calls; do not shorten automatically just because contact is involved. '
    'Shorten when observations or predicted frames show uncertainty, unstable contact, '
    'loss of grasp, collision risk, or a need to check progress sooner. '
    'Compare current measured RGB and real history with the 12 predicted frames. '
    'Frames at offsets up to execute_steps are the chosen prefix; later frames '
    'are hypothetical continuation and may change after replanning. '
    'Depth uses fixed [0,1] grayscale: bright=near, dark=far or invalid/padding. '
    'All future RGB/depth are model predictions, not measured or reliable metric geometry. '
    'Return only {"candidate_id": 0, "execute_steps": 24}, using an allowed length '
    'from the request. If no candidate has an acceptable prefix, return '
    '{"candidate_id": null, "execute_steps": 0} to stop. Never invent or edit joint actions.'
)


def validate_choice(choice, allowed):
    if choice is None:
        return None
    if not isinstance(choice, dict) or set(choice) != {'candidate_id', 'execute_steps'}:
        raise ValueError('review requires candidate_id and execute_steps')
    cid, count = choice['candidate_id'], choice['execute_steps']
    if cid is None and type(count) is int and count == 0:
        return None
    if type(cid) is not int or cid not in range(3) or type(count) is not int or count not in allowed:
        raise ValueError('invalid candidate or unsupported execution length')
    return dict(candidate_id=cid, execute_steps=count)


def mode3_astra_messages(packet):
    """Materialize every image into multimodal content, preserving exact time labels."""
    content = [{'type': 'text', 'text': packet['instruction']},
               {'type': 'text', 'text': 'Allowed execute_steps (prefer longer): ' +
                json.dumps(packet['allowed_execute_steps'])}]

    if packet.get('review_feedback'):
        content.append({'type':'text', 'text':'Previous approval was not executed: ' +
                        json.dumps(packet['review_feedback'], ensure_ascii=False)})
    if packet.get('skill_experience'):
        content.append({'type':'text', 'text':'Historical experience, not current observations: ' +
                        json.dumps(packet['skill_experience'], ensure_ascii=False)})

    def add_image(path, label):
        from PIL import Image
        import io
        # Canonical PNG: no file names or original image metadata enter model input.
        with Image.open(path) as source:
            buffer = io.BytesIO()
            source.convert('RGB').save(buffer, format='PNG')
        content.append({'type': 'text', 'text': label})
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}})

    for row in packet['history']:
        for view, path in enumerate(row['images']):
            add_image(path, f'Measured history step={row["step"]}, view={view}')
    for view, path in enumerate(packet['current_rgb']):
        add_image(path, f'Current measured RGB step={packet["decision_step"]}, view={view}')
    for candidate in packet['candidates']:
        for frame in candidate['frames']:
            label = (f'Candidate {candidate["candidate_id"]}; predicted; '
                     f'+{frame["offset_steps"]} steps / +{frame["offset_s"]:.2f}s; '
                     f'{frame["segment"]}')
            add_image(frame['path'], label + '; RGB')
            if 'depth_path' in frame:
                add_image(frame['depth_path'], label + '; normalized predicted depth')
    return [{'role': 'system', 'content': REVIEW_PROMPT}, {'role': 'user', 'content': content}]


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def _public_packet(packet, directory):
    """Export only visual review data; no source image paths/forecast sidecars."""
    from PIL import Image
    import copy
    directory = Path(directory)
    media = directory / 'media'
    media.mkdir(parents=True, exist_ok=False)
    index = 0
    def picture(path):
        nonlocal index
        target = media / f'{index:04d}.png'
        index += 1
        with Image.open(path) as source:
            # Fresh pixels strip source metadata, including paths or action annotations.
            clean = Image.new('RGB', source.size)
            clean.paste(source.convert('RGB'))
            clean.save(target)
        return str(target)
    keys = ('mode', 'review_id', 'instruction', 'decision_step', 'allowed_execute_steps',
            'control_hz', 'response', 'depth_available', 'depth_description',
            'review_feedback', 'skill_experience')
    public = {k:copy.deepcopy(packet[k]) for k in keys if k in packet}
    public['current_rgb'] = [picture(p) for p in packet['current_rgb']]
    public['history'] = [dict(step=r['step'], source='measured',
                              images=[picture(p) for p in r['images']]) for r in packet['history']]
    public['candidates'] = []
    for candidate in packet['candidates']:
        if {'actions', 'action_hash'} & set(candidate):
            raise ValueError('visual bridge cannot publish candidate actions')
        frames = []
        for frame in candidate['frames']:
            clean = {k:frame[k] for k in ('frame_index','control_step','offset_steps','segment','offset_s') if k in frame}
            clean['path'] = picture(frame['path'])
            if 'depth_path' in frame:
                clean['depth_path'] = picture(frame['depth_path'])
            frames.append(clean)
        public['candidates'].append(dict(candidate_id=candidate['candidate_id'], frames=frames))
    return public


class Mode3FileReviewer:
    """Current-session reviewer; response envelope is request_id + candidate_id.

    Timeout or malformed response stops without choosing a candidate. The request
    directory holds both a readable visual manifest and inline image messages.
    """
    def __init__(self, root, *, timeout_s=600):
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('finite positive review timeout required')
        self.root = Path(root)
        self.timeout_s = timeout_s

    def __call__(self, packet):
        request_id = uuid4().hex
        directory = self.root / request_id
        packet = _public_packet(packet, directory)
        messages = mode3_astra_messages(packet)  # Fail before publishing if media is missing.
        _save(directory / 'messages.json', messages)
        _save(directory / 'request.json', dict(request_id=request_id, prompt=REVIEW_PROMPT,
                                               **packet))
        _save(self.root / 'pending.json', dict(request_id=request_id, directory=str(directory)))
        deadline = time.monotonic() + self.timeout_s
        response = directory / 'response.json'
        try:
            while not response.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('mode 3 review timed out without selecting any action')
                time.sleep(0.2)
            value = json.loads(response.read_text())
            if (not isinstance(value, dict)
                    or set(value) != {'request_id', 'candidate_id', 'execute_steps'}
                    or value['request_id'] != request_id):
                raise ValueError('review response requires matching request_id and a choice')
            return validate_choice({key: value[key] for key in ('candidate_id', 'execute_steps')},
                                   packet['allowed_execute_steps'])
        finally:
            pending = self.root / 'pending.json'
            if pending.exists() and json.loads(pending.read_text()).get('request_id') == request_id:
                pending.unlink()


def eval_mode3_episode(task_env, client, *, root, reviewer=None, max_steps=800):
    """Compatibility entry: all physical execution now belongs to TaskSkillRunner."""
    if type(max_steps) is not int or max_steps <= 0 or max_steps % 4:
        raise ValueError('mode 3 action budget must be a positive multiple of 4')
    from environments.robodojo.framework import run_framework_episode
    return run_framework_episode(task_env,root=root,client=client,reviewer=reviewer,
                                 max_steps=max_steps,mode3=True)
