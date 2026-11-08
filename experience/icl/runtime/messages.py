"""Assemble current RGB + retrieved memory and deliver them to an existing client.

The caller owns model configuration, action parsing, simulator execution and
scoring. This module performs no network access without the supplied client.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json

SINGLE_CONTRACT = 'dexjoco/single-xyz-rotvec-hand16/v1'
DUAL_CONTRACT = 'dexjoco/dual-right-left-xyz-rotvec-hand16/v1'
from .memory_features import rgb_image
from .memory_schema import observation
from .render import ContextMemory


def _public_observation(obs: dict) -> dict:
    value = observation(obs)
    if not value['instruction'].strip() or not value['images']:
        raise ValueError('model input requires a task instruction and current camera images')
    layouts = {SINGLE_CONTRACT: (23, {'base', 'wrist'}),
               DUAL_CONTRACT: (46, {'wrist_left', 'wrist_right'})}
    if value['action_contract'] in layouts:
        dimension, cameras = layouts[value['action_contract']]
        if len(value['state']) != dimension or set(value['images']) != cameras:
            raise ValueError('provide the public DexJoCo state dimensions and camera names required by the action contract')
    elif value['action_contract'].startswith('dexjoco/'):
        raise ValueError('unknown DexJoCo action contract')
    return value


def build_messages(obs: dict, context: ContextMemory, *, system_prompt: str,
                             history=(), current_image_side: int | None = None) -> list[dict]:
    """Return ready-to-send chat messages with inline image bytes, not file links.

    Current images retain their original resolution by default. Any explicit
    resize setting is explicitly controlled by the caller.
    Pass prior conversation/execution feedback in history; do not pre-append
    this step's current observation or memory a second time.
    """
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError('provide the policy/action contract system prompt')
    if current_image_side is not None and (type(current_image_side) is not int or current_image_side < 32):
        raise ValueError('current_image_side must be None or an integer >= 32')
    obs = _public_observation(obs)
    for field in ('task', 'episode_id', 'scene_id'):
        if field in context.audit and context.audit[field] != obs[field]:
            raise ValueError('context identity differs from current observation: ' + field)
    if not isinstance(history, (tuple, list)) or any(not isinstance(m, dict) for m in history):
        raise ValueError('history must be a sequence of client messages')
    messages = [dict(role='system', content=system_prompt), *deepcopy(list(history))]
    historical = context.to_content()
    if historical:
        messages.append(dict(role='user', content=historical))
    # Put CURRENT last, so the latest observation is unambiguous after examples.
    public = {key: obs[key] for key in ('task', 'episode_id', 'instruction', 'action_contract', 'state')}
    content = [dict(type='text', text='CURRENT OBSERVATION. These camera images show the current scene.\n'
                    + json.dumps(public, ensure_ascii=False, allow_nan=False))]
    for camera, source in sorted(obs['images'].items()):
        pixels = rgb_image(source)
        if current_image_side is not None:
            pixels.thumbnail((current_image_side, current_image_side))
        stream = BytesIO()
        pixels.save(stream, format='PNG')
        payload = stream.getvalue()
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError('one current PNG exceeds 16 MiB; choose an explicit shared resize setting')
        content += [dict(type='text', text=f'CURRENT camera={camera} episode={obs["episode_id"]}'),
                    dict(type='image_url', image_url=dict(
                        url='data:image/png;base64,' + base64.b64encode(payload).decode('ascii'), detail='auto'))]
    messages.append(dict(role='user', content=content))
    return messages
