"""Versioned, policy-visible experience records; no simulator-only state."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import math


def vector(values, name: str) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if not result or not all(math.isfinite(x) for x in result):
        raise ValueError(f'{name} must be a nonempty finite vector')
    return result


def evidence_file_mapping(value) -> dict[str, str]:
    """Validate explicitly selected attachments; their paths are checked at ingestion."""
    if not isinstance(value, Mapping) or len(value) > 16:
        raise ValueError('evidence_files must map at most 16 labels to relative files')
    result = dict(value)
    if any(not isinstance(label, str) or not label.strip() or len(label) > 128
           or not isinstance(path, str) or not path.strip() or len(path) > 512
           for label, path in result.items()):
        raise ValueError('evidence_files requires nonempty text labels and relative paths')
    return result


@dataclass(frozen=True)
class MemoryStep:
    step: int
    state: tuple[float, ...]
    action: tuple[float, ...]
    images: Mapping[str, str]

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> 'MemoryStep':
        if type(value['step']) is not int or value['step'] < 0:
            raise ValueError('step must be a nonnegative integer')
        images = dict(value['images'])
        if not images or any(not isinstance(k, str) or not isinstance(v, str) for k, v in images.items()):
            raise ValueError('experience images must map camera names to relative file paths')
        return cls(value['step'], vector(value['state'], 'state'), vector(value['action'], 'action'), images)


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    task: str
    episode_id: str
    scene_id: str
    source_type: str
    source_split: str
    channel: str
    action_contract: str
    instruction: str
    outcome: str
    outcome_evidence: str
    note: str
    steps: tuple[MemoryStep, ...]
    evidence_files: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> 'MemoryRecord':
        fields = ('record_id', 'task', 'episode_id', 'scene_id', 'source_type', 'source_split',
                  'channel', 'action_contract', 'instruction', 'outcome')
        for name in fields:
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise ValueError(f'missing/non-text record field: {name}')
        if value['source_type'] not in ('human', 'agent'):
            raise ValueError('source_type must be human or agent')
        if value['source_split'] not in ('training', 'development'):
            raise ValueError('validation/test trajectories cannot enter the frozen memory pool')
        if value['channel'] not in ('icl', 'hmr'):
            raise ValueError('channel must be icl or hmr')
        if value['outcome'] not in ('success', 'failure', 'unknown'):
            raise ValueError('outcome must be success, failure or unknown')
        evidence, note = value.get('outcome_evidence', ''), value.get('note', '')
        if not isinstance(evidence, str) or not isinstance(note, str):
            raise ValueError('evidence and note must be text')
        if value['outcome'] != 'unknown' and not evidence.strip():
            raise ValueError('a success/failure label needs its evidence source')
        if len(note) > 4000 or len(evidence) > 2000 or len(value['instruction']) > 4000:
            raise ValueError('experience text exceeds ingestion limits')
        attachments = evidence_file_mapping(value.get('evidence_files', {}))
        steps = tuple(MemoryStep.parse(s) for s in value['steps'])
        if not steps or len(steps) > 16:
            raise ValueError('a snippet must contain 1..16 steps')
        if any(a.step >= b.step for a, b in zip(steps, steps[1:])):
            raise ValueError('snippet steps must strictly increase')
        if len({len(s.state) for s in steps}) != 1 or len({len(s.action) for s in steps}) != 1:
            raise ValueError('inconsistent dimensions within snippet')
        return cls(**{k: value[k] for k in fields}, outcome_evidence=evidence, note=note,
                   steps=steps, evidence_files=attachments)

    def to_dict(self) -> dict:
        return asdict(self)


def observation(value: Mapping[str, Any]) -> dict:
    """Only allowlisted fields survive; state must already be policy-visible."""
    result = {}
    for name in ('task', 'episode_id', 'action_contract'):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError(f'observation requires {name}')
        result[name] = value[name]
    result['state'] = vector(value['state'], 'observation state')
    result['images'] = dict(value.get('images', {}))
    result['instruction'] = str(value.get('instruction', value.get('prompt', '')))
    result['scene_id'] = str(value.get('scene_id', ''))
    return result
