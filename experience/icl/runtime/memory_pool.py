"""Build once, verify on load, and search a frozen memory pool without gradients."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np

from .memory_features import RGBEncoder, rgb_image, visual_features
from .memory_schema import MemoryRecord

EVIDENCE_SUFFIXES = frozenset(('.json', '.txt', '.md'))
MAX_EVIDENCE_BYTES = 1024 * 1024


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name or rel.as_posix() != name:
        raise ValueError(f'unsafe pool-relative path: {name}')
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('path outside pool')
    return target


def read_evidence_file(root: Path, name: str) -> bytes:
    """Read one explicit bounded text attachment, without fetching external references.

    Hashing preserves provenance bytes, not the truth of the result they describe.
    """
    path = inside(root, name)
    if path.suffix.lower() not in EVIDENCE_SUFFIXES:
        raise ValueError('evidence attachments must be .json, .txt or .md files')
    if not path.is_file():
        raise ValueError(f'missing evidence file: {name}')
    with path.open('rb') as stream:
        data = stream.read(MAX_EVIDENCE_BYTES + 1)
    if not data or len(data) > MAX_EVIDENCE_BYTES:
        raise ValueError('evidence attachment must contain 1 byte to 1 MiB')
    try:
        data.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise ValueError('evidence attachment must be UTF-8 text') from error
    return data


def build_pool(records: Sequence[dict], source_root: str | Path, output: str | Path, *,
               cameras=('base', 'wrist'), state_scale=None, encoder=None) -> Path:
    """Package source snippets and image assets. Existing pools cannot be overwritten."""
    parsed = [MemoryRecord.parse(r) for r in records]
    if not parsed or len({r.record_id for r in parsed}) != len(parsed):
        raise ValueError('pool requires nonempty records with unique IDs')
    cameras = tuple(cameras)
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError('unique camera names required')
    dims = {len(s.state) for r in parsed for s in r.steps}
    if len(dims) != 1:
        raise ValueError('one state layout per pool; create separate pools for other robots')
    contracts = {r.action_contract for r in parsed}
    if len(contracts) != 1:
        raise ValueError('one action contract per pool')
    action_dims = {len(s.action) for r in parsed for s in r.steps}
    if len(action_dims) != 1:
        raise ValueError('one action dimension per pool')
    dim = next(iter(dims))
    scales = np.ones(dim) if state_scale is None else np.asarray(state_scale, dtype=float)
    if scales.shape != (dim,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('state_scale must provide a positive finite scale per dimension')
    encoder = encoder or RGBEncoder()
    if not isinstance(encoder.revision, str) or not encoder.revision:
        raise ValueError('encoder needs a versioned identity')
    source_root, output = Path(source_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    inventory, exported, visual, states = {}, [], [], []
    for record in parsed:
        value = record.to_dict()
        attachments = {}
        for label, name in record.evidence_files.items():
            data = read_evidence_file(source_root, name)
            sha = hashlib.sha256(data).hexdigest()
            key = 'evidence/' + sha + PurePosixPath(name).suffix.lower()
            if key not in inventory:
                (output / 'evidence').mkdir(exist_ok=True)
                (output / key).write_bytes(data)
                inventory[key] = sha
            attachments[label] = key
            if value['outcome_evidence'] == name:
                value['outcome_evidence'] = key
        value['evidence_files'] = attachments
        for step in value['steps']:
            mapped = {}
            for camera in cameras:
                source = inside(source_root, step['images'][camera])
                # Re-encode as PNG, strip metadata, and deduplicate identical image bytes.
                from io import BytesIO
                stream = BytesIO()
                rgb_image(source).save(stream, format='PNG')
                data = stream.getvalue()
                key = 'images/' + hashlib.sha256(data).hexdigest() + '.png'
                if key not in inventory:
                    (output / key).write_bytes(data)
                    inventory[key] = hashlib.sha256(data).hexdigest()
                mapped[camera] = key
            step['images'] = mapped
        first = value['steps'][0]
        visual.append(visual_features({k: output / v for k, v in first['images'].items()}, cameras, encoder))
        states.append(first['state'])
        exported.append(value)
    with (output / 'records.jsonl').open('w', encoding='utf-8') as stream:
        for row in exported:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    np.savez_compressed(output / 'features.npz', visual=np.stack(visual), state=np.array(states, dtype=np.float32))
    inventory.update({name: digest(output / name) for name in ('records.jsonl', 'features.npz')})
    manifest = dict(schema='context-memory-pool/v1', encoder=encoder.revision, cameras=list(cameras),
                    state_scale=scales.tolist(), action_contract=next(iter(contracts)),
                    count=len(parsed), files=inventory)
    (output / 'memory_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return output


@dataclass(frozen=True)
class Hit:
    record: MemoryRecord
    score: float
    visual_score: float
    state_score: float


class MemoryPool:
    def __init__(self, root: str | Path, *, encoder=None):
        self.root = Path(root)
        manifest_path = self.root / 'memory_manifest.json'
        self.revision = digest(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['schema'] != 'context-memory-pool/v1':
            raise ValueError('unsupported memory pool')
        self.encoder = encoder or RGBEncoder()
        if self.encoder.revision != manifest['encoder']:
            raise ValueError('pool/query encoder identity mismatch')
        for name, expected in manifest['files'].items():
            if digest(inside(self.root, name)) != expected:
                raise ValueError(f'pool file hash mismatch: {name}')
        self.records = tuple(MemoryRecord.parse(json.loads(line)) for line in
                             (self.root / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())
        if len(self.records) != manifest['count'] or len({r.record_id for r in self.records}) != len(self.records):
            raise ValueError('pool count/ID mismatch')
        self.cameras = tuple(manifest['cameras'])
        self.scales = np.asarray(manifest['state_scale'], dtype=np.float32)
        self.action_contract = manifest['action_contract']
        if self.scales.ndim != 1 or not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError('invalid state scales')
        with np.load(self.root / 'features.npz', allow_pickle=False) as data:
            self.visual, self.states = data['visual'].copy(), data['state'].copy()
        if (self.states.shape != (len(self.records), len(self.scales)) or self.visual.ndim != 2
                or len(self.visual) != len(self.records) or not np.isfinite(self.states).all()
                or not np.isfinite(self.visual).all()):
            raise ValueError('invalid pool feature dimensions/values')
        for record in self.records:
            if record.action_contract != self.action_contract:
                raise ValueError('record/pool action contract mismatch')
            for name in record.evidence_files.values():
                if name not in manifest['files']:
                    raise ValueError('unverified evidence reference')
                read_evidence_file(self.root, name)
            for step in record.steps:
                if len(step.state) != len(self.scales) or set(step.images) != set(self.cameras):
                    raise ValueError('record layout differs from pool')
                if any(name not in manifest['files'] for name in step.images.values()):
                    raise ValueError('unverified image reference')
        self.visual.setflags(write=False)
        self.states.setflags(write=False)
        self.scales.setflags(write=False)

    def retrieve(self, obs: dict, *, channel: str, k: int = 2, visual_weight: float = .65,
                 min_score: float = .0, max_per_episode: int = 1, balance_outcomes: bool = True) -> list[Hit]:
        if (type(k) is not int or type(max_per_episode) is not int or k < 0 or max_per_episode < 1
                or not 0 <= visual_weight <= 1 or not 0 <= min_score <= 1):
            raise ValueError('invalid retrieval settings')
        if k == 0 or obs['action_contract'] != self.action_contract:
            return []
        candidates = [i for i, r in enumerate(self.records) if r.channel == channel and r.task == obs['task']
                      and r.episode_id != obs['episode_id']
                      and (not obs['scene_id'] or r.scene_id != obs['scene_id'])]
        if not candidates:
            return []
        state = np.asarray(obs['state'], dtype=np.float32)
        if state.shape != self.scales.shape:
            raise ValueError('query state dimension differs from pool; adapt public state explicitly')
        feature = visual_features(obs['images'], self.cameras, self.encoder)
        if feature.shape != self.visual.shape[1:]:
            raise ValueError('encoder feature dimension mismatch')
        indices = np.asarray(candidates)
        visual = np.clip(self.visual[indices] @ feature, 0, 1)
        delta = (self.states[indices] - state) / self.scales
        state_score = np.exp(-.5 * np.mean(delta * delta, axis=1))
        score = visual_weight * visual + (1 - visual_weight) * state_score
        # Partial selection is unnecessary for small pools; one vectorized scan and deterministic sorting.
        ranking = sorted((Hit(self.records[i], float(s), float(v), float(t))
                          for i, s, v, t in zip(indices, score, visual, state_score) if s >= min_score),
                         key=lambda h: (-h.score, h.record.record_id))
        chosen, counts = [], {}
        def add(hit):
            ep = hit.record.episode_id
            if hit not in chosen and counts.get(ep, 0) < max_per_episode and len(chosen) < k:
                chosen.append(hit)
                counts[ep] = counts.get(ep, 0) + 1
        if balance_outcomes and channel == 'hmr' and k >= 2:
            for outcome in ('success', 'failure'):
                match = next((h for h in ranking if h.record.outcome == outcome), None)
                if match is not None:
                    add(match)
        for hit in ranking:
            add(hit)
        return sorted(chosen, key=lambda h: (-h.score, h.record.record_id))
