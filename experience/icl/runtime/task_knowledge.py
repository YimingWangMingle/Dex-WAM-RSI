"""Frozen task guidance, phase evidence and numerical references; never executes code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

from .memory_pool import inside

KNOWLEDGE_SCHEMA = 'task-icl-knowledge/v1'
MANIFEST_SCHEMA = 'task-icl-manifest/v1'
MANIFEST_NAME = 'knowledge_manifest.json'
MAX_METADATA_BYTES = 16 * 1024 * 1024
_URL = re.compile(r'https?://[^\s<>"\[\]]+')


def _context_sources(doc: dict) -> dict[str, dict]:
    """Stable short IDs; full locations/hashes belong to audit, not model input."""
    sources = {}
    def add(source):
        if isinstance(source, str) and source and source not in sources:
            sources[source] = dict(source=source)
    for row in [*doc['phases'], *doc['recovery']]:
        add(row.get('source'))
    for row in doc['provenance'].get('source_files', []):
        if isinstance(row, dict) and isinstance(row.get('url'), str):
            add(row['url'])
            for key in ('path', 'sha256'):
                if key in row:
                    sources[row['url']][key] = row[key]
    def collect(value):
        if isinstance(value, str):
            for found in _URL.findall(value):
                add(found)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key not in ('source', 'path', 'sha256'):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect([doc['guidance'], doc['phases'], doc['recovery'], doc['references']])
    return {f'S{i}': value for i, value in enumerate(sources.values(), 1)}


def _model_text(value: str, sources: dict[str, dict]) -> str:
    identifiers = {row['source']: key for key, row in sources.items()}
    return _URL.sub(lambda match: '[' + identifiers[match.group()] + ']', value)


def _inventory(root: Path) -> dict[str, str]:
    files = {}
    if root.is_symlink():
        raise ValueError('knowledge root cannot be a symlink')
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('knowledge files cannot be symlinks')
        if path.is_file() and path != root / MANIFEST_NAME:
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def _document(value: dict) -> dict:
    if not isinstance(value, dict) or value.get('schema') != KNOWLEDGE_SCHEMA:
        raise ValueError('unsupported task knowledge schema')
    for key in ('task', 'action_contract', 'guidance'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError('task knowledge requires nonempty ' + key)
    for key in ('phases', 'recovery', 'references'):
        if not isinstance(value.get(key), list) or any(not isinstance(row, dict) for row in value[key]):
            raise ValueError('task knowledge requires a list of objects: ' + key)
    for key in ('provenance', 'coverage'):
        if not isinstance(value.get(key), dict):
            raise ValueError('task knowledge requires an object: ' + key)
    for phase in value['phases']:
        for key in ('id', 'name', 'instruction'):
            if not isinstance(phase.get(key), str) or not phase[key].strip():
                raise ValueError('task phase requires ' + key)
    if len({row['id'] for row in value['phases']}) != len(value['phases']):
        raise ValueError('task phase ids must be unique')
    for row in value['recovery']:
        for key in ('symptom', 'response', 'evidence_status', 'source'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('recovery guidance requires ' + key)
    identifiers = []
    for row in value['references']:
        for key in ('id', 'path', 'kind'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('task reference requires ' + key)
        if Path(row['path']).suffix.lower() not in ('.npz', '.npy', '.json'):
            raise ValueError('runtime references must be numerical .npz/.npy or .json data')
        identifiers.append(row['id'])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('task reference ids must be unique')
    return value


def build_task_knowledge(source, output) -> Path:
    """Seal a prepared source tree into a NEW directory, with full byte hashes.

    The source must contain knowledge.json and its referenced assets. It may also
    contain TASK.md/PHASES.json/RECOVERY.json, provenance and source documentation.
    Source provenance is retained, not independently authenticated by hashing.
    """
    source, output = Path(source), Path(output)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError('new knowledge output must be outside its source')
    if (source / MANIFEST_NAME).exists():
        raise ValueError('source is already sealed; do not mutate/reseal a published package')
    files = _inventory(source)
    if 'knowledge.json' not in files:
        raise ValueError('missing knowledge.json')
    doc = _document(json.loads((source / 'knowledge.json').read_text(encoding='utf-8')))
    for ref in doc['references']:
        inside(source, ref['path'])
        if ref['path'] not in files:
            raise ValueError('task reference missing from source: ' + ref['path'])
    output.mkdir(parents=True, exist_ok=False)
    for name in files:
        target = inside(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(source, name), target)
    (output / MANIFEST_NAME).write_text(json.dumps(dict(schema=MANIFEST_SCHEMA, files=files),
        indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
    TaskKnowledge(output)
    return output


class TaskKnowledge:
    """Verified task knowledge; reads may be routed through an episode audit reader."""

    def __init__(self, root, *, reader=None):
        self.root, self.reader = Path(root), reader
        manifest_data = (self.root / MANIFEST_NAME).read_bytes()
        if len(manifest_data) > MAX_METADATA_BYTES:
            raise ValueError('knowledge manifest exceeds metadata bound')
        manifest = json.loads(manifest_data)
        if manifest.get('schema') != MANIFEST_SCHEMA or not isinstance(manifest.get('files'), dict):
            raise ValueError('unsupported task knowledge manifest')
        self.files = manifest['files']
        for name in self.files:
            inside(self.root, name)
        if _inventory(self.root) != self.files or 'knowledge.json' not in self.files:
            raise ValueError('task knowledge file inventory/hash mismatch')
        self.revision = hashlib.sha256(manifest_data).hexdigest()
        self.doc = self.document()
        self.task, self.action_contract = self.doc['task'], self.doc['action_contract']
        for row in self.doc['references']:
            if row['path'] not in self.files:
                raise ValueError('reference is not included in task knowledge inventory')

    def read(self, name: str) -> bytes:
        if name not in self.files:
            raise ValueError('undeclared task knowledge file')
        data = self.reader(name) if self.reader else inside(self.root, name).read_bytes()
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != self.files[name]:
            raise ValueError('task knowledge changed after verification')
        return data

    def document(self) -> dict:
        data = self.read('knowledge.json')
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError('task knowledge exceeds metadata bound')
        return _document(json.loads(data))

    def references(self) -> tuple[dict, ...]:
        return tuple(dict(row, sha256=self.files[row['path']]) for row in self.doc['references'])

    def read_reference(self, reference_id: str) -> bytes:
        row = next((r for r in self.doc['references'] if r['id'] == reference_id), None)
        if row is None:
            raise ValueError('unknown task reference id')
        return self.read(row['path'])

    def context_sections(self) -> list[dict]:
        """Model-facing guidance only; provenance and file locations stay in audit."""
        doc = self.document()
        sources = _context_sources(doc)
        ids = {row['source']: key for key, row in sources.items()}
        text = lambda value: _model_text(value, sources)
        sections = [dict(section='guidance', text=text(doc['guidance']))]
        # Interleave the two categories so long phase catalogs cannot starve all
        # recovery guidance. Full provenance/coverage remain in the sealed source.
        for i in range(max(len(doc['phases']), len(doc['recovery']))):
            if i < len(doc['phases']):
                row = doc['phases'][i]
                phase = dict(section='phase', id=row['id'], name=text(row['name']),
                             instruction=text(row['instruction']),
                             observable_checks=[text(item) for item in row.get('observable_checks', [])])
                if row.get('evidence_status'):
                    phase['evidence_status'] = row['evidence_status']
                if row.get('source'):
                    phase['source_id'] = ids[row['source']]
                sections.append(phase)
            if i < len(doc['recovery']):
                row = doc['recovery'][i]
                sections.append(dict(section='recovery', symptom=text(row['symptom']),
                    response=text(row['response']), evidence_status=row['evidence_status'],
                    source_id=ids[row['source']]))
        for row in doc['references']:
            reference = dict(section='reference', id=row['id'], kind=row['kind'])
            if 'steps' in row:
                reference['steps'] = row['steps']
            sections.append(reference)
        return sections

    def context_sources(self) -> dict[str, dict]:
        """Return the full source lookup for logs/review, never for model messages."""
        return _context_sources(self.document())
