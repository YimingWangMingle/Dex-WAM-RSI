"""Task-scoped, audited multimodal ICL for the native Dex-WAM-RSI project.

Install this directory as ``experience/icl`` inside the project's ``experience``
package. The native catalog, package hashes and per-episode frozen workspace are
the authority for selecting and reading task material. No model is trained here.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .runtime.cleaning import clean_histories
from .runtime.memory_pool import MemoryPool, inside
from .runtime.messages import _public_observation, build_messages
from .runtime.render import ContextMemory, PromptBuilder as _Renderer
from .runtime.task_knowledge import TaskKnowledge


_active_icl = ContextVar('dex_wam_rsi_active_icl', default=None)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':'))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class _EventLog:
    def __init__(self, path, callback=None):
        self.path, self.callback, self.sequence = Path(path), callback, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # An episode must not silently inherit another episode's evidence.
        with self.path.open('x', encoding='utf-8'):
            pass

    def __call__(self, event):
        self.sequence += 1
        row = dict(event, sequence=self.sequence,
                   timestamp=datetime.now(timezone.utc).isoformat())
        with self.path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(_json(row) + '\n')
        if self.callback is not None:
            self.callback(deepcopy(row))


class ICLContext:
    """One task's native frozen package, bound to at most one current episode.

    Construct with :func:`open_icl`. ``run_options`` connects the same workspace
    to native ``run_episode``. ``wrap_client`` adds real historical image content
    to an existing client without replacing its current observation or tools.
    """

    def __init__(self, workspace, sink, *, task, skill, top_k, min_score,
                 max_chars, max_images):
        self.workspace, self._sink = workspace, sink
        self.task, self.skill = task, skill
        self.top_k, self.min_score = top_k, min_score
        self.max_chars, self.max_images = max_chars, max_images
        self._closed, self._episode_id, self._scene_id = False, None, None
        self.last_context = None
        self.last_messages = None

        manifest = json.loads(self._read_text_bytes('pool/memory_manifest.json'))
        self._pool_files = dict(manifest['files'])
        self._read_text_bytes('pool/records.jsonl')
        self.pool = MemoryPool(workspace.root / 'pool')
        self.knowledge = TaskKnowledge(workspace.root / 'knowledge', reader=self._read_knowledge)
        if (self.knowledge.task != task
                or self.knowledge.action_contract != self.pool.action_contract
                or any(record.task != task or record.channel != 'icl'
                       for record in self.pool.records)):
            raise ValueError('native task package contains a different task or action contract')
        if not self.pool.records:
            raise ValueError('native ICL package contains no historical examples')
        self._action_dim = len(self.pool.records[0].steps[0].action)
        if any(len(step.action) != self._action_dim
               for record in self.pool.records for step in record.steps):
            raise ValueError('task examples have inconsistent action dimensions')
        self.renderer = _Renderer(max_chars=max_chars, max_images=max_images,
                                  image_reader=self._read_pool_image)
        features = inside(self.pool.root, 'features.npz')
        self._emit('icl_index_loaded', path='pool/features.npz',
                   sha256=_sha(features.read_bytes()), bytes=features.stat().st_size,
                   records=len(self.pool.records), pool_revision=self.pool.revision,
                   purpose='observation similarity retrieval; not model input')
        self._emit('icl_open', workspace=self.workspace.identity(),
                   settings=dict(top_k=top_k, min_score=min_score,
                                 max_chars=max_chars, max_images=max_images))

    def _guard(self):
        if self._closed:
            raise RuntimeError('ICL episode context is closed; open a new context for another episode')

    def _emit(self, event, **fields):
        self._sink(dict(event=event, task=self.task, **fields))

    def _read_text_bytes(self, name):
        """Page through the native bounded reader without truncating UTF-8 JSON."""
        self._guard()
        inside(self.workspace.root, name)
        parts, offset = [], 0
        while True:
            part = self.workspace.read_text(name, offset=offset)
            parts.append(part)
            offset += len(part)
            if offset > 16 * 1024 * 1024:
                raise ValueError('ICL text metadata exceeds its 16 MiB character bound')
            if len(part) < self.workspace.max_text_chars:
                break
        data = ''.join(parts).encode('utf-8')
        if len(data) > 16 * 1024 * 1024:
            raise ValueError('ICL text metadata exceeds its 16 MiB byte bound')
        return data

    def _read_knowledge(self, name):
        """Route text through native paging; audit bounded binary reference reads.

        TaskKnowledge verifies the declared file SHA-256 before returning data.
        The native workspace has no generic binary reader, so this add-on records
        that read explicitly without treating a numerical NPZ as text or an image.
        """
        self._guard()
        relative = 'knowledge/' + name
        path = inside(self.workspace.root, relative)
        if path.suffix.lower() not in ('.npz', '.npy'):
            return self._read_text_bytes(relative)
        if not hasattr(self, 'knowledge') or name not in self.knowledge.files:
            raise ValueError('binary reference is not in the declared knowledge inventory')
        with path.open('rb') as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024 or _sha(data) != self.knowledge.files[name]:
            raise ValueError('binary reference size/hash validation failed')
        self._emit('icl_binary_reference_read', path=relative, bytes=len(data), sha256=_sha(data),
                   purpose='verify historical interval states, commands and source timestamps')
        return data

    def _read_pool_image(self, name):
        self._guard()
        inside(self.pool.root, name)
        if name not in self._pool_files:
            raise ValueError('historical image is not in the verified memory inventory')
        data, _mime = self.workspace.read_image('pool/' + name)
        if _sha(data) != self._pool_files[name]:
            raise ValueError('historical image changed after workspace verification')
        return data

    def _observation(self, obs):
        self._guard()
        value = _public_observation(obs)
        if value['task'] != self.task:
            raise ValueError('observation task differs from the task selected by the native catalog')
        if (value['action_contract'] != self.pool.action_contract
                or len(value['state']) != len(self.pool.scales)
                or set(value['images']) != set(self.pool.cameras)):
            raise ValueError('observation does not match the package state/action/camera layout')
        if (self._episode_id is not None
                and (value['episode_id'], value['scene_id']) != (self._episode_id, self._scene_id)):
            raise ValueError('one frozen ICL context is restricted to one episode_id and scene_id')
        return value

    @property
    def run_options(self):
        """Keyword arguments for native run_episode, using this exact workspace.

        Its ``root`` must be separate from ``open_icl``'s episode_dir because the
        native runner requires a new or empty output directory.
        """
        self._guard()
        return dict(experience=self.workspace, experience_task=self.task,
                    skill_experience={self.skill: dict(icl=['ICL.md'])})

    def get_context_memory(self, obs) -> ContextMemory:
        """Retrieve compatible examples and render task text plus inline RGB."""
        obs = self._observation(obs)
        hits = self.pool.retrieve(obs, channel='icl', k=self.top_k,
                                  min_score=self.min_score, visual_weight=.65,
                                  max_per_episode=1)
        context = self.renderer.build(obs, hits, self.pool.root,
            knowledge=self.knowledge,
            audit=dict(task=self.task, episode_id=obs['episode_id'],
                       scene_id=obs['scene_id'], selection='observation_similarity',
                       pool_revision=self.pool.revision,
                       experience=self.workspace.identity()))
        self._episode_id, self._scene_id = obs['episode_id'], obs['scene_id']
        self.last_context = context
        self._emit('icl_context_prepared', audit=context.audit,
                   content_sha256=_sha(_json(context.to_content()).encode('utf-8')),
                   historical_image_count=len(context.images))
        return context

    def _request_prepared(self, messages, tools, context, *, delivery):
        self._guard()
        self.last_messages = deepcopy(messages)
        self._emit('icl_client_request_prepared', delivery=delivery,
                   episode_id=self._episode_id,
                   serialization='UTF-8 JSON, ensure_ascii=false, sort_keys=true, separators=comma-colon',
                   payload_sha256=_sha(_json(dict(messages=messages, tools=tools)).encode('utf-8')),
                   context_sha256=_sha(_json(context.to_content()).encode('utf-8')),
                   historical_image_count=len(context.images),
                   evidence='request prepared locally; does not attest remote model consumption')

    def build_prompt(self, obs, system_prompt, history=()):
        """Complete client messages: caller policy, history, ICL and current RGB."""
        context = self.get_context_memory(obs)
        messages = build_messages(obs, context, system_prompt=system_prompt, history=history)
        self._request_prepared(messages, None, context, delivery='build_prompt')
        return messages

    def wrap_client(self, client, observation_provider, *, model_budget=None, role='agent'):
        """Wrap an existing ``complete(messages, tools)`` client.

        ``observation_provider()`` must return the policy-visible observation
        corresponding to the request. The caller keeps its model, action schema,
        tool definitions and current images; only historical ICL is inserted.

        Pass a raw client with ``model_budget=budget`` to use the native budget
        transport while preserving raw-client attributes such as last_metadata.
        Do not pass an already-budgeted client together with model_budget, which
        would count the same request twice. Existing budgeted clients can instead
        be supplied with model_budget omitted.
        """
        self._guard()
        if not callable(getattr(client, 'complete', None)) or not callable(observation_provider):
            raise TypeError('provide a complete(messages, tools) client and an observation callable')
        if not isinstance(role, str) or not role.strip():
            raise ValueError('budget role must be a nonempty string')
        transport = client
        if model_budget is not None:
            if not callable(getattr(model_budget, 'client', None)):
                raise TypeError('model_budget must provide the native client(client, role) adapter')
            transport = model_budget.client(client, role)
        return _ICLClient(client, self, observation_provider, transport=transport)

    def close(self):
        """Stop use while retaining the immutable workspace and logs for audit."""
        if not self._closed:
            self._closed = True
            self._emit('icl_close', episode_id=self._episode_id,
                       workspace_retained=True)


class _ICLClient:
    def __init__(self, client, context, observation_provider, *, transport):
        self.client, self.context = client, context
        self._transport = transport
        self.observation_provider = observation_provider
        self.last_context = None
        self.last_messages = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, messages, tools=None, **kwargs):
        self.context._guard()
        if not isinstance(messages, (tuple, list)) or any(not isinstance(m, dict) for m in messages):
            raise ValueError('messages must be a sequence of client message dictionaries')
        context = self.context.get_context_memory(self.observation_provider())
        original = deepcopy(list(messages))
        index = 0
        while index < len(original) and original[index].get('role') in ('system', 'developer'):
            index += 1
        content = context.to_content()
        prepared = original[:index] + ([dict(role='user', content=content)] if content else []) + original[index:]
        self.last_context, self.last_messages = context, deepcopy(prepared)
        self.context._request_prepared(prepared, tools, context, delivery='complete_wrapper')
        return self._transport.complete(messages=prepared, tools=tools, **kwargs)


@contextmanager
def open_icl(task, episode_dir, catalog_path=None, *, skill='direct_control',
             top_k=2, min_score=.3, max_chars=16000, max_images=8, on_event=None):
    """Select a native task package and freeze it for one episode.

    The default catalog is packaged next to this module. ``on_event`` optionally
    receives the same JSON-compatible events written to ``icl-events.jsonl``.
    This function performs no network requests and never edits the source package.
    """
    if not isinstance(task, str) or not task.strip() or not isinstance(skill, str) or not skill.strip():
        raise ValueError('task and skill must be nonempty strings')
    if (type(top_k) is not int or top_k < 0 or not isinstance(min_score, (int, float))
            or not 0 <= min_score <= 1):
        raise ValueError('invalid top_k or min_score')
    if on_event is not None and not callable(on_event):
        raise TypeError('on_event must be a callable')
    # Import the project's implementation, not a private fork of its package API.
    from experience.packages import ExperienceCatalog, materialize

    path = Path(catalog_path) if catalog_path else Path(__file__).resolve().parent / 'catalog.json'
    catalog = ExperienceCatalog.load(path)
    selection = catalog.select(task)
    if 'ICL.md' not in selection.entry_points:
        raise ValueError('selected task package must declare ICL.md as an entry point')
    # Validate renderer settings before materialization or creating the log.
    _Renderer(max_chars=max_chars, max_images=max_images)
    episode_dir = Path(episode_dir)
    sink = _EventLog(episode_dir / 'icl-events.jsonl', on_event)
    workspace = materialize(catalog, task, episode_dir, on_access=sink)
    context = ICLContext(workspace, sink, task=task, skill=skill, top_k=top_k,
                         min_score=min_score, max_chars=max_chars, max_images=max_images)
    token = _active_icl.set(context)
    try:
        yield context
    finally:
        try:
            context.close()
        finally:
            _active_icl.reset(token)


def _bound():
    context = _active_icl.get()
    if context is None:
        raise RuntimeError('call get_context_memory within with open_icl(task, episode_dir)')
    context._guard()
    return context


def get_context_memory(obs):
    """Standard group-two interface, bound to the active native episode workspace."""
    return _bound().get_context_memory(obs)


def build_prompt(obs, system_prompt, history=()):
    """Build complete multimodal messages within the active open_icl context."""
    return _bound().build_prompt(obs, system_prompt=system_prompt, history=history)
