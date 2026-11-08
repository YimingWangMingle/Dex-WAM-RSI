"""Ground a displayed interval in its sealed numerical training trajectory.

The later image is an observation at another recorded row, not proof that one
sampled command caused the change or that the episode reached its goal.
"""
from __future__ import annotations

from io import BytesIO
import zipfile

import numpy as np

from .memory_schema import MemoryRecord
from .task_knowledge import TaskKnowledge


def interval_evidence(record: MemoryRecord, knowledge: TaskKnowledge) -> dict:
    """Return verified timing and command samples between first/last observations.

    Uses the unique reference with the same source episode; filenames and snippet
    numbering are not used to infer identity. ``knowledge.read_reference`` must
    return original binary bytes and enforce its declared inventory/hash checks.
    """
    matches = [row for row in knowledge.references()
               if row.get('episode_id') == record.episode_id]
    if len(matches) != 1:
        raise ValueError('interval evidence requires exactly one reference for source episode: '
                         + record.episode_id)
    reference = matches[0]
    if (knowledge.task != record.task
            or knowledge.action_contract != record.action_contract
            or reference.get('action_contract') != record.action_contract):
        raise ValueError('interval reference task or action contract differs from the snippet')
    if not record.steps:
        raise ValueError('interval evidence requires at least one observation')
    state_dim = len(record.steps[0].state)
    action_dim = len(record.steps[0].action)
    indices = [step.step for step in record.steps]
    if (any(type(index) is not int or index < 0 for index in indices)
            or any(a >= b for a, b in zip(indices, indices[1:]))
            or any(len(step.state) != state_dim or len(step.action) != action_dim
                   for step in record.steps)):
        raise ValueError('snippet steps must increase and have consistent vector dimensions')

    data = knowledge.read_reference(reference['id'])
    if not isinstance(data, bytes) or len(data) > 16 * 1024 * 1024:
        raise ValueError('interval reference must be original binary bytes within 16 MiB')
    # A compressed file can be much larger when opened. Bound the declared
    # expansion before NumPy reads the arrays; ordinary task references are small.
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 128 * 1024 * 1024:
                raise ValueError('interval reference arrays exceed 128 MiB')
        with np.load(BytesIO(data), allow_pickle=False) as arrays:
            if not {'state', 'actions', 'timestamps'}.issubset(arrays.files):
                raise ValueError('interval reference requires state, actions and timestamps')
            state = arrays['state']
            actions = arrays['actions']
            timestamps = arrays['timestamps']
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError('interval reference is not a readable numerical NPZ') from error

    if any(array.dtype.kind not in 'fiu' for array in (state, actions, timestamps)):
        raise ValueError('interval reference arrays must contain real numerical values')
    if (state.ndim != 2 or actions.ndim != 2 or timestamps.ndim != 1
            or state.shape[1] != state_dim or actions.shape[1] != action_dim
            or len(state) != len(actions) or len(state) != len(timestamps)
            or len(state) < 1 or reference.get('steps') != len(state)
            or not all(np.isfinite(array).all() for array in (state, actions, timestamps))
            or not np.all(timestamps[1:] > timestamps[:-1])):
        raise ValueError('interval reference dimensions, values or timestamps are invalid')
    if indices[-1] >= len(state):
        raise ValueError('snippet step lies outside its source reference')
    for step in record.steps:
        if (not np.allclose(state[step.step], step.state, rtol=0, atol=1e-7)
                or not np.allclose(actions[step.step], step.action, rtol=0, atol=1e-7)):
            raise ValueError('snippet state/action differs from the source reference row')

    start, end = indices[0], indices[-1]
    elapsed = float(timestamps[end]) - float(timestamps[start])
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError('interval reference has an invalid elapsed duration')
    command_indices = sorted({start, end - 1}) if end > start else []
    samples = [dict(step=index, action=actions[index].tolist()) for index in command_indices]
    ranges = []
    arm_offsets = [('single', 0)] if action_dim == 22 else (
        [('right', 0), ('left', 22)] if action_dim == 44 else [])
    if end > start:
        for arm, offset in arm_offsets:
            targets = actions[start:end, offset:offset + 3]
            ranges.append(dict(arm=arm, min_m=targets.min(axis=0).tolist(),
                               max_m=targets.max(axis=0).tolist()))

    return dict(
        before_step=start,
        after_step=end,
        elapsed_seconds=elapsed,
        commands_count=end - start,
        command_samples=samples,
        omitted_command_count=end - start - len(samples),
        target_xyz_ranges=ranges,
        source_reference_id=reference['id'],
        source_reference_sha256=reference['sha256'],
        observation_timing='row i observation with command i; final displayed observation is before its own command',
        after_evidence='later_observation_not_terminal_success',
    )
