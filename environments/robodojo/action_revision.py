"""Validate explicit WAM review operations; revisions never execute implicitly."""
import copy
import math


def revised_plans(candidates, candidate_id, edits):
    if type(candidate_id) is not int:
        raise ValueError('edit requires an integer candidate ID')
    plans = [copy.deepcopy(c['actions']) for c in candidates]
    positions = [i for i, c in enumerate(candidates) if c['candidate_id'] == candidate_id]
    if len(positions) != 1 or not isinstance(edits, list) or not edits:
        raise ValueError('edit requires an existing candidate and nonempty edits')
    plan = plans[positions[0]]
    touched = set()
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'start', 'end', 'channel', 'value'}:
            raise ValueError('edit fields: start inclusive, end exclusive, channel, absolute value')
        start, end, channel, value = (edit[k] for k in ('start', 'end', 'channel', 'value'))
        if (any(type(x) is not int for x in (start, end, channel))
                or not 0 <= start < end <= len(plan) or not 0 <= channel < 14
                or type(value) not in (float, int) or not math.isfinite(value)):
            raise ValueError('invalid action edit range/channel/value')
        if channel in (6, 13) and not 0 <= value <= 1:
            raise ValueError('native gripper target must be in [0,1]')
        for row in range(start, end):
            key = (row, channel)
            if key in touched:
                raise ValueError('overlapping edits are ambiguous')
            touched.add(key)
            plan[row][channel] = float(value)
    if plans == [c['actions'] for c in candidates]:
        raise ValueError('revision must change the candidate')
    return plans


def validate_operation(choice):
    if not isinstance(choice, dict) or set(choice) != {'operation', 'candidate_id', 'execute_steps', 'edits', 'reason'}:
        raise ValueError('explicit operation requires operation/candidate_id/execute_steps/edits/reason')
    operation = choice['operation']
    if operation not in ('execute', 'resample', 'edit', 'stop') or not isinstance(choice['reason'], str) or not choice['reason'].strip():
        raise ValueError('valid operation and nonempty reason required')
    if not isinstance(choice['edits'], list):
        raise ValueError('edits must be a list')
    if operation != 'edit' and choice['edits']:
        raise ValueError('only edit can contain action modifications')
    if type(choice['execute_steps']) is not int:
        raise ValueError('execute_steps must be an integer')
    if operation != 'execute' and choice['execute_steps'] != 0:
        raise ValueError('revision/resample/stop cannot execute actions')
    if operation in ('stop', 'resample') and choice['candidate_id'] is not None:
        raise ValueError('stop/resample do not select a candidate')
    return operation
