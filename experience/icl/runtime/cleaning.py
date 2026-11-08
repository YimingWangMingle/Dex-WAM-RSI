"""Normalize supplied historical ICL records without changing the inputs."""
from copy import deepcopy
from .memory_schema import MemoryRecord, MemoryStep


def clean_histories(records, *, max_steps=16):
    """Normalize supplied ICL records; return accepted records and a cleaning report.

    Reject invalid records, deduplicate identical steps, sort by step ID, and
    sample long trajectories at evenly spaced positions including both ends.
    The input objects are never modified. Image bytes are checked by the builder.
    """
    if not isinstance(records, (list, tuple)) or type(max_steps) is not int or not 2 <= max_steps <= 16:
        raise ValueError('records must be a list; max_steps must be between 2 and 16')
    accepted, seen = [], set()
    report = dict(accepted=[], rejected=[], changes=[])
    for index, original in enumerate(records):
        record_id = original.get('record_id', '') if isinstance(original, dict) else ''
        try:
            if not isinstance(original, dict):
                raise ValueError('record must be an object')
            row = deepcopy(original)
            for key in ('record_id', 'task', 'episode_id', 'scene_id', 'source_type',
                        'source_split', 'action_contract', 'instruction', 'outcome', 'outcome_evidence', 'note'):
                if isinstance(row.get(key), str):
                    row[key] = row[key].strip()
            row.setdefault('channel', 'icl')
            row.setdefault('outcome', 'unknown')
            row.setdefault('outcome_evidence', '')
            if row['channel'] != 'icl':
                raise ValueError('only ICL examples are accepted')
            changes = []
            if row['outcome'] in ('success', 'failure') and not row['outcome_evidence']:
                row['outcome'] = 'unknown'
                changes.append('outcome changed to unknown: missing evidence')
            if not isinstance(row.get('steps'), (list, tuple)) or not row['steps']:
                raise ValueError('steps must be a nonempty list')
            unique = {}
            order = []
            for raw_step in row['steps']:
                step = MemoryStep.parse(raw_step)
                if step.step in unique:
                    if step != unique[step.step]:
                        raise ValueError('conflicting duplicate step ID: ' + str(step.step))
                    changes.append('removed identical duplicate step: ' + str(step.step))
                else:
                    unique[step.step] = step
                    order.append(step.step)
            steps = [unique[key] for key in sorted(unique)]
            if order != sorted(order):
                changes.append('sorted steps by step ID')
            if len({len(s.state) for s in steps}) != 1 or len({len(s.action) for s in steps}) != 1:
                raise ValueError('state/action dimensions vary within the trajectory')
            if len(steps) > max_steps:
                count = len(steps)
                steps = [steps[i * (count - 1) // (max_steps - 1)] for i in range(max_steps)]
                changes.append(f'sampled {max_steps} of {count} steps; omitted steps are not implied actions')
            row['steps'] = [dict(step=s.step, state=s.state, action=s.action, images=dict(s.images)) for s in steps]
            record = MemoryRecord.parse(row)
            if record.record_id in seen:
                raise ValueError('duplicate record_id')
            seen.add(record.record_id)
            accepted.append(record.to_dict())
            report['accepted'].append(record.record_id)
            if changes:
                report['changes'].append(dict(record_id=record.record_id, changes=changes))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            report['rejected'].append(dict(index=index, record_id=record_id, reason=str(error)))
    return accepted, report
