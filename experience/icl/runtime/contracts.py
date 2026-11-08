"""Describe the delivered DexJoCo data contracts without assuming a controller.

Conventions are documented in each package's attributed upstream task material,
not inferred from vector length. State and action have different layouts, most
notably in the dual-arm state where both poses precede both hand joint vectors.
No limits, control tools, timing, object motion or success labels are supplied by
these helpers. Measured changes describe robot state only.
"""
from __future__ import annotations

import math


SINGLE = 'dexjoco/single-xyz-rotvec-hand16/v1'
DUAL = 'dexjoco/dual-right-left-xyz-rotvec-hand16/v1'
_LAYOUTS = {SINGLE: (23, 22), DUAL: (46, 44)}
_FINGERS = ('index', 'middle', 'ring', 'thumb')


def _dimensions(contract):
    if not isinstance(contract, str) or contract not in _LAYOUTS:
        raise ValueError('unknown control contract: ' + str(contract))
    return _LAYOUTS[contract]


def _vector(values, dimension, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(name + ' must be a finite numeric vector')
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must be a finite numeric vector') from error
    if len(result) != dimension or not all(math.isfinite(value) for value in result):
        raise ValueError(f'{name} must contain {dimension} finite numbers')
    return result


def _number(value):
    if not math.isfinite(value):
        raise ValueError('computed contract value is not finite')
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0 else rounded


def _numbers(values):
    return [_number(value) for value in values]


def _hand(values):
    return {finger: _numbers(values[4 * i:4 * (i + 1)])
            for i, finger in enumerate(_FINGERS)}


def _unit_quaternion(values):
    norm = math.hypot(*values)
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError('state orientation must be a nonzero finite quaternion')
    return tuple(value / norm for value in values)


def _state_parts(state, contract):
    dimension, _ = _dimensions(contract)
    values = _vector(state, dimension, 'state')
    if contract == SINGLE:
        parts = {'arm': (values[:3], values[3:7], values[7:23])}
    else:
        parts = {'right': (values[:3], values[3:7], values[14:30]),
                 'left': (values[7:10], values[10:14], values[30:46])}
    for _position, quaternion, _joints in parts.values():
        _unit_quaternion(quaternion)
    return parts


def contract_description(action_contract) -> str:
    """Return self-contained layout and unit guidance for a supported contract."""
    _dimensions(action_contract)
    if action_contract == SINGLE:
        layout = (
            'Single arm. State has 23 values: [world xyz3, quaternion wxyz4, hand16]. '
            'Action has 22 values: [world xyz3, rotation-vector xyz3, hand16].'
        )
    else:
        layout = (
            'Dual arm. State has 46 values: [right world xyz3, right quaternion wxyz4, '
            'left world xyz3, left quaternion wxyz4, right hand16, left hand16]. '
            'Action has 44 values: [right world xyz3, right rotation-vector xyz3, '
            'right hand16, left world xyz3, left rotation-vector xyz3, left hand16].'
        )
    return (
        layout + '\n'
        'Positions use world coordinates in meters. Rotation vectors and hand joint angles use radians. '
        'Actions are absolute pose and finger-joint targets, not deltas, velocities or arm joint angles. '
        'Each hand16 is [index4, middle4, ring4, thumb4]; each finger quartet is '
        '[base spread, proximal flexion, middle flexion, distal flexion].\n'
        'State quaternions are wxyz, not action rotation vectors; q and -q describe the same orientation. '
        'Keep state and action layouts distinct. Use the caller\'s controlled reference point, action '
        'output schema, limits, tools and timing. This context provides no IK/calibration tool or '
        'additional action interface. Sample indices alone do not establish elapsed time.'
    )


def describe_state(state, contract) -> dict:
    """Name measured robot state fields; retain the supplied quaternion values."""
    return {arm: dict(position_world_m=_numbers(position),
                      orientation_quaternion_wxyz=_numbers(quaternion),
                      hand_joints_rad=_hand(joints))
            for arm, (position, quaternion, joints) in _state_parts(state, contract).items()}


def describe_action(action, contract) -> dict:
    """Name recorded absolute command fields; do not convert them into a policy."""
    _, dimension = _dimensions(contract)
    values = _vector(action, dimension, 'action')
    parts = {'arm': values} if contract == SINGLE else {'right': values[:22], 'left': values[22:]}
    return {arm: dict(target_position_world_m=_numbers(row[:3]),
                      target_rotation_vector_rad=_numbers(row[3:6]),
                      target_hand_joints_rad=_hand(row[6:22]))
            for arm, row in parts.items()}


def measured_change(before, after, contract) -> dict:
    """Summarize measured robot-state changes, without asserting action causality.

    The orientation magnitude is the shortest rotation between normalized
    quaternions, invariant to either quaternion's sign. It is not an angular
    velocity. Hand change is the largest absolute change of one recorded joint.
    """
    first, last = _state_parts(before, contract), _state_parts(after, contract)
    result = {}
    for arm, (start_position, start_quaternion, start_joints) in first.items():
        end_position, end_quaternion, end_joints = last[arm]
        delta = tuple(end - start for start, end in zip(start_position, end_position))
        start_unit, end_unit = _unit_quaternion(start_quaternion), _unit_quaternion(end_quaternion)
        cosine = min(1.0, max(0.0, abs(sum(a * b for a, b in zip(start_unit, end_unit)))))
        result[arm] = dict(
            position_delta_world_m=_numbers(delta),
            position_displacement_m=_number(math.hypot(*delta)),
            orientation_change_rad=_number(2 * math.acos(cosine)),
            max_hand_joint_change_rad=_number(max(abs(b - a) for a, b in zip(start_joints, end_joints))),
        )
    return result
