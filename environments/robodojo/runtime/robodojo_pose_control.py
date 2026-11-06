"""Bounded pose search and stateful guards for the direct RoboDojo pilot."""
import math
import numpy as np


def quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('invalid wxyz quaternion')
    return q / np.linalg.norm(q)


def quaternion_angle(a, b):
    return 2 * math.acos(float(np.clip(abs(quaternion(a) @ quaternion(b)), 0, 1)))


def multiply(a, b):
    aw, av = a[0], np.asarray(a[1:]); bw, bv = b[0], np.asarray(b[1:])
    return quaternion(np.r_[aw*bw - av @ bv, aw*bv + bw*av + np.cross(av, bv)])


def orientation_candidates(reference, max_tilt_deg=0):
    """World X/Y tilts, original first; never a silent unbounded search."""
    if type(max_tilt_deg) not in (int, float) or not math.isfinite(max_tilt_deg) or not 0 <= max_tilt_deg <= 45:
        raise ValueError('max_tilt_deg must be finite in [0,45]')
    reference = quaternion(reference)
    yield reference
    for degrees in (15, 30, 45):
        if degrees > max_tilt_deg:
            continue
        for sign in (1, -1):
            for axis in (0, 1):
                h = math.radians(sign*degrees)/2
                delta = np.zeros(4); delta[0] = math.cos(h); delta[axis+1] = math.sin(h)
                yield multiply(delta, reference)


class RejectionLedger:
    """Two planner rejections block repeats until geometry/state actually changes."""
    def __init__(self):
        self.counts = {}

    def key(self, arm, position, orientation, state, scene_revision, max_tilt_deg):
        q = quaternion(orientation)
        # q and -q denote the same pose.
        if q[np.argmax(np.abs(q))] < 0:
            q = -q
        values = np.r_[position, q, state]
        if not np.isfinite(values).all():
            raise ValueError('nonfinite rejection key')
        return (arm, tuple(np.round(values, 4)), scene_revision, float(max_tilt_deg))

    def blocked(self, key):
        return self.counts.get(key, 0) >= 2

    def reject(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1


def tracking_error(measured, expected):
    measured, expected = np.asarray(measured, float), np.asarray(expected, float)
    if measured.shape != expected.shape or not np.isfinite(measured).all() or not np.isfinite(expected).all():
        return float('inf')
    # Actual actuator coordinates; do not wrap a bounded joint through 2*pi.
    return float(np.max(np.abs(measured-expected)))


def validate_transfer_points(points):
    points = np.asarray(points, float)
    if points.shape != (4, 3) or not np.isfinite(points).all():
        raise ValueError('provide pregrasp, grasp, lift, transfer as four XYZ points')
    if np.any(points[:, 2] < .72) or np.any(points[:, 2] > 1.4):
        raise ValueError('transfer points outside height bound')
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > .35):
        raise ValueError('transfer segment exceeds 0.35m')
    if points[2, 2] < points[1, 2]+.01:
        raise ValueError('lift must be at least 1cm above grasp')
    return points
