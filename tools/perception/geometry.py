"""Small calibrated geometry algorithms, independently implemented with NumPy.

Convention: pixel (u,v), rectified native RGB, optical +Z forward, T_world_camera,
right-handed world coordinates in metres. No robot-specific table height priors.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


class GeometryUnavailable(ValueError):
    """Insufficient or inconsistent measurements, not a numerical fallback guess."""


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    K: np.ndarray
    T_world_camera: np.ndarray
    time_s: float
    calibration_revision: str
    rectified_native: bool
    calibration_image_hw: tuple

    def __post_init__(self):
        rgb = np.array(self.rgb, copy=True)
        k = np.array(self.K, dtype=float, copy=True)
        t = np.array(self.T_world_camera, dtype=float, copy=True)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("expected native uint8 RGB HxWx3")
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or not np.isclose(k[1, 0], 0)):
            raise ValueError("invalid pinhole intrinsics")
        if (t.shape != (4, 4) or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6)):
            raise ValueError("expected rigid camera-to-world transform (metres)")
        if (self.rectified_native is not True or not self.calibration_revision
                or not math.isfinite(self.time_s) or self.time_s < 0
                or tuple(self.calibration_image_hw) != rgb.shape[:2]):
            raise ValueError("rectified native images, calibration identity and timestamp required")
        object.__setattr__(self, "calibration_image_hw", tuple(self.calibration_image_hw))
        for name, value in (("rgb", rgb), ("K", k), ("T_world_camera", t)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def pixel_ray(camera: CameraFrame, uv):
    uv = np.asarray(uv, dtype=float)
    h, w = camera.rgb.shape[:2]
    if uv.shape != (2,) or not np.isfinite(uv).all() or not (0 <= uv[0] < w and 0 <= uv[1] < h):
        raise GeometryUnavailable("pixel_outside_native_image")
    direction = camera.T_world_camera[:3, :3] @ np.linalg.solve(camera.K, [*uv, 1.0])
    direction /= np.linalg.norm(direction)
    return camera.T_world_camera[:3, 3].copy(), direction


def intersect_horizontal(origin, direction, z):
    if not math.isfinite(z) or abs(direction[2]) < 1e-8:
        raise GeometryUnavailable("ray_parallel_to_plane_or_invalid_height")
    distance = (z - origin[2]) / direction[2]
    if distance <= 0:
        raise GeometryUnavailable("intersection_behind_camera")
    return origin + distance * direction


def triangulate(first: CameraFrame, uv, second: CameraFrame, uv2, *,
                min_angle_deg=3.0, max_gap_m=0.005):
    a, d = pixel_ray(first, uv)
    b, e = pixel_ray(second, uv2)
    sine = float(np.linalg.norm(np.cross(d, e)))
    if sine < math.sin(math.radians(min_angle_deg)):
        raise GeometryUnavailable("insufficient_ray_angle")
    distances = np.linalg.lstsq(np.column_stack([d, -e]), b - a, rcond=None)[0]
    if np.any(distances <= 0):
        raise GeometryUnavailable("triangulated_point_behind_camera")
    p, q = a + distances[0] * d, b + distances[1] * e
    gap = float(np.linalg.norm(p - q))
    if gap > max_gap_m:
        raise GeometryUnavailable("rays_disagree")
    return dict(point_world_m=((p + q) / 2).tolist(), ray_gap_m=gap,
                ray_angle_deg=math.degrees(math.asin(min(sine, 1.0))))


def project(camera, xyz):
    xyz = np.asarray(xyz, dtype=float)
    camera_xyz = (xyz - camera.T_world_camera[:3, 3]) @ camera.T_world_camera[:3, :3]
    pixels = camera_xyz @ camera.K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = pixels[:, :2] / pixels[:, 2:3]
    return uv, camera_xyz[:, 2] > 1e-8


def _gray(rgb):
    return np.asarray(rgb, dtype=float) @ np.array([0.299, 0.587, 0.114])


def _bilinear(image, uv):
    h, w = image.shape
    valid = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
             & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
    safe = np.where(valid[:, None], uv, 0)
    x, y = safe[:, 0], safe[:, 1]
    ix, iy = x.astype(int), y.astype(int)
    dx, dy = x - ix, y - iy
    value = ((1 - dx) * (1 - dy) * image[iy, ix] + dx * (1 - dy) * image[iy, ix + 1]
             + (1 - dx) * dy * image[iy + 1, ix] + dx * dy * image[iy + 1, ix + 1])
    return value, valid, ix, iy


def plane_sweep(first, second, mask1, mask2, *, z_min_m, z_max_m, step_m,
                min_points=64, max_samples=2048, min_ncc=0.6, min_prominence=0.08):
    """Fit one horizontal, textured surface inside two explicit image regions.

NCC and peak prominence are diagnostics, not calibrated confidence probabilities.
At least 80% of reference samples must remain visible in the second region.
"""
    if (not all(math.isfinite(v) for v in (z_min_m, z_max_m, step_m))
            or step_m <= 0 or z_min_m >= z_max_m):
        raise ValueError("finite ordered height bounds and positive step required")
    count = int(math.floor((z_max_m - z_min_m) / step_m)) + 1
    if count < 5 or count > 256:
        raise ValueError("height sweep must have 5..256 candidates")
    masks = [np.asarray(mask1), np.asarray(mask2)]
    for cam, mask in zip((first, second), masks):
        if mask.dtype != bool or mask.shape != cam.rgb.shape[:2]:
            raise ValueError("surface masks must be native-image bool HW")
    yy, xx = np.nonzero(masks[0])
    if len(xx) < min_points:
        raise GeometryUnavailable("insufficient_surface_pixels")
    idx = np.linspace(0, len(xx) - 1, min(len(xx), max_samples), dtype=int)
    xx, yy = xx[idx], yy[idx]
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=1)
    rays = pixels @ np.linalg.inv(first.K).T @ first.T_world_camera[:3, :3].T
    origin = first.T_world_camera[:3, 3]
    ref, other = _gray(first.rgb)[yy, xx], _gray(second.rgb)
    if float(np.std(ref)) < 3:
        raise GeometryUnavailable("surface_has_insufficient_texture")

    def score(z):
        with np.errstate(divide='ignore', invalid='ignore'):
            length = (z - origin[2]) / rays[:, 2]
            xyz = origin + length[:, None] * rays
        uv, front = project(second, xyz)
        vals, valid, ix, iy = _bilinear(other, uv)
        valid &= front & (length > 0) & masks[1][iy, ix]
        n = int(valid.sum())
        if n < max(min_points, math.ceil(0.8 * len(ref))):
            return float('nan'), n
        a, b = ref[valid], vals[valid]
        if min(float(a.std()), float(b.std())) < 3:
            return float('nan'), n
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), n

    heights = z_min_m + np.arange(count) * step_m
    scores = np.array([score(z)[0] for z in heights])
    valid = np.isfinite(scores)
    if valid.sum() < 5:
        raise GeometryUnavailable("insufficient_multiview_overlap_or_texture")
    best = int(np.nanargmax(scores))
    if best in (0, count - 1):
        raise GeometryUnavailable("height_peak_at_search_boundary")
    prominence = float(scores[best] - np.nanmedian(scores))
    far = valid & (np.abs(heights - heights[best]) > 2.1 * step_m)
    if scores[best] < min_ncc or prominence < min_prominence:
        raise GeometryUnavailable("weak_plane_alignment")
    if far.any() and scores[best] - np.max(scores[far]) < 0.01:
        raise GeometryUnavailable("ambiguous_plane_alignment")
    fine = np.linspace(heights[best - 1], heights[best + 1], 17)
    fine_scores = np.array([score(z)[0] for z in fine])
    chosen = int(np.nanargmax(fine_scores))
    z = float(fine[chosen])
    ncc, n = score(z)
    if chosen in (0, len(fine) - 1):
        raise GeometryUnavailable("refined_peak_at_search_boundary")
    return dict(plane_z_m=z, normal_world=[0.0, 0.0, 1.0], ncc=ncc,
                peak_prominence=prominence, valid_samples=n,
                search_resolution_m=step_m / 8, horizontal_surface_assumed=True)


def color_regions(rgb, lower_rgb, upper_rgb, min_area=40):
    lower, upper = np.asarray(lower_rgb), np.asarray(upper_rgb)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower < 0) or np.any(upper > 255)
            or np.any(lower > upper) or type(min_area) is not int or min_area < 1):
        raise ValueError("invalid RGB thresholds or area")
    mask = np.all((rgb >= lower) & (rgb <= upper), axis=2)
    h, w = mask.shape
    visited, regions = np.zeros_like(mask), []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack, points = [(int(y), int(x))], []
        visited[y, x] = True
        while stack:
            a, b = stack.pop()
            points.append((a, b))
            for u, v in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                if 0 <= u < h and 0 <= v < w and mask[u, v] and not visited[u, v]:
                    visited[u, v] = True
                    stack.append((u, v))
        if len(points) >= min_area:
            points = np.asarray(points)
            ys, xs = points[:, 0], points[:, 1]
            regions.append(dict(bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                                centroid_px=[float(xs.mean()), float(ys.mean())], area_px=len(points),
                                touches_border=bool(xs.min()==0 or ys.min()==0 or xs.max()==w-1 or ys.max()==h-1)))
    return sorted(regions, key=lambda r: r['centroid_px'][0])
