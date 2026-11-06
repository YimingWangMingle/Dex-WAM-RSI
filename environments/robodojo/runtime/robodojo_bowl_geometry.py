"""Conditional geometry of upright circular bowl lips from calibrated RGB only.

API
---
``RimCandidate(id, contour_px, complete=..., clipped=...)`` contains *observed*
Nx2 native rectified pixels on the outer white lip, not a completed fitted
ellipse. Points need not correspond between views. ``RimView(view_id, frame,
candidates)`` groups proposals. ``associate_bowl_rims(views, expected_bowls=None,
config=GeometryConfig())`` enumerates injective assignments for at most three
bowls. Without a count, all views must have the same number of candidates. Only
one globally feasible assignment is accepted; candidate IDs are view-local and
carry no identity information. Every accepted bowl has evidence in every view.
``fit_bowl_rim([RimObservation(view_id, frame, candidate), ...])`` checks a single
proposed association. These functions return dataclasses, never a fallback pose.
Unknown results have no geometry. Invalid configuration is a programmer error
(ValueError); unavailable measurements/optimization return status='unknown'.

World +Z is vertical, units are metres. The unknowns are (cx, cy, cz, radius).
There is no depth, GT object pose, nominal radius or tabletop input. Radius is
bounded to [0.02, 0.18]. ``z_bounds_m=None`` is unconstrained; optionally pass
WORKSPACE_Z_BOUNDS_M=(0.765, 1.2), an operational workspace bound, NOT a measured
table constraint. Solutions touching bounds are rejected. Static synchronized
views (default timestamp spread <=0.05 s) and correct calibration are required.

The solver uses scipy least_squares on signed pixel-normal (Sampson) distances
to each *projected 3-D circle conic*, jointly across cameras. No silhouette
extrema, contour samples or ellipse centers are treated as shared features.
Ellipse-center ray intersection is only a coarse initialization. Exact ellipse
distances verify every original point after fitting, with max error <=3 px.

Full contours are required by default. Opt-in allow_partial accepts >=240 deg
of measured coverage; gaps over max_sampling_gap_deg are missing evidence, not
filled arcs. Clipping is always rejected, including a projected rim crossing
the image border. Flags must describe the source observations honestly: this
module cannot distinguish an invented ellipse from real pixels.

Uncertainty is a LOCAL FIRST-ORDER, conditional perturbation bound, not a
statistical confidence interval or a calibration/grasp guarantee. For the
unweighted residual Jacobian J, each parameter bound is epsilon*||J^+ row||_1;
the center bound is the norm of its three coordinate bounds. This allows fully
correlated normal pixel errors (e.g. lip thickness), without a false sqrt(N)
gain. Epsilon is at least 1 px and the largest observed reprojection error.
Both center and radius bounds must be <=5 mm, with a well-conditioned rank-four
Jacobian. Calibration error, motion, noncircular/tilted lips and detector bias
larger than epsilon are NOT covered. Acceptance is only bounded geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from tools.perception.geometry import CameraFrame, pixel_ray

FloatArray = NDArray[np.float64]
WORKSPACE_Z_BOUNDS_M: tuple[float, float] = (0.765, 1.2)
_WARNINGS = (
    "Outer white lip radius can have radial thickness/segmentation bias (~1 px); "
    "it is not an inner opening or a grasp contact radius.",
    "Uncertainty is conditional on calibrated static views, an upright circular "
    "lip, bounded pixel-normal errors and a local linear approximation; "
    "it excludes calibration error and does not establish grasp success.",
)


@dataclass(frozen=True)
class GeometryConfig:
    z_bounds_m: tuple[float, float] | None = None
    allow_partial: bool = False
    min_partial_coverage_deg: float = 240.0
    min_full_coverage_deg: float = 340.0
    max_sampling_gap_deg: float = 20.0
    min_baseline_m: float = 0.05
    min_ray_angle_deg: float = 3.0
    max_reprojection_px: float = 3.0
    pixel_error_bound_px: float = 1.0
    max_uncertainty_m: float = 0.005
    max_jacobian_condition: float = 1e4
    max_time_spread_s: float = 0.05
    border_margin_px: float = 2.0
    min_points: int = 24
    max_fit_points_per_view: int = 128
    max_combinations: int = 729

    def __post_init__(self) -> None:
        positive = (self.min_baseline_m, self.min_ray_angle_deg,
                    self.max_reprojection_px, self.pixel_error_bound_px,
                    self.max_uncertainty_m, self.max_jacobian_condition,
                    self.max_sampling_gap_deg)
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError("finite positive geometry thresholds required")
        if not (240 <= self.min_partial_coverage_deg <= self.min_full_coverage_deg <= 360
                and 0 < self.max_sampling_gap_deg <= 20
                and self.max_reprojection_px <= 3
                and self.pixel_error_bound_px >= 1
                and self.max_uncertainty_m <= 0.005
                and self.min_ray_angle_deg < 90):
            raise ValueError("thresholds cannot relax coverage, 3px or 5mm safety limits")
        if (not math.isfinite(self.max_time_spread_s) or self.max_time_spread_s < 0
                or not math.isfinite(self.border_margin_px) or self.border_margin_px < 1):
            raise ValueError("invalid timestamp tolerance or border margin")
        for value in (self.min_points, self.max_fit_points_per_view, self.max_combinations):
            if type(value) is not int or value < 1:
                raise ValueError("positive integer sampling/search limits required")
        if not 24 <= self.min_points <= self.max_fit_points_per_view:
            raise ValueError("at least 24 fit points required")
        if self.z_bounds_m is not None:
            low, high = self.z_bounds_m
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("finite ordered workspace height bounds required")


@dataclass(frozen=True)
class RimCandidate:
    candidate_id: str
    contour_px: FloatArray
    complete: bool
    clipped: bool


@dataclass(frozen=True)
class RimView:
    view_id: str
    frame: CameraFrame
    candidates: Sequence[RimCandidate]


@dataclass(frozen=True)
class RimObservation:
    view_id: str
    frame: CameraFrame
    candidate: RimCandidate


@dataclass(frozen=True)
class ViewEvidence:
    view_id: str
    candidate_id: str
    calibration_revision: str
    time_s: float
    observed_points: int
    observed_coverage_deg: float
    complete: bool
    reprojection_rms_px: float
    reprojection_max_px: float


@dataclass(frozen=True)
class FitQuality:
    reprojection_rms_px: float
    reprojection_max_px: float
    max_baseline_m: float
    max_ray_angle_deg: float
    jacobian_condition: float
    warnings: tuple[str, ...] = _WARNINGS


@dataclass(frozen=True)
class ConditionalUncertainty:
    center_bound_m: float
    radius_bound_m: float
    coordinate_bounds_m: tuple[float, float, float]
    normal_error_bound_px: float
    method: str = "local_first_order_jacobian_Linf_to_coordinate_bounds"
    conditional: bool = True
    assumptions: tuple[str, ...] = _WARNINGS


@dataclass(frozen=True)
class BowlGeometryResult:
    status: Literal["accepted", "unknown"]
    reason: str
    center_xyz_m: tuple[float, float, float] | None = None
    radius_m: float | None = None
    quality: FitQuality | None = None
    uncertainty: ConditionalUncertainty | None = None
    view_evidence: tuple[ViewEvidence, ...] = ()
    warnings: tuple[str, ...] = _WARNINGS


@dataclass(frozen=True)
class BowlAssociationResult:
    status: Literal["accepted", "unknown"]
    reason: str
    bowls: tuple[BowlGeometryResult, ...] = ()
    assignments_evaluated: int = 0
    warnings: tuple[str, ...] = _WARNINGS


class _Unavailable(ValueError):
    pass


@dataclass(frozen=True)
class _Prepared:
    observation: RimObservation
    points: FloatArray
    fit_points: FloatArray
    ellipse_center: FloatArray
    coverage: float


def _ellipse(conic: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return center, semi-axes and orthonormal axis columns in pixel space."""
    quadratic = conic[:2, :2]
    center = -np.linalg.solve(quadratic, conic[:2, 2])
    level = float(center @ quadratic @ center - conic[2, 2])
    values, vectors = np.linalg.eigh(quadratic / level)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise _Unavailable("not_a_nondegenerate_ellipse")
    axes = 1 / np.sqrt(values)
    return center, axes, vectors


def _observed_ellipse(points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    # Normalize pixel coordinates before fitting a conic; this is only input QA
    # and initialization. The final model is the shared 3-D circle.
    origin = points.mean(axis=0)
    scale = float(np.sqrt(np.mean((points - origin) ** 2)))
    if scale < 2:
        raise _Unavailable("rim_too_small")
    x, y = ((points - origin) / scale).T
    design = np.column_stack((x*x, x*y, y*y, x, y, np.ones(len(x))))
    _, singular, vh = np.linalg.svd(design, full_matrices=False)
    if singular[-2] < 1e-8 * singular[0]:
        raise _Unavailable("degenerate_contour")
    a, b, c, d, e, f = vh[-1]
    conic = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])
    center, axes, vectors = _ellipse(conic)
    return origin + scale * center, scale * axes, vectors


def _distances(points: FloatArray, ellipse: tuple[FloatArray, FloatArray, FloatArray]) -> FloatArray:
    """Euclidean distance to an ellipse, using safeguarded quadrant iteration.

    For points outside, the Lagrange multiplier is positive; inside it is in
    (-minor_axis**2, 0). The minor-axis-zero special case also checks the
    stationary point on the major-axis interior and both ellipse vertices.
    """
    center, axes, vectors = ellipse
    xy = np.abs((points - center) @ vectors)
    # eigh above orders the axes from major to minor.
    a, b = axes
    x, y = xy.T
    outside = (x/a)**2 + (y/b)**2 >= 1
    low = np.where(outside, 0.0, -b*b * (1 - 1e-14))
    high = np.where(outside, np.hypot(a*x, b*y) + a*a, 0.0)
    for _ in range(64):
        mid = (low + high) / 2
        value = (a*x/(mid+a*a))**2 + (b*y/(mid+b*b))**2
        low = np.where(value > 1, mid, low)
        high = np.where(value > 1, high, mid)
    lam = (low + high) / 2
    qx, qy = a*a*x/(lam+a*a), b*b*y/(lam+b*b)
    distance = np.hypot(qx-x, qy-y)
    # At y=0 an interior stationary solution may lie at lambda=-b^2.
    special = (y < 1e-10) & ~outside
    if np.any(special):
        sx = np.clip(a*a*x / max(a*a-b*b, 1e-20), 0, a)
        sy = b * np.sqrt(np.maximum(0, 1-(sx/a)**2))
        candidates = np.minimum(np.hypot(sx-x, sy-y), np.hypot(a-x, y))
        distance = np.where(special, np.minimum(candidates, np.hypot(x, b-y)), distance)
    return distance


def _inside_image(ellipse: tuple[FloatArray, FloatArray, FloatArray],
                  frame: CameraFrame, margin: float) -> bool:
    center, axes, vectors = ellipse
    extent = np.sqrt((vectors**2) @ (axes**2))
    height, width = frame.rgb.shape[:2]
    return bool(np.all(center-extent >= margin)
                and np.all(center+extent <= np.array([width-1, height-1])-margin))


def _prepare(observation: RimObservation, cfg: GeometryConfig) -> _Prepared:
    candidate, frame = observation.candidate, observation.frame
    if not observation.view_id or not candidate.candidate_id:
        raise _Unavailable("missing_view_or_candidate_id")
    if candidate.clipped is not False:
        raise _Unavailable("clipped_or_unverified_contour")
    if candidate.complete is not True and not (candidate.complete is False and cfg.allow_partial):
        raise _Unavailable("partial_or_unverified_contour")
    points = np.array(candidate.contour_px, dtype=float, copy=True)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < cfg.min_points
            or not np.isfinite(points).all()):
        raise _Unavailable("invalid_or_insufficient_contour_points")
    if len(np.unique(points, axis=0)) < cfg.min_points:
        raise _Unavailable("insufficient_distinct_contour_points")
    height, width = frame.rgb.shape[:2]
    if (np.any(points < cfg.border_margin_px)
            or np.any(points > np.array([width-1, height-1])-cfg.border_margin_px)):
        raise _Unavailable("contour_touches_image_border")
    ellipse = _observed_ellipse(points)
    center, axes, vectors = ellipse
    if min(axes) < 3 or max(axes)/min(axes) > 100:
        raise _Unavailable("edge_on_or_unresolved_ellipse")
    if not _inside_image(ellipse, frame, cfg.border_margin_px):
        raise _Unavailable("ellipse_clipped_by_image")
    if float(_distances(points, ellipse).max()) > cfg.max_reprojection_px:
        raise _Unavailable("observed_contour_not_elliptical")
    xy = ((points-center) @ vectors) / axes
    angle = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2*np.pi)
    order = np.argsort(angle)
    gaps = np.rad2deg(np.diff(np.r_[angle[order], angle[order[0]] + 2*np.pi]))
    coverage = float(gaps[gaps <= cfg.max_sampling_gap_deg].sum())
    minimum = cfg.min_full_coverage_deg if candidate.complete else cfg.min_partial_coverage_deg
    if coverage + 1e-7 < minimum:
        raise _Unavailable("insufficient_observed_angular_coverage")
    # Subsample only real observations; no points in missing arcs are synthesized.
    indices = np.linspace(0, len(order)-1, min(len(order), cfg.max_fit_points_per_view), dtype=int)
    return _Prepared(observation, points, points[order[indices]], center, coverage)


def _projected_conic(frame: CameraFrame, parameters: FloatArray) -> FloatArray:
    rotation = frame.T_world_camera[:3, :3].T
    center = rotation @ (parameters[:3] - frame.T_world_camera[:3, 3])
    homography = frame.K @ np.column_stack((rotation[:, 0]*parameters[3],
                                            rotation[:, 1]*parameters[3], center))
    inverse = np.linalg.inv(homography)
    conic = inverse.T @ np.diag([1.0, 1.0, -1.0]) @ inverse
    return conic / np.linalg.norm(conic)


def _residual(parameters: FloatArray, prepared: Sequence[_Prepared]) -> FloatArray:
    pieces: list[FloatArray] = []
    for item in prepared:
        conic = _projected_conic(item.observation.frame, parameters)
        points = np.column_stack((item.fit_points, np.ones(len(item.fit_points))))
        normals = points @ conic
        denominator = 2*np.linalg.norm(normals[:, :2], axis=1)
        if np.any(denominator < 1e-15):
            raise _Unavailable("degenerate_projected_conic")
        pieces.append(np.sum(normals*points, axis=1) / denominator)
    return np.concatenate(pieces)


def _initializations(prepared: Sequence[_Prepared], cfg: GeometryConfig) -> list[FloatArray]:
    matrices, rhs = [], []
    for item in prepared:
        origin, direction = pixel_ray(item.observation.frame, item.ellipse_center)
        perpendicular = np.eye(3) - np.outer(direction, direction)
        matrices.append(perpendicular)
        rhs.append(perpendicular @ origin)
    # Apparent ellipse centers are NOT corresponding physical points. This
    # approximate ray intersection is never output or used as a residual.
    center = np.linalg.lstsq(np.vstack(matrices), np.concatenate(rhs), rcond=None)[0]
    heights = center[2] + np.array([0.0, -0.15, 0.15, -0.35, 0.35])
    if cfg.z_bounds_m is not None:
        low, high = cfg.z_bounds_m
        heights = np.r_[np.clip(heights, low+1e-5, high-1e-5), (low+high)/2]
    seeds: list[FloatArray] = []
    for z in heights:
        estimates = []
        for item in prepared:
            frame = item.observation.frame
            rays = (np.column_stack((item.fit_points, np.ones(len(item.fit_points))))
                    @ np.linalg.inv(frame.K).T @ frame.T_world_camera[:3, :3].T)
            origin = frame.T_world_camera[:3, 3]
            if np.any(np.abs(rays[:, 2]) < 1e-8):
                continue
            distances = (z-origin[2])/rays[:, 2]
            if np.any(distances <= 0):
                continue
            xy = (origin + distances[:, None]*rays)[:, :2]
            offset = xy.mean(axis=0)
            local = xy-offset
            fitted = np.linalg.lstsq(np.column_stack((2*local, np.ones(len(local)))),
                                     np.sum(local**2, axis=1), rcond=None)[0]
            radius2 = fitted[2] + np.sum(fitted[:2]**2)
            if radius2 > 0:
                estimates.append([*(offset+fitted[:2]), z, math.sqrt(radius2)])
        if estimates:
            seed = np.mean(estimates, axis=0)
            seed[3] = np.clip(seed[3], 0.02001, 0.17999)
            seeds.append(seed)
    if not seeds:
        raise _Unavailable("no_front_facing_initialization")
    return seeds


def _baseline(frames: Sequence[CameraFrame], center: FloatArray | None = None) -> tuple[float, float]:
    baseline, angle = 0.0, 0.0
    for first, second in combinations(frames, 2):
        a, b = first.T_world_camera[:3, 3], second.T_world_camera[:3, 3]
        baseline = max(baseline, float(np.linalg.norm(a-b)))
        if center is not None:
            da, db = center-a, center-b
            sine = np.linalg.norm(np.cross(da, db))/(np.linalg.norm(da)*np.linalg.norm(db))
            angle = max(angle, math.degrees(math.asin(float(np.clip(sine, 0, 1)))))
    return baseline, angle


def _check_views(observations: Sequence[RimObservation], cfg: GeometryConfig) -> None:
    if len(observations) < 2:
        raise _Unavailable("insufficient_views")
    if len({item.view_id for item in observations}) != len(observations):
        raise _Unavailable("duplicate_view_id")
    times = [item.frame.time_s for item in observations]
    if max(times)-min(times) > cfg.max_time_spread_s:
        raise _Unavailable("unsynchronized_views")
    baseline, _ = _baseline([item.frame for item in observations])
    if baseline < cfg.min_baseline_m:
        raise _Unavailable("insufficient_baseline")


def _evaluate(parameters: FloatArray, prepared: Sequence[_Prepared], cfg: GeometryConfig) -> BowlGeometryResult:
    if not np.isfinite(parameters).all():
        raise _Unavailable("nonfinite_solution")
    if min(parameters[3]-0.02, 0.18-parameters[3]) < 1e-5:
        raise _Unavailable("radius_at_workspace_bound")
    if cfg.z_bounds_m is not None and min(parameters[2]-cfg.z_bounds_m[0], cfg.z_bounds_m[1]-parameters[2]) < 1e-5:
        raise _Unavailable("height_at_workspace_bound")
    frames = [item.observation.frame for item in prepared]
    baseline, angle = _baseline(frames, parameters[:3])
    if angle < cfg.min_ray_angle_deg:
        raise _Unavailable("insufficient_ray_angle")
    evidence: list[ViewEvidence] = []
    all_errors = []
    for item in prepared:
        frame = item.observation.frame
        camera_center = (parameters[:3]-frame.T_world_camera[:3, 3]) @ frame.T_world_camera[:3, :3]
        z_extent = parameters[3]*np.linalg.norm(frame.T_world_camera[:2, 2])
        if camera_center[2]-z_extent <= 1e-6:
            raise _Unavailable("rim_not_in_front_of_camera")
        ellipse = _ellipse(_projected_conic(frame, parameters))
        if not _inside_image(ellipse, frame, cfg.border_margin_px):
            raise _Unavailable("projected_rim_clipped")
        errors = _distances(item.points, ellipse)
        if not np.isfinite(errors).all() or float(errors.max()) > cfg.max_reprojection_px:
            raise _Unavailable("reprojection_exceeds_3px_or_configured_limit")
        all_errors.append(errors)
        evidence.append(ViewEvidence(item.observation.view_id, item.observation.candidate.candidate_id,
                                     frame.calibration_revision, frame.time_s, len(item.points),
                                     item.coverage, item.observation.candidate.complete,
                                     float(np.sqrt(np.mean(errors**2))), float(errors.max())))
    errors = np.concatenate(all_errors)
    # Explicit unweighted central differences in metres; no active constraints
    # or robust-loss weights can make the uncertainty look artificially small.
    step = 1e-6
    jacobian = np.column_stack([(_residual(parameters+np.eye(4)[i]*step, prepared)
                                - _residual(parameters-np.eye(4)[i]*step, prepared))/(2*step)
                               for i in range(4)])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-12 else math.inf
    if not math.isfinite(condition) or condition > cfg.max_jacobian_condition:
        raise _Unavailable("ill_conditioned_geometry")
    epsilon = max(cfg.pixel_error_bound_px, float(errors.max()))
    bounds = epsilon * np.sum(np.abs(np.linalg.pinv(jacobian)), axis=1)
    center_bound = float(np.linalg.norm(bounds[:3]))
    if not np.isfinite(bounds).all() or max(center_bound, bounds[3]) > cfg.max_uncertainty_m:
        raise _Unavailable("conditional_uncertainty_exceeds_5mm_or_configured_limit")
    uncertainty = ConditionalUncertainty(center_bound, float(bounds[3]),
                                         tuple(float(x) for x in bounds[:3]), epsilon)
    quality = FitQuality(float(np.sqrt(np.mean(errors**2))), float(errors.max()), baseline, angle, condition)
    return BowlGeometryResult("accepted", "conditionally_resolved_rim",
                              tuple(float(x) for x in parameters[:3]), float(parameters[3]),
                              quality, uncertainty, tuple(evidence))


def _fit(prepared: Sequence[_Prepared], cfg: GeometryConfig) -> BowlGeometryResult:
    low = [-np.inf, -np.inf, -np.inf, 0.02]
    high = [np.inf, np.inf, np.inf, 0.18]
    if cfg.z_bounds_m is not None:
        low[2], high[2] = cfg.z_bounds_m
    solutions: list[tuple[float, FloatArray]] = []
    for seed in _initializations(prepared, cfg):
        try:
            fit = least_squares(_residual, seed, args=(prepared,), bounds=(low, high),
                                x_scale="jac", max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-9)
            if fit.success and np.isfinite(fit.fun).all():
                solutions.append((float(np.mean(fit.fun**2)), fit.x))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if not solutions:
        raise _Unavailable("optimization_unavailable")
    solutions.sort(key=lambda item: item[0])
    # A failed best fit is not rescued by selecting a poorer local minimum.
    accepted = _evaluate(solutions[0][1], prepared, cfg)
    for _, parameters in solutions[1:]:
        try:
            result = _evaluate(parameters, prepared, cfg)
        except _Unavailable:
            continue
        if (np.linalg.norm(np.array(result.center_xyz_m)-accepted.center_xyz_m) > 0.005
                or abs(result.radius_m-accepted.radius_m) > 0.005):
            raise _Unavailable("multiple_geometric_solutions")
    return accepted


def fit_bowl_rim(observations: Sequence[RimObservation], *,
                 config: GeometryConfig = GeometryConfig()) -> BowlGeometryResult:
    """Fit one proposed association; uniqueness against other instances needs associate_bowl_rims."""
    try:
        _check_views(observations, config)
        return _fit([_prepare(item, config) for item in observations], config)
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return BowlGeometryResult("unknown", str(error) or type(error).__name__)


def associate_bowl_rims(views: Sequence[RimView], *, expected_bowls: int | None = None,
                        config: GeometryConfig = GeometryConfig()) -> BowlAssociationResult:
    """Choose the only feasible disjoint assignment; two feasible assignments mean unknown.

    assignments_evaluated counts candidate tuples fitted, not permutations. Search
    is exhaustive within max_combinations; exceeding the cap never yields a pose.
    Explicit expected_bowls permits unused detector candidates (maximum 3/view).
    """
    evaluated = 0
    try:
        if len(views) < 2:
            raise _Unavailable("insufficient_views")
        counts = [len(view.candidates) for view in views]
        if any(count < 1 or count > 3 for count in counts):
            raise _Unavailable("one_to_three_candidates_per_view_required")
        if expected_bowls is None and len(set(counts)) != 1:
            raise _Unavailable("candidate_counts_disagree_specify_expected_bowls")
        count = counts[0] if expected_bowls is None else expected_bowls
        if type(count) is not int or not 1 <= count <= 3 or min(counts) < count:
            raise _Unavailable("invalid_or_unavailable_bowl_counts")
        if math.prod(counts) > config.max_combinations:
            raise _Unavailable("association_search_limit")
        _check_views([RimObservation(v.view_id, v.frame, v.candidates[0]) for v in views], config)
        prepared: list[list[_Prepared | None]] = []
        for view in views:
            if len({c.candidate_id for c in view.candidates}) != len(view.candidates):
                raise _Unavailable("duplicate_candidate_id_in_view")
            items: list[_Prepared | None] = []
            for candidate in view.candidates:
                try:
                    items.append(_prepare(RimObservation(view.view_id, view.frame, candidate), config))
                except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
                    items.append(None)
            prepared.append(items)
        feasible: list[tuple[tuple[int, ...], BowlGeometryResult]] = []
        for indices in product(*(range(n) for n in counts)):
            items = [prepared[v][index] for v, index in enumerate(indices)]
            if any(item is None for item in items):
                continue
            evaluated += 1
            try:
                result = _fit([item for item in items if item is not None], config)
                feasible.append((indices, result))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
        # Enumerate exact disjoint covers by increasing first-view index. Stop
        # at two: ambiguity cannot be resolved by arbitrarily sorting costs/IDs.
        assignments: list[tuple[BowlGeometryResult, ...]] = []

        def search(start: int, chosen: list[BowlGeometryResult], used: list[set[int]]) -> None:
            if len(assignments) >= 2:
                return
            if len(chosen) == count:
                assignments.append(tuple(chosen))
                return
            for position in range(start, len(feasible)):
                indices, result = feasible[position]
                if any(index in used[v] for v, index in enumerate(indices)):
                    continue
                search(position+1, [*chosen, result],
                       [used[v] | {index} for v, index in enumerate(indices)])

        search(0, [], [set() for _ in views])
        if not assignments:
            raise _Unavailable("no_feasible_multiview_assignment")
        if len(assignments) > 1:
            raise _Unavailable("ambiguous_multiview_assignment")
        return BowlAssociationResult("accepted", "unique_geometric_assignment", assignments[0], evaluated)
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return BowlAssociationResult("unknown", str(error) or type(error).__name__, assignments_evaluated=evaluated)
