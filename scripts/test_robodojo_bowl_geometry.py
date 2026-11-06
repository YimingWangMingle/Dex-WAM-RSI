"""Local synthetic checks; no simulator, network, detector, depth or GT input.

Run from any working directory with a Python containing NumPy and SciPy:
    python scripts/test_robodojo_bowl_geometry.py

Known synthetic centers/radii are used solely to generate pixels and assert
accuracy. The public fitting API receives only CameraFrame and observed pixels.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.perception.geometry import CameraFrame, project
from environments.robodojo.runtime.robodojo_bowl_geometry import (
    GeometryConfig, RimCandidate, RimObservation, RimView, WORKSPACE_Z_BOUNDS_M,
    associate_bowl_rims, fit_bowl_rim,
)

FloatArray = NDArray[np.float64]


def camera(origin: tuple[float, float, float], *, focal: float = 600,
           target: tuple[float, float, float] = (0, 0, 0.92), time: float = 1.0) -> CameraFrame:
    position = np.array(origin, dtype=float)
    forward = np.asarray(target) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, down, forward))
    transform[:3, 3] = position
    return CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8),
                       np.array([[focal, 0, 320], [0, focal*1.03, 240], [0, 0, 1]]),
                       transform, time, "synthetic-pinhole-v1", True, (480, 640))


def rim(frame: CameraFrame, center: tuple[float, float, float] = (0.025, -0.015, 0.94),
        radius: float = 0.068, *, identity: str = "rim", noise: float = 0,
        seed: int = 0, count: int = 120, arc_deg: float = 360,
        phase: float = 0, complete: bool = True) -> RimCandidate:
    angle = np.linspace(phase, phase+np.deg2rad(arc_deg), count, endpoint=False)
    xyz = np.asarray(center) + radius*np.column_stack((np.cos(angle), np.sin(angle), np.zeros(count)))
    uv, front = project(frame, xyz)
    if not front.all():
        raise AssertionError("invalid synthetic setup")
    uv += np.random.default_rng(seed).normal(0, noise, uv.shape)
    return RimCandidate(identity, uv, complete=complete, clipped=False)


class BowlGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = (camera((-0.38, -0.38, 1.50)), camera((0.38, -0.30, 1.65)))

    def observations(self, **kwargs: object) -> list[RimObservation]:
        return [RimObservation(f"v{i}", frame, rim(frame, seed=i, **kwargs))
                for i, frame in enumerate(self.frames)]

    def assert_unknown(self, result: object, reason: str | None = None) -> None:
        self.assertEqual(result.status, "unknown", result)
        if hasattr(result, "center_xyz_m"):
            self.assertIsNone(result.center_xyz_m)
            self.assertIsNone(result.radius_m)
        else:
            self.assertEqual(result.bowls, ())
        if reason is not None:
            self.assertIn(reason, result.reason)

    def test_exact_circle_recovers_free_center_and_radius(self) -> None:
        result = fit_bowl_rim(self.observations())
        self.assertEqual(result.status, "accepted", result)
        np.testing.assert_allclose(result.center_xyz_m, [0.025, -0.015, 0.94], atol=1e-7)
        self.assertAlmostEqual(result.radius_m, 0.068, places=7)
        self.assertLess(result.quality.reprojection_max_px, 1e-6)
        self.assertLessEqual(result.uncertainty.center_bound_m, 0.005)
        self.assertLessEqual(result.uncertainty.radius_bound_m, 0.005)
        self.assertTrue(result.uncertainty.conditional)
        self.assertEqual(len(result.view_evidence), 2)
        self.assertIn("thickness", " ".join(result.quality.warnings))

    def test_unequal_perspective_no_sample_correspondence(self) -> None:
        frames = [camera((-0.25, -0.32, 1.26), focal=420),
                  camera((0.65, 0.10, 2.05), focal=850),
                  camera((-0.55, 0.42, 1.63), focal=550)]
        observations = [RimObservation(str(i), frame, rim(frame, radius=0.11,
                         phase=i*1.13, count=90+i*47)) for i, frame in enumerate(frames)]
        # Destroy all ordering/correspondence while retaining observed pixels.
        for item in observations:
            np.random.default_rng(11).shuffle(item.candidate.contour_px)
        result = fit_bowl_rim(observations)
        self.assertEqual(result.status, "accepted", result)
        np.testing.assert_allclose(result.center_xyz_m, [0.025, -0.015, 0.94], atol=2e-7)
        self.assertAlmostEqual(result.radius_m, 0.11, places=7)

    def test_subpixel_noise(self) -> None:
        result = fit_bowl_rim(self.observations(noise=0.25))
        self.assertEqual(result.status, "accepted", result)
        self.assertLess(np.linalg.norm(np.array(result.center_xyz_m)-[0.025, -0.015, 0.94]), 0.001)
        self.assertLess(abs(result.radius_m-0.068), 0.001)
        self.assertLess(result.quality.reprojection_max_px, 3)

    def test_correlated_outer_lip_bias_is_reported_and_bounded(self) -> None:
        observations = self.observations()
        for item in observations:
            points = item.candidate.contour_px
            # A coherent outward offset rather than independent zero-mean noise.
            tangent = np.roll(points, -1, axis=0)-np.roll(points, 1, axis=0)
            normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            sign = np.sign(np.sum(normal*(points-points.mean(axis=0))))
            points += 0.8*sign*normal
        result = fit_bowl_rim(observations)
        self.assertEqual(result.status, "accepted", result)
        self.assertLess(abs(result.radius_m-0.068), result.uncertainty.radius_bound_m)
        self.assertLess(np.linalg.norm(np.array(result.center_xyz_m)-[0.025, -0.015, 0.94]),
                        result.uncertainty.center_bound_m)
        self.assertGreaterEqual(result.uncertainty.normal_error_bound_px, 1)

    def test_wrong_instance_pair_is_rejected(self) -> None:
        observations = [RimObservation("left", self.frames[0], rim(self.frames[0], center=(-0.10, 0, 0.88))),
                        RimObservation("right", self.frames[1], rim(self.frames[1], center=(0.10, 0.06, 1.05), radius=0.095))]
        self.assert_unknown(fit_bowl_rim(observations))

    def test_unique_three_bowl_assignment_ignores_candidate_order_and_ids(self) -> None:
        centers = [(-0.12, -0.035, 0.86), (0.11, -0.025, 0.96), (0.015, 0.14, 1.06)]
        radii = [0.052, 0.075, 0.062]
        frames = [*self.frames, camera((0.05, 0.48, 1.7))]
        orders = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
        views = [RimView(f"v{i}", frame,
                 [rim(frame, center=centers[k], radius=radii[k], identity=f"local-{j}", phase=i*.7)
                  for j, k in enumerate(orders[i])]) for i, frame in enumerate(frames)]
        result = associate_bowl_rims(views, expected_bowls=3)
        self.assertEqual(result.status, "accepted", result)
        self.assertEqual(result.assignments_evaluated, 27)
        self.assertEqual(len(result.bowls), 3)
        for bowl, center, radius in zip(result.bowls, centers, radii):
            np.testing.assert_allclose(bowl.center_xyz_m, center, atol=1e-6)
            self.assertAlmostEqual(bowl.radius_m, radius, places=6)
        self.assertEqual([e.candidate_id for e in result.bowls[0].view_evidence],
                         ["local-0", "local-1", "local-2"])

    def test_duplicate_proposals_are_association_ambiguity(self) -> None:
        views = [RimView(str(i), frame, [rim(frame, identity="a"), rim(frame, identity="b")])
                 for i, frame in enumerate(self.frames)]
        self.assert_unknown(associate_bowl_rims(views, expected_bowls=1), "ambiguous")
        self.assert_unknown(associate_bowl_rims(views, expected_bowls=2), "ambiguous")

    def test_three_head_candidates_one_wrist_candidate(self) -> None:
        centers = [(-0.12, -0.035, 0.86), (0.11, -0.025, 0.96), (0.015, 0.14, 1.06)]
        head = RimView("head", self.frames[0], [rim(self.frames[0], center=center,
                       identity=f"head-{i}") for i, center in enumerate(centers)])
        wrist = RimView("wrist", self.frames[1], [rim(self.frames[1], center=centers[2], identity="wrist-0")])
        result = associate_bowl_rims([head, wrist], expected_bowls=1)
        self.assertEqual(result.status, "accepted", result)
        self.assertEqual(result.assignments_evaluated, 3)
        self.assertEqual(result.bowls[0].view_evidence[0].candidate_id, "head-2")
        self.assert_unknown(associate_bowl_rims([head, wrist], expected_bowls=3), "count")

    def test_partial_points_inside_image_but_full_ellipse_clipped(self) -> None:
        observations = self.observations(arc_deg=280, complete=False)
        for i, item in enumerate(observations):
            points = item.candidate.contour_px.copy()
            points[:, 0] += 640-points[:, 0].mean()-30
            points = points[points[:, 0] < 637]
            observations[i] = replace(item, candidate=replace(item.candidate, contour_px=points))
        self.assert_unknown(fit_bowl_rim(observations, config=GeometryConfig(allow_partial=True)), "clipped")

    def test_nearby_distinct_instances_are_not_resolved_by_lowest_cost(self) -> None:
        # Differences below pixel error allowance cannot prove instance identity.
        views = [RimView(str(i), frame, [rim(frame, identity="a"),
                  rim(frame, center=(0.026, -0.015, 0.94), identity="b")])
                 for i, frame in enumerate(self.frames)]
        self.assert_unknown(associate_bowl_rims(views), "ambiguous")

    def test_insufficient_views_and_baseline(self) -> None:
        self.assert_unknown(fit_bowl_rim(self.observations()[:1]), "insufficient_views")
        frame = self.frames[0]
        observations = [RimObservation(str(i), frame, rim(frame)) for i in range(2)]
        self.assert_unknown(fit_bowl_rim(observations), "baseline")
        self.frames = (camera((-0.38, -0.38, 1.5)), camera((-0.379, -0.38, 1.5)))
        self.assert_unknown(fit_bowl_rim(self.observations()), "baseline")

    def test_large_translation_but_insufficient_parallax(self) -> None:
        center = (0.025, -0.015, 0.94)
        origin = np.array(center)+np.array([0, -0.3, 0.65])
        self.frames = (camera(tuple(origin), target=center),
                       camera(tuple(np.array(center)+1.4*(origin-center)), target=center))
        self.assert_unknown(fit_bowl_rim(self.observations()), "ray_angle")

    def test_clipped_flags_and_actual_image_clipping(self) -> None:
        observations = self.observations()
        observations[0] = replace(observations[0], candidate=replace(observations[0].candidate, clipped=True))
        self.assert_unknown(fit_bowl_rim(observations), "clipped")
        observations = self.observations()
        observations[0].candidate.contour_px[0, 0] = 0
        self.assert_unknown(fit_bowl_rim(observations), "border")

    def test_partial_default_rejects_opt_in_accepts_observed_270_degrees(self) -> None:
        observations = self.observations(arc_deg=280, complete=False, count=160)
        self.assert_unknown(fit_bowl_rim(observations), "partial")
        result = fit_bowl_rim(observations, config=GeometryConfig(allow_partial=True))
        self.assertEqual(result.status, "accepted", result)
        np.testing.assert_allclose(result.center_xyz_m, [0.025, -0.015, 0.94], atol=2e-7)
        for evidence in result.view_evidence:
            self.assertGreaterEqual(evidence.observed_coverage_deg, 240)
            self.assertLess(evidence.observed_coverage_deg, 330)
            self.assertEqual(evidence.observed_points, 160)
            self.assertFalse(evidence.complete)

    def test_claimed_full_but_missing_large_arc_is_rejected(self) -> None:
        self.assert_unknown(fit_bowl_rim(self.observations(arc_deg=270)), "coverage")

    def test_less_than_240_degree_arc_is_rejected_even_opt_in(self) -> None:
        result = fit_bowl_rim(self.observations(arc_deg=210, complete=False),
                             config=GeometryConfig(allow_partial=True))
        self.assert_unknown(result, "coverage")

    def test_disjoint_short_arcs_do_not_falsely_count_as_coverage(self) -> None:
        observations = self.observations(count=360, complete=False)
        for i, item in enumerate(observations):
            points = item.candidate.contour_px
            observed = np.vstack([points[0:30], points[120:150], points[240:270]])
            observations[i] = replace(item, candidate=replace(item.candidate, contour_px=observed))
        self.assert_unknown(fit_bowl_rim(observations, config=GeometryConfig(allow_partial=True)), "coverage")

    def test_reprojection_over_three_pixels_rejected(self) -> None:
        observations = self.observations()
        # Both inputs remain perfect ellipses, but no shared horizontal circle fits.
        observations[1].candidate.contour_px[:, 0] += 20
        self.assert_unknown(fit_bowl_rim(observations), "reprojection")

    def test_uncertainty_and_condition_rejections(self) -> None:
        self.assert_unknown(fit_bowl_rim(self.observations(), config=GeometryConfig(pixel_error_bound_px=10)),
                            "uncertainty")
        self.assert_unknown(fit_bowl_rim(self.observations(), config=GeometryConfig(max_jacobian_condition=1.01)),
                            "conditioned")

    def test_duplicate_samples_do_not_shrink_uncertainty(self) -> None:
        observations = self.observations(count=60)
        original = fit_bowl_rim(observations)
        repeated = [replace(item, candidate=replace(item.candidate,
                    contour_px=np.repeat(item.candidate.contour_px, 2, axis=0))) for item in observations]
        result = fit_bowl_rim(repeated)
        self.assertEqual(result.status, "accepted", result)
        self.assertAlmostEqual(original.uncertainty.center_bound_m, result.uncertainty.center_bound_m, places=7)

    def test_no_implicit_tabletop_or_nominal_radius(self) -> None:
        # Outside optional z range still solves with default unconstrained z.
        self.frames = (camera((-0.38, -0.38, 1.4), focal=850, target=(0, 0, 0.70)),
                       camera((0.38, -0.30, 1.55), focal=850, target=(0, 0, 0.70)))
        observations = self.observations(center=(0.025, -0.015, 0.70), radius=0.043)
        result = fit_bowl_rim(observations)
        self.assertEqual(result.status, "accepted", result)
        self.assertAlmostEqual(result.center_xyz_m[2], 0.70, places=7)
        self.assert_unknown(fit_bowl_rim(observations, config=GeometryConfig(z_bounds_m=WORKSPACE_Z_BOUNDS_M)))

    def test_radius_out_of_bounds_is_unknown(self) -> None:
        self.assert_unknown(fit_bowl_rim(self.observations(radius=0.019)))

    def test_nonfinite_degenerate_and_sparse_input(self) -> None:
        for points in [np.full((30, 2), np.nan), np.ones((30, 2))*200,
                       np.column_stack((np.arange(30)+200, np.ones(30)*200)), np.ones((4, 2))*200]:
            with self.subTest(points=points.shape):
                observations = self.observations()
                observations[0] = replace(observations[0], candidate=replace(observations[0].candidate, contour_px=points))
                self.assert_unknown(fit_bowl_rim(observations))

    def test_unsynchronized_and_duplicate_views(self) -> None:
        observations = self.observations()
        observations[1] = replace(observations[1], frame=replace(observations[1].frame, time_s=2))
        self.assert_unknown(fit_bowl_rim(observations), "unsynchronized")
        observations = self.observations()
        observations[1] = replace(observations[1], view_id="v0")
        self.assert_unknown(fit_bowl_rim(observations), "duplicate_view")

    def test_optimizer_failure_is_unknown(self) -> None:
        with patch("environments.robodojo.runtime.robodojo_bowl_geometry.least_squares", side_effect=ValueError("failure")):
            self.assert_unknown(fit_bowl_rim(self.observations()), "optimization_unavailable")

    def test_count_search_limits_and_duplicate_ids_fail_closed(self) -> None:
        views = [RimView(str(i), frame, [rim(frame, identity="a"), rim(frame, identity="b")])
                 for i, frame in enumerate(self.frames)]
        self.assert_unknown(associate_bowl_rims(views, config=GeometryConfig(max_combinations=2)), "search_limit")
        self.assert_unknown(associate_bowl_rims([views[0], replace(views[1], candidates=views[1].candidates[:1])]), "counts")
        self.assert_unknown(associate_bowl_rims(views, expected_bowls=4), "count")
        views[0] = replace(views[0], candidates=[views[0].candidates[0]]*2)
        self.assert_unknown(associate_bowl_rims(views), "duplicate_candidate")

    def test_config_cannot_relax_required_limits(self) -> None:
        for kwargs in [dict(max_reprojection_px=4), dict(max_uncertainty_m=0.01),
                       dict(pixel_error_bound_px=0.5), dict(min_partial_coverage_deg=200),
                       dict(z_bounds_m=(1.2, 0.7))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GeometryConfig(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
