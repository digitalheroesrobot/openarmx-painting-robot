#!/usr/bin/env python3
"""Numerical tests for coverage-driven spray path optimization primitives."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.spray_coverage_optimizer import (
    analyze_surface_deficits,
    build_deficit_correction_strokes,
    build_coverage_correction_plan,
)


class SurfaceDeficitAnalysisTests(unittest.TestCase):
    def test_area_weighted_metrics_include_unreached_target_samples(self):
        result = analyze_surface_deficits(
            sample_area_mm2=np.array([1., 3., 2., 4., 5.]),
            target_mask=np.array([True, True, True, True, False]),
            reached_mask=np.array([True, False, True, True, True]),
            thickness_um=np.array([0., 20., 50., 70., 1000.]),
            target_thickness_um=50.,
            lower_ratio=.9,
            upper_ratio=1.1,
        )
        self.assertAlmostEqual(result["target_area_mm2"], 10.)
        self.assertAlmostEqual(result["geometric_reach_percent"], 70.)
        self.assertAlmostEqual(result["no_film_area_percent"], 10.)
        self.assertAlmostEqual(result["below_target_area_percent"], 40.)
        self.assertAlmostEqual(result["in_target_band_area_percent"], 20.)
        self.assertAlmostEqual(result["over_target_area_percent"], 40.)
        self.assertAlmostEqual(result["area_weighted_mean_thickness_um"], 44.)
        np.testing.assert_array_equal(result["deficit_sample_indices"], [0, 1])
        np.testing.assert_array_equal(result["unreached_sample_indices"], [1])
        self.assertEqual(result["status_by_sample"][4], "EXCLUDED")

    def test_exact_band_boundaries_are_accepted(self):
        result = analyze_surface_deficits(
            sample_area_mm2=np.ones(4),
            target_mask=np.ones(4, dtype=bool),
            reached_mask=np.ones(4, dtype=bool),
            thickness_um=np.array([44.999, 45., 55., 55.001]),
            target_thickness_um=50.,
        )
        np.testing.assert_array_equal(
            result["status_by_sample"],
            ["UNDER_TARGET", "IN_BAND", "IN_BAND", "OVER_TARGET"],
        )

    def test_excluded_samples_never_enter_denominator_or_deficits(self):
        result = analyze_surface_deficits(
            sample_area_mm2=np.array([2., 1000.]),
            target_mask=np.array([True, False]),
            reached_mask=np.array([False, True]),
            thickness_um=np.array([0., 999.]),
            target_thickness_um=50.,
        )
        self.assertEqual(result["target_area_mm2"], 2.)
        self.assertEqual(result["geometric_reach_percent"], 0.)
        np.testing.assert_array_equal(result["deficit_sample_indices"], [0])
        self.assertEqual(result["excluded_sample_count"], 1)

    def test_rejects_mismatched_or_invalid_analysis_inputs(self):
        base = dict(
            sample_area_mm2=np.ones(2),
            target_mask=np.ones(2, dtype=bool),
            reached_mask=np.ones(2, dtype=bool),
            thickness_um=np.ones(2),
            target_thickness_um=50.,
        )
        with self.assertRaises(ValueError):
            analyze_surface_deficits(**{**base, "thickness_um": np.ones(3)})
        with self.assertRaises(ValueError):
            analyze_surface_deficits(**{**base, "sample_area_mm2": np.array([1., 0.])})
        with self.assertRaises(ValueError):
            analyze_surface_deficits(**{**base, "target_thickness_um": 0.})
        with self.assertRaises(ValueError):
            analyze_surface_deficits(**{**base, "lower_ratio": 1.1, "upper_ratio": 1.0})


class IntegratedCoverageCorrectionPlanTests(unittest.TestCase):
    def test_plan_uses_complete_target_and_only_underfilm_samples(self):
        points = np.array([
            [0., 0., 0.], [10., 0., 0.], [20., 0., 0.],
            [30., 0., 0.], [40., 0., 0.],
        ])
        normals = np.tile([0., 0., 1.], (5, 1))
        plan = build_coverage_correction_plan(
            sample_points_mm=points,
            sample_normals=normals,
            sample_area_mm2=np.ones(5),
            target_mask=np.array([True, True, True, True, False]),
            reached_mask=np.array([True, False, True, True, True]),
            thickness_um=np.array([0., 20., 50., 70., 999.]),
            target_thickness_um=50.,
            patch_ids=np.array([1, 1, 1, 1, 8]),
            standoff_mm=100.,
            max_gap_mm=15.,
        )
        self.assertEqual(plan["status"], "PREDICTED_UNCALIBRATED_CANDIDATES_ONLY")
        self.assertFalse(plan["execution_ready"])
        self.assertEqual(plan["analysis"]["target_sample_count"], 4)
        np.testing.assert_array_equal(
            plan["analysis"]["deficit_sample_indices"], [0, 1]
        )
        np.testing.assert_array_equal(
            plan["correction_strokes"]["source_sample_indices"], [0, 1]
        )
        self.assertEqual(
            plan["correction_strokes"]["candidate_status"],
            "CANDIDATE_REQUIRES_PHYSICAL_VALIDATION",
        )

class DeficitCorrectionStrokeTests(unittest.TestCase):
    def test_strokes_split_at_spatial_gaps_and_keep_source_mapping(self):
        points = np.array([
            [0., 0., 0.], [10., 0., 0.], [20., 0., 0.],
            [100., 0., 0.], [110., 0., 0.],
        ])
        normals = np.tile([0., 0., 1.], (5, 1))
        result = build_deficit_correction_strokes(
            sample_points_mm=points,
            sample_normals=normals,
            deficit_sample_indices=np.arange(5),
            standoff_mm=100.,
            max_gap_mm=15.,
        )
        self.assertEqual(len(result["stroke_ranges"]), 2)
        self.assertEqual(
            [item["waypoint_count"] for item in result["stroke_ranges"]],
            [3, 2],
        )
        np.testing.assert_array_equal(result["source_sample_indices"], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(
            result["nozzle_points_mm"] - points,
            np.tile([0., 0., 100.], (5, 1)),
        )
        np.testing.assert_allclose(
            result["spray_directions"],
            np.tile([0., 0., -1.], (5, 1)),
        )

    def test_strokes_never_join_different_stage1_patches(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        normals = np.tile([0., 0., 1.], (3, 1))
        result = build_deficit_correction_strokes(
            sample_points_mm=points,
            sample_normals=normals,
            deficit_sample_indices=np.array([0, 1, 2]),
            patch_ids=np.array([4, 4, 9]),
            standoff_mm=80.,
            max_gap_mm=100.,
        )
        self.assertEqual(len(result["stroke_ranges"]), 2)
        self.assertEqual(result["stroke_ranges"][0]["patch_id"], 4)
        self.assertEqual(result["stroke_ranges"][1]["patch_id"], 9)

    def test_strokes_split_on_large_normal_change(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])
        normals = np.array([[0., 0., 1.], [0., 1., 0.]])
        result = build_deficit_correction_strokes(
            sample_points_mm=points,
            sample_normals=normals,
            deficit_sample_indices=np.array([0, 1]),
            standoff_mm=50.,
            max_gap_mm=10.,
            max_normal_change_deg=30.,
        )
        self.assertEqual(len(result["stroke_ranges"]), 2)

    def test_empty_deficit_set_returns_empty_candidate(self):
        result = build_deficit_correction_strokes(
            sample_points_mm=np.zeros((2, 3)),
            sample_normals=np.tile([0., 0., 1.], (2, 1)),
            deficit_sample_indices=np.array([], dtype=np.int64),
            standoff_mm=50.,
            max_gap_mm=10.,
        )
        self.assertEqual(result["candidate_status"], "NO_DEFICITS")
        self.assertEqual(result["nozzle_points_mm"].shape, (0, 3))

    def test_rejects_zero_normals_and_invalid_indices(self):
        with self.assertRaises(ValueError):
            build_deficit_correction_strokes(
                sample_points_mm=np.zeros((1, 3)),
                sample_normals=np.zeros((1, 3)),
                deficit_sample_indices=np.array([0]),
                standoff_mm=50.,
                max_gap_mm=10.,
            )
        with self.assertRaises(ValueError):
            build_deficit_correction_strokes(
                sample_points_mm=np.zeros((1, 3)),
                sample_normals=np.array([[0., 0., 1.]]),
                deficit_sample_indices=np.array([1]),
                standoff_mm=50.,
                max_gap_mm=10.,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
