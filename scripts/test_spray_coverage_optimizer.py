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
    append_correction_strokes_to_trajectory,
    world_samples_to_product_cad_mm,
    analyze_surface_deficits,
    build_deficit_correction_strokes,
    build_coverage_correction_plan,
)


class CorrectionTrajectoryAppendTests(unittest.TestCase):
    def test_appends_cad_strokes_with_new_segments_and_monotonic_times(self):
        base = {
            "surface_points": np.array([[0., 0., 0.], [1., 0., 0.]]),
            "surface_normals": np.tile([0., 0., 1.], (2, 1)),
            "nozzle_points": np.array([[0., 0., 50.], [1., 0., 50.]]),
            "spray_directions": np.tile([0., 0., -1.], (2, 1)),
            "nozzle_frames": np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            "face_indices": np.array([2, 2]),
            "patch_ids": np.array([4, 4]),
            "segment_ids": np.array([0, 0]),
            "travel_speeds_mm_s": np.array([100., 100.]),
            "trajectory_time_s": np.array([0., .01]),
        }
        strokes = {
            "surface_points_mm": np.array([[10., 0., 0.], [20., 0., 0.], [40., 0., 0.]]),
            "surface_normals": np.tile([0., 0., 1.], (3, 1)),
            "nozzle_points_mm": np.array([[10., 0., 50.], [20., 0., 50.], [40., 0., 50.]]),
            "spray_directions": np.tile([0., 0., -1.], (3, 1)),
            "source_sample_indices": np.array([0, 1, 2]),
            "patch_ids": np.array([7, 7, 9]),
            "stroke_ranges": [
                {"start": 0, "stop": 2},
                {"start": 2, "stop": 3},
            ],
        }
        result = append_correction_strokes_to_trajectory(
            trajectory=base, correction_strokes=strokes,
            source_face_ids=np.array([11, 12, 13]),
            mm_per_unit=1., base_speed_mm_s=100.,
            minimum_speed_mm_s=50.)
        self.assertEqual(len(result["nozzle_points"]), 6)
        np.testing.assert_array_equal(result["face_indices"], [2, 2, 11, 12, 13, 13])
        np.testing.assert_array_equal(result["segment_ids"], [0, 0, 1, 1, 2, 2])
        self.assertTrue(np.all(np.diff(result["trajectory_time_s"]) > 0.0))
        self.assertEqual(result["correction_path_metadata"]["added_stroke_count"], 2)
        np.testing.assert_allclose(result["nozzle_points"][-1], [40., 0., 50.])

    def test_rejects_bad_source_mapping(self):
        base = {
            "surface_points": np.zeros((1, 3)), "surface_normals": np.array([[0., 0., 1.]]),
            "nozzle_points": np.array([[0., 0., 1.]]), "spray_directions": np.array([[0., 0., -1.]]),
            "segment_ids": np.array([0]), "travel_speeds_mm_s": np.array([10.]),
            "trajectory_time_s": np.array([0.]),
        }
        strokes = {
            "surface_points_mm": np.zeros((1, 3)), "surface_normals": np.array([[0., 0., 1.]]),
            "nozzle_points_mm": np.array([[0., 0., 1.]]), "spray_directions": np.array([[0., 0., -1.]]),
            "source_sample_indices": np.array([2]), "patch_ids": np.array([0]),
            "stroke_ranges": [{"start": 0, "stop": 1}],
        }
        with self.assertRaises(ValueError):
            append_correction_strokes_to_trajectory(
                trajectory=base, correction_strokes=strokes,
                source_face_ids=np.array([4]), mm_per_unit=1.,
                base_speed_mm_s=100., minimum_speed_mm_s=50.)


class ProductCadCoordinateTests(unittest.TestCase):
    def test_world_samples_are_inverted_to_product_cad_millimeters(self):
        angle = np.deg2rad(90.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                             [np.sin(angle), np.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [100.0, -20.0, 5.0]
        cad_mm = np.array([[10.0, 20.0, 30.0], [-5.0, 4.0, 2.0]])
        world = (cad_mm / 10.0) @ rotation.T + transform[:3, 3]
        normals_cad = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        normals_world = normals_cad @ rotation.T
        actual_points, actual_normals = world_samples_to_product_cad_mm(
            points_world=world, normals_world=normals_world,
            product_transform_4x4=transform, mm_per_unit=10.0)
        np.testing.assert_allclose(actual_points, cad_mm, atol=1e-10)
        np.testing.assert_allclose(actual_normals, normals_cad, atol=1e-10)

    def test_rejects_non_rigid_product_transform(self):
        transform = np.eye(4)
        transform[0, 0] = 2.0
        with self.assertRaises(ValueError):
            world_samples_to_product_cad_mm(
                points_world=np.zeros((1, 3)),
                normals_world=np.array([[0.0, 0.0, 1.0]]),
                product_transform_4x4=transform, mm_per_unit=1.0)


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


class ViewerCorrectionIntegrationTests(unittest.TestCase):
    def test_viewer_exposes_explicit_correction_preview_append(self):
        source = (ROOT / "ui" / "viewer_3d.py").read_text(encoding="utf-8")
        self.assertIn("APPEND CORRECTION STROKES TO PREVIEW", source)
        self.assertIn("def _apply_coverage_correction_preview(self):", source)
        self.assertIn("append_correction_strokes_to_trajectory(", source)
        self.assertIn("source_face_ids = np.repeat(", source)
    def test_viewer_runs_deficit_analysis_after_full_playback(self):
        source = (ROOT / "ui" / "viewer_3d.py").read_text(encoding="utf-8")
        self.assertIn("build_coverage_correction_plan", source)
        self.assertIn("def _update_coverage_correction_plan(self):", source)
        self.assertIn('cache["coverage_sample_hits"]', source)
        self.assertIn('cache["deposition_sample_thickness_um"]', source)
        self.assertIn('cache["sample_area_mm2"]', source)
        self.assertIn('cache["sample_patch_ids"]', source)
        self.assertIn("self._update_coverage_correction_plan()", source)

    def test_main_window_passes_stage1_patch_labels_to_viewer(self):
        source = (ROOT / "app" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn('self.current_stage1_result.get("patch_labels")', source)
        self.assertIn("patch_labels=patch_labels", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
