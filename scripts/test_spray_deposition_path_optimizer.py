import unittest
import numpy as np
import trimesh

from stages.spray_deposition_model import SprayDepositionConfig
from stages.spray_deposition_path_optimizer import (
    evaluate_raster_candidate,
    select_track_spacing_by_relative_deposition,
    _pose_durations,
    _trajectory_distance_metrics,
)


class RelativeDepositionPathOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box(extents=[100., 100., 20.])
        self.face_ids = np.arange(len(self.mesh.faces), dtype=np.int64)
        self.patch_ids = np.zeros(len(self.mesh.faces), dtype=np.int64)
        self.normals = np.asarray(self.mesh.face_normals, dtype=np.float64)
        self.config = SprayDepositionConfig(cone_angle_deg=50.)

    def test_evaluator_reports_area_weighted_relative_coverage(self):
        # Directly exercise the evaluation contract with a deterministic candidate.
        centers = self.mesh.triangles_center
        normals = self.mesh.face_normals
        nozzle = centers + normals * 80.
        directions = -normals
        candidate = {
            "face_indices": self.face_ids,
            "nozzle_points": nozzle,
            "spray_directions": directions,
            "segment_ids": np.zeros(len(self.face_ids), dtype=np.int64),
        }
        report = evaluate_raster_candidate(
            self.mesh, candidate, self.config, spray_speed=100.
        )
        self.assertTrue(report["valid"])
        self.assertGreaterEqual(report["area_weighted_relative_coverage"], 0.)
        self.assertLessEqual(report["area_weighted_relative_coverage"], 1.)
        self.assertEqual(report["calibration_status"], "UNCALIBRATED_RELATIVE_DOSE_ONLY")

    def test_spacing_search_returns_one_of_requested_candidates(self):
        result = select_track_spacing_by_relative_deposition(
            self.mesh, self.face_ids, self.patch_ids, self.normals,
            standoff=80., sample_spacing=20.,
            track_spacing_candidates=[20., 30., 40.],
            deposition_config=self.config, spray_speed=100.,
            minimum_area_coverage=0.0,
        )
        self.assertIn(result["selected_track_spacing"], [20., 30., 40.])
        self.assertEqual(result["evaluated_spacing_count"], 3)
        self.assertEqual(result["evaluated_orientation_count"], 4)
        self.assertEqual(result["evaluated_candidate_count"], 12)
        self.assertTrue(result["orientation_search_performed"])
        self.assertIn(result["selected_lane_orientation_angle_deg"], [0., 45., 90., 135.])
        self.assertIn("selected_candidate", result)


    def test_orientation_search_changes_geometry_and_path_cost(self):
        rectangle = trimesh.Trimesh(
            vertices=np.array([
                [0., 0., 0.], [180., 0., 0.],
                [180., 60., 0.], [0., 60., 0.],
            ]),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            process=False,
        )
        face_ids = np.arange(len(rectangle.faces), dtype=np.int64)
        patch_ids = np.zeros(len(rectangle.faces), dtype=np.int64)
        normals = np.asarray(rectangle.face_normals, dtype=np.float64)
        result = select_track_spacing_by_relative_deposition(
            rectangle, face_ids, patch_ids, normals,
            standoff=80., sample_spacing=10.,
            track_spacing_candidates=[25.],
            deposition_config=self.config, spray_speed=100.,
            minimum_area_coverage=0.0,
            lane_orientation_candidates_deg=[0., 90.],
        )
        evaluations = result["spacing_evaluations"]
        self.assertEqual(len(evaluations), 2)
        by_angle = {
            item["lane_orientation_angle_deg"]: item["evaluation"]
            for item in evaluations
        }
        self.assertEqual(set(by_angle), {0., 90.})
        self.assertNotEqual(
            by_angle[0.]["sample_count"],
            by_angle[90.]["sample_count"],
        )
        self.assertNotAlmostEqual(
            by_angle[0.]["spray_path_length_coordinate_units"],
            by_angle[90.]["spray_path_length_coordinate_units"],
            places=6,
        )
        self.assertEqual(result["orientation_search_scope"],
                         "PLANAR_PATCH_PCA_AXIS_ROTATION")


    def test_fixed_reference_threshold_does_not_follow_candidate_maximum(self):
        centers = self.mesh.triangles_center
        normals = self.mesh.face_normals
        nozzle = centers + normals * 80.
        candidate = {
            "face_indices": self.face_ids,
            "nozzle_points": nozzle,
            "spray_directions": -normals,
            "segment_ids": np.zeros(len(self.face_ids), dtype=np.int64),
        }
        fast = evaluate_raster_candidate(
            self.mesh, candidate, self.config, spray_speed=100.,
            reference_standoff=80.,
        )
        slow = evaluate_raster_candidate(
            self.mesh, candidate, self.config, spray_speed=50.,
            reference_standoff=80.,
        )
        self.assertAlmostEqual(
            fast["relative_dose_threshold"], slow["relative_dose_threshold"]
        )
        self.assertAlmostEqual(
            slow["maximum_relative_dose"],
            2.0 * fast["maximum_relative_dose"],
            places=10,
        )
        self.assertEqual(
            fast["coverage_threshold_basis"],
            "FIXED_CALLER_SUPPLIED_REFERENCE_STANDOFF",
        )

    def test_pose_time_weights_sum_to_stroke_length_over_speed(self):
        points = np.array([
            [0., 0., 0.], [3., 0., 0.], [3., 4., 0.],
            [20., 0., 0.], [20., 0., 6.],
        ])
        segments = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        durations = _pose_durations(points, segments, speed=2.)
        expected_length = 3. + 4. + 6.
        self.assertAlmostEqual(float(np.sum(durations)), expected_length / 2.)

    def test_total_trajectory_cost_includes_spray_off_transitions(self):
        points = np.array([
            [0., 0., 0.], [10., 0., 0.],
            [100., 0., 0.], [110., 0., 0.],
        ])
        segments = np.array([0, 0, 1, 1], dtype=np.int64)
        metrics = _trajectory_distance_metrics(points, segments, travel_speed=50.)
        self.assertAlmostEqual(metrics["spray_path_length_coordinate_units"], 20.)
        self.assertAlmostEqual(
            metrics["spray_off_transition_length_coordinate_units"], 90.
        )
        self.assertAlmostEqual(
            metrics["total_trajectory_length_coordinate_units"], 110.
        )
        self.assertAlmostEqual(metrics["spray_off_transition_time_s"], 1.8)

    def test_invalid_optimizer_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            select_track_spacing_by_relative_deposition(
                self.mesh, self.face_ids, self.patch_ids, self.normals,
                standoff=80., sample_spacing=20.,
                track_spacing_candidates=[0.],
                deposition_config=self.config, spray_speed=100.,
                lane_orientation_candidates_deg=[0., 45.],
            )

        with self.assertRaises(ValueError):
            select_track_spacing_by_relative_deposition(
                self.mesh, self.face_ids, self.patch_ids, self.normals,
                standoff=80., sample_spacing=20.,
                track_spacing_candidates=[20.],
                deposition_config=self.config, spray_speed=100.,
                lane_orientation_candidates_deg=[float("nan")],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
