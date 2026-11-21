import unittest
import numpy as np
import trimesh

from stages.spray_deposition_model import SprayDepositionConfig
from stages.spray_deposition_path_optimizer import (
    evaluate_raster_candidate,
    select_track_spacing_by_relative_deposition,
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
        self.assertFalse(result["orientation_search_performed"])
        self.assertIn("selected_candidate", result)

    def test_invalid_optimizer_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            select_track_spacing_by_relative_deposition(
                self.mesh, self.face_ids, self.patch_ids, self.normals,
                standoff=80., sample_spacing=20.,
                track_spacing_candidates=[0.],
                deposition_config=self.config, spray_speed=100.,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
