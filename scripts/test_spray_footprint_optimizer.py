"""Tests for 3-D cone-footprint overlap reduction."""
import unittest

import numpy as np

from stages.spray_footprint_optimizer import optimize_redundant_spray_points


class SprayFootprintOptimizerTests(unittest.TestCase):
    def _run_grid(self, size=5):
        x, y = np.meshgrid(np.arange(size, dtype=float) * 0.25, np.arange(size, dtype=float))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        nozzles = points + np.array([0.0, 0.0, 1.0])
        directions = np.tile([0.0, 0.0, -1.0], (len(points), 1))
        weights = np.ones(len(points), dtype=float)
        return optimize_redundant_spray_points(
            surface_points=points,
            nozzle_points=nozzles,
            spray_directions=directions,
            probe_points=points,
            probe_area_weights=weights,
            standoff_coordinate_units=1.0,
            cone_angle_deg=90.0,
            overlap_percent=50.0,
            sample_spacing_coordinate_units=0.25,
        )

    def test_redundant_interior_points_are_removed_without_coverage_loss(self):
        result = self._run_grid()
        self.assertGreater(result["removed_point_count"], 0)
        self.assertEqual(
            result["input_point_count"],
            result["removed_point_count"] + result["retained_point_count"],
        )
        self.assertTrue(result["coverage_preserved"])
        self.assertEqual(
            result["coverage_preservation_policy"],
            "NO_PREVIOUSLY_COVERED_SURFACE_SAMPLE_MAY_BECOME_UNCOVERED",
        )

    def test_overlap_threshold_and_area_method_are_reported(self):
        result = self._run_grid()
        self.assertEqual(result["overlap_threshold_percent"], 50.0)
        self.assertIn("TRIANGLE_AREA_WEIGHTED", result["area_measurement"])

    def test_invalid_inputs_are_rejected(self):
        points = np.array([[0.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            optimize_redundant_spray_points(
                surface_points=points,
                nozzle_points=points + [0.0, 0.0, 1.0],
                spray_directions=np.array([[0.0, 0.0, -1.0]]),
                probe_points=points,
                probe_area_weights=np.array([1.0]),
                standoff_coordinate_units=1.0,
                cone_angle_deg=0.0,
                overlap_percent=50.0,
                sample_spacing_coordinate_units=1.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
