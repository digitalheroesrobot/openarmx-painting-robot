"""Numerical tests for the uncalibrated spray deposition model."""
import unittest
import numpy as np

from stages.spray_deposition_model import (
    SprayDepositionConfig,
    evaluate_pose_deposition,
    integrate_spray_path_deposition,
)


class SprayDepositionModelTests(unittest.TestCase):
    def setUp(self):
        self.config = SprayDepositionConfig(
            cone_angle_deg=60.0,
            radial_sigma_fraction=0.45,
            incidence_exponent=1.0,
        )

    def test_center_receives_more_deposition_than_footprint_edge(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.45, 0.0],
            [0.0, 0.70, 0.0],
        ])
        normals = np.tile([0.0, 0.0, 1.0], (3, 1))
        result = evaluate_pose_deposition(
            points, normals, np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]), self.config,
        )
        flux = result["relative_flux"]
        self.assertGreater(flux[0], flux[1])
        self.assertEqual(flux[2], 0.0)
        self.assertTrue(result["inside_cone"][1])
        self.assertFalse(result["inside_cone"][2])

    def test_back_facing_and_behind_nozzle_points_receive_zero(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ])
        result = evaluate_pose_deposition(
            points, normals, np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]), self.config,
        )
        self.assertGreater(result["relative_flux"][0], 0.0)
        self.assertEqual(result["relative_flux"][1], 0.0)

    def test_grazing_incidence_reduces_relative_deposition(self):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        result = evaluate_pose_deposition(
            points, normals, np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]), self.config,
        )
        self.assertGreater(result["relative_flux"][0], result["relative_flux"][1])
        self.assertAlmostEqual(result["relative_flux"][1], 0.0)

    def test_path_dose_accumulates_with_pose_duration(self):
        points = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        nozzle = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        directions = np.tile([0.0, 0.0, -1.0], (2, 1))
        one = integrate_spray_path_deposition(
            points, normals, nozzle[:1], directions[:1],
            np.array([1.0]), self.config,
        )
        two = integrate_spray_path_deposition(
            points, normals, nozzle, directions,
            np.array([1.0, 1.0]), self.config,
        )
        self.assertAlmostEqual(two["relative_dose"][0], 2.0 * one["relative_dose"][0])
        self.assertFalse(two["physical_film_thickness_available"])

    def test_invalid_configuration_and_pose_are_rejected(self):
        with self.assertRaises(ValueError):
            SprayDepositionConfig(cone_angle_deg=0.0).validate()
        with self.assertRaises(ValueError):
            evaluate_pose_deposition(
                np.zeros((1, 3)), np.zeros((1, 3)),
                np.zeros(3), np.zeros(3), self.config,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
