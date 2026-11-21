"""Unit tests for the independent spray-gun envelope safety module."""
import unittest

import numpy as np
import trimesh

from stages.trajectory_safety import (
    SprayGunEnvelopeConfig,
    build_gun_body_centerline,
    evaluate_spray_gun_clearance,
)


def _downward_frame():
    # Right-handed frame whose +Z axis points down toward the product.
    return np.diag([1.0, -1.0, -1.0])


def _plane_mesh(z=0.0, extent=1000.0):
    vertices = np.array([
        [-extent, -extent, z],
        [ extent, -extent, z],
        [ extent,  extent, z],
        [-extent,  extent, z],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class SprayGunEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.config = SprayGunEnvelopeConfig(
            gun_body_length_mm=220.0,
            gun_body_diameter_mm=45.0,
            nozzle_length_mm=25.0,
            minimum_surface_clearance_mm=180.0,
            nozzle_outer_diameter_mm=None,
            dimensions_are_estimate=False,
        )

    def test_body_axis_extends_away_from_surface(self):
        tip = np.array([0.0, 0.0, 180.0])
        centerline = build_gun_body_centerline(
            tip, _downward_frame(), self.config, "mm"
        )
        self.assertAlmostEqual(centerline[0, 2], 205.0)
        self.assertAlmostEqual(centerline[-1, 2], 425.0)
        self.assertTrue(np.all(np.diff(centerline[:, 2]) > 0.0))

    def test_coordinate_unit_conversion_is_consistent(self):
        tip_mm = np.array([0.0, 0.0, 180.0])
        tip_m = tip_mm / 1000.0
        line_mm = build_gun_body_centerline(
            tip_mm, _downward_frame(), self.config, "mm"
        )
        line_m = build_gun_body_centerline(
            tip_m, _downward_frame(), self.config, "m"
        )
        np.testing.assert_allclose(line_mm, line_m * 1000.0)

    def test_body_clear_of_product_but_nozzle_unknown_is_uncertain(self):
        mesh = _plane_mesh()
        result = evaluate_spray_gun_clearance(
            mesh,
            np.array([[0.0, 0.0, 180.0]]),
            _downward_frame()[None, :, :],
            self.config,
            "mm",
        )
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["unsafe_sample_indices"], [])
        self.assertGreaterEqual(
            result["minimum_measured_body_clearance_mm"], 180.0
        )

    def test_body_entering_keepout_is_unsafe(self):
        mesh = _plane_mesh()
        # Flip +Z toward the product: body then extends into the keep-out zone.
        toward_surface_frame = np.diag([1.0, -1.0, -1.0])
        tip = np.array([[0.0, 0.0, 400.0]])
        config = SprayGunEnvelopeConfig(
            gun_body_length_mm=220.0,
            gun_body_diameter_mm=45.0,
            nozzle_length_mm=0.0,
            minimum_surface_clearance_mm=180.0,
            nozzle_outer_diameter_mm=12.0,
            dimensions_are_estimate=False,
        )
        result = evaluate_spray_gun_clearance(
            mesh, tip, toward_surface_frame[None, :, :], config, "mm"
        )
        self.assertEqual(result["status"], "UNSAFE")
        self.assertEqual(result["unsafe_sample_indices"], [0])

    def test_invalid_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            build_gun_body_centerline(
                np.zeros(3), np.eye(3) * 2.0, self.config, "mm"
            )

    def test_unknown_coordinate_unit_is_rejected(self):
        with self.assertRaises(ValueError):
            build_gun_body_centerline(
                np.zeros(3), _downward_frame(), self.config, "unknown"
            )


if __name__ == "__main__":
    unittest.main()
