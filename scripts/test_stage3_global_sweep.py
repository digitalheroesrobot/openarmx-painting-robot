"""Integration tests for the Stage 3 global spatial sweep generator."""
import unittest

import numpy as np
import trimesh

from stages.stage3_spray_trajectory import (
    FACE_PAINTABLE_CANDIDATE,
    generate_spray_trajectory,
)


class Stage3GlobalSweepTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.Trimesh(
            vertices=np.asarray([
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ]),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            process=False,
        )
        self.status = np.full(
            len(self.mesh.faces), FACE_PAINTABLE_CANDIDATE, dtype=np.int64
        )
        self.profile = {
            "pattern_type": "round_cone",
            "spray_cone_angle_deg": 45.0,
            "overlap_percent": 50.0,
        }

    def _generate(self, patch_labels):
        return generate_spray_trajectory(
            mesh=self.mesh.copy(),
            face_status_labels=self.status.copy(),
            patch_labels=np.asarray(patch_labels, dtype=np.int64),
            standoff_mm=100.0,
            sampling_spacing_mm=10.0,
            spray_angle_deg=45.0,
            coordinate_unit="mm",
            sprayer_profile=self.profile.copy(),
        )

    def test_face_and_patch_boundaries_do_not_change_sweep_order(self):
        one_patch = self._generate([7, 7])
        two_patches = self._generate([7, 9])

        first = one_patch
        second = two_patches
        np.testing.assert_allclose(
            first["surface_points"], second["surface_points"], atol=1e-10
        )
        np.testing.assert_array_equal(
            first["face_indices"], second["face_indices"]
        )
        np.testing.assert_array_equal(
            first["segment_ids"], second["segment_ids"]
        )
        np.testing.assert_allclose(
            first["spray_directions"], second["spray_directions"], atol=1e-10
        )
        self.assertEqual(
            second["surface_boundary_splits"]["status"],
            "NO_TOPOLOGY_BASED_SPLITS",
        )

    def test_every_candidate_sample_is_preserved_and_standoff_is_exact(self):
        result = self._generate([7, 9])
        count = len(result["surface_points"])
        self.assertGreater(count, 0)
        self.assertEqual(result["sampling"]["sample_count"], count)
        self.assertEqual(result["nozzle_points"].shape, (count, 3))
        self.assertEqual(result["nozzle_frames"].shape, (count, 3, 3))
        np.testing.assert_allclose(
            np.linalg.norm(
                result["nozzle_points"] - result["surface_points"], axis=1
            ),
            100.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result["nozzle_frames"][:, :, 2],
            result["spray_directions"],
            atol=1e-8,
        )
        for frame in result["nozzle_frames"]:
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(float(np.linalg.det(frame)), 1.0, places=8)

    def test_surface_sampling_spacing_uses_footprint_and_overlap(self):
        half_overlap_profile = self.profile.copy()
        high_overlap_profile = self.profile.copy()
        high_overlap_profile["overlap_percent"] = 75.0

        common = {
            "mesh": self.mesh.copy(),
            "face_status_labels": self.status.copy(),
            "patch_labels": np.asarray([7, 9], dtype=np.int64),
            "standoff_mm": 100.0,
            "sampling_spacing_mm": 20.0,
            "spray_angle_deg": 45.0,
            "coordinate_unit": "mm",
        }
        half_overlap = generate_spray_trajectory(
            **common, sprayer_profile=half_overlap_profile
        )
        high_overlap = generate_spray_trajectory(
            **common, sprayer_profile=high_overlap_profile
        )

        width = 2.0 * 100.0 * np.tan(np.deg2rad(45.0 / 2.0))
        expected_half_overlap_spacing = width * 0.5
        expected_high_overlap_spacing = width * 0.25

        self.assertAlmostEqual(
            half_overlap["sampling"]["coverage_spacing_mm"],
            expected_half_overlap_spacing,
            places=8,
        )
        self.assertAlmostEqual(
            high_overlap["sampling"]["coverage_spacing_mm"],
            expected_high_overlap_spacing,
            places=8,
        )
        self.assertGreater(
            high_overlap["sampling"]["source_sample_count_before_footprint_optimization"],
            half_overlap["sampling"]["source_sample_count_before_footprint_optimization"],
        )
        self.assertEqual(
            high_overlap["sampling"]["coverage_spacing_basis"],
            "SPRAY_FOOTPRINT_AND_OVERLAP",
        )
        self.assertAlmostEqual(
            high_overlap["rasterization"]["trajectory_sample_spacing_mm"],
            expected_high_overlap_spacing,
            places=8,
        )
        self.assertEqual(
            high_overlap["rasterization"]["trajectory_sample_spacing_basis"],
            "SPRAY_FOOTPRINT_AND_OVERLAP",
        )
        self.assertEqual(
            high_overlap["sampling"]["configured_sampling_spacing_mm"],
            20.0,
        )
        self.assertEqual(
            high_overlap["sampling"]["configured_sampling_spacing_role"],
            "FALLBACK_ONLY_WHEN_FOOTPRINT_UNAVAILABLE",
        )
        self.assertIn(
            "footprint_optimizer_removed_sample_count",
            high_overlap["sampling"],
        )
        self.assertTrue(
            high_overlap["rasterization"]["footprint_optimization"]["coverage_preserved"]
        )

        # Dense probes over the flat target must lie inside at least one
        # nominal circular footprint. This is a geometric test, not a paint
        # deposition or curved-surface certification.
        gx, gy = np.meshgrid(
            np.linspace(0.0, 100.0, 21),
            np.linspace(0.0, 100.0, 21),
        )
        probes = np.column_stack((
            gx.ravel(), gy.ravel(), np.zeros(gx.size)
        ))
        sample_points = half_overlap["surface_points"]
        nearest_distances = np.linalg.norm(
            probes[:, None, :] - sample_points[None, :, :],
            axis=2,
        ).min(axis=1)
        self.assertLessEqual(
            float(np.max(nearest_distances)),
            width * 0.5 + 1e-8,
        )

    def test_robot_waypoint_spacing_uses_speed_and_control_frequency(self):
        result = generate_spray_trajectory(
            mesh=self.mesh.copy(),
            face_status_labels=self.status.copy(),
            patch_labels=np.asarray([7, 9], dtype=np.int64),
            standoff_mm=100.0,
            sampling_spacing_mm=10.0,
            spray_angle_deg=45.0,
            coordinate_unit="mm",
            sprayer_profile=self.profile.copy(),
            coating_process={
                "base_travel_speed_mm_s": 240.0,
                "min_travel_speed_mm_s": 120.0,
                "robot_control_frequency_hz": 120.0,
            },
        )
        self.assertAlmostEqual(
            result["parameters"]["robot_waypoint_spacing_mm"],
            2.0,
            places=8,
        )
        self.assertAlmostEqual(
            result["rasterization"]["robot_waypoint_spacing_mm"],
            2.0,
            places=8,
        )
        expected_spray_pitch = (
            2.0 * 100.0 * np.tan(np.deg2rad(45.0 / 2.0)) * 0.5
        )
        self.assertAlmostEqual(
            result["rasterization"]["trajectory_sample_spacing_mm"],
            expected_spray_pitch,
            places=8,
        )
        self.assertNotAlmostEqual(
            result["rasterization"]["trajectory_sample_spacing_mm"],
            result["rasterization"]["robot_waypoint_spacing_mm"],
            places=6,
        )
        self.assertEqual(
            result["rasterization"]["robot_waypoint_sampling_basis"],
            "BASE_TRAVEL_SPEED_DIVIDED_BY_ROBOT_CONTROL_FREQUENCY",
        )

    def test_legacy_16_column_data_contract_is_unchanged(self):
        result = self._generate([7, 9])
        trajectory = result
        array = np.column_stack([
            trajectory["surface_points"],
            trajectory["surface_normals"],
            trajectory["nozzle_points"],
            trajectory["spray_directions"],
            trajectory["face_indices"],
            trajectory["patch_ids"],
            trajectory["segment_ids"],
            np.arange(len(trajectory["surface_points"])),
        ])
        self.assertEqual(array.shape[1], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
