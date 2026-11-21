"""Numerical tests for the isolated human-like sweep planner."""
import unittest

import numpy as np

from stages.human_like_spray_planner import (
    SweepPlanningConfig,
    build_nozzle_frames,
    plan_human_like_sweeps,
    _split_lane,
)


class HumanLikeSweepPlannerTests(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.arange(5.0), np.arange(3.0))
        self.points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        self.normals = np.tile([0.0, 0.0, 1.0], (len(self.points), 1))
        self.faces = np.arange(len(self.points), dtype=np.int64)
        self.patches = np.where(self.faces < 7, 1, 2)
        self.config = SweepPlanningConfig(
            lane_spacing=1.0,
            sample_spacing=1.0,
            max_gap_factor=2.5,
        )

    def test_every_sample_is_preserved_once(self):
        result = plan_human_like_sweeps(
            self.points, self.normals, self.faces, self.patches, self.config
        )
        np.testing.assert_array_equal(
            np.sort(result["sample_order"]), np.arange(len(self.points))
        )
        np.testing.assert_allclose(
            np.sort(result["surface_points"], axis=0),
            np.sort(self.points, axis=0),
        )

    def test_face_and_patch_boundaries_do_not_force_new_segments(self):
        points = np.column_stack((np.arange(8.0), np.zeros(8), np.zeros(8)))
        normals = np.tile([0.0, 0.0, 1.0], (8, 1))
        result = plan_human_like_sweeps(
            points,
            normals,
            np.arange(8),
            np.array([0, 0, 0, 1, 1, 2, 2, 2]),
            SweepPlanningConfig(2.0, 1.0, max_gap_factor=2.5),
        )
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(len(np.unique(result["segment_ids"])), 1)

    def test_real_spatial_gap_splits_path(self):
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0], [22.0, 0.0, 0.0],
        ])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        result = plan_human_like_sweeps(
            points, normals, np.arange(6), np.zeros(6, dtype=int),
            SweepPlanningConfig(5.0, 1.0, max_gap_factor=2.0),
        )
        self.assertGreaterEqual(result["segment_count"], 2)

    def test_frames_are_orthonormal_and_aim_toward_surface(self):
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        tangents = np.tile([1.0, 0.0, 0.0], (4, 1))
        frames = build_nozzle_frames(normals, tangents)
        for frame in frames:
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(float(np.linalg.det(frame)), 1.0, places=8)
        np.testing.assert_allclose(
            frames[:, :, 2], np.tile([0.0, 0.0, -1.0], (len(frames), 1))
        )

    def test_spatial_gap_is_measured_in_3d_not_only_in_projection(self):
        projected = np.column_stack((np.arange(4.0), np.zeros(4)))
        spatial = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 20.0],
        ])
        chunks = _split_lane(
            np.arange(4, dtype=np.int64),
            projected,
            spatial,
            max_gap=5.0,
        )
        self.assertEqual([chunk.tolist() for chunk in chunks], [[0, 1, 2], [3]])

    def test_lane_metadata_is_aligned_with_ordered_samples(self):
        result = plan_human_like_sweeps(
            self.points, self.normals, self.faces, self.patches, self.config
        )
        self.assertEqual(result["lane_ids"].shape, (len(self.points),))
        self.assertEqual(result["segment_ids"].shape, (len(self.points),))
        self.assertEqual(len(np.unique(result["lane_ids"])), result["lane_count"])
        for segment_id in np.unique(result["segment_ids"]):
            lane_ids = np.unique(result["lane_ids"][result["segment_ids"] == segment_id])
            self.assertEqual(len(lane_ids), 1)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            plan_human_like_sweeps(
                self.points, self.normals[:-1], self.faces, self.patches, self.config
            )
        with self.assertRaises(ValueError):
            SweepPlanningConfig(0.0, 1.0).validate()


if __name__ == "__main__":
    unittest.main()
