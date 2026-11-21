"""Numerical tests for patch-local spray raster candidate generation."""
import unittest

import numpy as np
import trimesh

from stages.spray_coverage_path_planner import (
    connected_paintable_components,
    generate_patchwise_raster_candidates,
)


class PatchwiseSprayPlannerTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.Trimesh(
            vertices=np.array([
                [0., 0., 0.], [100., 0., 0.],
                [100., 100., 0.], [0., 100., 0.],
            ]),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            process=False,
        )
        self.normals = np.tile([0., 0., 1.], (2, 1))

    def test_adjacent_faces_same_patch_form_one_component(self):
        groups = connected_paintable_components(
            self.mesh, np.array([0, 1]), np.array([4, 4]), self.normals
        )
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], [0, 1])

    def test_patch_boundary_splits_even_when_faces_are_adjacent(self):
        groups = connected_paintable_components(
            self.mesh, np.array([0, 1]), np.array([4, 9]), self.normals
        )
        self.assertEqual(len(groups), 2)

    def test_rasterizer_keeps_components_as_separate_spray_strokes(self):
        result = generate_patchwise_raster_candidates(
            self.mesh, np.array([0, 1]), np.array([4, 9]), self.normals,
            standoff=100., sample_spacing=10., track_spacing=40.,
        )
        self.assertEqual(result["component_count"], 2)
        self.assertEqual(result["rasterized_component_count"], 2)
        self.assertGreater(result["segment_count"], 1)
        self.assertFalse(result["spray_on_connectors_between_components"])
        self.assertFalse(result["global_pca_ordering_used"])
        for component in result["components"]:
            self.assertEqual(component["status"], "RASTERIZED")
        for segment_id in np.unique(result["segment_ids"]):
            ids = np.flatnonzero(result["segment_ids"] == segment_id)
            self.assertEqual(len(np.unique(np.array([4, 9])[result["face_indices"][ids]])), 1)
        np.testing.assert_allclose(
            np.linalg.norm(result["nozzle_points"] - result["surface_points"], axis=1),
            100., atol=1e-7,
        )

    def test_invalid_spacing_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_patchwise_raster_candidates(
                self.mesh, np.array([0, 1]), np.array([4, 4]), self.normals,
                standoff=100., sample_spacing=0., track_spacing=40.,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
