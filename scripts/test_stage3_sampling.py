#!/usr/bin/env python3
"""Numerical regression tests for Stage 3 triangular surface sampling."""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stages.stage3_spray_trajectory import (
    FACE_PAINTABLE_CANDIDATE,
    _sample_candidate_faces,
    _triangle_sample_points,
)


def reference_triangle_samples(triangle, spacing):
    """Original nested-loop algorithm retained as a test oracle."""
    a, b, c = np.asarray(triangle, dtype=np.float64)
    n = max(1, int(np.ceil(max(
        np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
    ) / spacing)))
    result = []
    def point(i, j):
        u, v = i / n, j / n
        return (1.0 - u - v) * a + u * b + v * c
    for i in range(n):
        for j in range(n - i):
            p0 = point(i, j)
            p1 = point(i + 1, j)
            p2 = point(i, j + 1)
            result.append((p0 + p1 + p2) / 3.0)
            if i + j < n - 1:
                p3 = point(i + 1, j + 1)
                result.append((p1 + p3 + p2) / 3.0)
    return np.asarray(result)


class TriangleSamplingTests(unittest.TestCase):
    def test_vectorized_sampler_matches_original_sample_locations(self):
        triangles = [
            np.array([[0., 0., 0.], [20., 0., 0.], [0., 20., 0.]]),
            np.array([[3., -4., 2.], [13., 5., 8.], [-2., 11., 4.]]),
            np.array([[0., 0., 0.], [0., 0., 17.], [0., 9., 0.]]),
        ]
        for triangle in triangles:
            for spacing in (20., 10., 5., 2.5):
                with self.subTest(spacing=spacing, triangle=triangle.tolist()):
                    actual = _triangle_sample_points(triangle, spacing)
                    expected = reference_triangle_samples(triangle, spacing)
                    self.assertEqual(len(actual), len(expected))
                    # Preserve exact legacy ordering as well as coordinates.
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-12, atol=1e-12
                    )

    def test_sample_count_is_square_of_subdivision_count(self):
        triangle = np.array([
            [0., 0., 0.], [30., 0., 0.],
            [15., 25.98076211353316, 0.],
        ])
        for n in (1, 2, 3, 5, 10):
            points = _triangle_sample_points(triangle, 30.001 / n)
            self.assertEqual(len(points), n * n)

    def test_all_samples_remain_inside_triangle(self):
        triangle = np.array([[1., 2., 3.], [31., 2., 3.], [1., 32., 3.]])
        points = _triangle_sample_points(triangle, 3.0)
        # For this right triangle, x + y <= 33 and both coordinates are bounded.
        self.assertTrue(np.all(points[:, 0] >= 1.0 - 1e-12))
        self.assertTrue(np.all(points[:, 1] >= 2.0 - 1e-12))
        self.assertTrue(np.all(points[:, 0] + points[:, 1] <= 33.0 + 1e-12))
        np.testing.assert_allclose(points[:, 2], 3.0)

    def test_invalid_and_degenerate_triangles_return_empty(self):
        self.assertEqual(len(_triangle_sample_points(np.zeros((3, 3)), 1.0)), 0)
        bad = np.array([[0., 0., 0.], [np.nan, 0., 0.], [0., 1., 0.]])
        self.assertEqual(len(_triangle_sample_points(bad, 1.0)), 0)

    def test_invalid_spacing_is_rejected(self):
        triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        for spacing in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(spacing=spacing):
                with self.assertRaises(ValueError):
                    _triangle_sample_points(triangle, spacing)


    def test_batched_candidate_validation_preserves_samples_and_source_order(self):
        vertices = np.array([
            [0., 0., 0.], [10., 0., 0.], [0., 10., 0.],
            [20., 0., 0.], [30., 0., 0.], [40., 0., 0.],
            [50., 0., 0.], [60., 0., 0.], [50., 10., 0.],
            [70., 0., 0.], [80., 0., 0.], [70., 10., 0.],
        ])
        faces = np.arange(12, dtype=np.int64).reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False, validate=False
        )
        status = np.full(4, FACE_PAINTABLE_CANDIDATE, dtype=np.int32)
        patches = np.array([11, 12, 13, 14], dtype=np.int32)
        normals = np.array([
            [0., 0., 1.],
            [np.nan, 0., 1.],  # skipped before geometry validation
            [0., 0., 1.],
            [0., 0., 1.],
        ])
        # Face 3 is deliberately degenerate; face 1 has an invalid normal.
        mesh.vertices[9:12] = np.array([
            [70., 0., 0.], [80., 0., 0.], [90., 0., 0.]
        ])
        mesh.faces[3] = [9, 10, 11]
        mesh.vertices[11] = mesh.vertices[10]

        result = _sample_candidate_faces(
            mesh, status, patches, normals, spacing_coordinate_units=5.0
        )
        expected = np.vstack([
            _triangle_sample_points(vertices[0:3], 5.0),
            _triangle_sample_points(vertices[6:9], 5.0),
        ])
        np.testing.assert_allclose(result["surface_points"], expected)
        self.assertEqual(result["candidate_face_count"], 4)
        self.assertEqual(result["skipped_invalid_normals"], 1)
        self.assertEqual(result["skipped_invalid_geometry"], 1)
        self.assertEqual(result["sample_count"], len(expected))
        self.assertEqual(
            result["face_indices"].tolist(),
            [0] * len(_triangle_sample_points(vertices[0:3], 5.0))
            + [2] * len(_triangle_sample_points(vertices[6:9], 5.0)),
        )
        self.assertEqual(
            result["patch_ids"].tolist(),
            [11] * len(_triangle_sample_points(vertices[0:3], 5.0))
            + [13] * len(_triangle_sample_points(vertices[6:9], 5.0)),
        )

    def test_candidate_sampling_enforces_total_sample_limit(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([
                [0., 0., 0.], [10., 0., 0.], [0., 10., 0.]
            ]),
            faces=np.array([[0, 1, 2]]),
            process=False,
            validate=False,
        )
        with self.assertRaisesRegex(RuntimeError, "more than"):
            _sample_candidate_faces(
                mesh,
                np.array([FACE_PAINTABLE_CANDIDATE]),
                np.array([0]),
                np.array([[0., 0., 1.]]),
                spacing_coordinate_units=1.0,
                max_total_samples=10,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
