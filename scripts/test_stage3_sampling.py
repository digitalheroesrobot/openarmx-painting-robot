#!/usr/bin/env python3
"""Numerical regression tests for Stage 3 triangular surface sampling."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stages.stage3_spray_trajectory import _triangle_sample_points


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


def sorted_rows(values):
    values = np.asarray(values)
    return values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))]


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
                    np.testing.assert_allclose(
                        sorted_rows(actual), sorted_rows(expected),
                        rtol=1e-12, atol=1e-12,
                    )

    def test_sample_count_is_square_of_subdivision_count(self):
        triangle = np.array([[0., 0., 0.], [30., 0., 0.], [0., 30., 0.]])
        for n in (1, 2, 3, 5, 10):
            points = _triangle_sample_points(triangle, 30.0 / n)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
