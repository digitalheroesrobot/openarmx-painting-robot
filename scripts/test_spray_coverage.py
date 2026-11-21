#!/usr/bin/env python3
"""Numerical regression tests for Stage 2-masked spray reach coverage."""
from __future__ import annotations

import unittest
import numpy as np

from core.spray_coverage import (
    build_paintable_face_mask,
    calculate_area_weighted_coverage,
    mark_sample_hits,
)


class SprayCoverageTests(unittest.TestCase):
    def test_only_stage2_paintable_faces_are_targets(self):
        labels = np.asarray([1, 0, 1, 2, 3], dtype=np.uint8)
        np.testing.assert_array_equal(
            build_paintable_face_mask(labels, 5),
            [True, False, True, False, False],
        )

    def test_missing_or_mismatched_stage2_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            build_paintable_face_mask(None, 3)
        with self.assertRaises(ValueError):
            build_paintable_face_mask(np.asarray([1, 1]), 3)

    def test_zero_reach_is_zero_percent(self):
        fractions, reached, target, percent = calculate_area_weighted_coverage(
            np.asarray([1.0, 1.0, 2.0]),
            np.zeros(12, dtype=bool),
            4,
        )
        np.testing.assert_allclose(fractions, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(reached, 0.0)
        self.assertAlmostEqual(target, 4.0)
        self.assertAlmostEqual(percent, 0.0)

    def test_excluded_area_is_not_in_the_coverage_denominator(self):
        labels = np.asarray([1, 0, 1], dtype=np.uint8)
        all_areas = np.asarray([1.0, 100.0, 1.0])
        target_mask = build_paintable_face_mask(labels, 3)
        target_areas = all_areas[target_mask]
        # First approved face is fully reached; second approved face is not.
        hits = np.asarray([True, True, True, True, False, False, False, False])
        _, reached, target, percent = calculate_area_weighted_coverage(
            target_areas, hits, 4
        )
        self.assertAlmostEqual(reached, 1.0)
        self.assertAlmostEqual(target, 2.0)
        self.assertAlmostEqual(percent, 50.0)

    def test_full_reach_is_one_hundred_percent_only_when_all_target_samples_hit(self):
        hits = np.ones(8, dtype=bool)
        _, reached, target, percent = calculate_area_weighted_coverage(
            np.asarray([2.5, 7.5]), hits, 4
        )
        self.assertAlmostEqual(reached, 10.0)
        self.assertAlmostEqual(target, 10.0)
        self.assertAlmostEqual(percent, 100.0)

    def test_half_reached_area_is_fifty_percent(self):
        hits = np.asarray([
            True, True, True, True,   # face 0: 1.0 / 1.0
            False, False, False, False,  # face 1: 0.0 / 1.0
            True, True, False, False,  # face 2: 0.5 / 2.0
        ], dtype=bool)
        _, reached, target, percent = calculate_area_weighted_coverage(
            np.asarray([1.0, 1.0, 2.0]), hits, 4
        )
        self.assertAlmostEqual(reached, 2.0)
        self.assertAlmostEqual(target, 4.0)
        self.assertAlmostEqual(percent, 50.0)

    def test_spray_off_does_not_accumulate_hits(self):
        hits = np.zeros(8, dtype=bool)
        returned = mark_sample_hits(hits, np.asarray([1, 3, 5]), spray_on=False)
        self.assertIs(returned, hits)
        self.assertEqual(int(np.count_nonzero(hits)), 0)

    def test_spray_on_accumulates_only_valid_hit_indices(self):
        hits = np.zeros(8, dtype=bool)
        mark_sample_hits(hits, np.asarray([1, 3, 5]), spray_on=True)
        np.testing.assert_array_equal(
            hits, [False, True, False, True, False, True, False, False]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
