"""Numerical tests for configured circular spray footprints."""

import math
import unittest

from stages.stage3_spray_trajectory import _calculate_spray_footprint


class CircularSprayFootprintTests(unittest.TestCase):
    def test_full_included_angle_sets_circular_diameter(self):
        profile = {
            "pattern_type": "round_cone",
            "spray_cone_angle_deg": 60.0,
            "overlap_percent": 25.0,
        }
        result = _calculate_spray_footprint(profile, standoff_mm=100.0)
        expected_diameter = 2.0 * 100.0 * math.tan(math.radians(30.0))
        self.assertAlmostEqual(result["width_mm"], expected_diameter, places=9)
        self.assertAlmostEqual(result["thickness_mm"], expected_diameter, places=9)
        self.assertAlmostEqual(
            result["nominal_track_spacing_mm"],
            expected_diameter * 0.75,
            places=9,
        )

    def test_profile_values_are_the_source_of_footprint_settings(self):
        profile = {
            "pattern_type": "round_cone",
            "spray_cone_angle_deg": 45.0,
            "overlap_percent": 50.0,
        }
        result = _calculate_spray_footprint(profile, standoff_mm=180.0)
        expected_diameter = 2.0 * 180.0 * math.tan(math.radians(22.5))
        self.assertAlmostEqual(result["working_distance_mm"], 180.0)
        self.assertAlmostEqual(result["included_angle_deg"], 45.0)
        self.assertAlmostEqual(result["width_mm"], expected_diameter, places=9)
        self.assertAlmostEqual(
            result["nominal_track_spacing_mm"],
            expected_diameter * 0.5,
            places=9,
        )

    def test_active_profile_values_produce_expected_point_spacing(self):
        profile = {
            "pattern_type": "round_cone",
            "spray_cone_angle_deg": 74.0,
            "overlap_percent": 50.0,
        }
        result = _calculate_spray_footprint(profile, standoff_mm=180.0)
        expected_diameter = 2.0 * 180.0 * math.tan(math.radians(37.0))
        self.assertAlmostEqual(result["diameter_mm"], expected_diameter, places=9)
        self.assertAlmostEqual(
            result["nominal_point_spacing_mm"],
            expected_diameter * 0.5,
            places=9,
        )
        self.assertAlmostEqual(result["nominal_point_spacing_mm"], 135.6397290185, places=8)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            _calculate_spray_footprint(
                {
                    "pattern_type": "round_cone",
                    "spray_cone_angle_deg": 45.0,
                    "overlap_percent": 100.0,
                },
                standoff_mm=180.0,
            )


if __name__ == "__main__":
    unittest.main()
