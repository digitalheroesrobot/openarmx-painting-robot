import math

import pytest

from stages.stage3_spray_trajectory import _calculate_spray_footprint


def test_default_sprayer_geometry_is_100_mm_and_50_degrees():
    footprint = _calculate_spray_footprint(
        {
            "pattern_type": "round_cone",
            "spray_cone_angle_deg": 50.0,
            "overlap_percent": 50.0,
        },
        100.0,
    )
    expected_width = 2.0 * 100.0 * math.tan(math.radians(25.0))
    assert footprint["working_distance_mm"] == 100.0
    assert footprint["included_angle_deg"] == 50.0
    assert math.isclose(footprint["width_mm"], expected_width)
    assert math.isclose(
        footprint["nominal_track_spacing_mm"],
        expected_width * 0.5,
    )


@pytest.mark.parametrize("overlap", [0.0, 25.0, 50.0, 75.0, 90.0])
def test_sample_spacing_equals_footprint_width_minus_overlap(overlap):
    footprint = _calculate_spray_footprint(
        {
            "pattern_type": "round_cone",
            "spray_cone_angle_deg": 50.0,
            "overlap_percent": overlap,
        },
        100.0,
    )
    expected = footprint["width_mm"] * (1.0 - overlap / 100.0)
    assert math.isclose(footprint["nominal_track_spacing_mm"], expected)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        _calculate_spray_footprint(
            {
                "pattern_type": "round_cone",
                "spray_cone_angle_deg": 50.0,
                "overlap_percent": 100.0,
            },
            100.0,
        )
