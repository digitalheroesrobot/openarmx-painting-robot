import math

import pytest

from core.coating_process import (
    DEFAULT_COATING_PROCESS,
    calculate_robot_waypoint_spacing_mm,
    validate_coating_process,
)


def test_robot_waypoint_spacing_uses_distance_per_control_cycle():
    assert math.isclose(calculate_robot_waypoint_spacing_mm(500.0, 100.0), 5.0)


def test_robot_waypoint_spacing_changes_with_speed_and_frequency():
    assert math.isclose(calculate_robot_waypoint_spacing_mm(250.0, 100.0), 2.5)
    assert math.isclose(calculate_robot_waypoint_spacing_mm(500.0, 250.0), 2.0)


def test_coating_process_does_not_own_robot_motion_settings():
    process = validate_coating_process(DEFAULT_COATING_PROCESS)
    assert "base_travel_speed_mm_s" not in process
    assert "robot_control_frequency_hz" not in process
    assert "min_travel_speed_mm_s" not in process
    assert "max_travel_acceleration_mm_s2" not in process


@pytest.mark.parametrize(
    "speed,frequency",
    [
        (0.0, 100.0), (-1.0, 100.0), (100.0, 0.0),
        (100.0, -1.0), (float("nan"), 100.0), (100.0, float("inf")),
    ],
)
def test_robot_waypoint_spacing_rejects_invalid_inputs(speed, frequency):
    with pytest.raises(ValueError):
        calculate_robot_waypoint_spacing_mm(speed, frequency)
