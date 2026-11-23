import math

import pytest

from core.coating_process import (
    DEFAULT_COATING_PROCESS,
    calculate_robot_waypoint_spacing_mm,
    validate_coating_process,
)


def test_robot_waypoint_spacing_uses_distance_per_control_cycle():
    assert math.isclose(
        calculate_robot_waypoint_spacing_mm(500.0, 100.0),
        5.0,
    )


def test_robot_waypoint_spacing_changes_with_speed_and_frequency():
    assert math.isclose(
        calculate_robot_waypoint_spacing_mm(250.0, 100.0),
        2.5,
    )
    assert math.isclose(
        calculate_robot_waypoint_spacing_mm(500.0, 250.0),
        2.0,
    )


def test_default_coating_process_has_robot_control_frequency():
    process = validate_coating_process(DEFAULT_COATING_PROCESS)
    assert process["robot_control_frequency_hz"] == 100.0
    assert math.isclose(
        calculate_robot_waypoint_spacing_mm(
            process["base_travel_speed_mm_s"],
            process["robot_control_frequency_hz"],
        ),
        5.0,
    )


@pytest.mark.parametrize(
    "speed,frequency",
    [
        (0.0, 100.0),
        (-1.0, 100.0),
        (100.0, 0.0),
        (100.0, -1.0),
        (float("nan"), 100.0),
        (100.0, float("inf")),
    ],
)
def test_robot_waypoint_spacing_rejects_non_positive_or_non_finite_inputs(
    speed,
    frequency,
):
    with pytest.raises(ValueError):
        calculate_robot_waypoint_spacing_mm(speed, frequency)


def test_control_frequency_validation_range():
    process = dict(DEFAULT_COATING_PROCESS)
    process["robot_control_frequency_hz"] = 0.0
    with pytest.raises(ValueError):
        validate_coating_process(process)
