import json

import pytest

from core.robot_settings import (
    DEFAULT_ROBOT_SETTINGS,
    load_robot_settings,
    save_robot_settings,
    validate_robot_settings,
)


def test_robot_settings_defaults_are_valid():
    settings = validate_robot_settings(DEFAULT_ROBOT_SETTINGS)
    assert settings["robot_control_frequency_hz"] == 100.0
    assert settings["base_travel_speed_mm_s"] == 500.0


def test_robot_settings_waypoint_spacing_inputs_are_independent():
    settings = validate_robot_settings({
        "base_travel_speed_mm_s": 240.0,
        "robot_control_frequency_hz": 120.0,
        "min_travel_speed_mm_s": 100.0,
        "max_linear_speed_mm_s": 300.0,
    })
    assert settings["base_travel_speed_mm_s"] / settings["robot_control_frequency_hz"] == 2.0


def test_robot_settings_migrate_legacy_coating_motion_values(tmp_path):
    path = tmp_path / "robot_settings.json"
    legacy = {
        "base_travel_speed_mm_s": 360.0,
        "robot_control_frequency_hz": 120.0,
        "min_travel_speed_mm_s": 80.0,
        "max_travel_acceleration_mm_s2": 700.0,
    }
    settings = load_robot_settings(path, legacy)
    assert settings["base_travel_speed_mm_s"] == 360.0
    assert settings["robot_control_frequency_hz"] == 120.0
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["robot"]["robot_control_frequency_hz"] == 120.0


def test_robot_settings_save_and_reload(tmp_path):
    path = tmp_path / "robot_settings.json"
    values = dict(DEFAULT_ROBOT_SETTINGS)
    values["name"] = "OpenArmX"
    values["robot_control_frequency_hz"] = 250.0
    save_robot_settings(path, values)
    loaded = load_robot_settings(path)
    assert loaded["name"] == "OpenArmX"
    assert loaded["robot_control_frequency_hz"] == 250.0


@pytest.mark.parametrize(
    "changes",
    [
        {"robot_control_frequency_hz": 0.0},
        {"base_travel_speed_mm_s": 0.0},
        {"min_travel_speed_mm_s": 600.0, "base_travel_speed_mm_s": 500.0},
        {"max_linear_speed_mm_s": 400.0, "base_travel_speed_mm_s": 500.0},
        {"max_angular_speed_deg_s": float("nan")},
    ],
)
def test_invalid_robot_limits_are_rejected(changes):
    values = dict(DEFAULT_ROBOT_SETTINGS)
    values.update(changes)
    with pytest.raises(ValueError):
        validate_robot_settings(values)
