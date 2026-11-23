import json

import pytest

from core.robot_settings import (
    DEFAULT_ROBOT_SETTINGS,
    export_robot_profile,
    import_robot_profile,
    load_robot_settings,
    save_robot_settings,
    validate_robot_settings,
)


def test_robot_settings_contains_only_two_motion_inputs():
    settings = validate_robot_settings(DEFAULT_ROBOT_SETTINGS)
    assert settings["base_travel_speed_mm_s"] == 500.0
    assert settings["robot_control_frequency_hz"] == 100.0
    assert set(settings) == {
        "robot_name", "product_name", "name",
        "base_travel_speed_mm_s", "robot_control_frequency_hz",
    }


def test_robot_profile_name_is_robot_name_plus_product_name():
    settings = validate_robot_settings({
        "robot_name": "OpenArmX",
        "product_name": "SteelBracket",
        "base_travel_speed_mm_s": 240.0,
        "robot_control_frequency_hz": 120.0,
    })
    assert settings["name"] == "OpenArmX_SteelBracket"


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
    assert "min_travel_speed_mm_s" not in settings
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["robot"]["robot_control_frequency_hz"] == 120.0


def test_robot_profile_save_and_reload(tmp_path):
    path = tmp_path / "robot_settings.json"
    values = {
        "robot_name": "OpenArmX",
        "product_name": "CubeCell",
        "base_travel_speed_mm_s": 360.0,
        "robot_control_frequency_hz": 250.0,
    }
    save_robot_settings(path, values)
    loaded = load_robot_settings(path)
    assert loaded["name"] == "OpenArmX_CubeCell"
    assert loaded["robot_control_frequency_hz"] == 250.0


def test_robot_profile_export_and_import(tmp_path):
    path = tmp_path / "OpenArmX_CubeCell.json"
    values = {
        "robot_name": "OpenArmX",
        "product_name": "CubeCell",
        "base_travel_speed_mm_s": 360.0,
        "robot_control_frequency_hz": 250.0,
    }
    export_robot_profile(path, values)
    loaded = import_robot_profile(path)
    assert loaded["name"] == "OpenArmX_CubeCell"
    assert loaded["base_travel_speed_mm_s"] == 360.0


@pytest.mark.parametrize(
    "changes",
    [
        {"robot_control_frequency_hz": 0.0},
        {"base_travel_speed_mm_s": 0.0},
        {"robot_control_frequency_hz": float("nan")},
    ],
)
def test_invalid_robot_profile_values_are_rejected(changes):
    values = dict(DEFAULT_ROBOT_SETTINGS)
    values.update(changes)
    with pytest.raises(ValueError):
        validate_robot_settings(values)
