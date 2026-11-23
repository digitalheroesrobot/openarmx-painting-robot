import json

from ui.sprayer_settings import DEFAULT_SPRAYER_PROFILE, load_sprayer_profiles


def test_default_sprayer_profile_uses_100_mm_and_50_degree_cone():
    assert DEFAULT_SPRAYER_PROFILE["standoff_mm"] == 100.0
    assert DEFAULT_SPRAYER_PROFILE["spray_cone_angle_deg"] == 50.0
    assert DEFAULT_SPRAYER_PROFILE["pattern_type"] == "round_cone"
    assert "travel_speed_mm_s" not in DEFAULT_SPRAYER_PROFILE
    assert "min_travel_speed_mm_s" not in DEFAULT_SPRAYER_PROFILE
    assert "max_travel_acceleration_mm_s2" not in DEFAULT_SPRAYER_PROFILE


def test_legacy_profile_migrates_old_defaults_and_removes_motion_fields(tmp_path):
    path = tmp_path / "sprayers.json"
    path.write_text(
        json.dumps({
            "profiles": [{
                "name": "default sprayer",
                "pattern_type": "round_cone",
                "standoff_mm": 180.0,
                "spray_cone_angle_deg": 45.0,
                "overlap_percent": 50.0,
                "travel_speed_mm_s": 150.0,
                "min_travel_speed_mm_s": 40.0,
                "max_travel_acceleration_mm_s2": 500.0,
            }],
            "active_profile": "default sprayer",
        }),
        encoding="utf-8",
    )
    profile = load_sprayer_profiles(path)[0]
    assert profile["standoff_mm"] == 100.0
    assert profile["spray_cone_angle_deg"] == 50.0
    assert "travel_speed_mm_s" not in profile
    assert "min_travel_speed_mm_s" not in profile
    assert "max_travel_acceleration_mm_s2" not in profile
    stored = json.loads(path.read_text(encoding="utf-8"))["profiles"][0]
    assert stored["standoff_mm"] == 100.0
    assert stored["spray_cone_angle_deg"] == 50.0
    assert "travel_speed_mm_s" not in stored


def test_measured_nondefault_standoff_and_angle_are_preserved(tmp_path):
    path = tmp_path / "sprayers.json"
    path.write_text(
        json.dumps({
            "profiles": [{
                "name": "Measured Gun",
                "pattern_type": "round_cone",
                "standoff_mm": 125.0,
                "spray_cone_angle_deg": 42.0,
                "overlap_percent": 60.0,
                "gun_body_length_mm": 210.0,
                "gun_body_diameter_mm": 48.0,
                "gun_body_dimensions_are_estimate": False,
            }],
            "active_profile": "Measured Gun",
        }),
        encoding="utf-8",
    )
    profiles = load_sprayer_profiles(path)
    profile = next(item for item in profiles if item["name"] == "Measured Gun")
    assert profile["standoff_mm"] == 125.0
    assert profile["spray_cone_angle_deg"] == 42.0
    assert profile["gun_body_dimensions_are_estimate"] is False

    # Loading also persists any default-profile insertion/migration. Verify
    # that the user's measured profile remains unchanged in the saved file.
    stored_profiles = json.loads(path.read_text(encoding="utf-8"))["profiles"]
    stored = next(item for item in stored_profiles if item["name"] == "Measured Gun")
    assert stored["standoff_mm"] == 125.0
    assert stored["spray_cone_angle_deg"] == 42.0
    assert stored["gun_body_dimensions_are_estimate"] is False
