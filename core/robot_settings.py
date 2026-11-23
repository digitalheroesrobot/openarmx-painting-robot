"""Robot motion and controller settings, stored independently of coating data."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

DEFAULT_ROBOT_SETTINGS = {
    "name": "default robot",
    "base_travel_speed_mm_s": 500.0,
    "robot_control_frequency_hz": 100.0,
    "min_travel_speed_mm_s": 300.0,
    "max_travel_acceleration_mm_s2": 1000.0,
    "max_linear_speed_mm_s": 2000.0,
    "max_angular_speed_deg_s": 180.0,
    "input_status": "UNVALIDATED_OPERATOR_INPUT",
}


def validate_robot_settings(parameters: dict) -> dict:
    if not isinstance(parameters, dict):
        raise TypeError("Robot settings must be a dictionary.")
    values = deepcopy(DEFAULT_ROBOT_SETTINGS)
    values.update(deepcopy(parameters))
    ranges = {
        "base_travel_speed_mm_s": (0.1, 2000.0),
        "robot_control_frequency_hz": (1.0, 1000.0),
        "min_travel_speed_mm_s": (0.1, 2000.0),
        "max_travel_acceleration_mm_s2": (0.1, 10000.0),
        "max_linear_speed_mm_s": (0.1, 5000.0),
        "max_angular_speed_deg_s": (0.1, 2000.0),
    }
    for key, (lower, upper) in ranges.items():
        try:
            value = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be between {lower} and {upper}.")
        values[key] = value
    if values["min_travel_speed_mm_s"] > values["base_travel_speed_mm_s"]:
        raise ValueError("Minimum travel speed cannot exceed painting speed.")
    if values["base_travel_speed_mm_s"] > values["max_linear_speed_mm_s"]:
        raise ValueError("Painting speed cannot exceed maximum linear speed.")
    values["name"] = str(values.get("name", "default robot")).strip()
    if not values["name"]:
        raise ValueError("Robot profile name cannot be empty.")
    return values


def load_robot_settings(config_path: str | Path, legacy_process: dict | None = None) -> dict:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        legacy = legacy_process if isinstance(legacy_process, dict) else {}
        values = deepcopy(DEFAULT_ROBOT_SETTINGS)
        for key in (
            "base_travel_speed_mm_s",
            "robot_control_frequency_hz",
            "min_travel_speed_mm_s",
            "max_travel_acceleration_mm_s2",
        ):
            if key in legacy:
                values[key] = legacy[key]
        save_robot_settings(path, values)
        return validate_robot_settings(values)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("robot", payload) if isinstance(payload, dict) else {}
        return validate_robot_settings(values)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Could not read robot settings: {path}\n{exc}") from exc


def save_robot_settings(config_path: str | Path, parameters: dict) -> None:
    path = Path(config_path)
    values = validate_robot_settings(parameters)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "robot": values}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
