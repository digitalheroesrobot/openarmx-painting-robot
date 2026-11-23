"""Minimal robot profile settings and JSON profile I/O."""
from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT_ROBOT_SETTINGS = {
    "robot_name": "Robot",
    "product_name": "Product",
    "name": "Robot_Product",
    "base_travel_speed_mm_s": 500.0,
    "robot_control_frequency_hz": 100.0,
}


def validate_robot_settings(parameters: dict) -> dict:
    if not isinstance(parameters, dict):
        raise TypeError("Robot settings must be a dictionary.")
    values = dict(DEFAULT_ROBOT_SETTINGS)
    values.update(parameters)
    for key, low, high in (
        ("base_travel_speed_mm_s", 0.1, 2000.0),
        ("robot_control_frequency_hz", 1.0, 1000.0),
    ):
        try:
            value = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        values[key] = value
    values["robot_name"] = str(values.get("robot_name", "")).strip()
    values["product_name"] = str(values.get("product_name", "")).strip()
    if not values["robot_name"] or not values["product_name"]:
        raise ValueError("Robot name and product name are required.")
    values["name"] = f"{values['robot_name']}_{values['product_name']}"
    return {key: values[key] for key in DEFAULT_ROBOT_SETTINGS}


def load_robot_settings(config_path: str | Path, legacy_process: dict | None = None) -> dict:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        legacy = legacy_process if isinstance(legacy_process, dict) else {}
        values = dict(DEFAULT_ROBOT_SETTINGS)
        for key in ("base_travel_speed_mm_s", "robot_control_frequency_hz"):
            if key in legacy:
                values[key] = legacy[key]
        save_robot_settings(path, values)
        return validate_robot_settings(values)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("robot", payload) if isinstance(payload, dict) else {}
        # Migrate profiles created before robot/product names were separated.
        if "robot_name" not in values:
            legacy_name = str(values.get("name", "Robot_Product"))
            values["robot_name"], _, values["product_name"] = legacy_name.partition("_")
            values["product_name"] = values["product_name"] or "Product"
        return validate_robot_settings(values)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Could not read robot settings: {path}\n{exc}") from exc


def save_robot_settings(config_path: str | Path, parameters: dict) -> None:
    path = Path(config_path)
    values = validate_robot_settings(parameters)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 2, "robot": values}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_robot_profile(file_path: str | Path, parameters: dict) -> None:
    path = Path(file_path)
    values = validate_robot_settings(parameters)
    payload = {"schema_version": 2, "robot": values}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_robot_profile(file_path: str | Path) -> dict:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("robot", payload) if isinstance(payload, dict) else {}
    return validate_robot_settings(values)
