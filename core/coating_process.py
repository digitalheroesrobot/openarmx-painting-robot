"""Coating-process inputs and unit-safe film-thickness primitives.

Values are operator-supplied process assumptions until validated against
the actual paint, sprayer, and film-thickness measurements.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

DEFAULT_COATING_PROCESS = {
    # Provisional engineering baseline only. Replace with site TDS/gun data
    # and measured coupon results when the production values arrive.
    "name": "generic industrial spray baseline (uncalibrated)",
    "target_dry_film_thickness_um": 50.0,
    "fluid_flow_ml_min": 300.0,
    "transfer_efficiency_percent": 50.0,
    "volume_solids_percent": 50.0,
    "base_travel_speed_mm_s": 500.0,
    "min_travel_speed_mm_s": 300.0,
    "max_travel_acceleration_mm_s2": 1000.0,
    "number_of_coats": 1,
    "spray_on_delay_ms": 0.0,
    "spray_off_delay_ms": 0.0,
    "input_status": "UNVALIDATED_OPERATOR_INPUT",
}


def validate_coating_process(parameters: dict) -> dict:
    """Return a normalized copy or raise ValueError for invalid inputs."""
    if not isinstance(parameters, dict):
        raise TypeError("Coating process parameters must be a dictionary.")
    values = deepcopy(DEFAULT_COATING_PROCESS)
    values.update(deepcopy(parameters))
    numeric_ranges = {
        "target_dry_film_thickness_um": (0.0, 2000.0),
        "fluid_flow_ml_min": (0.0, 10000.0),
        "transfer_efficiency_percent": (0.0, 100.0),
        "volume_solids_percent": (0.0, 100.0),
        "base_travel_speed_mm_s": (0.1, 2000.0),
        "min_travel_speed_mm_s": (0.1, 2000.0),
        "max_travel_acceleration_mm_s2": (0.1, 10000.0),
        "spray_on_delay_ms": (0.0, 10000.0),
        "spray_off_delay_ms": (0.0, 10000.0),
    }
    for key, (lower, upper) in numeric_ranges.items():
        try:
            value = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be between {lower} and {upper}.")
        values[key] = value
    try:
        coats = int(values["number_of_coats"])
    except (TypeError, ValueError) as exc:
        raise ValueError("number_of_coats must be an integer.") from exc
    if coats < 1 or coats > 20 or float(values["number_of_coats"]) != coats:
        raise ValueError("number_of_coats must be an integer from 1 to 20.")
    values["number_of_coats"] = coats
    if values["min_travel_speed_mm_s"] > values["base_travel_speed_mm_s"]:
        raise ValueError("Minimum travel speed cannot exceed base travel speed.")
    values["name"] = str(values.get("name", "default coating process")).strip()
    if not values["name"]:
        raise ValueError("Coating process name cannot be empty.")
    return values


def coating_model_readiness(parameters: dict) -> tuple[str, list[str]]:
    """Report whether the inputs support a numerical deposition prediction."""
    p = validate_coating_process(parameters)
    missing = []
    for key, label in (
        ("fluid_flow_ml_min", "fluid flow rate"),
        ("transfer_efficiency_percent", "transfer efficiency"),
        ("volume_solids_percent", "volume solids"),
        ("target_dry_film_thickness_um", "target dry film thickness"),
    ):
        if p[key] <= 0.0:
            missing.append(label)
    if missing:
        return "INPUTS_INCOMPLETE", missing
    return "INPUTS_READY_UNCALIBRATED", []


def uniform_dry_film_increment_um(
    fluid_flow_ml_min: float,
    duration_s: float,
    transfer_efficiency: float,
    volume_solids: float,
    area_mm2: float,
) -> float:
    """Uniform idealized dry-film increment over an area, in micrometres.

    Fractions are expected in [0, 1]. This is a mass/volume conservation
    reference calculation, not the spatial spray-distribution model.
    """
    vals = (fluid_flow_ml_min, duration_s, transfer_efficiency, volume_solids, area_mm2)
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError("Film calculation inputs must be finite.")
    q, dt, efficiency, solids, area = map(float, vals)
    if q < 0.0 or dt < 0.0 or area <= 0.0:
        raise ValueError("Flow and duration must be non-negative; area must be positive.")
    if not 0.0 <= efficiency <= 1.0 or not 0.0 <= solids <= 1.0:
        raise ValueError("Transfer efficiency and volume solids must be fractions in [0, 1].")
    deposited_wet_mm3 = q * dt / 60.0 * 1000.0 * efficiency
    deposited_dry_mm3 = deposited_wet_mm3 * solids
    return deposited_dry_mm3 / area * 1000.0


def load_coating_process(config_path: str | Path, legacy_sprayer_profile: dict | None = None) -> dict:
    """Load process settings, migrating legacy speed/flow values once."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        legacy = legacy_sprayer_profile if isinstance(legacy_sprayer_profile, dict) else {}
        values = deepcopy(DEFAULT_COATING_PROCESS)
        values["base_travel_speed_mm_s"] = legacy.get("travel_speed_mm_s", values["base_travel_speed_mm_s"])
        values["min_travel_speed_mm_s"] = min(
            float(legacy.get("min_travel_speed_mm_s", values["min_travel_speed_mm_s"])),
            float(values["base_travel_speed_mm_s"]),
        )
        values["max_travel_acceleration_mm_s2"] = legacy.get("max_travel_acceleration_mm_s2", values["max_travel_acceleration_mm_s2"])
        values["fluid_flow_ml_min"] = legacy.get("fluid_flow_ml_min", values["fluid_flow_ml_min"])
        save_coating_process(path, values)
        return validate_coating_process(values)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("process", payload) if isinstance(payload, dict) else {}
        values = validate_coating_process(values)

        # Upgrade only the exact factory-created incomplete profile shipped
        # before the provisional baseline. Preserve operator motion settings.
        is_legacy_factory_profile = (
            values.get("name") == "default coating process"
            and values["target_dry_film_thickness_um"] == 60.0
            and values["fluid_flow_ml_min"] == 0.0
            and values["transfer_efficiency_percent"] == 0.0
            and values["volume_solids_percent"] == 0.0
        )
        if is_legacy_factory_profile:
            values["name"] = DEFAULT_COATING_PROCESS["name"]
            for key in (
                "target_dry_film_thickness_um",
                "fluid_flow_ml_min",
                "transfer_efficiency_percent",
                "volume_solids_percent",
            ):
                values[key] = DEFAULT_COATING_PROCESS[key]
            # Update motion values only when they still equal the previous
            # untouched factory defaults; retain any customized legacy values.
            old_motion_defaults = {
                "base_travel_speed_mm_s": 150.0,
                "min_travel_speed_mm_s": 40.0,
                "max_travel_acceleration_mm_s2": 500.0,
            }
            for key, old_value in old_motion_defaults.items():
                if values[key] == old_value:
                    values[key] = DEFAULT_COATING_PROCESS[key]
            save_coating_process(path, values)
        return values
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Could not read coating process configuration: {path}\n{exc}") from exc


def save_coating_process(config_path: str | Path, parameters: dict) -> None:
    path = Path(config_path)
    values = validate_coating_process(parameters)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "process": values}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
