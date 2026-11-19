
from pathlib import Path

import numpy as np
import trimesh

from core.geometry_analyzer import analyze_geometry
from core.result_manager import ResultManager


# ============================================================
# SUPPORTED SOURCE UNITS
# ============================================================

SUPPORTED_UNITS = {
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",

    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",

    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",

    "in": "in",
    "inch": "in",
    "inches": "in",

    "ft": "ft",
    "foot": "ft",
    "feet": "ft",

    "unknown": "unknown",
}


# ============================================================
# UNIT NORMALIZATION
# ============================================================

def _normalize_unit(unit: str | None) -> str:

    if unit is None:
        return "unknown"

    if not isinstance(unit, str):
        raise TypeError(
            "source_unit must be a string or None."
        )

    normalized = unit.strip().lower()

    if not normalized:
        return "unknown"

    if normalized not in SUPPORTED_UNITS:
        supported = ", ".join(
            sorted(SUPPORTED_UNITS.keys())
        )

        raise ValueError(
            f"Unsupported coordinate unit: {unit!r}. "
            f"Supported units: {supported}"
        )

    return SUPPORTED_UNITS[normalized]


# ============================================================
# SOURCE UNIT RESOLUTION
# ============================================================

def _get_source_unit(
    mesh: trimesh.Trimesh,
    manual_unit: str | None = None,
) -> str:
    """
    Priority:
        1. Explicit user selection
        2. STL loader metadata
        3. Unknown

    Coordinates are never modified.
    """

    if manual_unit is not None:
        return _normalize_unit(manual_unit)

    metadata = getattr(mesh, "metadata", {})

    if not isinstance(metadata, dict):
        return "unknown"

    return _normalize_unit(
        metadata.get("source_unit")
    )


# ============================================================
# MEASUREMENT UNIT LABELS
# ============================================================

def _get_measurement_units(source_unit: str) -> dict:

    if source_unit == "unknown":
        coordinate_unit = "unknown"
        area_unit = "unknown^2"
        volume_unit = "unknown^3"

    else:
        coordinate_unit = source_unit
        area_unit = f"{source_unit}^2"
        volume_unit = f"{source_unit}^3"

    return {
        "coordinate_unit": coordinate_unit,
        "dimensions_unit": coordinate_unit,
        "surface_area_unit": area_unit,
        "volume_unit": volume_unit,
    }


# ============================================================
# STAGE 1 EXECUTION
# ============================================================

def execute_stage1(
    mesh: trimesh.Trimesh,
    stl_path: str | Path,
    result_manager: ResultManager,
    source_unit: str | None = None,
) -> dict:
    """
    Execute Stage 1 geometry analysis.

    Rules:
        - Explicit STL unit is used automatically.
        - Manual unit selection overrides STL metadata.
        - Unknown units are rejected.
        - No coordinate scaling or modification is performed.
    """

    # --------------------------------------------------------
    # 1. Validate mesh
    # --------------------------------------------------------

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            "Stage 1 requires a trimesh.Trimesh object."
        )

    if (
        len(mesh.vertices) == 0
        or len(mesh.faces) == 0
    ):
        raise ValueError(
            "Stage 1 received an empty mesh."
        )

    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(
            "Stage 1 received non-finite mesh coordinates."
        )

    # --------------------------------------------------------
    # 2. Validate STL path
    # --------------------------------------------------------

    stl_path = (
        Path(stl_path)
        .expanduser()
        .resolve()
    )

    if not stl_path.exists():
        raise FileNotFoundError(
            f"STL file not found: {stl_path}"
        )

    # --------------------------------------------------------
    # 3. Resolve coordinate unit
    # --------------------------------------------------------

    detected_unit = _get_source_unit(
        mesh,
        manual_unit=source_unit,
    )

    # IMPORTANT:
    # Do not run geometry analysis when unit is unknown.
    # The GUI must also disable the Stage 1 button.
    if detected_unit == "unknown":
        raise ValueError(
            "COORDINATE UNIT NOT SPECIFIED.\n\n"
            "The STL file does not contain a recognized "
            "physical length unit.\n\n"
            "Please select the actual coordinate unit "
            "in the application before running Geometry "
            "Inspection.\n\n"
            "The original STL coordinates have not been "
            "modified."
        )

    metadata = getattr(mesh, "metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    if source_unit is not None:
        unit_detection = "manual_user_selection"
    else:
        unit_detection = metadata.get(
            "unit_detection",
            "explicit_stl_metadata",
        )

    # --------------------------------------------------------
    # 4. Run geometry analyzer
    # --------------------------------------------------------

    # Pass the resolved unit to the analyzer.
    # This must not rescale or modify mesh coordinates.
    report, patch_labels = analyze_geometry(
        mesh,
        coordinate_unit=detected_unit,
    )

    # --------------------------------------------------------
    # 5. Source information
    # --------------------------------------------------------

    report["source_file"] = str(stl_path)

    report["part_name"] = (
        result_manager.part_name_from_stl(
            stl_path
        )
    )

    # --------------------------------------------------------
    # 6. Unit configuration
    # --------------------------------------------------------

    measurement_units = _get_measurement_units(
        detected_unit
    )

    report["unit_configuration"] = {
        "source_stl_coordinate_unit": detected_unit,

        "internal_geometry_unit": detected_unit,

        **measurement_units,

        "coordinate_conversion_applied": False,

        "original_stl_coordinates_preserved": True,

        "unit_detection": unit_detection,

        "manual_unit_override_applied": (
            source_unit is not None
        ),

        "manual_source_unit_argument": source_unit,
    }

    # --------------------------------------------------------
    # 7. Consistent analysis metadata
    # --------------------------------------------------------

    if not isinstance(report.get("analysis"), dict):
        report["analysis"] = {}

    report["analysis"].update({
        "stage": 1,
        "name": "Geometry Analysis",
        "coordinate_unit": detected_unit,
        "source_coordinates_modified": False,
        "physical_measurements_validated": False,
    })

    # --------------------------------------------------------
    # 8. Measurement notes
    # --------------------------------------------------------

    warnings = [
        "The selected coordinate unit defines the physical "
        "interpretation of the original STL coordinates.",

        "Changing the coordinate unit does not rescale "
        "the original mesh vertices.",

        "STL does not universally define a standardized "
        "physical length unit. Confirm manually selected "
        "units against the original CAD model when necessary.",
    ]

    report["measurement_warnings"] = warnings

    # --------------------------------------------------------
    # 9. Save report
    # --------------------------------------------------------

    json_path = result_manager.save_json(
        stl_path=stl_path,
        stage_number=1,
        filename="geometry_analysis.json",
        data=report,
    )

    # --------------------------------------------------------
    # 10. Return
    # --------------------------------------------------------

    return {
        "report": report,
        "patch_labels": patch_labels,
        "json_path": json_path,
    }


# ============================================================
# TESTS
# ============================================================

def run_stage1_unit_tests():

    print("=" * 65)
    print("STAGE 1 UNIT TESTS")
    print("=" * 65)

    mesh = trimesh.creation.box(
        extents=[10.0, 20.0, 30.0]
    )

    # Test 1: Unknown unit
    mesh.metadata["source_unit"] = "unknown"

    assert _get_source_unit(mesh) == "unknown"

    print("[PASS] Unknown source unit detected")

    # Test 2: STL metadata
    mesh.metadata["source_unit"] = "cm"

    assert _get_source_unit(mesh) == "cm"

    print("[PASS] STL metadata unit")

    # Test 3: Manual override
    mesh.metadata["source_unit"] = "mm"

    assert (
        _get_source_unit(mesh, "inch")
        == "in"
    )

    print("[PASS] Manual unit override")

    # Test 4: Manual unknown
    assert (
        _get_source_unit(mesh, "unknown")
        == "unknown"
    )

    print("[PASS] Manual unknown unit")

    # Test 5: Invalid unit
    try:
        _get_source_unit(mesh, "yard")

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Unsupported unit was not rejected."
        )

    print("[PASS] Unsupported unit validation")

    # Test 6: Coordinates unchanged
    original_vertices = mesh.vertices.copy()

    _get_source_unit(mesh, "cm")

    assert np.array_equal(
        original_vertices,
        mesh.vertices,
    )

    print("[PASS] Source coordinates unchanged")

    # Test 7: Unit labels
    units = _get_measurement_units("cm")

    assert units["coordinate_unit"] == "cm"
    assert units["surface_area_unit"] == "cm^2"
    assert units["volume_unit"] == "cm^3"

    print("[PASS] Measurement unit labels")

    # Test 8: Unknown unit must be rejected
    assert _get_source_unit(
        mesh,
        "unknown",
    ) == "unknown"

    print("[PASS] Unknown unit cannot be used for analysis")

    print("=" * 65)
    print("ALL STAGE 1 UNIT TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_stage1_unit_tests()
