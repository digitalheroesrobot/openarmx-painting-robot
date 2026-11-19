
from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import trimesh

from core.result_manager import ResultManager


# ============================================================
# STAGE 2 CONFIGURATION
# ============================================================

MIN_PATCH_AREA_MM2 = 100.0

MIN_NORMAL_COHERENCE = 0.85
INSPECTION_NORMAL_COHERENCE = 0.70

REVIEW_INTERNAL_CYLINDRICAL_SURFACES = True


# ============================================================
# UNIT CONVERSION
# ============================================================

# Millimeters represented by one STL coordinate unit.
UNIT_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
}


def _normalize_unit(unit: Any) -> str:
    if unit is None:
        return "unknown"

    value = str(unit).strip().lower()

    aliases = {
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "inch": "inch",
        "inches": "inch",
        "foot": "ft",
        "feet": "ft",
    }

    value = aliases.get(value, value)

    if value not in UNIT_TO_MM:
        return "unknown"

    return value


def _get_coordinate_unit(stage1_report: dict) -> str:
    """
    Resolve the original STL coordinate unit.

    Priority:
        1. Stage 1 unit_configuration
        2. Stage 1 analysis metadata
        3. Top-level source_unit
        4. Mesh metadata

    Unknown is never replaced with mm.
    """

    if not isinstance(stage1_report, dict):
        return "unknown"

    candidates = []

    unit_configuration = stage1_report.get(
        "unit_configuration"
    )

    if isinstance(unit_configuration, dict):
        candidates.extend([
            unit_configuration.get(
                "source_stl_coordinate_unit"
            ),
            unit_configuration.get("coordinate_unit"),
            unit_configuration.get("source_unit"),
        ])

    analysis = stage1_report.get("analysis")

    if isinstance(analysis, dict):
        candidates.extend([
            analysis.get("coordinate_unit"),
            analysis.get("source_unit"),
        ])

    candidates.append(stage1_report.get("source_unit"))

    mesh_metadata = stage1_report.get("mesh")

    if isinstance(mesh_metadata, dict):
        candidates.extend([
            mesh_metadata.get("coordinate_unit"),
            mesh_metadata.get("source_unit"),
        ])

    for candidate in candidates:
        normalized = _normalize_unit(candidate)

        if normalized != "unknown":
            return normalized

    return "unknown"


def _get_area_threshold_in_coordinate_units(
    coordinate_unit: str,
    minimum_area_mm2: float,
) -> float | None:

    factor = UNIT_TO_MM.get(coordinate_unit)

    if factor is None:
        return None

    return float(minimum_area_mm2 / (factor ** 2))


def _area_to_mm2(
    area: float | None,
    coordinate_unit: str,
) -> float | None:

    if area is None:
        return None

    factor = UNIT_TO_MM.get(coordinate_unit)

    if factor is None:
        return None

    return float(area * factor ** 2)


def _length_to_mm(
    length: float | None,
    coordinate_unit: str,
) -> float | None:

    if length is None:
        return None

    factor = UNIT_TO_MM.get(coordinate_unit)

    if factor is None:
        return None

    return float(length * factor)


# ============================================================
# FACE STATUS
# ============================================================

FACE_EXCLUDED = 0
FACE_PAINTABLE_CANDIDATE = 1
FACE_REQUIRES_INSPECTION = 2
FACE_INVALID = 3


CLASSIFICATION_NAMES = {
    FACE_EXCLUDED: "excluded",
    FACE_PAINTABLE_CANDIDATE: "paintable_candidate",
    FACE_REQUIRES_INSPECTION: "requires_inspection",
    FACE_INVALID: "invalid",
}


# ============================================================
# NUMERICAL UTILITIES
# ============================================================

def _safe_float(value: Any) -> float | None:

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _json_safe(value: Any):
    """Convert NumPy values and non-finite numbers to JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ============================================================
# PATCH RESULT
# ============================================================

def _make_patch_result(
    patch_id: int,
    geometry_class: str,
    status: str,
    face_status_value: int,
    area: float | None,
    coherence: float | None,
    max_deviation: float | None,
    reasons: list[str],
    coordinate_unit: str,
    advanced_features: dict | None = None,
) -> dict:

    return {
        "patch_id": int(patch_id),
        "geometry_class": geometry_class,
        "surface_status": status,
        "face_status_value": int(face_status_value),

        "area_coordinate_units2": area,
        "max_plane_deviation_coordinate_units": max_deviation,

        "area_mm2": _area_to_mm2(
            area,
            coordinate_unit,
        ),

        "max_plane_deviation_mm": _length_to_mm(
            max_deviation,
            coordinate_unit,
        ),

        "normal_coherence": coherence,
        "coordinate_unit": coordinate_unit,
        "advanced_features": advanced_features or {},
        "reasons": reasons,
    }


# ============================================================
# ADVANCED PATCH FEATURES
# ============================================================

def _get_advanced_patch_features(
    patch_id: int,
    stage1_report: dict,
    coordinate_unit: str,
) -> dict:

    segmentation = stage1_report.get(
        "surface_segmentation",
        {},
    )

    patches = segmentation.get("patches", [])

    patch_data = next(
        (
            item
            for item in patches
            if int(item.get("patch_id", -1)) == patch_id
        ),
        {},
    )

    cylinder = patch_data.get("cylindrical_fit", {})

    cylindrical_candidate = (
        cylinder.get("status") == "candidate"
    )

    surface_type = cylinder.get("surface_type")

    is_internal_cylinder = (
        cylindrical_candidate
        and surface_type == "internal_cylindrical_surface"
    )

    is_external_cylinder = (
        cylindrical_candidate
        and surface_type == "external_cylindrical_surface"
    )

    # Prefer explicitly named source-coordinate fields.
    # Stage 1 legacy *_mm fields may contain raw STL
    # coordinate values rather than physical millimeters.
    radius = _safe_float(
        cylinder.get("radius_coordinate_units")
    )

    diameter = _safe_float(
        cylinder.get("diameter_coordinate_units")
    )

    # Backward compatibility with the current Stage 1 schema:
    # legacy radius_mm / diameter_mm values are raw coordinates.
    if radius is None:
        radius = _safe_float(cylinder.get("radius_mm"))

    if diameter is None:
        diameter = _safe_float(cylinder.get("diameter_mm"))

    radius_mm = _length_to_mm(
        radius,
        coordinate_unit,
    )

    diameter_mm = _length_to_mm(
        diameter,
        coordinate_unit,
    )

    advanced_geometry = stage1_report.get(
        "advanced_geometry",
        {},
    )

    bend_section = advanced_geometry.get("bends", {})

    bend_candidates = []

    for bend in bend_section.get("candidates", []):

        patch_a = int(bend.get("patch_a", -1))
        patch_b = int(bend.get("patch_b", -1))

        if patch_id in (patch_a, patch_b):
            bend_candidates.append(bend)

    return {
        "cylindrical_surface_candidate": bool(
            cylindrical_candidate
        ),

        "cylindrical_surface_type": surface_type,

        "cylinder_radius_coordinate_units": radius,
        "cylinder_diameter_coordinate_units": diameter,

        "cylinder_radius_mm": radius_mm,
        "cylinder_diameter_mm": diameter_mm,

        "cylinder_fit_confidence": _safe_float(
            cylinder.get("confidence")
        ),

        "internal_cylindrical_surface": bool(
            is_internal_cylinder
        ),

        "external_cylindrical_surface": bool(
            is_external_cylinder
        ),

        "bend_candidate": bool(len(bend_candidates) > 0),
        "bend_candidates": bend_candidates,
    }


# ============================================================
# ADVANCED THICKNESS UNIT HANDLING
# ============================================================

def _normalize_thickness_summary(
    thickness_data: dict,
    coordinate_unit: str,
) -> dict:
    """
    Normalize Stage 1 thickness values.

    Stage 1 legacy keys such as minimum_mm, maximum_mm,
    mean_mm and median_mm may contain raw STL coordinate
    measurements. This function preserves the original
    values under coordinate-unit keys and calculates
    physical mm values separately.

    It does not modify the Stage 1 report.
    """

    if not isinstance(thickness_data, dict):
        return {}

    result = dict(thickness_data)

    field_names = (
        "minimum",
        "maximum",
        "mean",
        "median",
    )

    for name in field_names:

        coordinate_key = f"{name}_coordinate_units"
        legacy_key = f"{name}_mm"

        coordinate_value = _safe_float(
            thickness_data.get(coordinate_key)
        )

        if coordinate_value is None:
            # Current Stage 1 legacy schema:
            # *_mm fields contain raw coordinate values.
            coordinate_value = _safe_float(
                thickness_data.get(legacy_key)
            )

        result[coordinate_key] = coordinate_value

        result[legacy_key] = _length_to_mm(
            coordinate_value,
            coordinate_unit,
        )

    result["coordinate_unit"] = coordinate_unit

    result["measurement_unit_status"] = (
        "converted_to_mm"
        if coordinate_unit != "unknown"
        else "unknown_source_unit"
    )

    return result


# ============================================================
# PATCH CLASSIFICATION
# ============================================================

def _classify_patch(
    patch: dict,
    min_area_coordinate_units2: float | None,
    min_normal_coherence: float,
    inspection_normal_coherence: float,
    excluded_patch_ids: set[int],
    coordinate_unit: str,
    advanced_features: dict | None = None,
) -> dict:

    patch_id = int(patch["patch_id"])

    geometry_class = str(
        patch.get("classification", "invalid")
    ).lower()

    area = _safe_float(patch.get("area"))
    coherence = _safe_float(patch.get("normal_coherence"))
    max_deviation = _safe_float(
        patch.get("max_plane_deviation")
    )

    advanced_features = advanced_features or {}

    if patch_id in excluded_patch_ids:
        return _make_patch_result(
            patch_id,
            geometry_class,
            "excluded",
            FACE_EXCLUDED,
            area,
            coherence,
            max_deviation,
            ["manually_excluded"],
            coordinate_unit,
            advanced_features,
        )

    if geometry_class == "invalid":
        return _make_patch_result(
            patch_id,
            geometry_class,
            "invalid",
            FACE_INVALID,
            area,
            coherence,
            max_deviation,
            ["invalid_stage1_geometry"],
            coordinate_unit,
            advanced_features,
        )

    if area is None or area <= 0:
        return _make_patch_result(
            patch_id,
            geometry_class,
            "invalid",
            FACE_INVALID,
            area,
            coherence,
            max_deviation,
            ["invalid_or_zero_area"],
            coordinate_unit,
            advanced_features,
        )

    if geometry_class not in ("planar", "curved"):
        return _make_patch_result(
            patch_id,
            geometry_class,
            "requires_inspection",
            FACE_REQUIRES_INSPECTION,
            area,
            coherence,
            max_deviation,
            ["unknown_geometry_class"],
            coordinate_unit,
            advanced_features,
        )

    if coordinate_unit == "unknown":
        return _make_patch_result(
            patch_id,
            geometry_class,
            "requires_inspection",
            FACE_REQUIRES_INSPECTION,
            area,
            coherence,
            max_deviation,
            [
                "coordinate_unit_unknown",
                "coordinate_unit_unknown_area_threshold_not_applied",
            ],
            coordinate_unit,
            advanced_features,
        )

    if min_area_coordinate_units2 is not None:
        if area < min_area_coordinate_units2:
            return _make_patch_result(
                patch_id,
                geometry_class,
                "requires_inspection",
                FACE_REQUIRES_INSPECTION,
                area,
                coherence,
                max_deviation,
                [
                    "patch_area_below_minimum",
                    (
                        "minimum_area_coordinate_units2="
                        f"{min_area_coordinate_units2}"
                    ),
                ],
                coordinate_unit,
                advanced_features,
            )

    if geometry_class == "planar":

        if coherence is None:
            return _make_patch_result(
                patch_id,
                geometry_class,
                "requires_inspection",
                FACE_REQUIRES_INSPECTION,
                area,
                coherence,
                max_deviation,
                ["planar_normal_coherence_unavailable"],
                coordinate_unit,
                advanced_features,
            )

        if coherence < inspection_normal_coherence:
            return _make_patch_result(
                patch_id,
                geometry_class,
                "requires_inspection",
                FACE_REQUIRES_INSPECTION,
                area,
                coherence,
                max_deviation,
                ["very_low_planar_normal_coherence"],
                coordinate_unit,
                advanced_features,
            )

        if coherence < min_normal_coherence:
            return _make_patch_result(
                patch_id,
                geometry_class,
                "requires_inspection",
                FACE_REQUIRES_INSPECTION,
                area,
                coherence,
                max_deviation,
                ["low_planar_normal_coherence"],
                coordinate_unit,
                advanced_features,
            )

        status = "paintable_candidate"
        face_status = FACE_PAINTABLE_CANDIDATE
        reasons = ["valid_planar_candidate"]

    else:
        status = "paintable_candidate"
        face_status = FACE_PAINTABLE_CANDIDATE
        reasons = [
            "valid_curved_candidate",
            "global_normal_coherence_not_used_as_rejection",
        ]

    if (
        REVIEW_INTERNAL_CYLINDRICAL_SURFACES
        and advanced_features.get(
            "internal_cylindrical_surface",
            False,
        )
    ):
        status = "requires_inspection"
        face_status = FACE_REQUIRES_INSPECTION
        reasons.append(
            "internal_cylindrical_surface_requires_review"
        )

    if advanced_features.get("bend_candidate", False):
        reasons.append("bend_geometry_candidate")

    if advanced_features.get(
        "cylindrical_surface_candidate",
        False,
    ):
        reasons.append("cylindrical_surface_detected")

    return _make_patch_result(
        patch_id,
        geometry_class,
        status,
        face_status,
        area,
        coherence,
        max_deviation,
        reasons,
        coordinate_unit,
        advanced_features,
    )


# ============================================================
# ADVANCED GEOMETRY SUMMARY
# ============================================================

def _build_advanced_summary(
    stage1_report: dict,
    classified_patches: list[dict],
    coordinate_unit: str,
) -> dict:

    advanced = stage1_report.get("advanced_geometry", {})

    edge_data = advanced.get("edges", {})
    cylinder_data = advanced.get("cylindrical_features", {})
    hole_data = advanced.get("hole_candidates", {})
    bend_data = advanced.get("bends", {})
    component_data = advanced.get("components", {})
    thickness_data = advanced.get("thickness", {})

    internal_surface_count = sum(
        bool(
            patch.get("advanced_features", {}).get(
                "internal_cylindrical_surface",
                False,
            )
        )
        for patch in classified_patches
    )

    bend_patch_count = sum(
        bool(
            patch.get("advanced_features", {}).get(
                "bend_candidate",
                False,
            )
        )
        for patch in classified_patches
    )

    normalized_thickness = _normalize_thickness_summary(
        thickness_data,
        coordinate_unit,
    )

    return {
        "source": "stage1_geometry_analysis",

        "coordinate_unit": coordinate_unit,

        "boundary_edge_count": edge_data.get(
            "boundary_edge_count"
        ),

        "sharp_edge_count": edge_data.get(
            "sharp_edge_count"
        ),

        "cylindrical_feature_count": cylinder_data.get(
            "count"
        ),

        "hole_candidate_count": hole_data.get("count"),

        "bend_candidate_count": bend_data.get(
            "bend_candidate_count"
        ),

        "connected_component_count": component_data.get(
            "component_count"
        ),

        "thickness": normalized_thickness,

        "patch_review": {
            "internal_cylindrical_patch_count": int(
                internal_surface_count
            ),

            "bend_related_patch_count": int(
                bend_patch_count
            ),
        },

        "interpretation": {
            "holes_are_automatically_excluded": False,
            "bend_surfaces_are_automatically_excluded": False,
            "thickness_is_used_for_automatic_rejection": False,

            "reason": (
                "Advanced geometry features are retained for "
                "process review. Automatic exclusion requires "
                "explicit painting process rules."
            ),
        },
    }


# ============================================================
# MAIN IDENTIFICATION FUNCTION
# ============================================================

def identify_paintable_surfaces(
    mesh: trimesh.Trimesh,
    stage1_report: dict,
    patch_labels: np.ndarray,
    min_patch_area_mm2: float = MIN_PATCH_AREA_MM2,
    min_normal_coherence: float = MIN_NORMAL_COHERENCE,
    inspection_normal_coherence: float = INSPECTION_NORMAL_COHERENCE,
    excluded_patch_ids: set[int] | None = None,
    coordinate_unit: str | None = None,
) -> dict:

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            "mesh must be a trimesh.Trimesh instance."
        )

    if not isinstance(stage1_report, dict):
        raise TypeError(
            "stage1_report must be a dictionary."
        )

    minimum_area = _safe_float(min_patch_area_mm2)

    if minimum_area is None:
        raise ValueError(
            "min_patch_area_mm2 must be finite."
        )

    if minimum_area < 0:
        raise ValueError(
            "min_patch_area_mm2 cannot be negative."
        )

    minimum_coherence = _safe_float(min_normal_coherence)
    inspection_coherence = _safe_float(
        inspection_normal_coherence
    )

    if (
        minimum_coherence is None
        or inspection_coherence is None
        or not (
            0.0
            <= inspection_coherence
            <= minimum_coherence
            <= 1.0
        )
    ):
        raise ValueError(
            "Normal coherence thresholds must satisfy "
            "0 <= inspection <= minimum <= 1."
        )

    # Resolve coordinate unit.
    if coordinate_unit is not None:
        resolved_unit = _normalize_unit(coordinate_unit)
    else:
        resolved_unit = _get_coordinate_unit(stage1_report)

    min_area_coordinate_units2 = (
        _get_area_threshold_in_coordinate_units(
            resolved_unit,
            minimum_area,
        )
    )

    # Validate labels.
    patch_labels = np.asarray(
        patch_labels,
        dtype=np.int32,
    )

    if patch_labels.ndim != 1:
        raise ValueError(
            "patch_labels must be one-dimensional."
        )

    if len(patch_labels) != len(mesh.faces):
        raise ValueError(
            "patch_labels length does not match mesh faces."
        )

    if len(patch_labels) == 0:
        raise ValueError(
            "Mesh contains no triangle faces."
        )

    if np.any(patch_labels < 0):
        raise ValueError(
            "patch_labels contains negative values."
        )

    # Retrieve Stage 1 patches.
    stage1_patches = (
        stage1_report
        .get("surface_segmentation", {})
        .get("patches", [])
    )

    if not stage1_patches:
        raise ValueError(
            "Stage 1 report contains no surface patches."
        )

    report_patch_ids = {
        int(patch["patch_id"])
        for patch in stage1_patches
    }

    mesh_patch_ids = set(
        int(value)
        for value in np.unique(patch_labels)
    )

    if report_patch_ids != mesh_patch_ids:
        raise ValueError(
            "Stage 1 patch IDs do not match patch_labels."
        )

    if len(report_patch_ids) != len(stage1_patches):
        raise ValueError(
            "Duplicate patch IDs found in Stage 1 report."
        )

    excluded_patch_ids = set(
        int(value)
        for value in (excluded_patch_ids or set())
    )

    unknown_exclusions = (
        excluded_patch_ids - report_patch_ids
    )

    if unknown_exclusions:
        raise ValueError(
            f"Unknown excluded patch IDs: {sorted(unknown_exclusions)}"
        )

    # Classify patches.
    classified_patches = []
    patch_status_lookup = {}

    for patch in stage1_patches:

        patch_id = int(patch["patch_id"])

        advanced_features = _get_advanced_patch_features(
            patch_id,
            stage1_report,
            resolved_unit,
        )

        result = _classify_patch(
            patch=patch,

            min_area_coordinate_units2=(
                min_area_coordinate_units2
            ),

            min_normal_coherence=float(
                minimum_coherence
            ),

            inspection_normal_coherence=float(
                inspection_coherence
            ),

            excluded_patch_ids=excluded_patch_ids,

            coordinate_unit=resolved_unit,

            advanced_features=advanced_features,
        )

        classified_patches.append(result)

        patch_status_lookup[
            result["patch_id"]
        ] = result["face_status_value"]

    # Convert patch classification to face labels.
    face_status_labels = np.full(
        len(mesh.faces),
        FACE_INVALID,
        dtype=np.uint8,
    )

    for patch_id, status_value in patch_status_lookup.items():

        face_mask = patch_labels == patch_id

        face_status_labels[face_mask] = np.uint8(
            status_value
        )

    # Patch counts.
    patch_counts = {
        "paintable_candidate_patches": 0,
        "planar_candidate_patches": 0,
        "curved_candidate_patches": 0,
        "excluded_patches": 0,
        "requires_inspection_patches": 0,
        "invalid_patches": 0,
    }

    for patch in classified_patches:

        status = patch["surface_status"]
        geometry = patch["geometry_class"]

        if status == "paintable_candidate":

            patch_counts["paintable_candidate_patches"] += 1

            if geometry == "planar":
                patch_counts["planar_candidate_patches"] += 1

            elif geometry == "curved":
                patch_counts["curved_candidate_patches"] += 1

        elif status == "excluded":
            patch_counts["excluded_patches"] += 1

        elif status == "requires_inspection":
            patch_counts["requires_inspection_patches"] += 1

        elif status == "invalid":
            patch_counts["invalid_patches"] += 1

    # Face counts.
    face_counts = {
        "paintable_candidate_faces": int(
            np.count_nonzero(
                face_status_labels == FACE_PAINTABLE_CANDIDATE
            )
        ),

        "excluded_faces": int(
            np.count_nonzero(
                face_status_labels == FACE_EXCLUDED
            )
        ),

        "requires_inspection_faces": int(
            np.count_nonzero(
                face_status_labels == FACE_REQUIRES_INSPECTION
            )
        ),

        "invalid_faces": int(
            np.count_nonzero(
                face_status_labels == FACE_INVALID
            )
        ),
    }

    planar_candidate_ids = {
        patch["patch_id"]
        for patch in classified_patches
        if (
            patch["geometry_class"] == "planar"
            and patch["surface_status"] == "paintable_candidate"
        )
    }

    curved_candidate_ids = {
        patch["patch_id"]
        for patch in classified_patches
        if (
            patch["geometry_class"] == "curved"
            and patch["surface_status"] == "paintable_candidate"
        )
    }

    planar_candidate_faces = int(
        np.count_nonzero(
            np.isin(
                patch_labels,
                list(planar_candidate_ids),
            )
        )
    )

    curved_candidate_faces = int(
        np.count_nonzero(
            np.isin(
                patch_labels,
                list(curved_candidate_ids),
            )
        )
    )

    # Candidate surface area.
    candidate_area_coordinate_units2 = float(
        sum(
            patch["area_coordinate_units2"]
            for patch in classified_patches
            if (
                patch["surface_status"] == "paintable_candidate"
                and patch["area_coordinate_units2"] is not None
            )
        )
    )

    candidate_area_mm2 = _area_to_mm2(
        candidate_area_coordinate_units2,
        resolved_unit,
    )

    advanced_summary = _build_advanced_summary(
        stage1_report,
        classified_patches,
        resolved_unit,
    )

    if resolved_unit == "unknown":

        area_filter_status = (
            "not_applied_unknown_coordinate_unit"
        )

        unit_warning = (
            "The STL coordinate unit is unknown. The physical "
            "minimum patch area threshold in mm^2 was not applied. "
            "Select or confirm the source coordinate unit before "
            "using physical dimensions for production planning."
        )

    else:
        area_filter_status = "applied"
        unit_warning = None

    report = {
        "analysis": {
            "stage": 2,
            "name": "Paintable Surface Identification",

            "method": (
                "Stage 1 patch-based geometric classification "
                "with advanced geometry integration"
            ),

            "description": (
                "Identifies geometrically suitable planar and "
                "curved surface candidates for spray painting."
            ),

            "coordinate_unit": resolved_unit,

            "robot_reachability_checked": False,
            "spray_accessibility_checked": False,
            "fixture_interference_checked": False,
            "collision_clearance_checked": False,
            "standoff_distance_checked": False,
            "spray_orientation_checked": False,

            "unit_warning": unit_warning,

            "limitations": (
                "A paintable candidate is not proof of actual "
                "spray accessibility or robot reachability. "
                "Visibility, occlusion, spray gun orientation, "
                "standoff distance, collision clearance and "
                "fixture interference must be evaluated later."
            ),
        },

        "unit_configuration": {
            "source_stl_coordinate_unit": resolved_unit,

            "patch_area_coordinate_unit": (
                f"{resolved_unit}^2"
                if resolved_unit != "unknown"
                else "unknown"
            ),

            "physical_area_reference_unit": "mm^2",

            "minimum_patch_area_mm2": float(minimum_area),

            "minimum_patch_area_coordinate_units2": (
                min_area_coordinate_units2
            ),

            "area_filter_status": area_filter_status,

            "source_mesh_coordinates_modified": False,
        },

        "configuration": {
            "minimum_patch_area_mm2": float(minimum_area),

            "minimum_normal_coherence": float(
                minimum_coherence
            ),

            "inspection_normal_coherence": float(
                inspection_coherence
            ),

            "excluded_patch_ids": sorted(
                excluded_patch_ids
            ),

            "curved_surface_policy": (
                "Curved patches are not rejected based on "
                "global normal coherence."
            ),

            "internal_cylindrical_surface_policy": (
                "Internal cylindrical surface candidates "
                "require review and are not automatically excluded."
            ),
        },

        "summary": {
            **patch_counts,
            **face_counts,

            "planar_candidate_faces": planar_candidate_faces,
            "curved_candidate_faces": curved_candidate_faces,

            "total_candidate_area_coordinate_units2": (
                candidate_area_coordinate_units2
            ),

            "total_candidate_area_mm2": candidate_area_mm2,

            "coordinate_unit": resolved_unit,

            "total_patches": int(len(classified_patches)),
            "total_faces": int(len(mesh.faces)),
        },

        "advanced_geometry": advanced_summary,

        "classification_legend": {
            "0": "excluded",
            "1": "paintable_candidate",
            "2": "requires_inspection",
            "3": "invalid",
        },

        "patches": classified_patches,
    }

    report = _json_safe(report)

    return {
        "report": report,
        "face_status_labels": face_status_labels,
        "patch_labels": patch_labels.copy(),
    }


# ============================================================
# STAGE 2 EXECUTION
# ============================================================

def execute_stage2(
    mesh: trimesh.Trimesh,
    stl_path: str | Path,
    result_manager: ResultManager,
    stage1_result: dict,
    min_patch_area_mm2: float = MIN_PATCH_AREA_MM2,
    min_normal_coherence: float = MIN_NORMAL_COHERENCE,
    inspection_normal_coherence: float = INSPECTION_NORMAL_COHERENCE,
    excluded_patch_ids: set[int] | None = None,
    coordinate_unit: str | None = None,
) -> dict:

    if not isinstance(stage1_result, dict):
        raise TypeError(
            "stage1_result must be a dictionary."
        )

    if "report" not in stage1_result:
        raise ValueError(
            "stage1_result is missing 'report'."
        )

    if "patch_labels" not in stage1_result:
        raise ValueError(
            "stage1_result is missing 'patch_labels'."
        )

    result = identify_paintable_surfaces(
        mesh=mesh,
        stage1_report=stage1_result["report"],
        patch_labels=stage1_result["patch_labels"],
        min_patch_area_mm2=min_patch_area_mm2,
        min_normal_coherence=min_normal_coherence,
        inspection_normal_coherence=inspection_normal_coherence,
        excluded_patch_ids=excluded_patch_ids,
        coordinate_unit=coordinate_unit,
    )

    report = result["report"]

    report["source_file"] = str(
        Path(stl_path).resolve()
    )

    report["part_name"] = (
        result_manager.part_name_from_stl(stl_path)
    )

    json_path = result_manager.save_json(
        stl_path=stl_path,
        stage_number=2,
        filename="surface_identification.json",
        data=report,
    )

    stage_directory = Path(
        result_manager.stage_dir(
            stl_path=stl_path,
            stage_number=2,
        )
    )

    stage_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_path = (
        stage_directory / "face_status_labels.npy"
    )

    np.save(
        labels_path,
        result["face_status_labels"],
    )

    result["json_path"] = json_path
    result["labels_path"] = labels_path

    return result


# ============================================================
# NUMERICAL TESTS
# ============================================================

def run_stage2_numerical_tests():

    from core.geometry_analyzer import analyze_geometry

    print("=" * 65)
    print("STAGE 2 NUMERICAL TESTS")
    print("=" * 65)

    # Test 1: Basic classification.
    mesh = trimesh.creation.box(
        extents=[100.0, 100.0, 10.0]
    )

    stage1_report, patch_labels = analyze_geometry(mesh)

    stage1_report.setdefault(
        "unit_configuration",
        {},
    )

    # Explicit test metadata, not an assumption about STL files.
    stage1_report["unit_configuration"][
        "source_stl_coordinate_unit"
    ] = "mm"

    result = identify_paintable_surfaces(
        mesh=mesh,
        stage1_report=stage1_report,
        patch_labels=patch_labels,
    )

    assert len(result["face_status_labels"]) == len(mesh.faces)

    assert (
        result["report"]["summary"]["total_faces"]
        == len(mesh.faces)
    )

    print("[PASS] Basic Stage 2 classification")

    # Test 2: Manual exclusion.
    excluded_id = int(patch_labels[0])

    result = identify_paintable_surfaces(
        mesh=mesh,
        stage1_report=stage1_report,
        patch_labels=patch_labels,
        excluded_patch_ids={excluded_id},
    )

    assert np.all(
        result["face_status_labels"][
            patch_labels == excluded_id
        ] == FACE_EXCLUDED
    )

    print("[PASS] Manual patch exclusion")

    # Test 3: Advanced geometry integration.
    synthetic_report = {
        "unit_configuration": {
            "source_stl_coordinate_unit": "mm",
        },

        "surface_segmentation": {
            "patches": stage1_report[
                "surface_segmentation"
            ]["patches"],
        },

        "advanced_geometry": {
            "cylindrical_features": {
                "count": 1,
                "features": [],
            },

            "hole_candidates": {
                "count": 1,
                "features": [],
            },

            "bends": {
                "bend_candidate_count": 1,
                "candidates": [
                    {
                        "patch_a": int(patch_labels[0]),
                        "patch_b": int(patch_labels[-1]),
                        "mean_dihedral_angle_deg": 90.0,
                    }
                ],
            },

            "components": {
                "component_count": 1,
            },

            "thickness": {
                "status": "estimated",
                "mean_mm": 10.0,
            },

            "edges": {
                "boundary_edge_count": 0,
                "sharp_edge_count": 12,
            },
        },
    }

    result = identify_paintable_surfaces(
        mesh=mesh,
        stage1_report=synthetic_report,
        patch_labels=patch_labels,
    )

    advanced = result["report"]["advanced_geometry"]

    assert advanced["bend_candidate_count"] == 1
    assert advanced["sharp_edge_count"] == 12
    assert advanced["thickness"]["status"] == "estimated"

    # Legacy Stage 1 thickness value is treated as raw coordinates.
    assert np.isclose(
        advanced["thickness"]["mean_coordinate_units"],
        10.0,
    )

    assert np.isclose(
        advanced["thickness"]["mean_mm"],
        10.0,
    )

    print("[PASS] Advanced geometry integration")

    # Test 4: Unknown unit.
    legacy_report = {
        "surface_segmentation": {
            "patches": stage1_report[
                "surface_segmentation"
            ]["patches"],
        }
    }

    result = identify_paintable_surfaces(
        mesh=mesh,
        stage1_report=legacy_report,
        patch_labels=patch_labels,
    )

    assert (
        result["report"]["analysis"]["coordinate_unit"]
        == "unknown"
    )

    assert (
        result["report"]["unit_configuration"]["area_filter_status"]
        == "not_applied_unknown_coordinate_unit"
    )

    assert (
        result["report"]["summary"]["total_candidate_area_mm2"]
        is None
    )

    print("[PASS] Unknown-unit handling")

    # Test 5: cm conversion.
    test_mesh = trimesh.creation.box(
        extents=[1.0, 1.0, 1.0]
    )

    test_labels = np.zeros(
        len(test_mesh.faces),
        dtype=np.int32,
    )

    test_report = {
        "unit_configuration": {
            "source_stl_coordinate_unit": "cm",
        },

        "surface_segmentation": {
            "patches": [
                {
                    "patch_id": 0,
                    "classification": "planar",
                    "area": 2.0,
                    "normal_coherence": 1.0,
                    "max_plane_deviation": 0.0,
                }
            ]
        },
    }

    test_result = identify_paintable_surfaces(
        mesh=test_mesh,
        stage1_report=test_report,
        patch_labels=test_labels,
    )

    test_unit_config = (
        test_result["report"]["unit_configuration"]
    )

    assert (
        test_unit_config["source_stl_coordinate_unit"]
        == "cm"
    )

    assert np.isclose(
        test_unit_config[
            "minimum_patch_area_coordinate_units2"
        ],
        1.0,
    )

    assert np.isclose(
        test_result["report"]["patches"][0]["area_mm2"],
        200.0,
    )

    assert (
        test_result["report"]["patches"][0]["surface_status"]
        == "paintable_candidate"
    )

    print("[PASS] cm coordinate unit and area conversion")

    # Test 6: Inch conversion.
    inch_threshold = _get_area_threshold_in_coordinate_units(
        "inch",
        100.0,
    )

    assert np.isclose(
        inch_threshold,
        100.0 / (25.4 ** 2),
    )

    print("[PASS] inch coordinate unit conversion")

    # Test 7: Source mesh coordinates unchanged.
    original_vertices = test_mesh.vertices.copy()

    identify_paintable_surfaces(
        mesh=test_mesh,
        stage1_report=test_report,
        patch_labels=test_labels,
    )

    assert np.array_equal(
        original_vertices,
        test_mesh.vertices,
    )

    print("[PASS] Source mesh coordinates unchanged")

    # Test 8: JSON serialization.
    json.dumps(
        test_result["report"],
        allow_nan=False,
    )

    print("[PASS] JSON serialization")

    # Test 9: Cylinder values are converted from source units.
    cylinder_report = {
        "unit_configuration": {
            "source_stl_coordinate_unit": "cm",
        },

        "surface_segmentation": {
            "patches": [
                {
                    "patch_id": 0,
                    "classification": "curved",
                    "area": 2.0,
                    "normal_coherence": 1.0,
                    "max_plane_deviation": 0.0,
                    "cylindrical_fit": {
                        "status": "candidate",
                        "surface_type": "external_cylindrical_surface",
                        "radius_coordinate_units": 2.0,
                        "diameter_coordinate_units": 4.0,
                        "confidence": 0.95,
                    },
                }
            ]
        },
    }

    cylinder_result = identify_paintable_surfaces(
        mesh=test_mesh,
        stage1_report=cylinder_report,
        patch_labels=test_labels,
    )

    cylinder_features = (
        cylinder_result["report"]["patches"][0][
            "advanced_features"
        ]
    )

    assert np.isclose(
        cylinder_features["cylinder_radius_coordinate_units"],
        2.0,
    )

    assert np.isclose(
        cylinder_features["cylinder_radius_mm"],
        20.0,
    )

    assert np.isclose(
        cylinder_features["cylinder_diameter_mm"],
        40.0,
    )

    print("[PASS] Cylinder coordinate-to-mm conversion")

    # Test 10: Thickness values are converted from source units.
    thickness_result = _normalize_thickness_summary(
        {
            "minimum_mm": 0.5,
            "maximum_mm": 1.5,
            "mean_mm": 1.0,
            "median_mm": 1.0,
        },
        "cm",
    )

    assert np.isclose(
        thickness_result["mean_coordinate_units"],
        1.0,
    )

    assert np.isclose(
        thickness_result["mean_mm"],
        10.0,
    )

    print("[PASS] Thickness coordinate-to-mm conversion")

    # Test 11: Unknown unit must not create physical measurements.
    unknown_thickness = _normalize_thickness_summary(
        {"mean_mm": 1.0},
        "unknown",
    )

    assert unknown_thickness["mean_coordinate_units"] == 1.0
    assert unknown_thickness["mean_mm"] is None

    print("[PASS] Unknown-unit advanced measurements")

    print("=" * 65)
    print("ALL STAGE 2 TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_stage2_numerical_tests()
