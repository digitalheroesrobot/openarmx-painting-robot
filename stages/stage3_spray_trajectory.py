
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from core.result_manager import ResultManager
from core.coating_process import validate_coating_process
from stages.human_like_spray_planner import (
    SweepPlanningConfig,
    plan_human_like_sweeps,
)
from stages.spray_footprint_optimizer import optimize_redundant_spray_points


# ============================================================
# STAGE 3: SPRAY TRAJECTORY PLANNING
# ============================================================
#
# PURPOSE
#   Generate initial spray trajectories from Stage 2 paintable
#   surfaces and incorporate Stage 1 advanced geometry data.
#
# UNIT POLICY
#   - STL vertex coordinates are never rescaled or modified.
#   - UI parameters are expressed in physical millimeters.
#   - Physical parameters are converted into STL coordinate units
#     only for internal geometric calculations.
#   - Unknown coordinate units are rejected.
#
# SUPPORTED COORDINATE UNITS
#   mm, cm, m, in, ft
#
# OUTPUT
#   trajectory_points.npy: exactly 16 columns (legacy-compatible)
#   spray_trajectory.json
#
# IMPORTANT
#   Advanced geometry features are estimates, not CAD truth.
#   Robot reachability, collision, coverage and coating thickness
#   are not verified here.
#
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DEFAULT_STANDOFF_MM = 200.0

# Keep Stage 3 validation identical to the sprayer-settings UI range.
MIN_STANDOFF_MM = 20.0
MAX_STANDOFF_MM = 1000.0

MAX_TOTAL_SAMPLES = 250_000

# API-only default used by numerical tests and callers that omit a profile.
# The application always passes its active saved sprayer profile explicitly.
DEFAULT_CIRCULAR_SPRAYER_PROFILE = {
    "pattern_type": "round_cone",
    "spray_cone_angle_deg": 45.0,
    "overlap_percent": 50.0,
}

NORMAL_EPSILON = 1e-12

# Conversion factor: millimeters per one STL coordinate unit.
MM_PER_UNIT = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "ft": 304.8,
}

UNIT_ALIASES = {
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
}


# ------------------------------------------------------------
# Stage 2 face labels
# ------------------------------------------------------------

FACE_EXCLUDED = 0
FACE_PAINTABLE_CANDIDATE = 1
FACE_REQUIRES_INSPECTION = 2
FACE_INVALID = 3


# ------------------------------------------------------------
# Unit handling
# ------------------------------------------------------------

def _normalize_unit(unit: Any) -> str:
    if unit is None:
        return "unknown"

    value = str(unit).strip().lower()

    if not value or value in ("unknown", "none", "null"):
        return "unknown"

    normalized = UNIT_ALIASES.get(value)

    return normalized if normalized is not None else "unknown"


def _resolve_coordinate_unit(
    mesh: trimesh.Trimesh,
    stage1_report: dict | None,
    coordinate_unit: str | None,
) -> str:
    """
    Resolve the STL coordinate unit.

    Priority:
      1. Explicit coordinate_unit argument
      2. Stage 1 unit_configuration
      3. Stage 1 analysis metadata
      4. Mesh metadata

    Unknown units are rejected. No size-based inference is performed.
    """

    candidates = []

    if coordinate_unit is not None:
        candidates.append(coordinate_unit)

    if isinstance(stage1_report, dict):

        unit_configuration = stage1_report.get(
            "unit_configuration", {}
        )

        if isinstance(unit_configuration, dict):
            candidates.extend([
                unit_configuration.get(
                    "source_stl_coordinate_unit"
                ),
                unit_configuration.get("coordinate_unit"),
                unit_configuration.get("source_unit"),
            ])

        analysis = stage1_report.get("analysis", {})

        if isinstance(analysis, dict):
            candidates.extend([
                analysis.get("coordinate_unit"),
                analysis.get("source_unit"),
            ])

    metadata = getattr(mesh, "metadata", {})

    if isinstance(metadata, dict):
        candidates.extend([
            metadata.get("source_unit"),
            metadata.get("coordinate_unit"),
        ])

    for candidate in candidates:
        unit = _normalize_unit(candidate)

        if unit != "unknown":
            return unit

    raise ValueError(
        "STL coordinate unit is unknown. "
        "Select the actual coordinate unit in Product Setup "
        "before executing Stage 3. "
        "Supported units: mm, cm, m, in, ft."
    )


def _mm_to_coordinate_units(
    value_mm: float,
    coordinate_unit: str,
) -> float:
    """Convert a physical millimeter value to STL coordinate units."""

    unit = _normalize_unit(coordinate_unit)

    if unit not in MM_PER_UNIT:
        raise ValueError(
            f"Unsupported coordinate unit: {coordinate_unit}"
        )

    return float(value_mm) / MM_PER_UNIT[unit]


def _coordinate_units_to_mm(
    value: float,
    coordinate_unit: str,
) -> float:
    """Convert an STL-coordinate distance to millimeters."""

    unit = _normalize_unit(coordinate_unit)

    if unit not in MM_PER_UNIT:
        raise ValueError(
            f"Unsupported coordinate unit: {coordinate_unit}"
        )

    return float(value) * MM_PER_UNIT[unit]


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _validate_parameters(standoff_mm: float) -> float:
    standoff = _safe_float(standoff_mm)
    if standoff is None:
        raise ValueError("Standoff distance must be finite.")
    if not MIN_STANDOFF_MM <= standoff <= MAX_STANDOFF_MM:
        raise ValueError(
            f"Standoff must be between {MIN_STANDOFF_MM:.1f} "
            f"and {MAX_STANDOFF_MM:.1f} mm."
        )
    return standoff


def _calculate_spray_footprint(
    sprayer_profile: dict | None,
    standoff_mm: float,
) -> dict:
    """Calculate a circular-cone footprint and overlap-derived point pitch."""
    profile = sprayer_profile if isinstance(sprayer_profile, dict) else {}
    angle = _safe_float(profile.get("spray_cone_angle_deg"))
    if angle is None:
        raise ValueError("Sprayer profile must define spray_cone_angle_deg.")
    if not 0.0 < angle <= 150.0:
        raise ValueError("Circular spray angle must be greater than 0 and at most 150 degrees.")

    overlap = _safe_float(profile.get("overlap_percent"))
    if overlap is None:
        raise ValueError("Sprayer profile must define overlap_percent.")
    if not 0.0 <= overlap < 100.0:
        raise ValueError("Track overlap must be in the range [0, 100).")

    distance = float(standoff_mm)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("Working distance must be a positive finite value.")

    diameter = 2.0 * distance * math.tan(math.radians(angle * 0.5))
    point_spacing = diameter * (1.0 - overlap / 100.0)
    return {
        "pattern_type": "round_cone",
        "included_angle_deg": float(angle),
        "working_distance_mm": distance,
        "width_mm": float(diameter),
        "thickness_mm": float(diameter),
        "diameter_mm": float(diameter),
        "overlap_percent": float(overlap),
        "nominal_track_spacing_mm": float(point_spacing),
        "nominal_point_spacing_mm": float(point_spacing),
        "thickness_status": "circular_cone_geometry",
        "track_spacing_status": "calculated_from_diameter_and_overlap",
        "model_limitations": [
            "Nominal circular geometric footprint; not a calibrated droplet-deposition model.",
            "Actual footprint depends on sprayer, pressure, fluid, working distance and coating conditions.",
        ],
    }


# ------------------------------------------------------------
# Stage 1 advanced geometry extraction
# ------------------------------------------------------------

def _extract_stage1_features(
    stage1_report: dict | None,
) -> dict:

    if not isinstance(stage1_report, dict):
        return {
            "available": False,
            "patch_features": {},
            "advanced_geometry": {},
            "summary": {},
        }

    segmentation = stage1_report.get(
        "surface_segmentation", {}
    )

    if not isinstance(segmentation, dict):
        segmentation = {}

    patches = segmentation.get("patches", [])

    if not isinstance(patches, list):
        patches = []

    patch_features = {}

    for patch in patches:

        if not isinstance(patch, dict):
            continue

        patch_id = _safe_int(patch.get("patch_id"))

        if patch_id is None:
            continue

        cylindrical_fit = patch.get("cylindrical_fit", {})

        if not isinstance(cylindrical_fit, dict):
            cylindrical_fit = {}

        # New fields are preferred.
        # Legacy *_mm fields are raw coordinate values in the
        # current Geometry Analyzer implementation.
        radius = _safe_float(
            cylindrical_fit.get("radius_coordinate_units")
        )

        if radius is None:
            radius = _safe_float(
                cylindrical_fit.get("radius_mm")
            )

        diameter = _safe_float(
            cylindrical_fit.get("diameter_coordinate_units")
        )

        if diameter is None:
            diameter = _safe_float(
                cylindrical_fit.get("diameter_mm")
            )

        residual = _safe_float(
            cylindrical_fit.get("fit_residual_coordinate_units")
        )

        if residual is None:
            residual = _safe_float(
                cylindrical_fit.get("fit_residual_mm")
            )

        patch_features[patch_id] = {
            "patch_id": patch_id,
            "surface_type": patch.get("surface_type"),
            "classification": patch.get("classification"),
            "cylindrical_fit": {
                "candidate": bool(
                    cylindrical_fit.get("candidate", False)
                ),
                "radius_coordinate_units": radius,
                "diameter_coordinate_units": diameter,
                "fit_residual_coordinate_units": residual,
                "confidence": _safe_float(
                    cylindrical_fit.get("confidence")
                ),
                "surface_type": cylindrical_fit.get(
                    "surface_type"
                ),
            },
        }

    advanced = stage1_report.get("advanced_geometry", {})

    if not isinstance(advanced, dict):
        advanced = {}

    def _list_count(key: str) -> int:
        value = advanced.get(key, [])
        return len(value) if isinstance(value, list) else 0

    return {
        "available": True,
        "patch_features": patch_features,
        "advanced_geometry": advanced,
        "summary": {
            "patch_count": len(patches),
            "cylindrical_feature_count": _list_count(
                "cylindrical_features"
            ),
            "hole_candidate_count": _list_count(
                "hole_candidates"
            ),
            "bend_count": _list_count("bends"),
            "component_count": _list_count("components"),
        },
    }


def _build_patch_feature_summary(
    patch_ids: np.ndarray,
    patch_features: dict,
) -> list[dict]:

    results = []

    for patch_id in np.unique(patch_ids):

        patch_id = int(patch_id)
        feature = patch_features.get(patch_id, {})
        cylinder = feature.get("cylindrical_fit", {})

        results.append({
            "patch_id": patch_id,
            "surface_type": feature.get("surface_type"),
            "classification": feature.get("classification"),
            "cylindrical_candidate": bool(
                cylinder.get("candidate", False)
            ),
            "cylinder_radius_coordinate_units": cylinder.get(
                "radius_coordinate_units"
            ),
            "cylinder_diameter_coordinate_units": cylinder.get(
                "diameter_coordinate_units"
            ),
            "cylinder_fit_residual_coordinate_units": cylinder.get(
                "fit_residual_coordinate_units"
            ),
            "cylinder_confidence": cylinder.get("confidence"),
            "cylinder_surface_type": cylinder.get("surface_type"),
        })

    return results


# ------------------------------------------------------------
# Face normals
# ------------------------------------------------------------

def _get_face_normals(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, dict]:

    normals = np.asarray(
        mesh.face_normals,
        dtype=np.float64,
    ).copy()

    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("Mesh face normals have an invalid shape.")

    normal_lengths = np.linalg.norm(normals, axis=1)

    valid = (
        np.all(np.isfinite(normals), axis=1)
        & np.isfinite(normal_lengths)
        & (normal_lengths > NORMAL_EPSILON)
    )

    normals[valid] /= normal_lengths[valid, None]
    normals[~valid] = np.nan

    return normals, {
        "total_faces": int(len(normals)),
        "valid_normal_faces": int(np.count_nonzero(valid)),
        "invalid_normal_faces": int(np.count_nonzero(~valid)),
        "normal_source": "trimesh_face_normals",
    }


def _determine_normal_orientation(
    mesh: trimesh.Trimesh,
    normals: np.ndarray,
) -> tuple[np.ndarray, dict]:

    result = normals.copy()

    try:
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = False

    try:
        winding_consistent = bool(mesh.is_winding_consistent)
    except Exception:
        winding_consistent = False

    signed_volume = float("nan")

    if watertight and winding_consistent:
        try:
            signed_volume = float(mesh.volume)
        except Exception:
            signed_volume = float("nan")

    reliable = (
        watertight
        and winding_consistent
        and math.isfinite(signed_volume)
        and abs(signed_volume) > NORMAL_EPSILON
    )

    inverted = False

    if reliable and signed_volume < 0.0:
        result = -result
        inverted = True

    return result, {
        "watertight": watertight,
        "winding_consistent": winding_consistent,
        "signed_volume_coordinate_units3": (
            signed_volume if math.isfinite(signed_volume) else None
        ),
        "orientation_reliable": bool(reliable),
        "normals_inverted_for_outward_direction": bool(inverted),
        "description": (
            "Normals are treated as outward."
            if reliable
            else
            "Outward direction is uncertain. Verify before robot execution."
        ),
    }


# ------------------------------------------------------------
# Product pose and provisional nozzle frame
# ------------------------------------------------------------

def _validate_product_transform(product_transform: Any = None) -> np.ndarray:
    """Validate a rigid CAD-to-World transform in source coordinate units."""
    if product_transform is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(product_transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Product transform must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("Product transform must be a homogeneous rigid transform.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Product transform rotation must be orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("Product transform rotation must have determinant +1.")
    return matrix.copy()


def _build_nozzle_frames(spray_directions: np.ndarray) -> np.ndarray:
    """Build provisional right-handed frames; columns X/Y/Z, Z points toward surface."""
    directions = np.asarray(spray_directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("Spray directions must have shape (N, 3).")
    frames = np.empty((len(directions), 3, 3), dtype=np.float64)
    product_x = np.array([1.0, 0.0, 0.0])
    product_y = np.array([0.0, 1.0, 0.0])
    for index, direction in enumerate(directions):
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= NORMAL_EPSILON:
            raise ValueError("Cannot build a nozzle frame from an invalid direction.")
        z_axis = direction / length
        projected = product_x - float(np.dot(product_x, z_axis)) * z_axis
        if np.linalg.norm(projected) <= 1e-8:
            projected = product_y - float(np.dot(product_y, z_axis)) * z_axis
        x_axis = projected / np.linalg.norm(projected)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        frames[index] = np.column_stack((x_axis, y_axis, z_axis))
    return frames


def _build_parallel_transport_nozzle_frames(
    spray_directions: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    """
    Build minimum-twist right-handed nozzle frames along each stroke.

    Z follows the requested spray direction. X is parallel-transported from
    the previous sample using the smallest rotation that aligns consecutive
    Z axes. Frames restart at every segment boundary; no roll continuity is
    imposed across spray-OFF travel.
    """
    directions = np.asarray(spray_directions, dtype=np.float64)
    segments = np.asarray(segment_ids, dtype=np.int64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("Spray directions must have shape (N, 3).")
    if segments.shape != (len(directions),):
        raise ValueError("Segment IDs must have shape (N,).")
    if not np.all(np.isfinite(directions)):
        raise ValueError("Spray directions must be finite.")

    frames = np.empty((len(directions), 3, 3), dtype=np.float64)
    previous_segment = None
    previous_frame = None

    for index, direction in enumerate(directions):
        length = float(np.linalg.norm(direction))
        if length <= NORMAL_EPSILON:
            raise ValueError("Cannot build a frame from a zero spray direction.")
        z_axis = direction / length

        if previous_frame is None or int(segments[index]) != previous_segment:
            frames[index] = _build_nozzle_frames(z_axis[None, :])[0]
        else:
            previous_z = previous_frame[:, 2]
            cross = np.cross(previous_z, z_axis)
            sine = float(np.linalg.norm(cross))
            cosine = float(np.clip(np.dot(previous_z, z_axis), -1.0, 1.0))

            if sine > 1e-10:
                axis = cross / sine
                skew = np.asarray([
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ], dtype=np.float64)
                rotation = (
                    np.eye(3, dtype=np.float64)
                    + sine * skew
                    + (1.0 - cosine) * (skew @ skew)
                )
                x_axis = rotation @ previous_frame[:, 0]
            elif cosine >= 0.0:
                x_axis = previous_frame[:, 0].copy()
            else:
                # At a 180-degree spray-axis reversal, the minimum rotation
                # axis is ambiguous. Rotate about the previous X axis so the
                # gun's roll reference remains deterministic.
                x_axis = previous_frame[:, 0].copy()

            x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
            x_length = float(np.linalg.norm(x_axis))
            if x_length <= 1e-10:
                fallback = previous_frame[:, 1]
                x_axis = np.cross(fallback, z_axis)
                x_length = float(np.linalg.norm(x_axis))
            x_axis /= max(x_length, NORMAL_EPSILON)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= max(float(np.linalg.norm(y_axis)), NORMAL_EPSILON)
            frames[index] = np.column_stack((x_axis, y_axis, z_axis))

        previous_frame = frames[index]
        previous_segment = int(segments[index])

    return frames


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def _transform_vectors(vectors: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float64) @ transform[:3, :3].T


# ------------------------------------------------------------
# Triangle sampling
# ------------------------------------------------------------

def _triangle_sample_points(
    triangle: np.ndarray,
    spacing_coordinate_units: float,
) -> np.ndarray:
    """Vectorized triangular-lattice sampling for one mesh triangle.

    The generated points are the same two sub-triangle centroids as the
    previous nested-loop implementation. The two centroid groups are
    concatenated; their ordering is not used as a geometric authority.
    """
    triangle = np.asarray(triangle, dtype=np.float64)
    if triangle.shape != (3, 3):
        raise ValueError("Triangle must have shape (3, 3).")
    if not np.all(np.isfinite(triangle)):
        return np.empty((0, 3), dtype=np.float64)
    if (
        not math.isfinite(spacing_coordinate_units)
        or spacing_coordinate_units <= 0.0
    ):
        raise ValueError("Sampling spacing must be positive and finite.")

    a, b, c = triangle
    edge_ab = float(np.linalg.norm(b - a))
    edge_bc = float(np.linalg.norm(c - b))
    edge_ca = float(np.linalg.norm(a - c))
    longest_edge = max(edge_ab, edge_bc, edge_ca)
    if not math.isfinite(longest_edge) or longest_edge <= NORMAL_EPSILON:
        return np.empty((0, 3), dtype=np.float64)

    n = max(1, int(math.ceil(longest_edge / spacing_coordinate_units)))
    i_grid, j_grid = np.indices((n, n), dtype=np.int64)
    valid = (i_grid + j_grid) < n
    i = i_grid[valid].astype(np.float64) / n
    j = j_grid[valid].astype(np.float64) / n

    # First sub-triangle centroid for every triangular lattice cell.
    first = (
        (1.0 - i - j - 2.0 / (3.0 * n))[:, None] * a
        + (i + 1.0 / (3.0 * n))[:, None] * b
        + (j + 1.0 / (3.0 * n))[:, None] * c
    )

    # Preserve the original per-cell ordering (first centroid, then optional
    # second centroid) while computing all coordinates in vectorized batches.
    second_mask = (i_grid[valid] + j_grid[valid]) < (n - 1)
    if np.any(second_mask):
        i2 = i[second_mask]
        j2 = j[second_mask]
        second = (
            (1.0 - i2 - j2 - 4.0 / (3.0 * n))[:, None] * a
            + (i2 + 2.0 / (3.0 * n))[:, None] * b
            + (j2 + 2.0 / (3.0 * n))[:, None] * c
        )
        second_before = np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum(second_mask[:-1], dtype=np.int64),
        ))
        first_positions = np.arange(len(i), dtype=np.int64) + second_before
        samples = np.empty((n * n, 3), dtype=np.float64)
        samples[first_positions] = first
        samples[first_positions[second_mask] + 1] = second
        return samples
    return first


# ------------------------------------------------------------
# Candidate surface sampling
# ------------------------------------------------------------

def _sample_candidate_faces(
    mesh: trimesh.Trimesh,
    face_status_labels: np.ndarray,
    patch_labels: np.ndarray,
    normals: np.ndarray,
    spacing_coordinate_units: float,
    max_total_samples: int = MAX_TOTAL_SAMPLES,
) -> dict:

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    candidate_faces = np.flatnonzero(
        face_status_labels == FACE_PAINTABLE_CANDIDATE
    )
    if len(candidate_faces) == 0:
        raise RuntimeError(
            "No valid trajectory samples were generated. "
            "Check Stage 2 candidate labels and mesh normals."
        )

    # Validate all candidate triangles in contiguous NumPy batches. The
    # previous implementation repeated finite checks and cross products in
    # Python for every face; this preserves source-face order and the same
    # normal-first skip accounting while moving those operations to NumPy.
    candidate_triangles = vertices[faces[candidate_faces]]
    candidate_normals = np.asarray(normals, dtype=np.float64)[candidate_faces]
    valid_normal_mask = np.all(np.isfinite(candidate_normals), axis=1)
    skipped_invalid_normals = int(np.count_nonzero(~valid_normal_mask))

    geometry_triangles = candidate_triangles[valid_normal_mask]
    geometry_face_indices = candidate_faces[valid_normal_mask]
    edge_ab = geometry_triangles[:, 1] - geometry_triangles[:, 0]
    edge_ac = geometry_triangles[:, 2] - geometry_triangles[:, 0]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(edge_ab, edge_ac), axis=1
    )
    valid_geometry_mask = (
        np.all(np.isfinite(geometry_triangles), axis=(1, 2))
        & np.isfinite(triangle_areas)
        & (triangle_areas > NORMAL_EPSILON)
    )
    skipped_invalid_geometry = int(
        np.count_nonzero(~valid_geometry_mask)
    )

    valid_triangles = geometry_triangles[valid_geometry_mask]
    valid_face_indices = geometry_face_indices[valid_geometry_mask]
    valid_normals = candidate_normals[valid_normal_mask][valid_geometry_mask]

    # A live STEP context enables exact original-CAD surface evaluation.
    # STL inputs intentionally retain the existing triangle-based path.
    from core.step_loader import (
        get_step_brep_context,
        evaluate_step_triangle_sample,
    )
    step_brep_context = get_step_brep_context(mesh)

    all_points = []
    all_normals = []
    all_face_indices = []
    all_patch_ids = []
    sample_count = 0

    # Keep the per-face sampler and append order unchanged. Only validation
    # and area computation above are batched; each face still uses the
    # exact same triangular-lattice sampling routine.
    for triangle, normal, face_index in zip(
        valid_triangles, valid_normals, valid_face_indices
    ):
        points = _triangle_sample_points(
            triangle,
            spacing_coordinate_units,
        )
        if len(points) == 0:
            skipped_invalid_geometry += 1
            continue

        sample_count += len(points)
        if sample_count > max_total_samples:
            raise RuntimeError(
                f"Stage 3 would generate more than "
                f"{max_total_samples:,} samples. "
                f"Current spacing: {spacing_coordinate_units:.6g} "
                "STL coordinate units. Increase sampling spacing."
            )

        # STEP inputs use the original B-Rep Surface as the geometric
        # authority. The display triangle is retained only as a stable
        # triangle/Patch/Face index carrier and UV interpolation domain.
        if step_brep_context is not None:
            a, b, c = triangle
            edge_a = b - a
            edge_b = c - a
            gram_aa = float(np.dot(edge_a, edge_a))
            gram_ab = float(np.dot(edge_a, edge_b))
            gram_bb = float(np.dot(edge_b, edge_b))
            determinant = gram_aa * gram_bb - gram_ab * gram_ab
            if determinant <= NORMAL_EPSILON:
                skipped_invalid_geometry += len(points)
                continue
            offsets = points - a
            rhs_a = offsets @ edge_a
            rhs_b = offsets @ edge_b
            weight_b = (gram_bb * rhs_a - gram_ab * rhs_b) / determinant
            weight_c = (gram_aa * rhs_b - gram_ab * rhs_a) / determinant
            barycentric = np.column_stack((
                1.0 - weight_b - weight_c, weight_b, weight_c
            ))
            exact_points = np.empty_like(points)
            exact_normals = np.empty_like(points)
            for sample_index, weights in enumerate(barycentric):
                exact = evaluate_step_triangle_sample(
                    mesh, int(face_index), weights
                )
                exact_points[sample_index] = exact["point_mm"]
                exact_normal = np.asarray(exact["normal"], dtype=np.float64)
                # Preserve Stage 3's established outward-normal convention.
                if float(np.dot(exact_normal, normal)) < 0.0:
                    exact_normal *= -1.0
                exact_normals[sample_index] = exact_normal
            points = exact_points
            sample_normals = exact_normals
        else:
            sample_normals = np.repeat(normal[None, :], len(points), axis=0)

        all_points.append(points)
        all_normals.append(sample_normals)
        all_face_indices.append(
            np.full(len(points), face_index, dtype=np.int64)
        )
        all_patch_ids.append(
            np.full(
                len(points),
                int(patch_labels[face_index]),
                dtype=np.int64,
            )
        )

    if not all_points:
        raise RuntimeError(
            "No valid trajectory samples were generated. "
            "Check Stage 2 candidate labels and mesh normals."
        )

    return {
        "surface_points": np.vstack(all_points),
        "surface_normals": np.vstack(all_normals),
        "face_indices": np.concatenate(all_face_indices),
        "patch_ids": np.concatenate(all_patch_ids),
        "candidate_face_count": int(len(candidate_faces)),
        "surface_geometry_source": (
            "ORIGINAL_STEP_BREP_SURFACE"
            if step_brep_context is not None else "DISPLAY_MESH_TRIANGLES"
        ),
        "sample_count": int(sample_count),
        "skipped_invalid_normals": int(skipped_invalid_normals),
        "skipped_invalid_geometry": int(skipped_invalid_geometry),
    }


# ------------------------------------------------------------
# Trajectory ordering
# ------------------------------------------------------------

def _nearest_neighbor_order(points: np.ndarray) -> np.ndarray:

    count = len(points)

    if count <= 1:
        return np.arange(count, dtype=np.int64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)

        remaining = np.ones(count, dtype=bool)
        order = np.empty(count, dtype=np.int64)

        current = 0

        for index in range(count):

            order[index] = current
            remaining[current] = False

            if index == count - 1:
                break

            k = min(count, 64)

            _, indices = tree.query(points[current], k=k)
            indices = np.atleast_1d(indices)

            next_index = None

            for candidate in indices:
                candidate = int(candidate)

                if candidate < count and remaining[candidate]:
                    next_index = candidate
                    break

            if next_index is None:

                remaining_indices = np.flatnonzero(remaining)
                delta = points[remaining_indices] - points[current]

                squared_distances = np.einsum(
                    "ij,ij->i",
                    delta,
                    delta,
                )

                next_index = int(
                    remaining_indices[np.argmin(squared_distances)]
                )

            current = next_index

        return order

    except ImportError:

        if count > 10_000:
            raise RuntimeError(
                "SciPy is required for ordering more than 10,000 "
                "samples. Install it with: pip install scipy"
            )

        remaining = np.ones(count, dtype=bool)
        order = np.empty(count, dtype=np.int64)

        current = 0

        for index in range(count):

            order[index] = current
            remaining[current] = False

            if index == count - 1:
                break

            remaining_indices = np.flatnonzero(remaining)
            delta = points[remaining_indices] - points[current]

            squared_distances = np.einsum(
                "ij,ij->i",
                delta,
                delta,
            )

            current = int(
                remaining_indices[np.argmin(squared_distances)]
            )

        return order



def _generate_planar_raster_for_patch(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    normals: np.ndarray,
    standoff_coordinate_units: float,
    sample_spacing_coordinate_units: float,
    track_spacing_coordinate_units: float,
    planarity_angle_deg: float = 2.0,
    lane_orientation_angle_deg: float = 0.0,
) -> dict | None:
    """
    Generate clipped, serpentine raster lanes for one geometrically planar patch.

    Every lane is clipped against the union of its source triangles. Separate
    connected intervals remain separate segments; no spray-on connector is made.
    Returns None when the patch is not sufficiently planar.
    """
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if len(face_indices) == 0 or track_spacing_coordinate_units <= 0.0:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces[face_indices]]
    patch_normals = np.asarray(normals[face_indices], dtype=np.float64)
    mean_normal = np.mean(patch_normals, axis=0)
    normal_length = float(np.linalg.norm(mean_normal))
    if not math.isfinite(normal_length) or normal_length <= NORMAL_EPSILON:
        return None
    plane_normal = mean_normal / normal_length
    normal_cosine = math.cos(math.radians(planarity_angle_deg))
    if np.any(patch_normals @ plane_normal < normal_cosine):
        return None

    patch_vertices = triangles.reshape(-1, 3)
    origin = np.mean(patch_vertices, axis=0)
    signed_distances = (patch_vertices - origin) @ plane_normal
    plane_tolerance = max(sample_spacing_coordinate_units * 0.02, 1e-7)
    if float(np.max(np.abs(signed_distances))) > plane_tolerance:
        return None

    centered = patch_vertices - origin
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    lane_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    lane_axis -= float(np.dot(lane_axis, plane_normal)) * plane_normal
    lane_length = float(np.linalg.norm(lane_axis))
    if lane_length <= NORMAL_EPSILON:
        return None
    lane_axis /= lane_length
    if not math.isfinite(lane_orientation_angle_deg):
        return None
    angle = math.radians(float(lane_orientation_angle_deg))
    lane_axis = (math.cos(angle) * lane_axis
                 + math.sin(angle) * np.cross(plane_normal, lane_axis))
    lane_axis /= max(float(np.linalg.norm(lane_axis)), NORMAL_EPSILON)
    offset_axis = np.cross(plane_normal, lane_axis)
    offset_axis /= np.linalg.norm(offset_axis)

    projected = np.stack((
        centered @ lane_axis,
        centered @ offset_axis,
    ), axis=1)
    projected_triangles = projected.reshape(-1, 3, 2)
    min_offset = float(np.min(projected[:, 1]))
    max_offset = float(np.max(projected[:, 1]))
    extent = max_offset - min_offset
    if extent <= NORMAL_EPSILON:
        return None

    lane_spacing = float(track_spacing_coordinate_units)
    first_offset = min_offset + min(lane_spacing * 0.5, extent * 0.5)
    last_offset = max_offset - min(lane_spacing * 0.5, extent * 0.5)
    if last_offset < first_offset:
        lane_offsets = np.array([(min_offset + max_offset) * 0.5])
    else:
        lane_count = max(1, int(math.floor((last_offset - first_offset) / lane_spacing)) + 1)
        lane_offsets = first_offset + np.arange(lane_count, dtype=np.float64) * lane_spacing
        if len(lane_offsets) == 0:
            lane_offsets = np.array([(min_offset + max_offset) * 0.5])
        elif last_offset - float(lane_offsets[-1]) > lane_spacing * 0.5:
            lane_offsets = np.append(lane_offsets, last_offset)

    def triangle_interval(triangle_2d: np.ndarray, offset: float):
        intersections = []
        tolerance = 1e-9
        for edge_index in range(3):
            a = triangle_2d[edge_index]
            b = triangle_2d[(edge_index + 1) % 3]
            da = float(a[1] - offset)
            db = float(b[1] - offset)
            if abs(da) <= tolerance:
                intersections.append(float(a[0]))
            if da * db < -tolerance * tolerance:
                ratio = (offset - a[1]) / (b[1] - a[1])
                intersections.append(float(a[0] + ratio * (b[0] - a[0])))
        if len(intersections) < 2:
            return None
        values = np.unique(np.round(np.asarray(intersections), decimals=10))
        if len(values) < 2 or float(values[-1] - values[0]) <= 1e-9:
            return None
        return float(values[0]), float(values[-1])

    def barycentric_2d(point: np.ndarray, triangle: np.ndarray):
        """Return projected-triangle barycentric weights, including edge points."""
        a, b, c = np.asarray(triangle, dtype=np.float64)
        v0 = b - a
        v1 = c - a
        v2 = np.asarray(point, dtype=np.float64) - a
        denominator = float(v0[0] * v1[1] - v1[0] * v0[1])
        if not math.isfinite(denominator) or abs(denominator) <= 1e-14:
            return None
        beta = float((v2[0] * v1[1] - v1[0] * v2[1]) / denominator)
        gamma = float((v0[0] * v2[1] - v0[1] * v2[0]) / denominator)
        weights = np.asarray([1.0 - beta - gamma, beta, gamma], dtype=np.float64)
        if (
            not np.all(np.isfinite(weights))
            or float(np.min(weights)) < -1e-8
            or float(np.max(weights)) > 1.0 + 1e-8
        ):
            return None
        weights = np.clip(weights, 0.0, 1.0)
        total = float(np.sum(weights))
        if total <= NORMAL_EPSILON:
            return None
        return weights / total

    def point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray) -> bool:
        return barycentric_2d(point, triangle) is not None

    out_surface = []
    out_normals = []
    out_nozzle = []
    out_directions = []
    out_faces = []
    out_segments = []
    segment_id = 0
    lane_records = []

    for lane_index, offset in enumerate(lane_offsets):
        intervals = []
        for local_index, triangle_2d in enumerate(projected_triangles):
            interval = triangle_interval(triangle_2d, float(offset))
            if interval is not None:
                intervals.append((interval[0], interval[1], local_index))
        if not intervals:
            continue
        intervals.sort(key=lambda item: (item[0], item[1]))
        merged = []
        for start, end, local_index in intervals:
            if not merged or start > merged[-1][1] + 1e-8:
                merged.append([start, end, [local_index]])
            else:
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2].append(local_index)

        if lane_index % 2:
            merged = list(reversed(merged))

        for start, end, source_local_indices in merged:
            distance = max(0.0, end - start)
            sample_count = max(2, int(math.ceil(distance / sample_spacing_coordinate_units)) + 1)
            along_values = np.linspace(start, end, sample_count)
            if lane_index % 2:
                along_values = along_values[::-1]
            segment_points = []
            segment_faces = []
            for along in along_values:
                point_2d = np.array([along, float(offset)], dtype=np.float64)
                source_local_index = None
                source_weights = None
                for local_index in source_local_indices:
                    weights = barycentric_2d(
                        point_2d, projected_triangles[local_index]
                    )
                    if weights is not None:
                        source_local_index = int(local_index)
                        source_weights = weights
                        break

                if source_local_index is None:
                    # Merged 2-D intervals can contain a tiny numerical gap.
                    # Search the complete patch before rejecting the planar
                    # raster; never assign a point to an arbitrary first triangle.
                    for local_index, projected_triangle in enumerate(projected_triangles):
                        weights = barycentric_2d(point_2d, projected_triangle)
                        if weights is not None:
                            source_local_index = int(local_index)
                            source_weights = weights
                            break

                if source_local_index is None or source_weights is None:
                    # Do not invent a source-face association. Reject this planar
                    # patch so the curved raster / explicit fallback can handle it.
                    return None

                source_face = int(face_indices[source_local_index])
                # Reconstruct on the actual source triangle, not on the mean
                # PCA plane. This preserves a valid 3-D barycentric mapping even
                # when the patch is within the allowed planar tolerance but is
                # not mathematically perfectly coplanar.
                point_3d = source_weights @ triangles[source_local_index]
                segment_points.append(point_3d)
                segment_faces.append(source_face)

            segment_points = np.asarray(segment_points, dtype=np.float64)
            segment_normals = np.repeat(plane_normal[None, :], len(segment_points), axis=0)
            segment_directions = -segment_normals
            segment_nozzles = segment_points + standoff_coordinate_units * segment_normals
            out_surface.extend(segment_points)
            out_normals.extend(segment_normals)
            out_directions.extend(segment_directions)
            out_nozzle.extend(segment_nozzles)
            out_faces.extend(segment_faces)
            out_segments.extend([segment_id] * len(segment_points))
            out_segments[-len(segment_points):] = [segment_id] * len(segment_points)
            lane_records.append({
                "segment_id": int(segment_id),
                "lane_index": int(lane_index),
                "sample_count": int(len(segment_points)),
                "length_coordinate_units": float(distance),
                "spray_enabled_connector": False,
            })
            segment_id += 1

    if not out_surface:
        return None
    surface_array = np.asarray(out_surface, dtype=np.float64)
    normal_array = np.asarray(out_normals, dtype=np.float64)
    direction_array = np.asarray(out_directions, dtype=np.float64)
    nozzle_array = np.asarray(out_nozzle, dtype=np.float64)
    face_array = np.asarray(out_faces, dtype=np.int64)
    segment_array = np.asarray(out_segments, dtype=np.int64)
    return {
        "surface_points": surface_array,
        "surface_normals": normal_array,
        "nozzle_points": nozzle_array,
        "spray_directions": direction_array,
        "nozzle_frames": _build_nozzle_frames(direction_array),
        "face_indices": face_array,
        "segment_ids": segment_array,
        "segments": lane_records,
        "lane_count": int(len(lane_records)),
        "lane_spacing_coordinate_units": lane_spacing,
        "lane_direction": lane_axis.tolist(),
        "offset_direction": offset_axis.tolist(),
    }



def _split_raster_lane_at_discontinuities(
    lane_points: np.ndarray,
    maximum_step_coordinate_units: float,
) -> list[tuple[int, int]]:
    """Return half-open sample ranges separated by oversized 3-D lane jumps."""
    points = np.asarray(lane_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Raster lane points must have shape (N, 3).")
    maximum_step = float(maximum_step_coordinate_units)
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("Maximum raster step must be finite and positive.")
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [(0, 1)]
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    split_after = (np.flatnonzero(step_lengths > maximum_step) + 1).tolist()
    boundaries = [0, *split_after, len(points)]
    return [
        (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]


def _smooth_normals_with_deviation_limit(
    normals: np.ndarray,
    max_deviation_deg: float = 8.0,
    iterations: int = 6,
) -> tuple[np.ndarray, float]:
    """
    Smooth sample-to-sample normal noise while staying inside a cone around
    every original source normal. Surface positions are never modified.
    """
    original = np.asarray(normals, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3:
        raise ValueError("Normals must have shape (N, 3).")
    if not np.all(np.isfinite(original)):
        raise ValueError("Normals must contain only finite values.")
    if not math.isfinite(float(max_deviation_deg)) or not 0.0 <= max_deviation_deg <= 45.0:
        raise ValueError("Normal smoothing deviation must be between 0 and 45 degrees.")
    if int(iterations) < 0 or int(iterations) > 50:
        raise ValueError("Normal smoothing iterations must be between 0 and 50.")
    if len(original) == 0:
        return original.copy(), 0.0

    lengths = np.linalg.norm(original, axis=1)
    if np.any(lengths <= NORMAL_EPSILON):
        raise ValueError("Cannot smooth a zero-length surface normal.")
    reference = original / lengths[:, None]
    smoothed = reference.copy()
    max_angle_rad = math.radians(float(max_deviation_deg))

    for _ in range(int(iterations)):
        if len(smoothed) < 3:
            break
        padded = np.pad(smoothed, ((1, 1), (0, 0)), mode="edge")
        candidate = (
            0.25 * padded[:-2]
            + 0.50 * padded[1:-1]
            + 0.25 * padded[2:]
        )
        candidate_lengths = np.linalg.norm(candidate, axis=1)
        candidate /= np.maximum(candidate_lengths[:, None], NORMAL_EPSILON)

        dots = np.clip(np.einsum("ij,ij->i", reference, candidate), -1.0, 1.0)
        angles = np.arccos(dots)
        limited = candidate.copy()
        for index, angle in enumerate(angles):
            if float(angle) <= max_angle_rad or float(angle) <= 1e-12:
                continue
            fraction = max_angle_rad / float(angle)
            sin_angle = math.sin(float(angle))
            if abs(sin_angle) <= 1e-10:
                limited[index] = reference[index]
                continue
            limited[index] = (
                math.sin((1.0 - fraction) * float(angle)) / sin_angle
            ) * reference[index] + (
                math.sin(fraction * float(angle)) / sin_angle
            ) * candidate[index]
        limited /= np.maximum(
            np.linalg.norm(limited, axis=1, keepdims=True),
            NORMAL_EPSILON,
        )
        smoothed = limited

    final_dots = np.clip(
        np.einsum("ij,ij->i", reference, smoothed), -1.0, 1.0
    )
    final_deviation = np.degrees(np.arccos(final_dots))
    return smoothed, float(np.max(final_deviation)) if len(final_deviation) else 0.0


def _generate_surface_following_raster_for_patch(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    normals: np.ndarray,
    standoff_coordinate_units: float,
    sample_spacing_coordinate_units: float,
    track_spacing_coordinate_units: float,
    diagnostics: dict | None = None,
) -> dict | None:
    """
    Generate serpentine lanes on a smoothly curved triangulated patch.

    The patch is projected into its two dominant PCA directions only to define
    lane coordinates. Every output point is reconstructed by barycentric
    interpolation on an actual source triangle, and its nozzle offset follows
    that triangle's local normal. This is a surface-following approximation,
    not a globally valid UV/geodesic parameterization. Patches that fold over
    in the projection are rejected so the caller can use a safe fallback.
    """
    def reject(reason: str):
        if diagnostics is not None:
            diagnostics["reason"] = reason
        return None

    face_indices = np.unique(np.asarray(face_indices, dtype=np.int64))
    if (
        len(face_indices) == 0
        or track_spacing_coordinate_units <= 0.0
        or sample_spacing_coordinate_units <= 0.0
    ):
        return reject("invalid_input_or_spacing")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles_3d = vertices[faces[face_indices]]
    patch_normals = np.asarray(normals[face_indices], dtype=np.float64)
    valid_normals = (
        np.all(np.isfinite(patch_normals), axis=1)
        & (np.linalg.norm(patch_normals, axis=1) > NORMAL_EPSILON)
    )
    if not np.all(valid_normals):
        return reject("invalid_face_normals")

    mean_normal = np.mean(patch_normals, axis=0)
    mean_length = float(np.linalg.norm(mean_normal))
    if not math.isfinite(mean_length) or mean_length < 0.15:
        return reject("inconsistent_or_cancelling_normals")
    mean_normal /= mean_length
    normal_alignment = patch_normals @ mean_normal
    # A patch with strongly opposing normals is likely folded or incorrectly
    # segmented; a single 2-D projection would be ambiguous.
    if float(np.min(normal_alignment)) < -0.25:
        return reject("folded_patch_opposing_normals")

    patch_vertices = triangles_3d.reshape(-1, 3)
    origin = np.mean(patch_vertices, axis=0)
    centered = patch_vertices - origin
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    lane_axis = eigenvectors[:, int(np.argmax(eigenvalues))].copy()
    offset_axis = eigenvectors[:, int(np.argsort(eigenvalues)[-2])].copy()
    lane_axis /= max(float(np.linalg.norm(lane_axis)), NORMAL_EPSILON)
    offset_axis -= float(np.dot(offset_axis, lane_axis)) * lane_axis
    offset_length = float(np.linalg.norm(offset_axis))
    if offset_length <= NORMAL_EPSILON:
        return reject("degenerate_pca_axes")
    offset_axis /= offset_length

    projected = np.stack((
        centered @ lane_axis,
        centered @ offset_axis,
    ), axis=1)
    projected_triangles = projected.reshape(-1, 3, 2)

    # Reject triangles that collapse in the PCA plane. A large number of
    # collapsed projections indicates a folded/near-edge-on patch.
    projected_edge_a = (
        projected_triangles[:, 1] - projected_triangles[:, 0]
    )
    projected_edge_b = (
        projected_triangles[:, 2] - projected_triangles[:, 0]
    )
    # 2-D scalar cross product (determinant). Avoid np.cross on 2-D
    # vectors, which raises in newer NumPy versions.
    projected_areas = np.abs(
        projected_edge_a[:, 0] * projected_edge_b[:, 1]
        - projected_edge_a[:, 1] * projected_edge_b[:, 0]
    )
    source_areas = np.linalg.norm(
        np.cross(
            triangles_3d[:, 1] - triangles_3d[:, 0],
            triangles_3d[:, 2] - triangles_3d[:, 0],
        ),
        axis=1,
    )
    usable = (
        (projected_areas > 1e-10)
        & (source_areas > 1e-10)
        & np.isfinite(projected_areas)
    )
    # A patch containing exactly one source triangle is still a valid
    # raster domain: its three vertices define a non-degenerate 2-D region.
    # Requiring two usable triangles unconditionally rejects every
    # single-triangle patch and sends it to nearest-neighbor fallback.
    minimum_usable_triangles = (
        1 if len(face_indices) == 1
        else max(2, int(math.ceil(len(face_indices) * 0.5)))
    )
    if np.count_nonzero(usable) < minimum_usable_triangles:
        return reject("too_many_degenerate_projected_triangles")

    projected_triangles = projected_triangles[usable]
    triangles_3d = triangles_3d[usable]
    source_face_indices = face_indices[usable]
    source_normals = patch_normals[usable]

    min_offset = float(np.min(projected_triangles[:, :, 1]))
    max_offset = float(np.max(projected_triangles[:, :, 1]))
    extent = max_offset - min_offset
    if not math.isfinite(extent) or extent <= NORMAL_EPSILON:
        return reject("zero_projected_lane_extent")

    lane_spacing = float(track_spacing_coordinate_units)
    first_offset = min_offset + min(lane_spacing * 0.5, extent * 0.5)
    last_offset = max_offset - min(lane_spacing * 0.5, extent * 0.5)
    if last_offset < first_offset:
        lane_offsets = np.asarray([(min_offset + max_offset) * 0.5])
    else:
        lane_count = max(
            1,
            int(math.floor((last_offset - first_offset) / lane_spacing)) + 1,
        )
        lane_offsets = first_offset + np.arange(lane_count) * lane_spacing
        if last_offset - float(lane_offsets[-1]) > lane_spacing * 0.5:
            lane_offsets = np.append(lane_offsets, last_offset)

    def line_interval(triangle_2d: np.ndarray, offset: float):
        hits = []
        eps = 1e-9
        for edge in range(3):
            a = triangle_2d[edge]
            b = triangle_2d[(edge + 1) % 3]
            da = float(a[1] - offset)
            db = float(b[1] - offset)
            if abs(da) <= eps:
                hits.append(float(a[0]))
            if da * db < -(eps * eps):
                ratio = (offset - a[1]) / (b[1] - a[1])
                hits.append(float(a[0] + ratio * (b[0] - a[0])))
        if len(hits) < 2:
            return None
        hits = np.unique(np.round(np.asarray(hits), decimals=10))
        if len(hits) < 2 or float(hits[-1] - hits[0]) <= eps:
            return None
        return float(hits[0]), float(hits[-1])

    # A single global PCA grid can fall entirely inside a gap between
    # disconnected projected components. If none of its lanes intersects any
    # triangle, retry with one guaranteed interior scanline per projected
    # triangle. The centroid's offset coordinate lies strictly inside every
    # non-degenerate triangle, so this recovers narrow/disconnected bands
    # without inventing geometry or relaxing barycentric validation.
    regular_grid_has_intersection = any(
        line_interval(triangle_2d, float(offset)) is not None
        for offset in lane_offsets
        for triangle_2d in projected_triangles
    )
    adaptive_scanline_offsets_used = False
    if not regular_grid_has_intersection:
        adaptive_offsets = np.mean(projected_triangles[:, :, 1], axis=1)
        adaptive_offsets = np.unique(
            np.round(adaptive_offsets.astype(np.float64), decimals=10)
        )
        if len(adaptive_offsets) > 0:
            lane_offsets = adaptive_offsets
            adaptive_scanline_offsets_used = True
            if diagnostics is not None:
                diagnostics["adaptive_scanline_offsets_used"] = True
                diagnostics["adaptive_scanline_count"] = int(len(lane_offsets))
                diagnostics["regular_scanline_count"] = int(
                    len(np.unique(np.round(
                        np.asarray([
                            min_offset + min(lane_spacing * 0.5, extent * 0.5)
                            + i * lane_spacing
                            for i in range(max(
                                1,
                                int(math.floor(
                                    (last_offset - first_offset) / lane_spacing
                                )) + 1,
                            ))
                        ], dtype=np.float64),
                        decimals=10,
                    )))
                )

    def barycentric_2d(point: np.ndarray, triangle: np.ndarray):
        a, b, c = triangle
        v0 = b - a
        v1 = c - a
        v2 = point - a
        denominator = float(v0[0] * v1[1] - v1[0] * v0[1])
        if abs(denominator) <= 1e-14:
            return None
        beta = float((v2[0] * v1[1] - v1[0] * v2[1]) / denominator)
        gamma = float((v0[0] * v2[1] - v2[0] * v0[1]) / denominator)
        alpha = 1.0 - beta - gamma
        weights = np.asarray([alpha, beta, gamma], dtype=np.float64)
        if float(np.min(weights)) < -1e-6:
            return None
        return weights

    out_surface = []
    out_normals = []
    out_nozzle = []
    out_directions = []
    out_faces = []
    out_segments = []
    segment_records = []
    segment_id = 0
    maximum_normal_smoothing_deviation_deg = 0.0

    for lane_index, offset in enumerate(lane_offsets):
        intervals = []
        for local_face, triangle_2d in enumerate(projected_triangles):
            interval = line_interval(triangle_2d, float(offset))
            if interval is not None:
                intervals.append((interval[0], interval[1], local_face))
        if not intervals:
            continue

        intervals.sort(key=lambda item: (item[0], item[1]))
        merged = []
        for start, end, local_face in intervals:
            if not merged or start > merged[-1][1] + 1e-8:
                merged.append([start, end, [local_face]])
            else:
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2].append(local_face)
        if lane_index % 2:
            merged.reverse()

        for start, end, candidate_faces in merged:
            sample_count = max(
                2,
                int(math.ceil((end - start) / sample_spacing_coordinate_units)) + 1,
            )
            along_values = np.linspace(start, end, sample_count)
            if lane_index % 2:
                along_values = along_values[::-1]

            lane_points = []
            lane_normals = []
            lane_face_ids = []
            for along in along_values:
                point_2d = np.asarray([along, float(offset)], dtype=np.float64)
                best = None
                best_margin = -float("inf")
                for local_face in candidate_faces:
                    weights = barycentric_2d(
                        point_2d, projected_triangles[local_face]
                    )
                    if weights is None:
                        continue
                    margin = float(np.min(weights))
                    if margin > best_margin:
                        best = (local_face, weights)
                        best_margin = margin
                if best is None:
                    # A tiny numerical gap at a triangle boundary is not
                    # allowed to create an invented point off the mesh.
                    return reject("barycentric_reconstruction_failed")
                local_face, weights = best
                point_3d = weights @ triangles_3d[local_face]
                normal = source_normals[local_face].copy()
                normal_length = float(np.linalg.norm(normal))
                if normal_length <= NORMAL_EPSILON:
                    return reject("invalid_reconstructed_surface_normal")
                normal /= normal_length
                lane_points.append(point_3d)
                lane_normals.append(normal)
                lane_face_ids.append(int(source_face_indices[local_face]))

            lane_points = np.asarray(lane_points, dtype=np.float64)
            lane_normals = np.asarray(lane_normals, dtype=np.float64)
            lane_face_ids = np.asarray(lane_face_ids, dtype=np.int64)

            # A PCA projection can make distant 3-D surface regions overlap
            # in 2-D. Never keep an oversized reconstructed 3-D jump inside
            # one spray stroke: split it into separate strokes (spray OFF
            # between them) while preserving every original surface sample.
            maximum_step_coordinate_units = max(
                3.0 * float(sample_spacing_coordinate_units),
                2.0 * float(track_spacing_coordinate_units),
                1e-9,
            )
            lane_nozzle_points_for_split = (
                lane_points
                + float(standoff_coordinate_units) * lane_normals
            )
            lane_chunks = _split_raster_lane_at_discontinuities(
                lane_nozzle_points_for_split,
                maximum_step_coordinate_units,
            )
            for chunk_index, (chunk_start, chunk_end) in enumerate(lane_chunks):
                chunk_points = lane_points[chunk_start:chunk_end]
                chunk_normals = lane_normals[chunk_start:chunk_end].copy()
                chunk_face_ids = lane_face_ids[chunk_start:chunk_end]

                # Smooth normal noise only inside this continuous 3-D
                # stroke. Each result remains within 8 degrees of its source
                # face normal; positions and physical standoff stay unchanged.
                chunk_normals, chunk_normal_deviation_deg = (
                    _smooth_normals_with_deviation_limit(
                        chunk_normals,
                        max_deviation_deg=8.0,
                        iterations=6,
                    )
                )
                maximum_normal_smoothing_deviation_deg = max(
                    maximum_normal_smoothing_deviation_deg,
                    chunk_normal_deviation_deg,
                )

                chunk_directions = -chunk_normals
                chunk_nozzles = (
                    chunk_points
                    + float(standoff_coordinate_units) * chunk_normals
                )
                out_surface.extend(chunk_points)
                out_normals.extend(chunk_normals)
                out_directions.extend(chunk_directions)
                out_nozzle.extend(chunk_nozzles)
                out_faces.extend(chunk_face_ids.tolist())
                out_segments.extend([segment_id] * len(chunk_points))
                segment_records.append({
                    "segment_id": int(segment_id),
                    "lane_index": int(lane_index),
                    "projection_interval_index": int(
                        len(segment_records)
                    ),
                    "sample_count": int(len(chunk_points)),
                    "length_coordinate_units": float(
                        np.sum(np.linalg.norm(
                            np.diff(chunk_points, axis=0), axis=1
                        ))
                    ) if len(chunk_points) > 1 else 0.0,
                    "path_type": "curved_surface_raster",
                    "projection_discontinuity_split": bool(
                        len(lane_chunks) > 1
                    ),
                    "split_chunk_index": int(chunk_index),
                    "split_trigger_distance_coordinate_units": (
                        float(np.linalg.norm(
                            lane_nozzle_points_for_split[chunk_start]
                            - lane_nozzle_points_for_split[chunk_start - 1]
                        ))
                        if chunk_index > 0 else None
                    ),
                    "maximum_allowed_step_coordinate_units": float(
                        maximum_step_coordinate_units
                    ),
                    "max_normal_smoothing_deviation_deg": float(
                        chunk_normal_deviation_deg
                    ),
                    "spray_enabled_connector": False,
                })
                segment_id += 1

    if not out_surface:
        return reject("no_intersecting_scanlines")

    surface_array = np.asarray(out_surface, dtype=np.float64)
    normal_array = np.asarray(out_normals, dtype=np.float64)
    direction_array = np.asarray(out_directions, dtype=np.float64)
    nozzle_array = np.asarray(out_nozzle, dtype=np.float64)
    return {
        "surface_points": surface_array,
        "surface_normals": normal_array,
        "nozzle_points": nozzle_array,
        "spray_directions": direction_array,
        "nozzle_frames": _build_nozzle_frames(direction_array),
        "face_indices": np.asarray(out_faces, dtype=np.int64),
        "segment_ids": np.asarray(out_segments, dtype=np.int64),
        "segments": segment_records,
        "lane_count": int(len(segment_records)),
        "lane_spacing_coordinate_units": lane_spacing,
        "lane_direction": lane_axis.tolist(),
        "offset_direction": offset_axis.tolist(),
        "projection_method": "pca_surface_triangle_barycentric",
        "projection_is_approximate": True,
        "adaptive_scanline_offsets_used": bool(adaptive_scanline_offsets_used),
        "adaptive_scanline_count": int(len(lane_offsets)) if adaptive_scanline_offsets_used else 0,
        "max_normal_smoothing_deviation_deg": float(
            maximum_normal_smoothing_deviation_deg
        ),
    }



def _merge_single_face_normal_groups(
    groups: list[np.ndarray],
    face_indices: np.ndarray,
    unit_normals: np.ndarray,
    max_merge_angle_deg: float = 70.0,
) -> tuple[list[np.ndarray], int]:
    """
    Merge singleton normal-cone groups into the closest compatible larger
    group. A merge is accepted only when every resulting normal remains within
    max_merge_angle_deg of the merged group's mean direction.
    """
    if not groups:
        return groups, 0

    face_to_local = {
        int(face_id): int(local_index)
        for local_index, face_id in enumerate(face_indices)
    }
    merged_groups = [np.asarray(group, dtype=np.int64).copy() for group in groups]
    singleton_groups = [
        group for group in merged_groups if len(group) == 1
    ]
    merged_groups = [
        group for group in merged_groups if len(group) >= 2
    ]
    if not singleton_groups:
        return merged_groups, 0

    cosine_limit = math.cos(math.radians(float(max_merge_angle_deg)))
    merged_count = 0
    unresolved = []

    for singleton in singleton_groups:
        face_id = int(singleton[0])
        local_index = face_to_local[face_id]
        normal = unit_normals[local_index]
        best_index = None
        best_alignment = -float("inf")

        for group_index, group in enumerate(merged_groups):
            member_indices = np.asarray(
                [face_to_local[int(member)] for member in group],
                dtype=np.int64,
            )
            group_mean = np.mean(unit_normals[member_indices], axis=0)
            mean_length = float(np.linalg.norm(group_mean))
            if mean_length <= NORMAL_EPSILON:
                continue
            group_mean /= mean_length
            alignment = float(np.dot(normal, group_mean))
            if alignment > best_alignment:
                best_alignment = alignment
                best_index = group_index

        if best_index is None or best_alignment < cosine_limit:
            unresolved.append(singleton)
            continue

        candidate = np.concatenate(
            [merged_groups[best_index], singleton]
        ).astype(np.int64)
        candidate_indices = np.asarray(
            [face_to_local[int(member)] for member in candidate],
            dtype=np.int64,
        )
        candidate_mean = np.mean(unit_normals[candidate_indices], axis=0)
        candidate_mean_length = float(np.linalg.norm(candidate_mean))
        if candidate_mean_length <= NORMAL_EPSILON:
            unresolved.append(singleton)
            continue
        candidate_mean /= candidate_mean_length
        candidate_alignment = unit_normals[candidate_indices] @ candidate_mean
        if float(np.min(candidate_alignment)) < cosine_limit:
            unresolved.append(singleton)
            continue

        merged_groups[best_index] = candidate
        merged_count += 1

    merged_groups.extend(unresolved)
    return merged_groups, merged_count


def _generate_partitioned_surface_raster_for_patch(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    normals: np.ndarray,
    standoff_coordinate_units: float,
    sample_spacing_coordinate_units: float,
    track_spacing_coordinate_units: float,
    max_normal_cone_angle_deg: float = 45.0,
    diagnostics: dict | None = None,
) -> dict | None:
    """
    Split a broad curved patch into directionally coherent normal cones, then
    rasterize each cone independently in its own local PCA projection.

    Normals are never flipped or averaged across opposing directions. This is
    intentionally conservative: if any sub-patch cannot be rasterized, the
    caller retains the original nearest-neighbor fallback for the whole patch.
    """
    face_indices = np.unique(np.asarray(face_indices, dtype=np.int64))
    if len(face_indices) < 2:
        if diagnostics is not None:
            diagnostics["reason"] = "normal_partition_too_few_faces"
        return None

    patch_normals = np.asarray(normals[face_indices], dtype=np.float64)
    lengths = np.linalg.norm(patch_normals, axis=1)
    valid = (
        np.all(np.isfinite(patch_normals), axis=1)
        & np.isfinite(lengths)
        & (lengths > NORMAL_EPSILON)
    )
    if not np.all(valid):
        if diagnostics is not None:
            diagnostics["reason"] = "normal_partition_invalid_normals"
        return None
    unit_normals = patch_normals / lengths[:, None]

    angle = float(max_normal_cone_angle_deg)
    if not math.isfinite(angle) or not 5.0 <= angle <= 80.0:
        if diagnostics is not None:
            diagnostics["reason"] = "normal_partition_invalid_angle"
        return None
    cosine_threshold = math.cos(math.radians(angle))

    # Greedy deterministic spherical grouping. Every face is assigned once,
    # and each group stays within the configured angle of its seed normal.
    unassigned = np.ones(len(face_indices), dtype=bool)
    groups = []
    while np.any(unassigned):
        seed_local = int(np.flatnonzero(unassigned)[0])
        seed_normal = unit_normals[seed_local]
        remaining_local = np.flatnonzero(unassigned)
        alignment = unit_normals[remaining_local] @ seed_normal
        members = remaining_local[alignment >= cosine_threshold]
        if len(members) == 0:
            members = np.asarray([seed_local], dtype=np.int64)
        groups.append(face_indices[members])
        unassigned[members] = False

    groups, merged_singleton_count = _merge_single_face_normal_groups(
        groups=groups,
        face_indices=face_indices,
        unit_normals=unit_normals,
        max_merge_angle_deg=70.0,
    )

    if len(groups) <= 1:
        if diagnostics is not None:
            diagnostics["reason"] = "normal_partition_did_not_split_patch"
        return None

    remaining_small_group = next(
        (
            group_index
            for group_index, group_faces in enumerate(groups)
            if len(group_faces) < 2
        ),
        None,
    )
    if remaining_small_group is not None:
        if diagnostics is not None:
            diagnostics["reason"] = (
                f"normal_partition_subpatch_too_small:{remaining_small_group}"
            )
            diagnostics["merged_singleton_group_count"] = int(
                merged_singleton_count
            )
        return None

    parts = []
    for group_index, group_faces in enumerate(groups):
        part_diagnostics = {}
        part = _generate_surface_following_raster_for_patch(
            mesh=mesh,
            face_indices=group_faces,
            normals=normals,
            standoff_coordinate_units=standoff_coordinate_units,
            sample_spacing_coordinate_units=sample_spacing_coordinate_units,
            track_spacing_coordinate_units=track_spacing_coordinate_units,
            diagnostics=part_diagnostics,
        )
        if part is None:
            if diagnostics is not None:
                diagnostics["reason"] = (
                    "normal_partition_subpatch_failed:"
                    + str(part_diagnostics.get("reason", "unknown"))
                )
                diagnostics["failed_normal_cluster"] = int(group_index)
                diagnostics["normal_cluster_count"] = int(len(groups))
            return None
        parts.append(part)

    array_keys = (
        "surface_points",
        "surface_normals",
        "nozzle_points",
        "spray_directions",
        "nozzle_frames",
        "face_indices",
    )
    combined = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in array_keys
    }
    segment_ids = []
    segment_records = []
    next_segment_id = 0
    for cluster_index, part in enumerate(parts):
        local_ids = np.asarray(part["segment_ids"], dtype=np.int64)
        segment_ids.append(local_ids + next_segment_id)
        for segment in part["segments"]:
            segment_records.append({
                **segment,
                "segment_id": int(segment["segment_id"] + next_segment_id),
                "normal_cluster_id": int(cluster_index),
            })
        next_segment_id = int(np.max(local_ids)) + next_segment_id + 1

    combined["segment_ids"] = np.concatenate(segment_ids)
    combined["segments"] = segment_records
    combined["lane_count"] = int(len(segment_records))
    combined["normal_cluster_count"] = int(len(parts))
    combined["max_normal_smoothing_deviation_deg"] = float(max(
        part.get("max_normal_smoothing_deviation_deg", 0.0)
        for part in parts
    ))
    combined["normal_cluster_max_angle_deg"] = angle
    combined["normal_cluster_merged_singleton_count"] = int(
        merged_singleton_count
    )
    combined["projection_method"] = "normal_cone_partitioned_pca_triangle_barycentric"
    combined["projection_is_approximate"] = True
    if diagnostics is not None:
        diagnostics["normal_cluster_count"] = int(len(parts))
        diagnostics["reason"] = "recovered_by_normal_cone_partition"
    return combined


def _step_surface_types_for_mesh_triangles(
    mesh: trimesh.Trimesh,
    triangle_indices: np.ndarray,
) -> tuple[str, ...] | None:
    """Return authoritative STEP surface types for selected display triangles."""
    from core.step_loader import get_step_brep_context

    context = get_step_brep_context(mesh)
    if context is None:
        return None

    indices = np.unique(np.asarray(triangle_indices, dtype=np.int64))
    triangle_face_ids = np.asarray(context["triangle_face_ids"], dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(triangle_face_ids)):
        raise ValueError("STEP surface classification received an invalid mesh triangle index.")

    face_records = context["face_records"]
    surface_types = set()
    for triangle_index in indices:
        face_id = int(triangle_face_ids[int(triangle_index)])
        if face_id < 1 or face_id > len(face_records):
            raise ValueError("STEP triangle mapping references an invalid source Face ID.")
        record = face_records[face_id - 1]
        surface_type = str(record.get("surface_type", "OTHER")).strip().upper()
        surface_types.add(surface_type or "OTHER")
    return tuple(sorted(surface_types))


def _step_patch_requires_curved_raster(
    mesh: trimesh.Trimesh,
    triangle_indices: np.ndarray,
) -> tuple[bool, tuple[str, ...] | None]:
    """Use exact CAD surface type, not a flat display facet, to choose raster mode."""
    surface_types = _step_surface_types_for_mesh_triangles(mesh, triangle_indices)
    if surface_types is None:
        return False, None
    # A display triangle is always planar. Only an exact STEP PLANE is a
    # physically planar source surface; cylinders, tori, splines, cones, and
    # mixed/unknown patches must use the surface-following raster path.
    return any(value != "PLANE" for value in surface_types), surface_types


def _order_trajectory_with_planar_raster(
    mesh: trimesh.Trimesh,
    face_normals: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    nozzle_points: np.ndarray,
    spray_directions: np.ndarray,
    nozzle_frames: np.ndarray,
    face_indices: np.ndarray,
    patch_ids: np.ndarray,
    standoff_coordinate_units: float,
    sampling_spacing_coordinate_units: float,
    track_spacing_coordinate_units: float,
    coordinate_unit: str = "mm",
) -> dict:
    """Rasterize planar and smooth curved patches with explicit fallback metadata."""
    keys = (
        "surface_points", "surface_normals", "nozzle_points",
        "spray_directions", "nozzle_frames", "face_indices",
    )
    collected = {key: [] for key in keys}
    collected["patch_ids"] = []
    collected["segment_ids"] = []
    segment_metadata = []
    next_segment_id = 0
    rasterized_patch_ids = []
    curved_rasterized_patch_ids = []
    normal_partitioned_patch_ids = []
    fallback_patch_ids = []
    fallback_reason_counts = {}
    fallback_patch_reasons = []
    curved_normal_smoothing_deviation_max_deg = 0.0
    step_curvature_forced_patch_ids = []
    step_curvature_surface_types_by_patch = {}

    for patch_id in np.unique(patch_ids):
        indices = np.flatnonzero(patch_ids == patch_id)
        patch_faces = np.unique(face_indices[indices])
        force_curved_raster, exact_step_surface_types = (
            _step_patch_requires_curved_raster(mesh, patch_faces)
        )
        if force_curved_raster:
            step_curvature_forced_patch_ids.append(int(patch_id))
            step_curvature_surface_types_by_patch[str(int(patch_id))] = list(
                exact_step_surface_types or ()
            )
            raster = None
        else:
            raster = _generate_planar_raster_for_patch(
                mesh=mesh,
                face_indices=patch_faces,
                normals=np.asarray(face_normals, dtype=np.float64),
                standoff_coordinate_units=standoff_coordinate_units,
                sample_spacing_coordinate_units=sampling_spacing_coordinate_units,
                track_spacing_coordinate_units=track_spacing_coordinate_units,
            )
        if raster is not None:
            patch_result = raster
            rasterized_patch_ids.append(int(patch_id))
            for segment in raster["segments"]:
                segment_metadata.append({
                    **segment,
                    "segment_id": int(segment["segment_id"] + next_segment_id),
                    "patch_id": int(patch_id),
                    "path_type": "planar_raster",
                })
            local_segments = raster["segment_ids"] + next_segment_id
        else:
            curved_diagnostics = {}
            curved = _generate_surface_following_raster_for_patch(
                mesh=mesh,
                face_indices=patch_faces,
                normals=np.asarray(face_normals, dtype=np.float64),
                standoff_coordinate_units=standoff_coordinate_units,
                sample_spacing_coordinate_units=sampling_spacing_coordinate_units,
                track_spacing_coordinate_units=track_spacing_coordinate_units,
                diagnostics=curved_diagnostics,
            )
            used_normal_partition = False
            if curved is None and curved_diagnostics.get("reason") in {
                "inconsistent_or_cancelling_normals",
                "folded_patch_opposing_normals",
            }:
                partition_diagnostics = {}
                partitioned = _generate_partitioned_surface_raster_for_patch(
                    mesh=mesh,
                    face_indices=patch_faces,
                    normals=np.asarray(face_normals, dtype=np.float64),
                    standoff_coordinate_units=standoff_coordinate_units,
                    sample_spacing_coordinate_units=sampling_spacing_coordinate_units,
                    track_spacing_coordinate_units=track_spacing_coordinate_units,
                    max_normal_cone_angle_deg=45.0,
                    diagnostics=partition_diagnostics,
                )
                if partitioned is not None:
                    curved = partitioned
                    used_normal_partition = True
                    curved_diagnostics.update(partition_diagnostics)
                else:
                    curved_diagnostics["partition_attempt_reason"] = (
                        partition_diagnostics.get("reason", "unknown")
                    )

            if curved is not None:
                patch_result = curved
                curved_normal_smoothing_deviation_max_deg = max(
                    curved_normal_smoothing_deviation_max_deg,
                    float(curved.get("max_normal_smoothing_deviation_deg", 0.0)),
                )
                curved_rasterized_patch_ids.append(int(patch_id))
                if used_normal_partition:
                    normal_partitioned_patch_ids.append(int(patch_id))
                for segment in curved["segments"]:
                    segment_metadata.append({
                        **segment,
                        "segment_id": int(segment["segment_id"] + next_segment_id),
                        "patch_id": int(patch_id),
                        "path_type": "curved_surface_raster",
                    })
                local_segments = curved["segment_ids"] + next_segment_id
            else:
                fallback_patch_ids.append(int(patch_id))
                fallback_reason = curved_diagnostics.get(
                    "partition_attempt_reason",
                    curved_diagnostics.get("reason", "unspecified_rejection"),
                )
                fallback_reason_counts[fallback_reason] = (
                    fallback_reason_counts.get(fallback_reason, 0) + 1
                )
                fallback_patch_reasons.append({
                    "patch_id": int(patch_id),
                    "reason": fallback_reason,
                    "source_face_count": int(len(patch_faces)),
                    "source_sample_count": int(len(indices)),
                })
                local_order = _nearest_neighbor_order(nozzle_points[indices])
                ordered_indices = indices[local_order]
                patch_result = {
                    "surface_points": surface_points[ordered_indices],
                    "surface_normals": surface_normals[ordered_indices],
                    "nozzle_points": nozzle_points[ordered_indices],
                    "spray_directions": spray_directions[ordered_indices],
                    "nozzle_frames": nozzle_frames[ordered_indices],
                    "face_indices": face_indices[ordered_indices],
                }
                local_segments = np.full(len(ordered_indices), next_segment_id, dtype=np.int64)
                segment_metadata.append({
                    "segment_id": int(next_segment_id),
                    "patch_id": int(patch_id),
                    "sample_count": int(len(ordered_indices)),
                    "path_type": "nearest_neighbor_fallback_projection_rejected",
                })

        for key in keys:
            collected[key].append(patch_result[key])
        collected["patch_ids"].append(np.full(len(patch_result["surface_points"]), int(patch_id), dtype=np.int64))
        collected["segment_ids"].append(local_segments)
        next_segment_id = int(np.max(local_segments)) + 1

    if not collected["surface_points"]:
        raise RuntimeError("No trajectory segments were generated.")

    result = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    result, step_surface_refinement = _refine_trajectory_with_step_brep(
        mesh=mesh,
        trajectory=result,
        standoff_coordinate_units=standoff_coordinate_units,
        coordinate_unit=coordinate_unit,
    )
    result["step_surface_refinement"] = step_surface_refinement
    if step_surface_refinement["status"] == "PASS":
        print(
            "STEP B-REP PATH REFINEMENT | "
            f"{step_surface_refinement['evaluated_sample_count']:,}/"
            f"{step_surface_refinement['sample_count']:,} samples | "
            f"position correction median/P95/max "
            f"{step_surface_refinement['median_position_correction_mm']:.6f}/"
            f"{step_surface_refinement['p95_position_correction_mm']:.6f}/"
            f"{step_surface_refinement['maximum_position_correction_mm']:.6f} mm | "
            f"max normal correction "
            f"{step_surface_refinement['maximum_mesh_to_brep_normal_change_deg']:.3f} deg "
            f"at sample {step_surface_refinement['maximum_normal_change_sample_index']} "
            f"STEP Face {step_surface_refinement['maximum_normal_change_step_face_id']} "
            f"({step_surface_refinement['maximum_normal_change_step_surface_type']})"
        )
        print(
            "STEP SURFACE TYPE SAMPLE COUNTS | "
            + ", ".join(f"{key}: {value}" for key, value in step_surface_refinement['step_surface_type_sample_counts'].items())
        )
    elif step_surface_refinement["status"] not in {"NOT_APPLICABLE"}:
        print(
            "STEP B-REP PATH REFINEMENT | "
            f"{step_surface_refinement['status']} | "
            f"{step_surface_refinement.get('failure_reason', step_surface_refinement.get('reason', 'unknown'))}"
        )
    step_geometry_report = mesh.metadata.get("step_geometry_report", {})
    step_topology = (
        step_geometry_report.get("topology", {})
        if isinstance(step_geometry_report, dict) else {}
    )
    step_brep = (
        step_topology.get("face_edge_adjacency", {})
        if isinstance(step_topology, dict) else {}
    )
    step_face_adjacency = (
        step_brep.get("face_adjacency")
        if isinstance(step_brep, dict) else None
    )
    triangle_step_face_ids = mesh.metadata.get("triangle_face_ids")
    has_exact_step_topology = (
        isinstance(triangle_step_face_ids, (list, tuple, np.ndarray))
        and isinstance(step_face_adjacency, dict)
        and bool(step_face_adjacency)
    )
    result, segment_metadata, surface_boundary_splits = (
        _split_strokes_at_surface_discontinuities(
            mesh=mesh,
            trajectory=result,
            segment_metadata=segment_metadata,
            maximum_normal_change_deg=30.0,
            source_face_ids=(
                np.asarray(triangle_step_face_ids, dtype=np.int64)
                if has_exact_step_topology else None
            ),
            source_face_adjacency=(
                step_face_adjacency if has_exact_step_topology else None
            ),
        )
    )
    result, segment_metadata, travel_ordering = _optimize_stroke_order_for_travel(
        trajectory=result,
        segment_metadata=segment_metadata,
    )
    # Rebuild X/Y axes after stroke ordering and reversal. This avoids the
    # independent global-reference projection that can cause 180-degree roll
    # flips while preserving each sample's spray-axis direction.
    result["nozzle_frames"] = _build_parallel_transport_nozzle_frames(
        result["spray_directions"],
        result["segment_ids"],
    )
    result["segments"] = segment_metadata
    result["surface_boundary_splits"] = surface_boundary_splits
    result["travel_ordering"] = travel_ordering
    result["max_normal_smoothing_deviation_deg"] = float(
        curved_normal_smoothing_deviation_max_deg
    )
    result["rasterized_planar_patch_ids"] = rasterized_patch_ids
    result["rasterized_curved_patch_ids"] = curved_rasterized_patch_ids
    result["normal_partitioned_patch_ids"] = normal_partitioned_patch_ids
    result["nearest_neighbor_fallback_patch_ids"] = fallback_patch_ids

    # Keep sample and segment counts by the actual generator used for each
    # patch. Playback's total is the final output sample count, not lane count.
    segment_types_by_patch = {}
    for segment in segment_metadata:
        patch_id = int(segment.get("patch_id", -1))
        path_type = str(segment.get("path_type", "unknown"))
        segment_types_by_patch.setdefault(patch_id, path_type)

    output_patch_ids = result["patch_ids"]
    per_patch_statistics = []
    for patch_id in np.unique(output_patch_ids):
        patch_id = int(patch_id)
        patch_mask = output_patch_ids == patch_id
        patch_segments = [
            segment for segment in segment_metadata
            if int(segment.get("patch_id", -1)) == patch_id
        ]
        path_type = segment_types_by_patch.get(patch_id, "unknown")
        per_patch_statistics.append({
            "patch_id": patch_id,
            "path_type": path_type,
            "sample_count": int(np.count_nonzero(patch_mask)),
            "segment_count": int(len(patch_segments)),
        })

    def _count_path_type_samples(path_type):
        patch_id_set = {
            item["patch_id"] for item in per_patch_statistics
            if item["path_type"] == path_type
        }
        return int(np.count_nonzero(np.isin(output_patch_ids, list(patch_id_set))))

    path_type_names = {
        "planar_raster": "planar",
        "curved_surface_raster": "curved",
        "nearest_neighbor_fallback_projection_rejected": "fallback",
    }
    path_statistics = {
        "source_candidate_sample_count": int(len(surface_points)),
        "playback_sample_count": int(len(result["surface_points"])),
        "planar_sample_count": _count_path_type_samples("planar_raster"),
        "curved_sample_count": _count_path_type_samples("curved_surface_raster"),
        "fallback_sample_count": _count_path_type_samples(
            "nearest_neighbor_fallback_projection_rejected"
        ),
        "planar_segment_count": int(sum(
            1 for item in segment_metadata
            if item.get("path_type") == "planar_raster"
        )),
        "curved_segment_count": int(sum(
            1 for item in segment_metadata
            if item.get("path_type") == "curved_surface_raster"
        )),
        "fallback_segment_count": int(sum(
            1 for item in segment_metadata
            if item.get("path_type") == "nearest_neighbor_fallback_projection_rejected"
        )),
        "planar_patch_ids": [int(value) for value in rasterized_patch_ids],
        "curved_patch_ids": [int(value) for value in curved_rasterized_patch_ids],
        "normal_partitioned_patch_ids": [
            int(value) for value in normal_partitioned_patch_ids
        ],
        "fallback_patch_ids": [int(value) for value in fallback_patch_ids],
        "fallback_reason_counts": fallback_reason_counts,
        "fallback_patch_reasons": fallback_patch_reasons,
        "step_curvature_forced_patch_ids": [
            int(value) for value in step_curvature_forced_patch_ids
        ],
        "step_curvature_forced_patch_count": int(
            len(step_curvature_forced_patch_ids)
        ),
        "step_curvature_surface_types_by_patch": (
            step_curvature_surface_types_by_patch
        ),
        "per_patch": per_patch_statistics,
        "note": (
            "Playback samples are generated path points. Source candidate samples "
            "are pre-rasterization samples; their counts are diagnostic and are "
            "not expected to match one-to-one."
        ),
    }
    result["path_statistics"] = path_statistics

    if step_curvature_forced_patch_ids:
        surface_type_counts = {}
        for values in step_curvature_surface_types_by_patch.values():
            for surface_type in values:
                surface_type_counts[surface_type] = (
                    surface_type_counts.get(surface_type, 0) + 1
                )
        print(
            "STEP SURFACE-AWARE RASTER | "
            f"{len(step_curvature_forced_patch_ids)} exact non-planar/mixed patches "
            "routed away from planar raster | "
            + ", ".join(
                f"{key}: {value}"
                for key, value in sorted(surface_type_counts.items())
            )
        )

    print("\n" + "=" * 68)
    print("STAGE 3 PATH GENERATION DIAGNOSTICS")
    print("=" * 68)
    print(
        f"Samples | Planar: {path_statistics['planar_sample_count']:,} | "
        f"Curved: {path_statistics['curved_sample_count']:,} | "
        f"Fallback: {path_statistics['fallback_sample_count']:,} | "
        f"Total: {path_statistics['playback_sample_count']:,}"
    )
    print(
        f"Segments | Planar: {path_statistics['planar_segment_count']} | "
        f"Curved: {path_statistics['curved_segment_count']} | "
        f"Fallback: {path_statistics['fallback_segment_count']}"
    )
    print(
        f"Patches  | Planar: {len(rasterized_patch_ids)} | "
        f"Curved: {len(curved_rasterized_patch_ids)} | "
        f"Fallback: {len(fallback_patch_ids)}"
    )
    print(
        f"Curved recovery | Normal-cone partitioned patches: "
        f"{len(normal_partitioned_patch_ids)}"
    )
    print(
        "SURFACE ADJACENCY AUTHORITY | "
        f"{surface_boundary_splits['adjacency_authority']}"
    )
    print(
        "SURFACE BOUNDARY SPLITS | "
        f"{surface_boundary_splits['added_stroke_count']} additional strokes | "
        f"non-adjacent faces: "
        f"{surface_boundary_splits['split_transition_counts']['non_adjacent_source_faces']} | "
        f"sharp creases: "
        f"{surface_boundary_splits['split_transition_counts']['sharp_source_mesh_crease']} "
        "(spray OFF at each boundary)"
    )
    print(
        "TRAVEL ORDER OPTIMIZATION | "
        f"{travel_ordering['stroke_count']} strokes | "
        f"{travel_ordering['original_travel_distance_coordinate_units']:.3f} -> "
        f"{travel_ordering['optimized_travel_distance_coordinate_units']:.3f} "
        f"coordinate units | {travel_ordering['reduction_percent']:.1f}% reduction "
        "(greedy nearest endpoint; collision check NOT performed)"
    )
    if fallback_reason_counts:
        print("Fallback rejection reasons:")
        for reason, count in sorted(
            fallback_reason_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"  - {reason}: {count} patch(es)")
        print("Fallback patch details:")
        for item in fallback_patch_reasons:
            print(
                f"  - Patch {item['patch_id']}: {item['reason']} | "
                f"faces={item['source_face_count']} | "
                f"samples={item['source_sample_count']}"
            )
    print("=" * 68 + "\n")
    return result


def _order_trajectory_by_patch(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    nozzle_points: np.ndarray,
    spray_directions: np.ndarray,
    nozzle_frames: np.ndarray,
    face_indices: np.ndarray,
    patch_ids: np.ndarray,
) -> dict:

    ordered_indices = []
    segment_lengths = []

    unique_patch_ids = np.unique(patch_ids)

    for segment_id, patch_id in enumerate(unique_patch_ids):

        patch_indices = np.flatnonzero(patch_ids == patch_id)

        if len(patch_indices) == 0:
            continue

        local_order = _nearest_neighbor_order(
            nozzle_points[patch_indices]
        )

        ordered_patch_indices = patch_indices[local_order]
        ordered_indices.append(ordered_patch_indices)

        segment_lengths.append({
            "segment_id": int(segment_id),
            "patch_id": int(patch_id),
            "sample_count": int(len(ordered_patch_indices)),
        })

    if not ordered_indices:
        raise RuntimeError("No trajectory segments were generated.")

    order = np.concatenate(ordered_indices)

    ordered_segment_ids = np.empty(len(order), dtype=np.int64)

    cursor = 0

    for segment in segment_lengths:

        length = segment["sample_count"]

        ordered_segment_ids[cursor:cursor + length] = (
            segment["segment_id"]
        )

        cursor += length

    return {
        "surface_points": surface_points[order],
        "surface_normals": surface_normals[order],
        "nozzle_points": nozzle_points[order],
        "spray_directions": spray_directions[order],
        "nozzle_frames": nozzle_frames[order],
        "face_indices": face_indices[order],
        "patch_ids": patch_ids[order],
        "segment_ids": ordered_segment_ids,
        "segments": segment_lengths,
    }


# ------------------------------------------------------------
# Path metrics
# ------------------------------------------------------------

def _calculate_path_metrics(
    nozzle_points: np.ndarray,
    segment_ids: np.ndarray,
    coordinate_unit: str,
) -> dict:

    total_length_coordinate_units = 0.0
    segment_results = []

    for segment_id in np.unique(segment_ids):

        indices = np.flatnonzero(segment_ids == segment_id)
        points = nozzle_points[indices]

        if len(points) < 2:
            segment_length = 0.0
        else:
            distances = np.linalg.norm(
                np.diff(points, axis=0),
                axis=1,
            )

            segment_length = float(np.sum(distances))

        total_length_coordinate_units += segment_length

        segment_results.append({
            "segment_id": int(segment_id),
            "sample_count": int(len(points)),
            "path_length_coordinate_units": segment_length,
            "path_length_mm": _coordinate_units_to_mm(
                segment_length,
                coordinate_unit,
            ),
        })

    return {
        "coordinate_unit": coordinate_unit,
        "total_path_length_coordinate_units": float(
            total_length_coordinate_units
        ),
        "total_path_length_mm": _coordinate_units_to_mm(
            total_length_coordinate_units,
            coordinate_unit,
        ),
        "segment_count": int(len(segment_results)),
        "segments": segment_results,
        "measurement_note": (
            "Path length includes movements within each patch only. "
            "Travel between separate patches is excluded."
        ),
    }



def _barycentric_coordinates_on_triangle(
    point: np.ndarray,
    triangle: np.ndarray,
    tolerance: float = 1e-6,
) -> np.ndarray | None:
    """Return barycentric weights for a 3D point projected onto a triangle plane."""
    point = np.asarray(point, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)
    if point.shape != (3,) or triangle.shape != (3, 3):
        raise ValueError("Point and triangle must have shapes (3,) and (3, 3).")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(triangle)):
        raise ValueError("Point and triangle coordinates must be finite.")

    edge_a = triangle[1] - triangle[0]
    edge_b = triangle[2] - triangle[0]
    offset = point - triangle[0]
    d00 = float(np.dot(edge_a, edge_a))
    d01 = float(np.dot(edge_a, edge_b))
    d11 = float(np.dot(edge_b, edge_b))
    d20 = float(np.dot(offset, edge_a))
    d21 = float(np.dot(offset, edge_b))
    denominator = d00 * d11 - d01 * d01
    scale = d00 * d11
    if (
        not math.isfinite(denominator)
        or not math.isfinite(scale)
        or scale <= np.finfo(float).tiny
        or abs(denominator) <= 1e-14 * scale
    ):
        return None

    weight_b = (d11 * d20 - d01 * d21) / denominator
    weight_c = (d00 * d21 - d01 * d20) / denominator
    weights = np.asarray(
        [1.0 - weight_b - weight_c, weight_b, weight_c],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(weights))
        or float(np.min(weights)) < -float(tolerance)
        or float(np.max(weights)) > 1.0 + float(tolerance)
    ):
        return None
    weights = np.clip(weights, 0.0, 1.0)
    total = float(np.sum(weights))
    if total <= NORMAL_EPSILON:
        return None
    return weights / total


def _refine_trajectory_with_step_brep(
    mesh: trimesh.Trimesh,
    trajectory: dict,
    standoff_coordinate_units: float,
    coordinate_unit: str,
) -> tuple[dict, dict]:
    """Replace mesh-interpolated path samples with exact STEP B-Rep evaluations.

    Refinement is atomic: if any sample cannot be mapped to its original STEP
    face, the complete original trajectory is retained.
    """
    summary = {
        "status": "NOT_APPLICABLE",
        "method": "OPENCASCADE_BREP_ADAPTOR_D1_WITH_TRIANGULATION_UV_MAPPING",
        "sample_count": int(len(trajectory.get("surface_points", []))),
        "evaluated_sample_count": 0,
        "failed_sample_count": 0,
        "median_position_correction_mm": None,
        "p95_position_correction_mm": None,
        "maximum_position_correction_mm": None,
        "maximum_mesh_to_brep_normal_change_deg": None,
        "normal_source": "DISPLAY_MESH",
        "runtime_context_attached": bool(
            getattr(mesh, "_step_brep_context", None) is not None
        ),
        "mesh_metadata_type": type(
            getattr(mesh, "metadata", None)
        ).__name__,
    }
    if _normalize_unit(coordinate_unit) != "mm":
        summary["status"] = "SKIPPED_NON_MM_COORDINATE_UNIT"
        summary["reason"] = (
            "STEP B-Rep coordinates are canonical millimetres; exact path "
            "refinement is enabled only when Stage 3 coordinates are configured as mm."
        )
        return dict(trajectory), summary

    try:
        from core.step_loader import (
            evaluate_step_triangle_sample,
            get_step_brep_context,
        )
        context = get_step_brep_context(mesh)
    except Exception as exc:
        summary["status"] = "FALLBACK_INVALID_STEP_CONTEXT"
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        return dict(trajectory), summary
    if context is None:
        summary["status"] = "MISSING_STEP_BREP_CONTEXT"
        metadata = getattr(mesh, "metadata", None)
        if not isinstance(metadata, dict):
            summary["reason"] = (
                "Mesh metadata is not a dictionary; STEP runtime context "
                "lookup cannot proceed."
            )
        elif not summary["runtime_context_attached"]:
            summary["reason"] = (
                "The mesh reaching Stage 3 has no _step_brep_context runtime attribute."
            )
        else:
            summary["reason"] = (
                "STEP runtime context lookup returned None despite an attached attribute."
            )
        return dict(trajectory), summary

    points = np.asarray(trajectory["surface_points"], dtype=np.float64)
    normals = np.asarray(trajectory["surface_normals"], dtype=np.float64)
    triangle_indices = np.asarray(trajectory["face_indices"], dtype=np.int64)
    segment_ids = np.asarray(trajectory["segment_ids"], dtype=np.int64)
    if (
        points.shape != (len(triangle_indices), 3)
        or normals.shape != points.shape
        or segment_ids.shape != (len(points),)
    ):
        raise ValueError("STEP refinement requires aligned trajectory sample arrays.")

    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    exact_points = np.empty_like(points)
    exact_normals = np.empty_like(normals)
    position_corrections = np.empty(len(points), dtype=np.float64)
    normal_changes = np.empty(len(points), dtype=np.float64)
    step_face_ids = np.empty(len(points), dtype=np.int64)
    step_surface_types = np.empty(len(points), dtype=object)

    for sample_index, (point, triangle_index) in enumerate(zip(points, triangle_indices)):
        triangle_index = int(triangle_index)
        if triangle_index < 0 or triangle_index >= len(mesh_faces):
            summary["status"] = "FALLBACK_SAMPLE_MAPPING_FAILED"
            summary["failed_sample_count"] = 1
            summary["failed_sample_index"] = int(sample_index)
            summary["failure_reason"] = "invalid_source_triangle_index"
            return dict(trajectory), summary
        triangle = mesh_vertices[mesh_faces[triangle_index]]
        weights = _barycentric_coordinates_on_triangle(
            point, triangle, tolerance=2e-5
        )
        if weights is None:
            summary["status"] = "FALLBACK_SAMPLE_MAPPING_FAILED"
            summary["failed_sample_count"] = 1
            summary["failed_sample_index"] = int(sample_index)
            summary["failure_reason"] = "sample_outside_source_triangle"
            return dict(trajectory), summary
        try:
            evaluated = evaluate_step_triangle_sample(mesh, triangle_index, weights)
        except Exception as exc:
            summary["status"] = "FALLBACK_EXACT_SURFACE_EVALUATION_FAILED"
            summary["failed_sample_count"] = 1
            summary["failed_sample_index"] = int(sample_index)
            summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
            return dict(trajectory), summary

        exact_point = np.asarray(evaluated["point_mm"], dtype=np.float64)
        exact_normal = np.asarray(evaluated["normal"], dtype=np.float64)
        exact_normal_length = float(np.linalg.norm(exact_normal))
        reference_normal = normals[sample_index]
        reference_length = float(np.linalg.norm(reference_normal))
        if (
            exact_point.shape != (3,)
            or exact_normal.shape != (3,)
            or not np.all(np.isfinite(exact_point))
            or not np.all(np.isfinite(exact_normal))
            or exact_normal_length <= NORMAL_EPSILON
            or reference_length <= NORMAL_EPSILON
        ):
            summary["status"] = "FALLBACK_EXACT_SURFACE_EVALUATION_FAILED"
            summary["failed_sample_count"] = 1
            summary["failed_sample_index"] = int(sample_index)
            summary["failure_reason"] = "non_finite_or_degenerate_exact_surface_result"
            return dict(trajectory), summary

        exact_normal /= exact_normal_length
        reference_normal = reference_normal / reference_length
        # Preserve the established outward-normal convention if the CAD face
        # orientation differs from the display mesh's global orientation.
        if float(np.dot(exact_normal, reference_normal)) < 0.0:
            exact_normal *= -1.0
        exact_points[sample_index] = exact_point
        exact_normals[sample_index] = exact_normal
        step_face_ids[sample_index] = int(evaluated["face_id"])
        step_surface_types[sample_index] = str(evaluated["surface_type"])
        position_corrections[sample_index] = float(np.linalg.norm(exact_point - point))
        normal_changes[sample_index] = math.degrees(math.acos(float(np.clip(
            np.dot(exact_normal, reference_normal), -1.0, 1.0
        ))))

    # Smooth exact normals only within each existing spray stroke. Raw CAD
    # normals remain available for crease classification and audit reporting.
    smoothed_normals = exact_normals.copy()
    maximum_smoothing_deviation = 0.0
    for segment_id in np.unique(segment_ids):
        indices = np.flatnonzero(segment_ids == segment_id)
        local_normals, local_deviation = _smooth_normals_with_deviation_limit(
            exact_normals[indices], max_deviation_deg=8.0, iterations=6
        )
        smoothed_normals[indices] = local_normals
        maximum_smoothing_deviation = max(
            maximum_smoothing_deviation, float(local_deviation)
        )

    refined = dict(trajectory)
    refined["surface_points"] = exact_points
    refined["source_surface_normals"] = exact_normals
    refined["step_face_ids"] = step_face_ids
    refined["step_surface_types"] = step_surface_types
    refined["surface_normals"] = smoothed_normals
    refined["spray_directions"] = -smoothed_normals
    refined["nozzle_points"] = (
        exact_points + float(standoff_coordinate_units) * smoothed_normals
    )
    refined["nozzle_frames"] = _build_nozzle_frames(refined["spray_directions"])
    summary.update({
        "status": "PASS",
        "evaluated_sample_count": int(len(points)),
        "failed_sample_count": 0,
        "median_position_correction_mm": float(np.median(position_corrections)),
        "p95_position_correction_mm": float(np.percentile(position_corrections, 95)),
        "maximum_position_correction_mm": float(np.max(position_corrections)),
        "maximum_mesh_to_brep_normal_change_deg": float(np.max(normal_changes)),
        "maximum_normal_change_sample_index": int(np.argmax(normal_changes)),
        "maximum_normal_change_step_face_id": int(step_face_ids[int(np.argmax(normal_changes))]),
        "maximum_normal_change_step_surface_type": str(step_surface_types[int(np.argmax(normal_changes))]),
        "step_surface_type_sample_counts": {
            str(surface_type): int(np.count_nonzero(step_surface_types == surface_type))
            for surface_type in sorted(set(step_surface_types.tolist()))
        },
        "mean_mesh_to_brep_normal_change_deg": float(np.mean(normal_changes)),
        "maximum_normal_smoothing_deviation_deg": float(maximum_smoothing_deviation),
        "normal_source": "EXACT_STEP_BREP_FIRST_DERIVATIVES",
        "position_source": "EXACT_STEP_BREP_SURFACE_EVALUATION",
        "atomic_fallback": False,
    })
    return refined, summary



def _split_strokes_at_surface_discontinuities(
    mesh: trimesh.Trimesh,
    trajectory: dict,
    segment_metadata: list[dict],
    maximum_normal_change_deg: float = 30.0,
    source_face_ids: np.ndarray | None = None,
    source_face_adjacency: dict | None = None,
) -> tuple[dict, list[dict], dict]:
    """Split strokes using exact STEP face topology when it is available.

    Trajectory face indices refer to display-mesh triangles. source_face_ids
    maps those triangles to original STEP Face IDs. Exact B-Rep adjacency is
    authoritative; triangle vertex sharing is only a fallback.
    """
    face_indices = np.asarray(trajectory["face_indices"], dtype=np.int64)
    old_segments = np.asarray(trajectory["segment_ids"], dtype=np.int64)
    if face_indices.shape != old_segments.shape:
        raise ValueError("Face indices and segment IDs must have matching shapes.")
    threshold = float(maximum_normal_change_deg)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 180.0:
        raise ValueError("Maximum normal change must be in (0, 180] degrees.")

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    source_sample_normals = np.asarray(
        trajectory.get("source_surface_normals", np.empty((0, 3))), dtype=np.float64
    )
    has_exact_sample_normals = source_sample_normals.shape == (len(face_indices), 3)
    use_exact_brep = source_face_ids is not None and isinstance(source_face_adjacency, dict)
    cad_face_ids = None
    brep_adjacency_pairs: set[tuple[int, int]] = set()
    if use_exact_brep:
        cad_face_ids = np.asarray(source_face_ids, dtype=np.int64)
        if cad_face_ids.ndim != 1 or len(cad_face_ids) != len(mesh.faces):
            raise ValueError("source_face_ids must map every display triangle to one STEP Face ID.")
        for face_key, neighbors in source_face_adjacency.items():
            try:
                face_id = int(face_key)
                for neighbor in neighbors:
                    neighbor_id = int(neighbor)
                    if face_id != neighbor_id:
                        brep_adjacency_pairs.add(tuple(sorted((face_id, neighbor_id))))
            except (TypeError, ValueError):
                continue
        if not brep_adjacency_pairs:
            raise ValueError("Exact STEP adjacency was supplied but contains no adjacent face pairs.")

    metadata_by_id = {
        int(item["segment_id"]): dict(item)
        for item in segment_metadata
        if "segment_id" in item
    }
    starts = np.r_[0, np.flatnonzero(np.diff(old_segments) != 0) + 1]
    ends = np.r_[starts[1:], len(old_segments)]
    blocks = []
    split_reasons = {
        "non_adjacent_source_faces": 0,
        "sharp_source_mesh_crease": 0,
    }

    for start, end in zip(starts, ends):
        split_after = []
        for index in range(int(start), int(end) - 1):
            face_a = int(face_indices[index])
            face_b = int(face_indices[index + 1])
            if not (
                0 <= face_a < len(face_normals)
                and 0 <= face_b < len(face_normals)
            ):
                continue

            if use_exact_brep:
                cad_face_a = int(cad_face_ids[face_a])
                cad_face_b = int(cad_face_ids[face_b])
                if cad_face_a == cad_face_b:
                    continue
                if tuple(sorted((cad_face_a, cad_face_b))) not in brep_adjacency_pairs:
                    split_after.append(index + 1)
                    split_reasons["non_adjacent_source_faces"] += 1
                    continue
                # Prefer exact first-derivative normals for crease angles;
                # exact B-Rep shared-edge topology remains authoritative.
                if has_exact_sample_normals:
                    normal_a = source_sample_normals[index]
                    normal_b = source_sample_normals[index + 1]
                else:
                    normal_a = face_normals[face_a]
                    normal_b = face_normals[face_b]
            else:
                if face_a == face_b:
                    continue
                shared_vertices = int(
                    len(np.intersect1d(mesh.faces[face_a], mesh.faces[face_b]))
                )
                # Two triangles are edge-adjacent only when they share
                # at least two vertices. Sharing a single vertex is merely a
                # point contact and must not be treated as a continuous surface.
                if shared_vertices < 2:
                    split_after.append(index + 1)
                    split_reasons["non_adjacent_source_faces"] += 1
                    continue
                normal_a = face_normals[face_a]
                normal_b = face_normals[face_b]
            length_a = float(np.linalg.norm(normal_a))
            length_b = float(np.linalg.norm(normal_b))
            if length_a <= NORMAL_EPSILON or length_b <= NORMAL_EPSILON:
                continue
            cosine = float(np.clip(
                np.dot(normal_a / length_a, normal_b / length_b),
                -1.0,
                1.0,
            ))
            angle = math.degrees(math.acos(cosine))
            if angle > threshold:
                split_after.append(index + 1)
                split_reasons["sharp_source_mesh_crease"] += 1

        boundaries = [int(start), *split_after, int(end)]
        old_id = int(old_segments[int(start)])
        base_metadata = metadata_by_id.get(old_id, {"segment_id": old_id})
        for chunk_index, (chunk_start, chunk_end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            if chunk_end <= chunk_start:
                continue
            blocks.append((
                int(chunk_start),
                int(chunk_end),
                old_id,
                chunk_index,
                len(split_after),
                base_metadata,
            ))

    sample_keys = (
        "surface_points", "surface_normals", "nozzle_points",
        "spray_directions", "nozzle_frames", "face_indices", "patch_ids",
        "source_surface_normals",
        "step_face_ids",
        "step_surface_types",
    )
    updated = dict(trajectory)
    new_ids = np.empty(len(old_segments), dtype=np.int64)
    new_metadata = []
    for new_id, (start, end, old_id, chunk_index, split_count, base) in enumerate(blocks):
        new_ids[start:end] = new_id
        item = dict(base)
        item.update({
            "original_segment_id": int(old_id),
            "segment_id": int(new_id),
            "surface_discontinuity_split": bool(split_count),
            "surface_discontinuity_chunk_index": int(chunk_index),
            "surface_discontinuity_split_count_in_original_stroke": int(split_count),
            "sample_count": int(end - start),
        })
        new_metadata.append(item)

    updated["segment_ids"] = new_ids
    for key in sample_keys:
        if key in trajectory and len(trajectory[key]) != len(old_segments):
            raise ValueError(f"{key} does not match trajectory sample count.")

    stats = {
        "method": (
            "split_at_exact_step_brep_boundaries_and_sharp_face_creases"
            if use_exact_brep else
            "split_at_non_adjacent_mesh_faces_and_sharp_mesh_creases"
        ),
        "adjacency_authority": (
            "EXACT_STEP_BREP_SHARED_EDGES"
            if use_exact_brep else "DISPLAY_MESH_SHARED_EDGES_FALLBACK"
        ),
        "maximum_source_face_normal_change_deg": threshold,
        "original_stroke_count": int(len(starts)),
        "resulting_stroke_count": int(len(blocks)),
        "added_stroke_count": int(max(0, len(blocks) - len(starts))),
        "split_transition_counts": split_reasons,
        "samples_preserved": True,
        "surface_points_or_normals_modified": False,
        "note": (
            "Spray is OFF at each inserted boundary. Source positions and normals "
            "are unchanged; travel links remain straight-line estimates only."
        ),
    }
    return updated, new_metadata, stats


def _optimize_stroke_order_for_travel(
    trajectory: dict,
    segment_metadata: list[dict],
) -> tuple[dict, list[dict], dict]:
    """Greedily order complete strokes by nearest endpoint, reversing strokes when useful.

    The stroke geometry and per-sample nozzle frames are preserved. Only the
    order of complete spray strokes changes; no spray-on connector is created.
    The first stroke is selected near the overall path centroid because no
    robot home pose is available at Stage 3.
    """
    sample_keys = (
        "surface_points",
        "surface_normals",
        "nozzle_points",
        "spray_directions",
        "nozzle_frames",
        "face_indices",
        "patch_ids",
        "source_surface_normals",
    )
    segments = np.asarray(trajectory["segment_ids"], dtype=np.int64)
    if segments.ndim != 1 or len(segments) == 0:
        return trajectory, segment_metadata, {
            "method": "greedy_nearest_endpoint",
            "stroke_count": 0,
            "original_travel_distance_coordinate_units": 0.0,
            "optimized_travel_distance_coordinate_units": 0.0,
            "reduction_percent": 0.0,
        }

    starts = np.r_[0, np.flatnonzero(np.diff(segments) != 0) + 1]
    ends = np.r_[starts[1:], len(segments)]
    blocks = [
        np.arange(int(start), int(end), dtype=np.int64)
        for start, end in zip(starts, ends)
    ]
    if len(blocks) <= 1:
        return trajectory, segment_metadata, {
            "method": "greedy_nearest_endpoint",
            "stroke_count": int(len(blocks)),
            "original_travel_distance_coordinate_units": 0.0,
            "optimized_travel_distance_coordinate_units": 0.0,
            "reduction_percent": 0.0,
        }

    points = np.asarray(trajectory["nozzle_points"], dtype=np.float64)
    if points.shape != (len(segments), 3) or not np.all(np.isfinite(points)):
        raise ValueError("Stroke ordering requires finite N×3 nozzle points.")

    old_distance = 0.0
    for left, right in zip(blocks[:-1], blocks[1:]):
        old_distance += float(np.linalg.norm(points[right[0]] - points[left[-1]]))

    centers = np.asarray([
        0.5 * (points[block[0]] + points[block[-1]])
        for block in blocks
    ])
    centroid = np.mean(centers, axis=0)
    seed = int(np.argmin(np.linalg.norm(centers - centroid, axis=1)))
    first_block = blocks[seed]
    if np.linalg.norm(points[first_block[-1]] - centroid) < np.linalg.norm(
        points[first_block[0]] - centroid
    ):
        first_block = first_block[::-1]

    remaining = set(range(len(blocks)))
    remaining.remove(seed)
    ordered_blocks = [first_block]
    ordered_indices = [seed]
    current_endpoint = points[first_block[-1]]

    while remaining:
        best_index = None
        best_reverse = False
        best_distance = float("inf")
        for candidate_index in remaining:
            candidate = blocks[candidate_index]
            distance_start = float(np.linalg.norm(
                points[candidate[0]] - current_endpoint
            ))
            distance_end = float(np.linalg.norm(
                points[candidate[-1]] - current_endpoint
            ))
            if distance_start < best_distance:
                best_index = candidate_index
                best_reverse = False
                best_distance = distance_start
            if distance_end < best_distance:
                best_index = candidate_index
                best_reverse = True
                best_distance = distance_end

        chosen = blocks[best_index]
        if best_reverse:
            chosen = chosen[::-1]
        ordered_blocks.append(chosen)
        ordered_indices.append(best_index)
        current_endpoint = points[chosen[-1]]
        remaining.remove(best_index)

    candidate_distance = 0.0
    for left_block, right_block in zip(ordered_blocks[:-1], ordered_blocks[1:]):
        candidate_distance += float(np.linalg.norm(
            points[right_block[0]] - points[left_block[-1]]
        ))
    if candidate_distance > old_distance + 1e-9:
        return trajectory, segment_metadata, {
            "method": "greedy_nearest_endpoint_with_original_order_guard",
            "stroke_count": int(len(blocks)),
            "original_travel_distance_coordinate_units": float(old_distance),
            "optimized_travel_distance_coordinate_units": float(old_distance),
            "reduction_percent": 0.0,
            "used_original_order": True,
            "collision_check": "NOT_PERFORMED",
            "note": "Greedy candidate was longer, so the original order was retained.",
        }

    ordered_sample_indices = np.concatenate(ordered_blocks)
    optimized = dict(trajectory)
    for key in sample_keys:
        if key in trajectory:
            values = np.asarray(trajectory[key])
            if len(values) != len(segments):
                raise ValueError(f"{key} does not match trajectory sample count.")
            optimized[key] = values[ordered_sample_indices].copy()

    optimized_segment_ids = np.empty(len(segments), dtype=np.int64)
    old_to_new = {}
    for new_id, block in enumerate(ordered_blocks):
        old_id = int(segments[block[0]])
        old_to_new[old_id] = int(new_id)
        optimized_segment_ids[
            sum(len(item) for item in ordered_blocks[:new_id]):
            sum(len(item) for item in ordered_blocks[:new_id + 1])
        ] = new_id
    optimized["segment_ids"] = optimized_segment_ids

    metadata_by_id = {
        int(item["segment_id"]): dict(item)
        for item in segment_metadata
        if "segment_id" in item
    }
    reordered_metadata = []
    for new_id, old_index in enumerate(ordered_indices):
        old_id = int(segments[blocks[old_index][0]])
        item = metadata_by_id.get(old_id, {"segment_id": old_id})
        item["original_segment_id"] = old_id
        item["segment_id"] = int(new_id)
        item["travel_order_index"] = int(new_id)
        item["direction_reversed_for_travel"] = bool(
            ordered_blocks[new_id][0] == blocks[old_index][-1]
        )
        reordered_metadata.append(item)

    new_points = np.asarray(optimized["nozzle_points"], dtype=np.float64)
    new_starts = np.r_[0, np.flatnonzero(np.diff(optimized_segment_ids) != 0) + 1]
    new_ends = np.r_[new_starts[1:], len(optimized_segment_ids)]
    new_distance = 0.0
    for left_end, right_start in zip(new_ends[:-1], new_starts[1:]):
        new_distance += float(np.linalg.norm(
            new_points[int(right_start)] - new_points[int(left_end - 1)]
        ))

    reduction = (
        100.0 * (old_distance - new_distance) / old_distance
        if old_distance > 1e-12 else 0.0
    )
    metadata = {
        "method": "greedy_nearest_endpoint_with_optional_stroke_reversal",
        "stroke_count": int(len(blocks)),
        "original_travel_distance_coordinate_units": float(old_distance),
        "optimized_travel_distance_coordinate_units": float(new_distance),
        "reduction_percent": float(reduction),
        "start_selection": "stroke midpoint nearest overall centroid; robot home pose unavailable",
        "collision_check": "NOT_PERFORMED",
        "note": (
            "Complete spray strokes are reordered and may be reversed. Stroke "
            "geometry is unchanged. This is a geometric travel heuristic, not "
            "collision-free or robot-joint-space motion planning."
        ),
    }
    return optimized, reordered_metadata, metadata


def _build_inter_segment_travel_plan(
    nozzle_points: np.ndarray,
    segment_ids: np.ndarray,
    coordinate_unit: str,
) -> dict:
    """Describe direct spray-OFF connector distances without timing assumptions."""
    points = np.asarray(nozzle_points, dtype=np.float64)
    segments = np.asarray(segment_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(segments):
        raise ValueError("Travel planning requires matching (N, 3) points and segment IDs.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Travel planning points must be finite.")
    mm_per_unit = MM_PER_UNIT.get(_normalize_unit(coordinate_unit))
    if mm_per_unit is None:
        raise ValueError(f"Unsupported travel-plan coordinate unit: {coordinate_unit}")
    starts = np.r_[0, np.flatnonzero(np.diff(segments) != 0) + 1] if len(segments) else np.empty(0, dtype=int)
    ends = np.r_[starts[1:], len(segments)] if len(starts) else np.empty(0, dtype=int)
    transitions = []
    total_distance_mm = 0.0
    for block_index in range(max(0, len(starts) - 1)):
        end_index = int(ends[block_index] - 1)
        start_index = int(starts[block_index + 1])
        distance_mm = float(np.linalg.norm(points[start_index] - points[end_index]) * mm_per_unit)
        transitions.append({
            "from_segment_id": int(segments[end_index]),
            "to_segment_id": int(segments[start_index]),
            "from_sample_index": end_index,
            "to_sample_index": start_index,
            "start_point_coordinate_units": points[end_index].tolist(),
            "end_point_coordinate_units": points[start_index].tolist(),
            "distance_mm": distance_mm,
            "spray_state": "OFF",
            "path_type": "direct_straight_line_visualization_only",
        })
        total_distance_mm += distance_mm
    return {
        "method": "direct_straight_line_geometry_only",
        "transition_count": len(transitions),
        "total_distance_mm": float(total_distance_mm),
        "transitions": transitions,
        "safety_status": "NOT_COLLISION_CHECKED",
        "note": "Connector distances are geometric only; no speed, time, collision or robot-motion estimate is made.",
    }


# ------------------------------------------------------------
# Advanced geometry assessment
# ------------------------------------------------------------

def _build_advanced_geometry_analysis(
    feature_data: dict,
    patch_ids: np.ndarray,
    coordinate_unit: str,
) -> dict:

    if not feature_data.get("available", False):
        return {
            "available": False,
            "coordinate_unit": coordinate_unit,
            "description": (
                "Stage 1 report was not supplied. "
                "Trajectory generation used Stage 2 face labels only."
            ),
            "trajectory_patch_features": [],
            "limitations": [
                "Cylinder and bend information was not available.",
                "No advanced feature-based trajectory adaptation was applied.",
            ],
        }

    patch_features = feature_data["patch_features"]
    advanced = feature_data["advanced_geometry"]

    trajectory_features = _build_patch_feature_summary(
        patch_ids,
        patch_features,
    )

    internal_cylinders = [
        item for item in trajectory_features
        if (
            item["cylindrical_candidate"]
            and str(item["cylinder_surface_type"]).lower()
            in ("internal", "internal_cylinder", "inner")
        )
    ]

    uncertain_cylinders = [
        item for item in trajectory_features
        if (
            item["cylindrical_candidate"]
            and item["cylinder_surface_type"] is None
        )
    ]

    bends = advanced.get("bends", [])
    holes = advanced.get("hole_candidates", [])
    edges = advanced.get("edges", [])

    if not isinstance(bends, list):
        bends = []

    if not isinstance(holes, list):
        holes = []

    if not isinstance(edges, dict):
        edges = {}

    return {
        "available": True,
        "source": "Stage 1 geometry analysis",
        "coordinate_unit": coordinate_unit,
        "summary": feature_data["summary"],
        "trajectory_patch_features": trajectory_features,
        "internal_cylinder_patches_in_trajectory": len(
            internal_cylinders
        ),
        "uncertain_cylinder_patches_in_trajectory": len(
            uncertain_cylinders
        ),
        "bend_candidates": bends,
        "hole_candidates": holes,
        "edge_analysis": edges,
        "interpretation": [
            (
                "Cylinder radius and classification are estimated "
                "from triangulated surface geometry."
            ),
            (
                "Cylinder dimensions are expressed in STL coordinate units."
            ),
            (
                "Bend candidates are retained as geometry metadata. "
                "They do not yet generate a separate bend-following path."
            ),
            (
                "Internal cylindrical patches should be reviewed "
                "before painting. Stage 2 labels remain authoritative."
            ),
            (
                "Hole candidates are not automatically painted or masked. "
                "Stage 2 face labels control inclusion."
            ),
        ],
        "limitations": [
            "No CAD feature semantics are available in STL.",
            "Cylinder fit confidence does not prove a true cylindrical feature.",
            "Hole candidates do not prove through-hole or blind-hole geometry.",
            "Bend radius and angle are not used to modify nozzle orientation.",
            "Nozzle orientation is still based on local triangle normals.",
        ],
    }


# ------------------------------------------------------------
# Engineering analysis
# ------------------------------------------------------------

def _build_analysis(
    sampling_result: dict,
    normal_info: dict,
    orientation_info: dict,
    path_metrics: dict,
    standoff_mm: float,
    spacing_mm: float,
    standoff_coordinate_units: float,
    spacing_coordinate_units: float,
    coordinate_unit: str,
    advanced_analysis: dict,
) -> dict:

    sample_count = sampling_result["sample_count"]
    candidate_face_count = sampling_result["candidate_face_count"]

    skipped_normals = sampling_result["skipped_invalid_normals"]
    skipped_geometry = sampling_result["skipped_invalid_geometry"]

    warnings = []

    if not orientation_info["orientation_reliable"]:
        warnings.append(
            "Surface outward direction is uncertain. "
            "Verify mesh orientation before robot execution."
        )

    if skipped_normals > 0:
        warnings.append(
            f"{skipped_normals} candidate faces were skipped "
            "because their normals were invalid."
        )

    if skipped_geometry > 0:
        warnings.append(
            f"{skipped_geometry} candidate faces were skipped "
            "because their geometry was invalid."
        )

    if not advanced_analysis["available"]:
        warnings.append(
            "Stage 1 advanced geometry report was not supplied."
        )

    if advanced_analysis.get(
        "internal_cylinder_patches_in_trajectory", 0
    ) > 0:
        warnings.append(
            "An internal cylindrical patch appears in the trajectory. "
            "Review Stage 2 classification before painting."
        )

    if advanced_analysis.get(
        "uncertain_cylinder_patches_in_trajectory", 0
    ) > 0:
        warnings.append(
            "Some cylindrical patches have uncertain internal/external "
            "classification."
        )

    warnings.extend([
        "Surface samples and normals currently come from the STL triangle mesh, not exact STEP B-Rep faces.",
        "The nozzle roll axis is a provisional geometric convention, not a calibrated sprayer TCP orientation.",
        "Robot reachability has not been checked.",
        "Robot collision and fixture clearance have not been checked.",
        "Spray coverage and coating thickness have not been verified.",
        "The simplified nozzle reference point is not a calibrated TCP.",
        "Nearest-neighbor trajectory ordering is not globally optimized.",
        "Travel between separate patch segments is not generated.",
    ])

    geometric_generation_pass = (
        sample_count > 0
        and candidate_face_count > 0
        and skipped_normals == 0
        and skipped_geometry == 0
    )

    orientation_reliable = bool(
        orientation_info["orientation_reliable"]
    )

    geometric_status = (
        "PASS" if geometric_generation_pass else "REVIEW"
    )

    overall_status = (
        "PASS"
        if geometric_generation_pass and orientation_reliable
        else "REVIEW"
    )

    explanation = [
        {
            "metric": "Candidate faces",
            "value": candidate_face_count,
            "meaning": (
                "Number of Stage 2 triangles selected for trajectory sampling."
            ),
        },
        {
            "metric": "Generated samples",
            "value": sample_count,
            "meaning": (
                "Number of surface positions where the initial spray "
                "trajectory is evaluated."
            ),
        },
        {
            "metric": "Sampling spacing",
            "value": f"{spacing_mm:.2f} mm",
            "coordinate_value": spacing_coordinate_units,
            "coordinate_unit": coordinate_unit,
            "meaning": (
                "Coverage-oriented surface lattice interval derived from the "
                "nominal spray footprint and overlap, converted to STL coordinates. "
                "It does not prove calibrated paint deposition."
            ),
        },
        {
            "metric": "Standoff distance",
            "value": f"{standoff_mm:.2f} mm",
            "coordinate_value": standoff_coordinate_units,
            "coordinate_unit": coordinate_unit,
            "meaning": (
                "Physical distance from each surface sample to its "
                "simplified nozzle reference position."
            ),
        },
        {
            "metric": "Trajectory length",
            "value": (
                f"{path_metrics['total_path_length_mm']:.2f} mm"
            ),
            "coordinate_value": (
                path_metrics["total_path_length_coordinate_units"]
            ),
            "coordinate_unit": coordinate_unit,
            "meaning": (
                "Sum of distances within each patch. Travel between patches "
                "is excluded."
            ),
        },
        {
            "metric": "Trajectory segments",
            "value": path_metrics["segment_count"],
            "meaning": "Number of spatially continuous serpentine sweep segments.",
        },
        {
            "metric": "Invalid face normals",
            "value": normal_info["invalid_normal_faces"],
            "meaning": "Faces whose normals could not be normalized.",
        },
    ]

    return {
        "overall_status": overall_status,
        "geometric_generation_status": geometric_status,
        "engineering_readiness": "REVIEW",
        "coordinate_unit": coordinate_unit,
        "checks": {
            "trajectory_samples_generated": sample_count > 0,
            "candidate_faces_available": candidate_face_count > 0,
            "all_candidate_normals_valid": skipped_normals == 0,
            "all_candidate_geometry_valid": skipped_geometry == 0,
            "normal_orientation_reliable": orientation_reliable,
            "product_pose_recorded": True,
            "nozzle_frame_tcp_calibrated": False,
            "exact_step_brep_normals_used": False,
            "advanced_geometry_available": advanced_analysis["available"],
            "robot_reachability_checked": False,
            "collision_checked": False,
            "coverage_verified": False,
        },
        "metrics_explanation": explanation,
        "warnings": warnings,
        "recommendations": [
            "Confirm the outward direction of product surface normals.",
            "Review internal cylindrical surfaces and hole candidates.",
            "Use cylinder radius and bend information for future adaptive path planning.",
            "Provide the actual spray-gun CAD model and calibrated TCP offset.",
            "Add robot-base and product coordinate transformations.",
            "Check robot reachability for every target pose.",
            "Check robot, spray-gun, fixture and product collisions.",
            "Define circular spray footprint diameter and overlap.",
            "Validate coating coverage and thickness experimentally.",
            "Optimize travel movements between separate patch segments.",
        ],
    }


# ------------------------------------------------------------
# Generated trajectory inspection
# ------------------------------------------------------------

def _inspect_generated_trajectory(
    mesh,
    source_face_indices: np.ndarray,
    trajectory: dict,
    coordinate_unit: str,
    sample_spacing_mm: float,
    track_spacing_mm: float,
) -> dict:
    """Run geometric path checks and report explicit spray stroke boundaries."""
    face_count = len(mesh.faces)
    source_faces = np.unique(np.asarray(source_face_indices, dtype=np.int64))
    path_faces = np.unique(np.asarray(trajectory["face_indices"], dtype=np.int64))
    source_faces = source_faces[(source_faces >= 0) & (source_faces < face_count)]
    path_faces = path_faces[(path_faces >= 0) & (path_faces < face_count)]
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    target_area = float(np.sum(face_areas[source_faces])) if len(source_faces) else 0.0
    represented_area = float(np.sum(face_areas[np.intersect1d(source_faces, path_faces)])) if len(source_faces) else 0.0
    face_coverage_percent = (
        100.0 * represented_area / target_area if target_area > 1e-12 else 0.0
    )

    points = np.asarray(trajectory["nozzle_points"], dtype=np.float64)
    directions = np.asarray(trajectory["spray_directions"], dtype=np.float64)
    nozzle_frames = np.asarray(
        trajectory.get(
            "nozzle_frames",
            np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(points), axis=0),
        ),
        dtype=np.float64,
    )
    segments = np.asarray(trajectory["segment_ids"], dtype=np.int64)
    mm_per_unit = MM_PER_UNIT[coordinate_unit]
    jump_limit_mm = max(3.0 * float(sample_spacing_mm), 2.0 * float(track_spacing_mm), 1.0)
    jumps_mm = []
    orientation_changes_deg = []
    nozzle_frame_changes_deg = []
    segment_blocks = []
    patch_ids = np.asarray(
        trajectory.get("patch_ids", np.full(len(points), -1)), dtype=np.int64
    )
    face_indices = np.asarray(
        trajectory.get("face_indices", np.full(len(points), -1)), dtype=np.int64
    )
    segment_metadata_by_id = {
        int(item.get("segment_id", -1)): item
        for item in trajectory.get("segments", [])
        if isinstance(item, dict)
    }
    continuity_outliers = []
    orientation_outlier_details = []
    surface_normals_for_inspection = np.asarray(
        trajectory.get("surface_normals", np.full((len(points), 3), np.nan)),
        dtype=np.float64,
    )
    source_face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    step_face_ids_for_inspection = np.asarray(trajectory.get("step_face_ids", np.full(len(points), -1)), dtype=np.int64)
    step_surface_types_for_inspection = np.asarray(trajectory.get("step_surface_types", np.full(len(points), "unknown", dtype=object)), dtype=object)
    if len(segments):
        starts = np.r_[0, np.flatnonzero(np.diff(segments) != 0) + 1]
        ends = np.r_[starts[1:], len(segments)]
        for start, end in zip(starts, ends):
            segment_id = int(segments[start])
            segment_blocks.append((int(start), int(end), segment_id))
            if end - start > 1:
                local_steps = np.linalg.norm(np.diff(points[start:end], axis=0), axis=1) * mm_per_unit
                jumps_mm.extend(local_steps.tolist())
                local_dots = np.einsum(
                    "ij,ij->i", directions[start:end - 1], directions[start + 1:end]
                )
                local_orientation_angles = np.degrees(
                    np.arccos(np.clip(local_dots, -1.0, 1.0))
                )
                orientation_changes_deg.extend(local_orientation_angles.tolist())
                for local_offset, angle_deg in enumerate(local_orientation_angles):
                    if float(angle_deg) <= 30.0:
                        continue
                    sample_index = int(start + local_offset)
                    next_index = sample_index + 1
                    face_a = int(face_indices[sample_index]) if len(face_indices) == len(points) else -1
                    face_b = int(face_indices[next_index]) if len(face_indices) == len(points) else -1
                    patch_a = int(patch_ids[sample_index]) if len(patch_ids) == len(points) else -1
                    patch_b = int(patch_ids[next_index]) if len(patch_ids) == len(points) else -1
                    shared_vertices = None
                    source_face_angle = None
                    sample_normal_angle = None
                    deviation_a = None
                    deviation_b = None
                    if 0 <= face_a < face_count and 0 <= face_b < face_count:
                        shared_vertices = int(
                            len(np.intersect1d(mesh.faces[face_a], mesh.faces[face_b]))
                        )
                        normal_a = source_face_normals[face_a]
                        normal_b = source_face_normals[face_b]
                        if (
                            np.all(np.isfinite(normal_a))
                            and np.all(np.isfinite(normal_b))
                            and np.linalg.norm(normal_a) > NORMAL_EPSILON
                            and np.linalg.norm(normal_b) > NORMAL_EPSILON
                        ):
                            normal_a = normal_a / np.linalg.norm(normal_a)
                            normal_b = normal_b / np.linalg.norm(normal_b)
                            source_face_angle = math.degrees(math.acos(float(np.clip(
                                np.dot(normal_a, normal_b), -1.0, 1.0
                            ))))
                    if surface_normals_for_inspection.shape == (len(points), 3):
                        normal_a = surface_normals_for_inspection[sample_index]
                        normal_b = surface_normals_for_inspection[next_index]
                        if (
                            np.all(np.isfinite(normal_a))
                            and np.all(np.isfinite(normal_b))
                            and np.linalg.norm(normal_a) > NORMAL_EPSILON
                            and np.linalg.norm(normal_b) > NORMAL_EPSILON
                        ):
                            normal_a = normal_a / np.linalg.norm(normal_a)
                            normal_b = normal_b / np.linalg.norm(normal_b)
                            sample_normal_angle = math.degrees(math.acos(float(np.clip(
                                np.dot(normal_a, normal_b), -1.0, 1.0
                            ))))
                        for sample_i, face_i, key in (
                            (sample_index, face_a, "a"),
                            (next_index, face_b, "b"),
                        ):
                            if 0 <= face_i < face_count:
                                sample_normal = surface_normals_for_inspection[sample_i]
                                face_normal = source_face_normals[face_i]
                                if (
                                    np.all(np.isfinite(sample_normal))
                                    and np.all(np.isfinite(face_normal))
                                    and np.linalg.norm(sample_normal) > NORMAL_EPSILON
                                    and np.linalg.norm(face_normal) > NORMAL_EPSILON
                                ):
                                    value = math.degrees(math.acos(float(np.clip(
                                        np.dot(
                                            sample_normal / np.linalg.norm(sample_normal),
                                            face_normal / np.linalg.norm(face_normal),
                                        ),
                                        -1.0,
                                        1.0,
                                    ))))
                                    if key == "a":
                                        deviation_a = value
                                    else:
                                        deviation_b = value
                    if patch_a != patch_b:
                        cause = "patch_boundary_inside_stroke"
                    elif face_a == face_b and face_a >= 0:
                        cause = "same_source_face_sample_normal_change"
                    elif shared_vertices == 0:
                        cause = "non_adjacent_source_faces"
                    elif source_face_angle is not None and source_face_angle > 30.0:
                        cause = "sharp_source_mesh_crease"
                    elif source_face_angle is not None:
                        cause = "large_axis_change_without_sharp_face_crease"
                    else:
                        cause = "source_face_normal_unavailable"
                    segment_meta = segment_metadata_by_id.get(int(segments[sample_index]), {})
                    orientation_outlier_details.append({
                        "step_face_id_from": int(step_face_ids_for_inspection[sample_index]) if len(step_face_ids_for_inspection) == len(points) else -1,
                        "step_face_id_to": int(step_face_ids_for_inspection[next_index]) if len(step_face_ids_for_inspection) == len(points) else -1,
                        "step_surface_type_from": str(step_surface_types_for_inspection[sample_index]) if len(step_surface_types_for_inspection) == len(points) else "unknown",
                        "step_surface_type_to": str(step_surface_types_for_inspection[next_index]) if len(step_surface_types_for_inspection) == len(points) else "unknown",
                        "sample_index_from": sample_index,
                        "sample_index_to": next_index,
                        "segment_id": int(segments[sample_index]),
                        "patch_id_from": patch_a,
                        "patch_id_to": patch_b,
                        "face_index_from": face_a,
                        "face_index_to": face_b,
                        "shared_source_face_vertices": shared_vertices,
                        "spray_axis_change_deg": float(angle_deg),
                        "source_face_normal_change_deg": source_face_angle,
                        "sample_surface_normal_change_deg": sample_normal_angle,
                        "sample_a_deviation_from_source_face_deg": deviation_a,
                        "sample_b_deviation_from_source_face_deg": deviation_b,
                        "point_step_mm": float(
                            np.linalg.norm(points[next_index] - points[sample_index]) * mm_per_unit
                        ),
                        "cause_classification": cause,
                        "path_type": str(segment_meta.get("path_type", "unknown")),
                    })
                if nozzle_frames.shape == (len(points), 3, 3):
                    for frame_index in range(start, end - 1):
                        relative_rotation = (
                            nozzle_frames[frame_index].T
                            @ nozzle_frames[frame_index + 1]
                        )
                        cosine = float(np.clip(
                            (np.trace(relative_rotation) - 1.0) * 0.5,
                            -1.0,
                            1.0,
                        ))
                        nozzle_frame_changes_deg.append(
                            math.degrees(math.acos(cosine))
                        )
                for local_offset, step_mm in enumerate(local_steps):
                    if float(step_mm) <= jump_limit_mm:
                        continue
                    sample_index = int(start + local_offset)
                    next_index = sample_index + 1
                    patch_a = int(patch_ids[sample_index]) if len(patch_ids) == len(points) else -1
                    patch_b = int(patch_ids[next_index]) if len(patch_ids) == len(points) else -1
                    face_a = int(face_indices[sample_index]) if len(face_indices) == len(points) else -1
                    face_b = int(face_indices[next_index]) if len(face_indices) == len(points) else -1
                    shared_vertices = None
                    if (
                        0 <= face_a < face_count
                        and 0 <= face_b < face_count
                    ):
                        shared_vertices = int(
                            len(np.intersect1d(mesh.faces[face_a], mesh.faces[face_b]))
                        )
                    if patch_a != patch_b:
                        cause = "different_patches_inside_one_stroke"
                    elif face_a == face_b and face_a >= 0:
                        cause = "large_step_on_same_source_face"
                    elif shared_vertices == 0:
                        cause = "non_adjacent_source_faces_inside_one_stroke"
                    elif shared_vertices is not None:
                        cause = "large_step_between_vertex_adjacent_faces"
                    else:
                        cause = "source_face_adjacency_unavailable"
                    segment_meta = segment_metadata_by_id.get(segment_id, {})
                    continuity_outliers.append({
                        "sample_index_from": sample_index,
                        "sample_index_to": next_index,
                        "segment_id": segment_id,
                        "patch_id_from": patch_a,
                        "patch_id_to": patch_b,
                        "face_index_from": face_a,
                        "face_index_to": face_b,
                        "shared_source_face_vertices": shared_vertices,
                        "step_distance_mm": float(step_mm),
                        "allowed_step_mm": float(jump_limit_mm),
                        "excess_over_limit_mm": float(step_mm - jump_limit_mm),
                        "cause_classification": cause,
                        "path_type": str(segment_meta.get("path_type", "unknown")),
                        "point_from_coordinate_units": points[sample_index].astype(float).tolist(),
                        "point_to_coordinate_units": points[next_index].astype(float).tolist(),
                        "step_vector_coordinate_units": (
                            points[next_index] - points[sample_index]
                        ).astype(float).tolist(),
                    })

    continuity_outliers.sort(
        key=lambda item: item["step_distance_mm"], reverse=True
    )
    orientation_outlier_details.sort(
        key=lambda item: item["spray_axis_change_deg"], reverse=True
    )
    jumps = np.asarray(jumps_mm, dtype=np.float64)
    angles = np.asarray(orientation_changes_deg, dtype=np.float64)
    frame_angles = np.asarray(nozzle_frame_changes_deg, dtype=np.float64)
    jump_outliers = int(np.count_nonzero(jumps > jump_limit_mm))
    max_jump = float(np.max(jumps)) if len(jumps) else 0.0
    p95_jump = float(np.percentile(jumps, 95)) if len(jumps) else 0.0
    max_angle = float(np.max(angles)) if len(angles) else 0.0
    p95_angle = float(np.percentile(angles, 95)) if len(angles) else 0.0
    max_frame_angle = (
        float(np.max(frame_angles)) if len(frame_angles) else max_angle
    )
    p95_frame_angle = (
        float(np.percentile(frame_angles, 95))
        if len(frame_angles) else p95_angle
    )
    orientation_over_30 = int(np.count_nonzero(angles > 30.0))
    frame_orientation_over_30 = int(np.count_nonzero(frame_angles > 30.0))
    max_orientation_change = max(max_angle, max_frame_angle)
    segment_count = len(segment_blocks)
    spray_off_transitions = max(0, segment_count - 1)
    spray_on_samples = max(0, len(points) - spray_off_transitions)

    coverage_status = "PASS" if face_coverage_percent >= 95.0 else "WARN"
    continuity_status = "PASS" if jump_outliers == 0 else "WARN"
    orientation_warning_threshold_deg = 30.0
    orientation_status = (
        "PASS"
        if max_orientation_change <= orientation_warning_threshold_deg
        else "WARN"
    )
    spray_status = "PASS" if segment_count > 0 else "FAIL"

    report = {
        "surface_face_representation_percent": face_coverage_percent,
        "target_face_count": int(len(source_faces)),
        "represented_face_count": int(len(np.intersect1d(source_faces, path_faces))),
        "target_area_coordinate_units2": target_area,
        "represented_area_coordinate_units2": represented_area,
        "continuity": {
            "status": continuity_status,
            "within_segment_sample_pairs": int(len(jumps)),
            "maximum_step_mm": max_jump,
            "p95_step_mm": p95_jump,
            "jump_limit_mm": jump_limit_mm,
            "jump_outlier_count": jump_outliers,
            "outlier_details": continuity_outliers,
            "outlier_detail_count": int(len(continuity_outliers)),
            "outlier_detail_policy": (
                "Every within-stroke adjacent sample pair above the limit is listed "
                "with sample, segment, patch, source-face, distance and geometric "
                "adjacency details. Face adjacency is estimated by shared mesh vertices."
            ),
        },
        "orientation": {
            "status": orientation_status,
            "maximum_change_deg": max_orientation_change,
            "p95_change_deg": max(p95_angle, p95_frame_angle),
            "spray_axis_maximum_change_deg": max_angle,
            "spray_axis_p95_change_deg": p95_angle,
            "spray_axis_changes_over_30_deg": orientation_over_30,
            "full_nozzle_frame_maximum_change_deg": max_frame_angle,
            "full_nozzle_frame_p95_change_deg": p95_frame_angle,
            "full_nozzle_frame_changes_over_30_deg": frame_orientation_over_30,
            "changes_over_30_deg": int(
                max(orientation_over_30, frame_orientation_over_30)
            ),
            "warning_threshold_deg": orientation_warning_threshold_deg,
            "max_normal_smoothing_deviation_deg": float(
                trajectory.get("max_normal_smoothing_deviation_deg", 0.0)
            ),
            "surface_normal_deviation_limit_deg": 8.0,
            "over_30_degree_change_details": orientation_outlier_details,
            "over_30_degree_change_detail_count": int(len(orientation_outlier_details)),
            "over_30_degree_change_classifications": {
                cause: sum(
                    1 for item in orientation_outlier_details
                    if item["cause_classification"] == cause
                )
                for cause in sorted({
                    item["cause_classification"]
                    for item in orientation_outlier_details
                })
            },
        },
        "spray_control": {
            "status": spray_status,
            "spray_on_sample_count": int(spray_on_samples),
            "spray_off_boundary_sample_count": int(spray_off_transitions),
            "spray_off_transition_count": int(spray_off_transitions),
            "stroke_segment_count": int(segment_count),
            "policy": (
                "Spray ON within a stroke; OFF at every segment boundary. "
                "Direct straight-line OFF travel estimates are reported separately; "
                "they are not collision-checked robot paths."
            ),
        },
        "coverage_status": coverage_status,
        "note": (
            "Surface face representation is the area of source candidate mesh faces "
            "referenced by generated path samples. It is not physical spray-envelope "
            "coverage, paint deposition, or film-thickness validation."
        ),
    }

    print("\n" + "=" * 68)
    print("STAGE 3 GENERATED TRAJECTORY INSPECTION")
    print("=" * 68)
    print(
        f"[{coverage_status}] Surface face representation: "
        f"{face_coverage_percent:.2f}% "
        f"({report['represented_face_count']}/{report['target_face_count']} faces)"
    )
    print(
        f"[{continuity_status}] Path continuity within strokes: "
        f"max {max_jump:.2f} mm | P95 {p95_jump:.2f} mm | "
        f"limit {jump_limit_mm:.2f} mm | outliers {jump_outliers}"
    )
    if continuity_outliers:
        print("CONTINUITY OUTLIER DETAILS (largest first)")
        for item in continuity_outliers:
            print(
                f"  samples {item['sample_index_from']}->{item['sample_index_to']} | "
                f"segment {item['segment_id']} | "
                f"patch {item['patch_id_from']}->{item['patch_id_to']} | "
                f"faces {item['face_index_from']}->{item['face_index_to']} | "
                f"step {item['step_distance_mm']:.2f} mm "
                f"(limit {jump_limit_mm:.2f}) | {item['cause_classification']} | "
                f"path={item['path_type']}"
            )
    print(
        f"[{orientation_status}] Nozzle orientation: "
        f"axis max {max_angle:.2f} deg / P95 {p95_angle:.2f} deg | "
        f"full-frame max {max_frame_angle:.2f} deg / "
        f"P95 {p95_frame_angle:.2f} deg | "
        f"axis >30 deg {orientation_over_30} | "
        f"frame >30 deg {frame_orientation_over_30} | "
        f"max smoothing deviation: "
        f"{report['orientation']['max_normal_smoothing_deviation_deg']:.2f} deg "
        f"(limit 8.00 deg)"
    )
    if orientation_outlier_details:
        print("ORIENTATION CHANGE DETAILS (>30 deg, largest first; first 20)")
        for item in orientation_outlier_details[:20]:
            source_angle = (
                f"{item['source_face_normal_change_deg']:.2f} deg"
                if item["source_face_normal_change_deg"] is not None else "n/a"
            )
            print(
                f"  samples {item['sample_index_from']}->{item['sample_index_to']} | "
                f"segment {item['segment_id']} | "
                f"patch {item['patch_id_from']}->{item['patch_id_to']} | "
                f"mesh faces {item['face_index_from']}->{item['face_index_to']} | "
                f"STEP faces {item['step_face_id_from']}->{item['step_face_id_to']} "
                f"({item['step_surface_type_from']}->{item['step_surface_type_to']}) | "
                f"axis {item['spray_axis_change_deg']:.2f} deg | "
                f"source-face normals {source_angle} | "
                f"step {item['point_step_mm']:.2f} mm | "
                f"{item['cause_classification']} | path={item['path_type']}"
            )
        print(
            "  Classification counts: "
            + ", ".join(
                f"{key}={value}"
                for key, value in report["orientation"][
                    "over_30_degree_change_classifications"
                ].items()
            )
        )
    print(
        f"[{spray_status}] Spray control: ON samples {spray_on_samples:,} | "
        f"OFF boundary samples {spray_off_transitions:,} | "
        f"stroke segments {segment_count:,}"
    )
    print(
        "NOTE: Spray is OFF between strokes. Straight-line travel estimates are "
        "reported separately; collision checking and IK are NOT performed."
    )
    print(report["note"])
    print("=" * 68)
    return report


# Main trajectory calculation
# ------------------------------------------------------------

def generate_spray_trajectory(
    mesh: trimesh.Trimesh,
    face_status_labels: np.ndarray,
    patch_labels: np.ndarray,
    standoff_mm: float = DEFAULT_STANDOFF_MM,
    stage1_report: dict | None = None,
    coordinate_unit: str | None = None,
    sprayer_profile: dict | None = None,
    coating_process: dict | None = None,
    product_transform: np.ndarray | list | None = None,
) -> dict:

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh object.")

    standoff_mm = _validate_parameters(standoff_mm)

    product_transform_matrix = _validate_product_transform(product_transform)

    coating_process_data = validate_coating_process(coating_process or {})

    # Resolve unit before performing any physical-distance calculation.
    source_unit = _resolve_coordinate_unit(
        mesh=mesh,
        stage1_report=stage1_report,
        coordinate_unit=coordinate_unit,
    )

    standoff_coordinate_units = _mm_to_coordinate_units(
        standoff_mm,
        source_unit,
    )

    # The active sprayer profile defines the nominal footprint. Surface
    # sampling is then driven by footprint diameter and overlap, not by an
    # arbitrary fixed point interval alone.
    profile_for_path = (
        dict(sprayer_profile)
        if isinstance(sprayer_profile, dict)
        else dict(DEFAULT_CIRCULAR_SPRAYER_PROFILE)
    )
    spray_footprint = _calculate_spray_footprint(
        profile_for_path,
        standoff_mm,
    )
    spray_angle_deg = float(spray_footprint["included_angle_deg"])
    path_spray_width_mm = float(spray_footprint["width_mm"])
    path_track_spacing_mm = float(
        spray_footprint["nominal_track_spacing_mm"]
    )

    # There is no arbitrary point-spacing fallback. Both the surface sampling
    # interval and lane pitch come exclusively from the active circular
    # footprint and requested overlap.
    if not math.isfinite(path_track_spacing_mm) or path_track_spacing_mm <= 0.0:
        raise ValueError("Circular spray footprint produced an invalid point spacing.")
    coverage_spacing_mm = path_track_spacing_mm
    coverage_spacing_basis = "SPRAY_FOOTPRINT_AND_OVERLAP"

    spacing_coordinate_units = _mm_to_coordinate_units(
        coverage_spacing_mm,
        source_unit,
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(faces) == 0:
        raise ValueError("Mesh contains no triangles.")

    if not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh contains non-finite vertex coordinates.")

    face_status_labels = np.asarray(
        face_status_labels,
        dtype=np.int64,
    )

    patch_labels = np.asarray(
        patch_labels,
        dtype=np.int64,
    )

    if face_status_labels.shape != (len(faces),):
        raise ValueError(
            "Stage 2 face_status_labels must contain exactly "
            "one label per mesh triangle."
        )

    if patch_labels.shape != (len(faces),):
        raise ValueError(
            "patch_labels must contain exactly one patch ID "
            "per mesh triangle."
        )

    valid_labels = [
        FACE_EXCLUDED,
        FACE_PAINTABLE_CANDIDATE,
        FACE_REQUIRES_INSPECTION,
        FACE_INVALID,
    ]

    if not np.all(np.isin(face_status_labels, valid_labels)):
        raise ValueError(
            "Stage 2 contains an unknown face classification."
        )

    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Mesh contains invalid triangle vertex indices.")

    # Preserve the source mesh coordinate values.
    original_vertices = vertices.copy()

    # 1. Extract Stage 1 advanced geometry metadata.
    feature_data = _extract_stage1_features(stage1_report)

    # 2. Calculate local face normals.
    normals, normal_info = _get_face_normals(mesh)

    # 3. Determine outward normal direction when reliable.
    normals, orientation_info = _determine_normal_orientation(
        mesh,
        normals,
    )

    # 4. Sample only Stage 2 paintable candidate faces.
    sampling_result = _sample_candidate_faces(
        mesh=mesh,
        face_status_labels=face_status_labels,
        patch_labels=patch_labels,
        normals=normals,
        spacing_coordinate_units=spacing_coordinate_units,
    )
    from core.step_loader import (
        get_step_brep_context,
        build_step_patch_face_mapping,
    )
    step_context = get_step_brep_context(mesh)
    surface_mapping = (
        build_step_patch_face_mapping(
            mesh, patch_labels, face_status_labels
        )
        if step_context is not None else None
    )

    surface_points = sampling_result["surface_points"]
    surface_normals = sampling_result["surface_normals"]

    face_indices = sampling_result["face_indices"]
    patch_ids = sampling_result["patch_ids"]

    # 5. Calculate nozzle reference positions in STL coordinates.
    #
    # P_g = P_s + d_source * N
    #
    # d_source is the requested physical standoff converted into
    # the STL source coordinate unit.

    nozzle_points = (
        surface_points
        + standoff_coordinate_units * surface_normals
    )

    # 6. Spray direction points from nozzle toward surface.
    spray_directions = -surface_normals.copy()

    # 6a. Construct a provisional geometric nozzle frame at each sample.
    nozzle_frames = _build_nozzle_frames(spray_directions)

    # 7. Build one spatially connected global serpentine sweep plan.
    # CAD face and patch IDs remain metadata; they do not create stroke
    # boundaries. Only a true 3D spatial gap separates sweep segments.
    # A zero/unspecified cone angle cannot define a footprint lane spacing.
    # In that case, use the configured point spacing as a geometric fallback
    # and report that coverage spacing is not physically established.
    effective_lane_spacing_mm = path_track_spacing_mm
    path_track_spacing_units = _mm_to_coordinate_units(
        effective_lane_spacing_mm, source_unit
    )
    sweep_config = SweepPlanningConfig(
        # Use the working-distance / circular-angle / overlap-derived pitch
        # along and across the spray path. No robot timing or speed is applied.
        lane_spacing=path_track_spacing_units,
        sample_spacing=spacing_coordinate_units,
        max_gap_factor=2.75,
        tangent_blend=0.35,
    )
    sweep = plan_human_like_sweeps(
        surface_points=surface_points,
        surface_normals=surface_normals,
        face_indices=face_indices,
        patch_ids=patch_ids,
        config=sweep_config,
    )
    ordered = {
        "surface_points": sweep["surface_points"],
        "surface_normals": sweep["surface_normals"],
        "face_indices": sweep["face_indices"],
        "patch_ids": sweep["patch_ids"],
        "segment_ids": sweep["segment_ids"],
        "spray_directions": sweep["spray_directions"],
        "nozzle_frames": sweep["nozzle_frames"],
    }
    # Preserve the exact requested stand-off at every generated point.
    ordered["nozzle_points"] = (
        ordered["surface_points"]
        + standoff_coordinate_units * ordered["surface_normals"]
    )

    # Evaluate the actual 3-D spray cone against area-weighted probes on the
    # product mesh. Delete a candidate only when its footprint overlap meets
    # Track overlap (%) and no previously covered probe becomes uncovered.
    cone_angle_for_optimization = float(spray_footprint["included_angle_deg"])
    if cone_angle_for_optimization > 0.0:
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        # Use an independent mesh-probe set at one-eighth of candidate spacing
        # for footprint-area integration. This reduces the chance of missing
        # narrow uncovered regions between the retained candidate centers.
        probe_spacing_coordinate_units = max(
            spacing_coordinate_units * 0.125, 1e-6
        )
        try:
            probe_sampling_result = _sample_candidate_faces(
                mesh=mesh,
                face_status_labels=face_status_labels,
                patch_labels=patch_labels,
                normals=normals,
                spacing_coordinate_units=probe_spacing_coordinate_units,
            )
        except RuntimeError as exc:
            if "more than" not in str(exc):
                raise
            probe_spacing_coordinate_units = spacing_coordinate_units * 0.5
            try:
                probe_sampling_result = _sample_candidate_faces(
                    mesh=mesh,
                    face_status_labels=face_status_labels,
                    patch_labels=patch_labels,
                    normals=normals,
                    spacing_coordinate_units=probe_spacing_coordinate_units,
                )
            except RuntimeError as retry_exc:
                if "more than" not in str(retry_exc):
                    raise
                probe_sampling_result = sampling_result
                probe_spacing_coordinate_units = spacing_coordinate_units

        probe_face_counts = np.bincount(
            probe_sampling_result["face_indices"], minlength=len(mesh.faces)
        )
        probe_face_indices = probe_sampling_result["face_indices"]
        probe_face_areas = face_areas[probe_face_indices]
        probe_face_sample_counts = probe_face_counts[probe_face_indices]
        probe_area_weights = np.zeros(len(probe_face_indices), dtype=np.float64)
        valid_probe_areas = (
            np.isfinite(probe_face_areas)
            & (probe_face_areas > 0.0)
            & (probe_face_sample_counts > 0)
        )
        probe_area_weights[valid_probe_areas] = (
            probe_face_areas[valid_probe_areas]
            / probe_face_sample_counts[valid_probe_areas]
        )
        footprint_optimization = optimize_redundant_spray_points(
            surface_points=ordered["surface_points"],
            nozzle_points=ordered["nozzle_points"],
            spray_directions=ordered["spray_directions"],
            probe_points=probe_sampling_result["surface_points"],
            probe_area_weights=probe_area_weights,
            standoff_coordinate_units=standoff_coordinate_units,
            cone_angle_deg=cone_angle_for_optimization,
            overlap_percent=float(spray_footprint["overlap_percent"]),
            sample_spacing_coordinate_units=probe_spacing_coordinate_units,
        )
        keep_indices = footprint_optimization.pop("keep_indices")
        removed_count = len(ordered["surface_points"]) - len(keep_indices)
        if removed_count:
            for key in (
                "surface_points", "surface_normals", "face_indices",
                "patch_ids", "segment_ids", "spray_directions",
                "nozzle_frames", "nozzle_points",
            ):
                ordered[key] = ordered[key][keep_indices]
            sweep["lane_ids"] = sweep["lane_ids"][keep_indices]
        footprint_optimization["removed_point_count"] = int(removed_count)
        footprint_optimization["retained_point_count"] = int(len(keep_indices))
    else:
        footprint_optimization = {
            "status": "SKIPPED_ZERO_CONE_ANGLE",
            "overlap_threshold_percent": float(spray_footprint["overlap_percent"]),
            "removed_point_count": 0,
            "retained_point_count": int(len(ordered["surface_points"])),
            "coverage_preserved": None,
            "area_measurement": "UNAVAILABLE_ZERO_CONE_ANGLE",
        }

    segment_metadata = []
    for segment_id in np.unique(ordered["segment_ids"]):
        indices = np.flatnonzero(ordered["segment_ids"] == segment_id)
        segment_patch_ids = np.unique(ordered["patch_ids"][indices])
        segment_metadata.append({
            "segment_id": int(segment_id),
            "start_index": int(indices[0]),
            "end_index_exclusive": int(indices[-1] + 1),
            "sample_count": int(len(indices)),
            "lane_id": int(sweep["lane_ids"][indices[0]]),
            "patch_ids": [int(value) for value in segment_patch_ids],
            "path_type": "global_spatial_serpentine_sweep",
        })
    ordered["segments"] = segment_metadata
    ordered["surface_boundary_splits"] = {
        "status": "NO_TOPOLOGY_BASED_SPLITS",
        "adjacency_authority": "3D_SPATIAL_GAP_ONLY",
        "policy": (
            "CAD face and patch boundaries do not create spray-OFF breaks; "
            "only spatially discontinuous ordered samples start a new segment."
        ),
        "added_stroke_count": 0,
        "split_transition_counts": {},
    }
    ordered["step_surface_refinement"] = {
        "status": "NOT_RUN_GLOBAL_SWEEP_V1",
        "normal_source": "STL_TRIANGLE_FACE_NORMALS",
        "reason": (
            "The global sweep preserves the sampled PRODUCT_CAD surface points "
            "and their local mesh normals without applying a separate B-Rep path "
            "reconstruction pass."
        ),
    }
    ordered["travel_ordering"] = {
        "algorithm": sweep["algorithm"],
        "lane_count": int(len(np.unique(sweep["lane_ids"]))),
        "segment_count": int(len(segment_metadata)),
        "ordering_optimization": "PCA_PROJECTED_SERPENTINE",
        "gap_split_threshold_coordinate_units": float(
            sweep["gap_split_threshold"]
        ),
        "face_and_patch_boundaries_used_for_splitting": False,
    }
    ordered["max_normal_smoothing_deviation_deg"] = 0.0
    ordered["rasterized_planar_patch_ids"] = []
    ordered["rasterized_curved_patch_ids"] = []
    ordered["normal_partitioned_patch_ids"] = []
    ordered["nearest_neighbor_fallback_patch_ids"] = []

    per_patch_statistics = []
    for patch_id in np.unique(ordered["patch_ids"]):
        patch_mask = ordered["patch_ids"] == patch_id
        touching_segments = [
            item for item in segment_metadata
            if int(patch_id) in item["patch_ids"]
        ]
        per_patch_statistics.append({
            "patch_id": int(patch_id),
            "path_type": "global_spatial_serpentine_sweep",
            "sample_count": int(np.count_nonzero(patch_mask)),
            "segment_count": int(len(touching_segments)),
        })
    ordered["path_statistics"] = {
        "source_candidate_sample_count": int(len(surface_points)),
        "footprint_optimizer_removed_sample_count": int(
            footprint_optimization.get("removed_point_count", 0)
        ),
        "playback_sample_count": int(len(ordered["surface_points"])),
        "global_sweep_sample_count": int(len(ordered["surface_points"])),
        "global_sweep_segment_count": int(len(segment_metadata)),
        "lane_count": int(len(np.unique(sweep["lane_ids"]))),
        "planar_sample_count": 0,
        "curved_sample_count": 0,
        "fallback_sample_count": 0,
        "planar_segment_count": 0,
        "curved_segment_count": 0,
        "fallback_segment_count": 0,
        "planar_patch_ids": [],
        "curved_patch_ids": [],
        "fallback_patch_ids": [],
        "per_patch": per_patch_statistics,
        "algorithm": sweep["algorithm"],
        "note": (
            "Candidate samples are reduced only when their sampled cone "
            "footprints are redundant without reducing sampled surface coverage. "
            "Path segments are separated only by 3D spatial gaps, not by "
            "source face or patch identity. This geometric ordering does not "
            "constitute a verified spray-coverage result."
        ),
    }

    print(
        "GLOBAL SPRAY SWEEP | "
        f"{len(ordered['surface_points']):,} samples | "
        f"{len(np.unique(sweep['lane_ids']))} lanes | "
        f"{len(segment_metadata)} spatial segments | "
        f"lane spacing {effective_lane_spacing_mm:.2f} mm"
    )

    # 7a. Transform CAD-local samples and nozzle frames into World XYZ.
    # The source mesh and CAD-local arrays remain unchanged.
    world_surface_points = _transform_points(ordered['surface_points'], product_transform_matrix)
    world_surface_normals = _transform_vectors(ordered['surface_normals'], product_transform_matrix)
    world_nozzle_points = _transform_points(ordered['nozzle_points'], product_transform_matrix)
    world_spray_directions = _transform_vectors(ordered['spray_directions'], product_transform_matrix)
    world_nozzle_frames = np.einsum('ij,njk->nik', product_transform_matrix[:3, :3], ordered['nozzle_frames'])

    # 8. Describe spray-OFF connectors geometrically only.
    inter_segment_travel_plan = _build_inter_segment_travel_plan(
        nozzle_points=ordered["nozzle_points"],
        segment_ids=ordered["segment_ids"],
        coordinate_unit=source_unit,
    )

    # 9. Calculate path metrics in both coordinate units and mm.
    path_metrics = _calculate_path_metrics(
        nozzle_points=ordered["nozzle_points"],
        segment_ids=ordered["segment_ids"],
        coordinate_unit=source_unit,
    )

    trajectory_validation = _inspect_generated_trajectory(
        mesh=mesh,
        source_face_indices=face_indices,
        trajectory=ordered,
        coordinate_unit=source_unit,
        sample_spacing_mm=coverage_spacing_mm,
        track_spacing_mm=path_track_spacing_mm,
    )

    # 10. Associate Stage 1 geometry features.
    advanced_analysis = _build_advanced_geometry_analysis(
        feature_data=feature_data,
        patch_ids=ordered["patch_ids"],
        coordinate_unit=source_unit,
    )

    # 11. Numerical validation.
    for key in (
        "surface_points",
        "surface_normals",
        "nozzle_points",
        "spray_directions",
    ):
        if not np.all(np.isfinite(ordered[key])):
            raise RuntimeError(
                f"Generated trajectory contains invalid values: {key}"
            )

    normal_lengths = np.linalg.norm(
        ordered["surface_normals"],
        axis=1,
    )

    direction_lengths = np.linalg.norm(
        ordered["spray_directions"],
        axis=1,
    )

    if not np.allclose(normal_lengths, 1.0, atol=1e-6):
        raise RuntimeError(
            "Some generated surface normals are not unit vectors."
        )

    if not np.allclose(direction_lengths, 1.0, atol=1e-6):
        raise RuntimeError(
            "Some spray directions are not unit vectors."
        )

    actual_standoffs_coordinate_units = np.linalg.norm(
        ordered["nozzle_points"] - ordered["surface_points"],
        axis=1,
    )

    if not np.allclose(
        actual_standoffs_coordinate_units,
        standoff_coordinate_units,
        atol=max(1e-9, abs(standoff_coordinate_units) * 1e-8),
    ):
        raise RuntimeError(
            "Calculated nozzle positions do not satisfy the requested "
            "physical standoff."
        )

    actual_standoffs_mm = (
        actual_standoffs_coordinate_units * MM_PER_UNIT[source_unit]
    )

    if not np.allclose(
        actual_standoffs_mm,
        standoff_mm,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Physical nozzle standoff validation failed."
        )

    # Confirm that no source mesh vertex was modified.
    if not np.array_equal(
        np.asarray(mesh.vertices, dtype=np.float64),
        original_vertices,
    ):
        raise RuntimeError(
            "Source mesh vertices changed during Stage 3."
        )

    # 12. Engineering analysis.
    analysis = _build_analysis(
        sampling_result=sampling_result,
        normal_info=normal_info,
        orientation_info=orientation_info,
        path_metrics=path_metrics,
        standoff_mm=standoff_mm,
        spacing_mm=coverage_spacing_mm,
        standoff_coordinate_units=standoff_coordinate_units,
        spacing_coordinate_units=spacing_coordinate_units,
        coordinate_unit=source_unit,
        advanced_analysis=advanced_analysis,
    )

    profile_data = (
        dict(sprayer_profile)
        if isinstance(sprayer_profile, dict)
        else {}
    )
    spray_width_at_standoff_mm = spray_footprint["width_mm"]
    nominal_track_spacing_mm = spray_footprint["nominal_track_spacing_mm"]

    return {
        "parameters": {
            "standoff_mm": standoff_mm,
            "sampling_spacing_mm": coverage_spacing_mm,
            "effective_coverage_sampling_spacing_mm": coverage_spacing_mm,
            "coverage_spacing_basis": coverage_spacing_basis,
            "spray_angle_deg": spray_angle_deg,
            "sprayer_profile": profile_data,
            "coating_process": coating_process_data,
            "product_transform_4x4": product_transform_matrix.tolist(),
            "product_coordinate_frame": "PRODUCT_CAD",
            "world_coordinate_frame": "WORLD_XYZ",
            "nozzle_frame_convention": {
                "matrix_columns": ["X", "Y", "Z"],
                "Z_axis": "spray direction from nozzle toward surface",
                "X_axis": "PRODUCT_CAD +X projected onto plane normal to Z; +Y fallback",
                "status": (
                    "PROVISIONAL_GEOMETRIC_FRAME"
                    if orientation_info["orientation_reliable"]
                    else "REVIEW_REQUIRED_OUTWARD_NORMAL_UNCERTAIN"
                ),
                "tcp_calibration": "NOT_APPLIED",
            },
            "spray_width_at_standoff_mm": spray_width_at_standoff_mm,
            "nominal_track_spacing_from_overlap_mm": nominal_track_spacing_mm,
            "spray_footprint": spray_footprint,
            "coordinate_unit": source_unit,
            "internal_coordinate_unit": source_unit,
            "standoff_coordinate_units": standoff_coordinate_units,
            "sampling_spacing_coordinate_units": spacing_coordinate_units,
            "normal_direction": "outward",
            "spray_direction": "toward_surface",
            "nozzle_position_formula": "P_g = P_s + d_source * N",
            "spray_direction_formula": "D = -N",
            "unit_conversion": (
                "source coordinate value = physical millimeter value "
                "/ millimeters per source unit"
            ),
        },
        "surface_points": ordered["surface_points"],
        "surface_normals": ordered["surface_normals"],
        "nozzle_points": ordered["nozzle_points"],
        "spray_directions": ordered["spray_directions"],
        "nozzle_frames": ordered["nozzle_frames"],
        "surface_points_world": world_surface_points,
        "surface_normals_world": world_surface_normals,
        "nozzle_points_world": world_nozzle_points,
        "spray_directions_world": world_spray_directions,
        "nozzle_frames_world": world_nozzle_frames,
        "face_indices": ordered["face_indices"],
        "patch_ids": ordered["patch_ids"],
        "segment_ids": ordered["segment_ids"],
        "segments": ordered["segments"],
        "surface_boundary_splits": ordered.get("surface_boundary_splits", {}),
        "step_surface_refinement": ordered.get("step_surface_refinement", {
            "status": (
                "EXACT_BREP_SURFACE_SAMPLING"
                if sampling_result.get("surface_geometry_source") == "ORIGINAL_STEP_BREP_SURFACE"
                else "NOT_APPLICABLE"
            ),
            "normal_source": (
                "ORIGINAL_STEP_BREP_SURFACE"
                if sampling_result.get("surface_geometry_source") == "ORIGINAL_STEP_BREP_SURFACE"
                else "DISPLAY_MESH"
            ),
        }),
        "surface_geometry_source": sampling_result.get(
            "surface_geometry_source", "DISPLAY_MESH_TRIANGLES"
        ),
        "patch_brep_face_mapping": surface_mapping,
        "path_statistics": ordered.get("path_statistics", {}),
        "trajectory_validation": trajectory_validation,
        "inter_segment_travel_plan": inter_segment_travel_plan,
        "rasterization": {
            "algorithm": "PCA_SPATIAL_SERPENTINE_SWEEP",
            "rasterized_planar_patch_ids": ordered.get("rasterized_planar_patch_ids", []),
            "rasterized_curved_patch_ids": ordered.get("rasterized_curved_patch_ids", []),
            "nearest_neighbor_fallback_patch_ids": ordered.get("nearest_neighbor_fallback_patch_ids", []),
            "lane_spacing_mm": effective_lane_spacing_mm,
            "surface_sample_spacing_mm": coverage_spacing_mm,
            "trajectory_sample_spacing_mm": coverage_spacing_mm,
            "trajectory_sample_spacing_basis": coverage_spacing_basis,
            "coverage_spacing_basis": coverage_spacing_basis,
            "nominal_footprint_track_spacing_mm": path_track_spacing_mm,
            "lane_spacing_coordinate_units": path_track_spacing_units,
            "lane_direction_policy": "global_principal_axis_projection",
            "topology_boundary_policy": "NO_SPLIT_ON_FACE_OR_PATCH_ID",
            "spatial_gap_policy": "split_only_when_ordered_3d_sample_gap_exceeds_threshold",
            "orientation_policy": "local_surface_normal_with_path_tangent_frame_blending",
            "coverage_spacing_status": (
                "FOOTPRINT_AND_OVERLAP_BASED"
                if path_track_spacing_mm > 0.0
                else "INTERNAL_SPACING_FALLBACK_NO_PHYSICAL_FOOTPRINT"
            ),
            "travel_moves_between_segments": "DIRECT_STRAIGHT_LINE_VISUALIZATION_ONLY",
            "coverage_verified": False,
            "sampled_coverage_preserved": bool(
                footprint_optimization.get("coverage_preserved", False)
            ),
            "footprint_optimization": footprint_optimization,
        },
        "sampling": {
            "coordinate_unit": source_unit,
            "sampling_spacing_mm": coverage_spacing_mm,
            "coverage_spacing_mm": coverage_spacing_mm,
            "coverage_spacing_basis": coverage_spacing_basis,
            "sampling_spacing_coordinate_units": spacing_coordinate_units,
            "candidate_face_count": sampling_result["candidate_face_count"],
            "sample_count": int(len(ordered["surface_points"])),
            "source_sample_count_before_footprint_optimization": int(
                sampling_result["sample_count"]
            ),
            "footprint_optimizer_removed_sample_count": int(
                footprint_optimization.get("removed_point_count", 0)
            ),
            "skipped_invalid_normals": sampling_result[
                "skipped_invalid_normals"
            ],
            "skipped_invalid_geometry": sampling_result[
                "skipped_invalid_geometry"
            ],
        },
        "normal_analysis": normal_info,
        "orientation_analysis": orientation_info,
        "path_metrics": path_metrics,
        "inter_segment_travel_plan": inter_segment_travel_plan,
        "advanced_geometry_analysis": advanced_analysis,
        "analysis": analysis,
    }


# ------------------------------------------------------------
# JSON conversion
# ------------------------------------------------------------

def _make_json_safe(value: Any) -> Any:

    if isinstance(value, dict):
        return {
            str(key): _make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        result = float(value)
        return result if math.isfinite(result) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ------------------------------------------------------------
# Saved report
# ------------------------------------------------------------

def _build_saved_report(
    trajectory: dict,
    stl_path: str | Path,
    result_manager: ResultManager,
) -> dict:

    sample_count = len(trajectory["surface_points"])
    coordinate_unit = trajectory["parameters"]["coordinate_unit"]

    return {
        "stage": 3,
        "stage_name": "Spray Trajectory Planning",
        "source_file": str(Path(stl_path).resolve()),
        "part_name": result_manager.part_name_from_stl(stl_path),
        "coordinate_unit": coordinate_unit,
        "parameters": trajectory["parameters"],
        "sampling": trajectory["sampling"],
        "normal_analysis": trajectory["normal_analysis"],
        "orientation_analysis": trajectory["orientation_analysis"],
        "path_metrics": trajectory["path_metrics"],
        "inter_segment_travel_plan": trajectory.get("inter_segment_travel_plan", {}),
        "surface_boundary_splits": trajectory.get("surface_boundary_splits", {}),
        "surface_geometry": {
            "geometry_source": trajectory.get(
                "surface_geometry_source", "DISPLAY_MESH_TRIANGLES"
            ),
            "patch_brep_face_mapping": trajectory.get(
                "patch_brep_face_mapping"
            ),
            "mapping_scope": (
                "MANY_TO_MANY; Stage 2 statuses remain per source triangle"
            ),
        },
        "segments": trajectory["segments"],
        "path_statistics": trajectory.get("path_statistics", {}),
        "trajectory_validation": trajectory.get("trajectory_validation", {}),
        "trajectory_safety": trajectory.get("trajectory_safety", {}),
        "rasterization": trajectory.get("rasterization", {}),
        "advanced_geometry_analysis": trajectory[
            "advanced_geometry_analysis"
        ],
        "analysis": trajectory["analysis"],
        "trajectory_data": {
            "sample_count": sample_count,
            "coordinate_unit": coordinate_unit,
            "columns": [
                "surface_x_coordinate_units",
                "surface_y_coordinate_units",
                "surface_z_coordinate_units",
                "normal_x",
                "normal_y",
                "normal_z",
                "nozzle_x_coordinate_units",
                "nozzle_y_coordinate_units",
                "nozzle_z_coordinate_units",
                "spray_direction_x",
                "spray_direction_y",
                "spray_direction_z",
                "face_index",
                "patch_id",
                "segment_id",
                "sample_index",
            ],
            "array_file": "trajectory_points.npy",
            "world_array_file": "trajectory_points_world.npy",
            "world_array_columns": [
                "world_surface_x_coordinate_units",
                "world_surface_y_coordinate_units",
                "world_surface_z_coordinate_units",
                "world_normal_x", "world_normal_y", "world_normal_z",
                "world_nozzle_x_coordinate_units",
                "world_nozzle_y_coordinate_units",
                "world_nozzle_z_coordinate_units",
                "world_spray_direction_x",
                "world_spray_direction_y",
                "world_spray_direction_z",
                "face_index", "patch_id", "segment_id", "sample_index",
            ],
            "nozzle_frames_file": "trajectory_nozzle_frames.npy",
            "world_nozzle_frames_file": "trajectory_nozzle_frames_world.npy",
            "nozzle_frame_columns": [
                "x_axis_x", "x_axis_y", "x_axis_z",
                "y_axis_x", "y_axis_y", "y_axis_z",
                "z_axis_x", "z_axis_y", "z_axis_z",
            ],
            "world_coordinate_frame": "WORLD_XYZ",
            "column_compatibility": (
                "The 16-column order is unchanged from the previous "
                "Stage 3 format. Coordinate columns now explicitly "
                "identify their values as STL coordinate units."
            ),
        },
        "engineering_limitations": [
            "A sampled cylindrical gun-body clearance check is performed in PRODUCT_CAD coordinates and reported separately from the trajectory arrays.",
            "The gun-body envelope uses configured dimensions; provisional dimensions are explicitly marked uncertain.",
            "Nozzle shaft radial clearance, robot links, wrist, hoses, fixtures, and external obstacles are not yet checked by the safety module.",
            "The spray-gun clearance result is not a certified robot safety assessment and cannot establish full robot reachability.",
            "Robot reachability is not checked.",
            "Robot collision is not checked.",
            "Fixture and hook clearance are not checked.",
            "Spray coverage is not verified.",
            "Coating thickness is not calculated.",
            "No actual spray-gun CAD or calibrated TCP offset is applied.",
            "The nozzle roll axis is a provisional geometric reference, not a manufacturer-defined tool frame.",
            (
                "Surface sample positions and normals are evaluated on original STEP B-Rep Faces; triangle indices are retained only as source mapping metadata."
                if trajectory.get("surface_geometry_source") == "ORIGINAL_STEP_BREP_SURFACE"
                else "Surface samples and normals come from display-mesh triangles because STL inputs do not contain original CAD surfaces."
            ),
            "Open or non-watertight mesh normals may have uncertain outward direction and require review.",
            "The product CAD-to-World rigid transform is recorded and world-coordinate arrays are exported.",
            "The initial generator uses global PCA-projected serpentine sweeps over all sampled paintable surfaces; local mesh normals guide nozzle orientation, and only 3D spatial gaps split segments.",
            "Surface sample spacing and nominal raster lane spacing are derived from the configured cone footprint and overlap; this is geometric coverage planning, not calibrated coating-deposition verification.",
            "Inter-segment travel is represented as direct straight-line, spray-OFF visualization estimates only; collision and reachability are not checked.",
            "Travel movements between separate patches are also represented by the same unvalidated straight-line estimates.",
            "Bend and cylinder estimates do not yet modify nozzle orientation.",
        ],
    }


# ------------------------------------------------------------
# Execute Stage 3 and save
# ------------------------------------------------------------

def execute_stage3(
    mesh: trimesh.Trimesh,
    stl_path: str | Path,
    result_manager: ResultManager,
    stage2_result: dict | None = None,
    face_status_labels: np.ndarray | None = None,
    patch_labels: np.ndarray | None = None,
    standoff_mm: float = DEFAULT_STANDOFF_MM,
    stage1_report: dict | None = None,
    coordinate_unit: str | None = None,
    sprayer_profile: dict | None = None,
    coating_process: dict | None = None,
    product_transform: np.ndarray | list | None = None,
) -> dict:

    if stage2_result is not None:

        if face_status_labels is None:
            face_status_labels = stage2_result.get(
                "face_status_labels"
            )

        if patch_labels is None:
            patch_labels = stage2_result.get("patch_labels")

    if face_status_labels is None:
        raise ValueError(
            "Stage 3 requires Stage 2 face_status_labels."
        )

    if patch_labels is None:
        raise ValueError(
            "Stage 3 requires Stage 1 patch_labels."
        )

    trajectory = generate_spray_trajectory(
        mesh=mesh,
        face_status_labels=face_status_labels,
        patch_labels=patch_labels,
        standoff_mm=standoff_mm,
        stage1_report=stage1_report,
        coordinate_unit=coordinate_unit,
        sprayer_profile=sprayer_profile,
        coating_process=coating_process,
        product_transform=product_transform,
    )

    # This first Stage 3 action is a spray-point / circular-footprint
    # visualization preview. Do not run the expensive gun-envelope
    # clearance analysis in this UI worker: it can take substantially
    # longer than point generation and prevents the preview from appearing.
    # Safety remains a separate follow-up operation and is not implied here.

    stage_directory = result_manager.stage_dir(
        stl_path,
        stage_number=3,
    )

    sample_count = len(trajectory["surface_points"])
    sample_indices = np.arange(sample_count, dtype=np.float64)

    # Preserve the original 16-column order.
    trajectory_array = np.column_stack([
        trajectory["surface_points"],
        trajectory["surface_normals"],
        trajectory["nozzle_points"],
        trajectory["spray_directions"],
        trajectory["face_indices"].astype(np.float64),
        trajectory["patch_ids"].astype(np.float64),
        trajectory["segment_ids"].astype(np.float64),
        sample_indices,
    ])

    if trajectory_array.shape != (sample_count, 16):
        raise RuntimeError(
            "Trajectory NPY array must contain exactly 16 columns."
        )

    array_path = stage_directory / "trajectory_points.npy"
    np.save(array_path, trajectory_array)

    world_trajectory_array = np.column_stack([
        trajectory["surface_points_world"],
        trajectory["surface_normals_world"],
        trajectory["nozzle_points_world"],
        trajectory["spray_directions_world"],
        trajectory["face_indices"].astype(np.float64),
        trajectory["patch_ids"].astype(np.float64),
        trajectory["segment_ids"].astype(np.float64),
        sample_indices,
    ])
    if world_trajectory_array.shape != (sample_count, 16):
        raise RuntimeError("World trajectory NPY array must contain exactly 16 columns.")
    world_array_path = stage_directory / "trajectory_points_world.npy"
    np.save(world_array_path, world_trajectory_array)

    nozzle_frames_path = stage_directory / "trajectory_nozzle_frames.npy"
    np.save(nozzle_frames_path, trajectory["nozzle_frames"].reshape(sample_count, 9))

    world_nozzle_frames_path = stage_directory / "trajectory_nozzle_frames_world.npy"
    np.save(
        world_nozzle_frames_path,
        trajectory["nozzle_frames_world"].reshape(sample_count, 9),
    )

    report = _build_saved_report(
        trajectory=trajectory,
        stl_path=stl_path,
        result_manager=result_manager,
    )

    report = _make_json_safe(report)

    json_path = result_manager.save_json(
        stl_path=stl_path,
        stage_number=3,
        filename="spray_trajectory.json",
        data=report,
    )

    return {
        "trajectory": trajectory,
        "report": report,
        "json_path": json_path,
        "array_path": array_path,
        "world_array_path": world_array_path,
        "nozzle_frames_path": nozzle_frames_path,
        "world_nozzle_frames_path": world_nozzle_frames_path,
        "sample_count": sample_count,
        "stage_directory": stage_directory,
    }


# ------------------------------------------------------------
# Numerical tests
# ------------------------------------------------------------

def _make_test_mesh(
    coordinate_scale: float = 1.0,
) -> trimesh.Trimesh:

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
        ],
        dtype=np.float64,
    ) * coordinate_scale

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )


def run_stage3_numerical_tests() -> dict:

    # Circular-cone footprint uses the configured working distance, angle and overlap.
    circular_footprint = _calculate_spray_footprint({
        "pattern_type": "round_cone",
        "spray_cone_angle_deg": 74.0,
        "overlap_percent": 50.0,
    }, 180.0)
    expected_diameter = 2.0 * 180.0 * math.tan(math.radians(37.0))
    assert math.isclose(circular_footprint["diameter_mm"], expected_diameter, rel_tol=1e-10)
    assert math.isclose(circular_footprint["nominal_point_spacing_mm"], expected_diameter * 0.5, rel_tol=1e-10)
    assert circular_footprint["pattern_type"] == "round_cone"

    # 3D barycentric mapping used to transfer path samples to STEP UV nodes.
    barycentric_test = _barycentric_coordinates_on_triangle(
        np.asarray([0.25, 0.25, 0.0], dtype=np.float64),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    )
    assert barycentric_test is not None
    assert np.allclose(barycentric_test, [0.5, 0.25, 0.25], atol=1e-9)
    tiny_triangle_test = _barycentric_coordinates_on_triangle(
        np.asarray([2.5e-5, 2.5e-5, 0.0], dtype=np.float64),
        np.asarray([
            [0.0, 0.0, 0.0],
            [1e-4, 0.0, 0.0],
            [0.0, 1e-4, 0.0],
        ], dtype=np.float64),
    )
    assert tiny_triangle_test is not None
    assert np.allclose(tiny_triangle_test, [0.5, 0.25, 0.25], atol=1e-9)
    assert _barycentric_coordinates_on_triangle(
        np.asarray([1.5, 1.5, 0.0], dtype=np.float64),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    ) is None

    test_mesh = _make_test_mesh(1.0)

    # STEP refinement must never hide an atomic fallback behind the default
    # NOT_APPLICABLE status. Simulate an attached runtime context and force
    # two deterministic source-triangle mapping failures.
    from unittest.mock import patch

    refinement_test_trajectory = {
        "surface_points": np.asarray([[0.2, 0.2, 0.0]], dtype=np.float64),
        "surface_normals": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        "face_indices": np.asarray([999], dtype=np.int64),
        "segment_ids": np.asarray([0], dtype=np.int64),
    }
    with patch("core.step_loader.get_step_brep_context", return_value=object()):
        invalid_triangle_result, invalid_triangle_summary = (
            _refine_trajectory_with_step_brep(
                test_mesh, refinement_test_trajectory, 10.0, "mm"
            )
        )
    assert invalid_triangle_summary["status"] == "FALLBACK_SAMPLE_MAPPING_FAILED"
    assert invalid_triangle_summary["failure_reason"] == "invalid_source_triangle_index"
    assert invalid_triangle_summary["failed_sample_count"] == 1
    assert np.array_equal(
        invalid_triangle_result["surface_points"],
        refinement_test_trajectory["surface_points"],
    )

    outside_triangle_trajectory = dict(refinement_test_trajectory)
    outside_triangle_trajectory["surface_points"] = np.asarray(
        [[200.0, 200.0, 0.0]], dtype=np.float64
    )
    outside_triangle_trajectory["face_indices"] = np.asarray([0], dtype=np.int64)
    with patch("core.step_loader.get_step_brep_context", return_value=object()):
        outside_triangle_result, outside_triangle_summary = (
            _refine_trajectory_with_step_brep(
                test_mesh, outside_triangle_trajectory, 10.0, "mm"
            )
        )
    assert outside_triangle_summary["status"] == "FALLBACK_SAMPLE_MAPPING_FAILED"
    assert outside_triangle_summary["failure_reason"] == "sample_outside_source_triangle"
    assert outside_triangle_summary["failed_sample_count"] == 1
    assert np.array_equal(
        outside_triangle_result["surface_points"],
        outside_triangle_trajectory["surface_points"],
    )

    labels = np.array(
        [FACE_PAINTABLE_CANDIDATE] * 2,
        dtype=np.int64,
    )

    patches = np.array([0, 0], dtype=np.int64)

    original_vertices = np.asarray(
        test_mesh.vertices,
        dtype=np.float64,
    ).copy()

    # --------------------------------------------------------
    # Test 1: Original mm sample count.
    # --------------------------------------------------------

    result = generate_spray_trajectory(
        mesh=test_mesh,
        face_status_labels=labels,
        patch_labels=patches,
        standoff_mm=200.0,
        coordinate_unit="mm",
    )

    surface_points = result["surface_points"]
    normals = result["surface_normals"]
    nozzle_points = result["nozzle_points"]
    directions = result["spray_directions"]

    assert len(surface_points) > 0, (
        "Circular-footprint sampling must generate at least one surface point."
    )

    # --------------------------------------------------------
    # Test 2: Physical standoff.
    # --------------------------------------------------------

    measured_standoff = np.linalg.norm(
        nozzle_points - surface_points,
        axis=1,
    )

    assert np.allclose(
        measured_standoff,
        200.0,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 3: Unit normals.
    # --------------------------------------------------------

    assert np.allclose(
        np.linalg.norm(normals, axis=1),
        1.0,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 4: Unit spray directions.
    # --------------------------------------------------------

    assert np.allclose(
        np.linalg.norm(directions, axis=1),
        1.0,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 5: Direction opposite to normal.
    # --------------------------------------------------------

    assert np.allclose(
        directions,
        -normals,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 6: Nozzle position formula.
    # --------------------------------------------------------

    assert np.allclose(
        nozzle_points,
        surface_points + 200.0 * normals,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 7: Open mesh orientation.
    # --------------------------------------------------------

    orientation = result["orientation_analysis"]

    assert orientation["watertight"] is False
    assert orientation["orientation_reliable"] is False

    # --------------------------------------------------------
    # Test 8: Stage 1 advanced geometry integration.
    # Legacy *_mm values are interpreted as coordinate values.
    # --------------------------------------------------------

    synthetic_stage1 = {
        "unit_configuration": {
            "source_stl_coordinate_unit": "mm",
        },
        "surface_segmentation": {
            "patches": [
                {
                    "patch_id": 0,
                    "surface_type": "curved",
                    "classification": "curved",
                    "cylindrical_fit": {
                        "candidate": True,
                        "radius_mm": 25.0,
                        "diameter_mm": 50.0,
                        "confidence": 0.95,
                        "surface_type": "external",
                        "fit_residual_mm": 0.2,
                    },
                }
            ]
        },
        "advanced_geometry": {
            "cylindrical_features": [{}],
            "hole_candidates": [{}],
            "bends": [{}],
            "components": [{}],
            "edges": {},
        },
    }

    feature_result = generate_spray_trajectory(
        mesh=test_mesh,
        face_status_labels=labels,
        patch_labels=patches,
        standoff_mm=200.0,
        stage1_report=synthetic_stage1,
    )

    advanced = feature_result["advanced_geometry_analysis"]

    assert advanced["available"] is True
    assert advanced["summary"]["cylindrical_feature_count"] == 1
    assert advanced["summary"]["hole_candidate_count"] == 1
    assert advanced["summary"]["bend_count"] == 1

    patch_feature = advanced["trajectory_patch_features"][0]

    assert patch_feature["cylindrical_candidate"] is True
    assert patch_feature["cylinder_radius_coordinate_units"] == 25.0
    assert patch_feature["cylinder_surface_type"] == "external"

    # --------------------------------------------------------
    # Test 9: Inspection faces are excluded.
    # --------------------------------------------------------

    inspection_labels = np.array(
        [FACE_REQUIRES_INSPECTION] * 2,
        dtype=np.int64,
    )

    try:
        generate_spray_trajectory(
            mesh=test_mesh,
            face_status_labels=inspection_labels,
            patch_labels=patches,
            coordinate_unit="mm",
        )
    except RuntimeError:
        inspection_faces_excluded = True
    else:
        inspection_faces_excluded = False

    assert inspection_faces_excluded

    # --------------------------------------------------------
    # Test 10: Legacy compatibility without Stage 1 report.
    # --------------------------------------------------------

    assert result["advanced_geometry_analysis"]["available"] is False

    # --------------------------------------------------------
    # Test 11: JSON serialization.
    # --------------------------------------------------------

    json_text = json.dumps(
        _make_json_safe(feature_result),
        allow_nan=False,
    )

    assert len(json_text) > 0

    # --------------------------------------------------------
    # Test 12: Unknown unit must be rejected.
    # --------------------------------------------------------

    unknown_unit_rejected = False

    try:
        generate_spray_trajectory(
            mesh=test_mesh,
            face_status_labels=labels,
            patch_labels=patches,
        )
    except ValueError as exc:
        unknown_unit_rejected = "coordinate unit is unknown" in str(exc)

    assert unknown_unit_rejected

    # --------------------------------------------------------
    # Test 13: Equivalent physical standoff for all units.
    #
    # The same physical geometry is represented in different
    # coordinate systems. The physical nozzle distance must
    # remain exactly 200 mm.
    # --------------------------------------------------------

    unit_test_results = {}

    for unit, mm_per_unit in MM_PER_UNIT.items():

        scale = 1.0 / mm_per_unit
        unit_mesh = _make_test_mesh(scale)

        unit_result = generate_spray_trajectory(
            mesh=unit_mesh,
            face_status_labels=labels,
            patch_labels=patches,
            standoff_mm=200.0,
            coordinate_unit=unit,
        )

        coordinate_distance = np.linalg.norm(
            unit_result["nozzle_points"]
            - unit_result["surface_points"],
            axis=1,
        )

        physical_distance_mm = (
            coordinate_distance * mm_per_unit
        )

        assert np.allclose(
            physical_distance_mm,
            200.0,
            atol=1e-6,
        )

        expected_coordinate_standoff = 200.0 / mm_per_unit

        assert math.isclose(
            unit_result["parameters"][
                "standoff_coordinate_units"
            ],
            expected_coordinate_standoff,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )

        unit_test_results[unit] = {
            "standoff_coordinate_units": expected_coordinate_standoff,
            "standoff_mm": float(np.mean(physical_distance_mm)),
            "status": "PASS",
        }

    # --------------------------------------------------------
    # Test 14: Equivalent physical sampling spacing.
    # --------------------------------------------------------

    sampling_test_results = {}

    for unit, mm_per_unit in MM_PER_UNIT.items():

        scale = 1.0 / mm_per_unit
        unit_mesh = _make_test_mesh(scale)

        unit_result = generate_spray_trajectory(
            mesh=unit_mesh,
            face_status_labels=labels,
            patch_labels=patches,
            standoff_mm=200.0,
            coordinate_unit=unit,
        )

        expected_spacing_mm = (
            2.0 * 200.0 * math.tan(math.radians(45.0 * 0.5)) * 0.5
        )
        expected_spacing = expected_spacing_mm / mm_per_unit

        assert math.isclose(
            unit_result["parameters"][
                "sampling_spacing_coordinate_units"
            ],
            expected_spacing,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )

        sampling_test_results[unit] = {
            "spacing_coordinate_units": expected_spacing,
            "spacing_mm": expected_spacing_mm,
            "status": "PASS",
        }

    # --------------------------------------------------------
    # Test 15: Source mesh vertices are unchanged.
    # --------------------------------------------------------

    assert np.array_equal(
        np.asarray(test_mesh.vertices, dtype=np.float64),
        original_vertices,
    )

    # --------------------------------------------------------
    # Test 16: 16-column NPY-compatible data layout.
    # --------------------------------------------------------

    sample_count = len(result["surface_points"])

    trajectory_array = np.column_stack([
        result["surface_points"],
        result["surface_normals"],
        result["nozzle_points"],
        result["spray_directions"],
        result["face_indices"].astype(np.float64),
        result["patch_ids"].astype(np.float64),
        result["segment_ids"].astype(np.float64),
        np.arange(sample_count, dtype=np.float64),
    ])

    assert trajectory_array.shape == (sample_count, 16)

    # --------------------------------------------------------
    # Test 17: Sprayer spray-envelope and overlap metadata.
    # --------------------------------------------------------

    test_sprayer_profile = {
        "name": "test sprayer",
        "spray_cone_angle_deg": 60.0,
        "overlap_percent": 50.0,
        "pattern_type": "round_cone",
    }
    sprayer_result = generate_spray_trajectory(
        mesh=test_mesh,
        face_status_labels=labels,
        patch_labels=patches,
        standoff_mm=200.0,
        coordinate_unit="mm",
        sprayer_profile=test_sprayer_profile,
    )
    expected_spray_width = (
        2.0 * 200.0 * math.tan(math.radians(30.0))
    )
    expected_track_spacing = expected_spray_width * 0.5
    assert math.isclose(
        sprayer_result["parameters"]["spray_width_at_standoff_mm"],
        expected_spray_width,
        rel_tol=1e-10,
    )
    assert math.isclose(
        sprayer_result["parameters"]["nominal_track_spacing_from_overlap_mm"],
        expected_track_spacing,
        rel_tol=1e-10,
    )
    assert (
        sprayer_result["parameters"]["sprayer_profile"]["name"]
        == "test sprayer"
    )

    # --------------------------------------------------------
    # Test 18: Product CAD pose and provisional nozzle frames.
    # --------------------------------------------------------

    angle = math.radians(90.0)
    test_transform = np.array([
        [math.cos(angle), -math.sin(angle), 0.0, 10.0],
        [math.sin(angle),  math.cos(angle), 0.0, 20.0],
        [0.0,              0.0,             1.0, 30.0],
        [0.0,              0.0,             0.0, 1.0],
    ])
    posed_result = generate_spray_trajectory(
        mesh=test_mesh,
        face_status_labels=labels,
        patch_labels=patches,
        standoff_mm=200.0,
        coordinate_unit="mm",
        product_transform=test_transform,
    )
    expected_world_surface = (
        posed_result["surface_points"] @ test_transform[:3, :3].T
        + test_transform[:3, 3]
    )
    expected_world_nozzle = (
        posed_result["nozzle_points"] @ test_transform[:3, :3].T
        + test_transform[:3, 3]
    )
    assert np.allclose(posed_result["surface_points_world"], expected_world_surface)
    assert np.allclose(posed_result["nozzle_points_world"], expected_world_nozzle)
    assert np.allclose(
        posed_result["nozzle_frames_world"],
        np.einsum("ij,njk->nik", test_transform[:3, :3], posed_result["nozzle_frames"]),
    )
    frame_gram = np.einsum(
        "nji,njk->nik",
        posed_result["nozzle_frames"],
        posed_result["nozzle_frames"],
    )
    assert np.allclose(
        frame_gram,
        np.repeat(np.eye(3)[None, :, :], len(frame_gram), axis=0),
        atol=1e-6,
    )
    assert np.allclose(np.linalg.det(posed_result["nozzle_frames"]), 1.0, atol=1e-6)

    invalid_transform_rejected = False
    invalid_transform = test_transform.copy()
    invalid_transform[0, 0] = 2.0
    try:
        generate_spray_trajectory(
            mesh=test_mesh,
            face_status_labels=labels,
            patch_labels=patches,
            coordinate_unit="mm",
            product_transform=invalid_transform,
        )
    except ValueError:
        invalid_transform_rejected = True
    assert invalid_transform_rejected

    # --------------------------------------------------------
    # Test 19: Planar raster uses fan width and overlap.
    # --------------------------------------------------------

    raster_profile = {
        "name": "raster test sprayer",
        "spray_cone_angle_deg": 20.0,
        "overlap_percent": 50.0,
        "pattern_type": "round_cone",
    }
    raster_result = generate_spray_trajectory(
        mesh=test_mesh,
        face_status_labels=labels,
        patch_labels=patches,
        standoff_mm=200.0,
        coordinate_unit="mm",
        sprayer_profile=raster_profile,
    )
    expected_raster_width = 2.0 * 200.0 * math.tan(math.radians(10.0))
    expected_raster_spacing = expected_raster_width * 0.5
    raster_info = raster_result["rasterization"]
    assert raster_info["rasterized_planar_patch_ids"] == [0]
    assert raster_info["nearest_neighbor_fallback_patch_ids"] == []
    assert math.isclose(raster_info["lane_spacing_mm"], expected_raster_spacing, rel_tol=1e-10)
    assert len(np.unique(raster_result["segment_ids"])) >= 2
    assert np.allclose(
        np.linalg.norm(raster_result["nozzle_points"] - raster_result["surface_points"], axis=1),
        200.0, atol=1e-6,
    )
    assert np.allclose(raster_result["surface_points"][:, 2], 0.0, atol=1e-7)
    assert np.all(
        (raster_result["surface_points"][:, 0] >= -1e-7)
        & (raster_result["surface_points"][:, 0] <= 100.0 + 1e-7)
        & (raster_result["surface_points"][:, 1] >= -1e-7)
        & (raster_result["surface_points"][:, 1] <= 100.0 + 1e-7)
    )

    # --------------------------------------------------------
    # Test 21: Round-cone footprint is circular.
    # --------------------------------------------------------
    round_cone = _calculate_spray_footprint({
        "pattern_type": "round_cone",
        "spray_cone_angle_deg": 40.0,
        "overlap_percent": 25.0,
    }, 200.0)
    assert math.isclose(
        round_cone["width_mm"],
        round_cone["thickness_mm"],
        rel_tol=1e-12,
    )

    # --------------------------------------------------------
    # Test 22: Larger spray angle reduces planar pass count.
    # --------------------------------------------------------
    wide_angle_profile = {
        "name": "wide angle test",
        "pattern_type": "round_cone",
        "spray_cone_angle_deg": 60.0,
        "overlap_percent": 50.0,
    }
    wide_angle_result = generate_spray_trajectory(
        mesh=test_mesh,
        face_status_labels=labels,
        patch_labels=patches,
        standoff_mm=200.0,
        coordinate_unit="mm",
        sprayer_profile=wide_angle_profile,
    )
    narrow_pass_count = len(np.unique(raster_result["segment_ids"]))
    wide_pass_count = len(np.unique(wide_angle_result["segment_ids"]))
    assert wide_pass_count < narrow_pass_count, (
        f"Wider spray angle should reduce planar passes: "
        f"{wide_pass_count} !< {narrow_pass_count}"
    )

    # --------------------------------------------------------
    # Test 22A: Curved surface raster reconstructs samples on a cylinder.
    # --------------------------------------------------------
    curved_test_mesh = trimesh.creation.cylinder(
        radius=40.0,
        height=100.0,
        sections=48,
    )
    curved_test_normals = np.asarray(
        curved_test_mesh.face_normals, dtype=np.float64
    )
    curved_side_faces = np.flatnonzero(
        (np.abs(curved_test_normals[:, 2]) < 0.1)
        & (curved_test_normals[:, 0] > 0.3)
    )
    curved_raster = _generate_surface_following_raster_for_patch(
        mesh=curved_test_mesh,
        face_indices=curved_side_faces,
        normals=curved_test_normals,
        standoff_coordinate_units=25.0,
        sample_spacing_coordinate_units=5.0,
        track_spacing_coordinate_units=8.0,
    )
    assert curved_raster is not None
    assert len(curved_raster["surface_points"]) > 10
    assert len(np.unique(curved_raster["segment_ids"])) >= 2
    radial_distances = np.linalg.norm(
        curved_raster["surface_points"][:, :2], axis=1
    )
    # The test cylinder is a 48-sided triangulated approximation. Surface
    # points reconstructed barycentrically lie on its planar facets (chords),
    # not on the ideal analytic radius except at polygon vertices.
    polygon_apothem = 40.0 * math.cos(math.pi / 48.0)
    assert np.all(radial_distances >= polygon_apothem - 1e-5), (
        f"Curved raster points fell inside the cylinder mesh: "
        f"minimum radius {float(np.min(radial_distances)):.6f} mm, "
        f"mesh apothem {polygon_apothem:.6f} mm"
    )
    assert np.all(radial_distances <= 40.0 + 1e-5), (
        f"Curved raster points exceeded the cylinder mesh radius: "
        f"maximum radius {float(np.max(radial_distances)):.6f} mm"
    )
    assert np.allclose(
        np.linalg.norm(
            curved_raster["nozzle_points"] - curved_raster["surface_points"],
            axis=1,
        ),
        25.0,
        atol=1e-6,
    )
    assert np.allclose(
        np.linalg.norm(curved_raster["surface_normals"], axis=1),
        1.0,
        atol=1e-6,
    )


    # --------------------------------------------------------
    # Test 22B: Full cylindrical patch is split into coherent normal cones.
    # --------------------------------------------------------
    full_cylinder_faces = np.flatnonzero(
        np.abs(curved_test_normals[:, 2]) < 0.1
    )
    partition_diagnostics = {}
    partitioned_cylinder = _generate_partitioned_surface_raster_for_patch(
        mesh=curved_test_mesh,
        face_indices=full_cylinder_faces,
        normals=curved_test_normals,
        standoff_coordinate_units=25.0,
        sample_spacing_coordinate_units=5.0,
        track_spacing_coordinate_units=8.0,
        max_normal_cone_angle_deg=45.0,
        diagnostics=partition_diagnostics,
    )
    assert partitioned_cylinder is not None, (
        "A full cylindrical side patch should rasterize after normal-cone "
        f"partitioning: {partition_diagnostics}"
    )
    assert partitioned_cylinder["normal_cluster_count"] > 1
    assert len(partitioned_cylinder["surface_points"]) > 10
    assert np.allclose(
        np.linalg.norm(
            partitioned_cylinder["nozzle_points"]
            - partitioned_cylinder["surface_points"],
            axis=1,
        ),
        25.0,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 22E: Projected 2-D overlap cannot create a large 3-D spray stroke.
    # --------------------------------------------------------
    discontinuous_lane_test = np.asarray([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [300.0, 0.0, 0.0],
        [310.0, 0.0, 0.0],
        [320.0, 0.0, 0.0],
        [330.0, 0.0, 0.0],
    ], dtype=np.float64)
    discontinuous_chunks = _split_raster_lane_at_discontinuities(
        discontinuous_lane_test,
        maximum_step_coordinate_units=30.0,
    )
    assert discontinuous_chunks == [(0, 3), (3, 7)]
    for chunk_start, chunk_end in discontinuous_chunks:
        if chunk_end - chunk_start > 1:
            chunk_steps = np.linalg.norm(
                np.diff(
                    discontinuous_lane_test[chunk_start:chunk_end],
                    axis=0,
                ),
                axis=1,
            )
            assert np.all(chunk_steps <= 30.0)

    # --------------------------------------------------------
    # Test 22G: Parallel transport prevents arbitrary roll flips when the
    # spray axis is unchanged and follows curved spray axes with minimum twist.
    # --------------------------------------------------------
    transport_z = np.asarray([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, math.sin(math.radians(20.0)), math.cos(math.radians(20.0))],
        [0.0, math.sin(math.radians(40.0)), math.cos(math.radians(40.0))],
    ], dtype=np.float64)
    transport_segments = np.asarray([0, 0, 0, 0], dtype=np.int64)
    transported_frames = _build_parallel_transport_nozzle_frames(
        transport_z,
        transport_segments,
    )
    transport_step_angles = []
    for frame_index in range(len(transported_frames) - 1):
        relative = (
            transported_frames[frame_index].T
            @ transported_frames[frame_index + 1]
        )
        transport_step_angles.append(math.degrees(math.acos(float(np.clip(
            (np.trace(relative) - 1.0) * 0.5, -1.0, 1.0
        )))))
    assert abs(transport_step_angles[0]) < 1e-6
    assert abs(transport_step_angles[1] - 20.0) < 1e-6
    assert abs(transport_step_angles[2] - 20.0) < 1e-6
    assert np.allclose(
        np.linalg.det(transported_frames),
        1.0,
        atol=1e-6,
    )
    restarted_frames = _build_parallel_transport_nozzle_frames(
        np.asarray([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
        np.asarray([0, 1], dtype=np.int64),
    )
    assert np.allclose(restarted_frames[0], restarted_frames[1], atol=1e-6)

    # --------------------------------------------------------
    # Test 22F: Normal smoothing reduces zig-zag while preserving a strict
    # angular bound from every original source normal.
    # --------------------------------------------------------
    zigzag_angles = np.radians(np.asarray(
        [0.0, 40.0, 0.0, 40.0, 0.0, 40.0, 0.0],
        dtype=np.float64,
    ))
    zigzag_normals = np.column_stack((
        np.cos(zigzag_angles),
        np.sin(zigzag_angles),
        np.zeros(len(zigzag_angles)),
    ))
    smoothed_zigzag, max_zigzag_deviation = (
        _smooth_normals_with_deviation_limit(
            zigzag_normals,
            max_deviation_deg=8.0,
            iterations=6,
        )
    )
    raw_zigzag_steps = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", zigzag_normals[:-1], zigzag_normals[1:]),
        -1.0,
        1.0,
    )))
    smooth_zigzag_steps = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", smoothed_zigzag[:-1], smoothed_zigzag[1:]),
        -1.0,
        1.0,
    )))
    assert float(np.max(smooth_zigzag_steps)) < float(np.max(raw_zigzag_steps))
    assert max_zigzag_deviation <= 8.0 + 1e-6
    assert np.allclose(
        np.linalg.norm(smoothed_zigzag, axis=1), 1.0, atol=1e-6
    )

    # --------------------------------------------------------
    # Test 22D: A small normal group is merged only when directionally safe.
    # --------------------------------------------------------
    merge_test_normals = np.asarray([
        [1.0, 0.0, 0.0],
        [math.cos(math.radians(40.0)), math.sin(math.radians(40.0)), 0.0],
        [math.cos(math.radians(80.0)), math.sin(math.radians(80.0)), 0.0],
    ], dtype=np.float64)
    merge_test_groups, merge_test_count = _merge_single_face_normal_groups(
        groups=[
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        ],
        face_indices=np.asarray([0, 1, 2], dtype=np.int64),
        unit_normals=merge_test_normals,
        max_merge_angle_deg=70.0,
    )
    assert merge_test_count == 1
    assert len(merge_test_groups) == 1
    assert set(merge_test_groups[0].tolist()) == {0, 1, 2}

    # --------------------------------------------------------
    # Test 22C: Adaptive scanlines recover separated projected bands.
    # The regular global lane falls in the gap between the two triangles.
    # --------------------------------------------------------
    disconnected_band_mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 10.0, 0.0], [100.0, 10.0, 0.0], [0.0, 11.0, 0.0],
        ], dtype=np.float64),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        process=False,
    )
    disconnected_band_normals = np.asarray(
        disconnected_band_mesh.face_normals, dtype=np.float64
    )
    adaptive_scanline_diagnostics = {}
    adaptive_band_raster = _generate_surface_following_raster_for_patch(
        mesh=disconnected_band_mesh,
        face_indices=np.asarray([0, 1], dtype=np.int64),
        normals=disconnected_band_normals,
        standoff_coordinate_units=5.0,
        sample_spacing_coordinate_units=2.0,
        track_spacing_coordinate_units=100.0,
        diagnostics=adaptive_scanline_diagnostics,
    )
    assert adaptive_band_raster is not None, (
        "Adaptive scanlines should recover disconnected projected bands: "
        f"{adaptive_scanline_diagnostics}"
    )
    assert adaptive_band_raster["adaptive_scanline_offsets_used"] is True
    assert set(np.unique(adaptive_band_raster["face_indices"]).tolist()) == {0, 1}
    assert len(np.unique(adaptive_band_raster["segment_ids"])) >= 2
    assert np.allclose(
        np.linalg.norm(
            adaptive_band_raster["nozzle_points"]
            - adaptive_band_raster["surface_points"],
            axis=1,
        ),
        5.0,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 22H: Planar raster samples must retain a valid source-triangle
    # barycentric association even when the accepted patch is slightly warped.
    # The raster must reconstruct points on the selected 3-D triangle rather
    # than on the average PCA plane.
    # --------------------------------------------------------
    warped_planar_mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 100.0, 0.20],
            [0.0, 100.0, 0.20],
        ], dtype=np.float64),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    warped_planar_normals = np.asarray(
        warped_planar_mesh.face_normals, dtype=np.float64
    )
    warped_planar_raster = _generate_planar_raster_for_patch(
        mesh=warped_planar_mesh,
        face_indices=np.asarray([0, 1], dtype=np.int64),
        normals=warped_planar_normals,
        standoff_coordinate_units=10.0,
        sample_spacing_coordinate_units=10.0,
        track_spacing_coordinate_units=25.0,
    )
    assert warped_planar_raster is not None
    assert len(warped_planar_raster["surface_points"]) > 0
    for point, face_index in zip(
        warped_planar_raster["surface_points"],
        warped_planar_raster["face_indices"],
    ):
        source_triangle = warped_planar_mesh.vertices[
            warped_planar_mesh.faces[int(face_index)]
        ]
        weights = _barycentric_coordinates_on_triangle(
            point, source_triangle, tolerance=2e-5
        )
        assert weights is not None, (
            "Planar raster emitted a sample outside its selected source triangle."
        )
        reconstructed = weights @ source_triangle
        assert np.allclose(point, reconstructed, atol=1e-7, rtol=1e-9)

    # --------------------------------------------------------
    # Test 24: Inter-segment travel links are spray-OFF and measured.
    # --------------------------------------------------------
    travel_test = _build_inter_segment_travel_plan(
        nozzle_points=np.asarray([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 5.0, 0.0],
            [13.0, 9.0, 0.0],
        ], dtype=np.float64),
        segment_ids=np.asarray([0, 0, 1, 1, 2], dtype=np.int64),
        coordinate_unit="mm",
    )
    assert travel_test["transition_count"] == 2
    assert travel_test["transitions"][0]["distance_mm"] == 0.0
    assert abs(travel_test["transitions"][1]["distance_mm"] - 5.0) < 1e-9
    assert travel_test["transitions"][0]["spray_state"] == "OFF"
    assert travel_test["safety_status"] == "NOT_COLLISION_CHECKED"

    # --------------------------------------------------------
    # Test 24B: Greedy stroke ordering reduces a deliberately poor route.
    # --------------------------------------------------------
    ordering_test_trajectory = {
        "surface_points": np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0], [101.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
            [102.0, 0.0, 0.0], [103.0, 0.0, 0.0],
        ], dtype=np.float64),
        "surface_normals": np.tile([0.0, 0.0, 1.0], (8, 1)),
        "nozzle_points": np.asarray([
            [0.0, 0.0, 10.0], [1.0, 0.0, 10.0],
            [100.0, 0.0, 10.0], [101.0, 0.0, 10.0],
            [2.0, 0.0, 10.0], [3.0, 0.0, 10.0],
            [102.0, 0.0, 10.0], [103.0, 0.0, 10.0],
        ], dtype=np.float64),
        "spray_directions": np.tile([0.0, 0.0, -1.0], (8, 1)),
        "nozzle_frames": np.tile(np.eye(3), (8, 1, 1)),
        "face_indices": np.arange(8, dtype=np.int64),
        "patch_ids": np.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
        "segment_ids": np.repeat(np.arange(4, dtype=np.int64), 2),
    }
    ordering_test_metadata = [
        {"segment_id": index, "patch_id": index, "path_type": "test"}
        for index in range(4)
    ]
    optimized_test, optimized_metadata, ordering_test = (
        _optimize_stroke_order_for_travel(
            ordering_test_trajectory, ordering_test_metadata
        )
    )
    assert ordering_test["stroke_count"] == 4
    assert ordering_test["optimized_travel_distance_coordinate_units"] < (
        ordering_test["original_travel_distance_coordinate_units"]
    )
    assert np.array_equal(
        optimized_test["segment_ids"],
        np.repeat(np.arange(4, dtype=np.int64), 2),
    )
    assert len(optimized_metadata) == 4
    assert np.allclose(
        np.sort(optimized_test["nozzle_points"][:, 0]),
        np.sort(ordering_test_trajectory["nozzle_points"][:, 0]),
    )

    inspection_test_mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ], dtype=np.float64),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    inspection_directions = np.asarray([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, math.sin(math.radians(10.0)), -math.cos(math.radians(10.0))],
        [0.0, math.sin(math.radians(10.0)), -math.cos(math.radians(10.0))],
    ], dtype=np.float64)
    inspection_result = _inspect_generated_trajectory(
        mesh=inspection_test_mesh,
        source_face_indices=np.asarray([0, 1], dtype=np.int64),
        trajectory={
            "face_indices": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "nozzle_points": np.asarray([
                [0.0, 0.0, 180.0],
                [10.0, 0.0, 180.0],
                [0.0, 10.0, 180.0],
                [10.0, 10.0, 180.0],
            ], dtype=np.float64),
            "spray_directions": inspection_directions,
            "segment_ids": np.asarray([0, 0, 1, 1], dtype=np.int64),
        },
        coordinate_unit="mm",
        sample_spacing_mm=10.0,
        track_spacing_mm=20.0,
    )
    assert abs(inspection_result["surface_face_representation_percent"] - 100.0) < 1e-9
    assert inspection_result["continuity"]["jump_outlier_count"] == 0
    assert inspection_result["orientation"]["maximum_change_deg"] < 11.0

    # --------------------------------------------------------
    # Test 24D: Full nozzle-frame inspection detects roll changes even when
    # the spray axis itself remains unchanged.
    # --------------------------------------------------------
    base_nozzle_frame = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    roll_angle = math.radians(40.0)
    local_roll = np.asarray([
        [math.cos(roll_angle), -math.sin(roll_angle), 0.0],
        [math.sin(roll_angle), math.cos(roll_angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    roll_inspection = _inspect_generated_trajectory(
        mesh=inspection_test_mesh,
        source_face_indices=np.asarray([0, 1], dtype=np.int64),
        trajectory={
            "face_indices": np.asarray([0, 0], dtype=np.int64),
            "nozzle_points": np.asarray([
                [0.0, 0.0, 180.0],
                [1.0, 0.0, 180.0],
            ], dtype=np.float64),
            "spray_directions": np.asarray([
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ], dtype=np.float64),
            "nozzle_frames": np.asarray([
                base_nozzle_frame,
                base_nozzle_frame @ local_roll,
            ], dtype=np.float64),
            "segment_ids": np.asarray([0, 0], dtype=np.int64),
        },
        coordinate_unit="mm",
        sample_spacing_mm=10.0,
        track_spacing_mm=20.0,
    )
    assert roll_inspection["orientation"]["spray_axis_maximum_change_deg"] < 1e-6
    assert abs(
        roll_inspection["orientation"]["full_nozzle_frame_maximum_change_deg"]
        - 40.0
    ) < 1e-6
    assert roll_inspection["orientation"]["status"] == "WARN"

    # --------------------------------------------------------
    # Test 24E: Orientation diagnostics distinguish a sharp source-mesh crease
    # from a large spray-axis change on otherwise smooth adjacent faces.
    # --------------------------------------------------------
    crease_mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ], dtype=np.float64),
        faces=np.asarray([[0, 1, 2], [1, 0, 3]], dtype=np.int64),
        process=False,
    )
    crease_normals = np.asarray(crease_mesh.face_normals, dtype=np.float64)
    crease_directions = -crease_normals
    crease_frames = _build_parallel_transport_nozzle_frames(
        crease_directions,
        np.asarray([0, 0], dtype=np.int64),
    )
    crease_inspection = _inspect_generated_trajectory(
        mesh=crease_mesh,
        source_face_indices=np.asarray([0, 1], dtype=np.int64),
        trajectory={
            "face_indices": np.asarray([0, 1], dtype=np.int64),
            "patch_ids": np.asarray([3, 3], dtype=np.int64),
            "surface_normals": crease_normals,
            "nozzle_points": np.asarray([
                [0.0, 0.0, 180.0],
                [1.0, 0.0, 180.0],
            ], dtype=np.float64),
            "spray_directions": crease_directions,
            "nozzle_frames": crease_frames,
            "segment_ids": np.asarray([0, 0], dtype=np.int64),
            "segments": [{
                "segment_id": 0,
                "patch_id": 3,
                "path_type": "test_crease",
            }],
        },
        coordinate_unit="mm",
        sample_spacing_mm=10.0,
        track_spacing_mm=20.0,
    )
    crease_detail = crease_inspection["orientation"]["over_30_degree_change_details"][0]
    assert crease_detail["cause_classification"] == "sharp_source_mesh_crease"
    assert crease_detail["shared_source_face_vertices"] == 2
    assert crease_detail["source_face_normal_change_deg"] > 30.0

    # --------------------------------------------------------
    # Test 24F: Surface topology and sharp creases insert spray-OFF boundaries
    # without modifying or dropping any trajectory samples.
    # --------------------------------------------------------
    split_test_mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [20.0, 0.0, 0.0],
            [20.0, 10.0, 0.0],
        ], dtype=np.float64),
        faces=np.asarray([
            [0, 1, 2],
            [1, 0, 3],
            [3, 4, 5],
        ], dtype=np.int64),
        process=False,
    )
    split_test_points = np.asarray([
        [0.0, 0.0, 180.0],
        [1.0, 0.0, 180.0],
        [2.0, 0.0, 180.0],
    ], dtype=np.float64)
    split_test_normals = np.asarray(split_test_mesh.face_normals, dtype=np.float64)
    split_test_trajectory = {
        "surface_points": np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64),
        "surface_normals": split_test_normals.copy(),
        "nozzle_points": split_test_points.copy(),
        "spray_directions": -split_test_normals.copy(),
        "nozzle_frames": _build_parallel_transport_nozzle_frames(
            -split_test_normals,
            np.asarray([0, 0, 0], dtype=np.int64),
        ),
        "face_indices": np.asarray([0, 1, 2], dtype=np.int64),
        "patch_ids": np.asarray([9, 9, 9], dtype=np.int64),
        "segment_ids": np.asarray([0, 0, 0], dtype=np.int64),
    }
    split_test_result, split_test_metadata, split_test_stats = (
        _split_strokes_at_surface_discontinuities(
            split_test_mesh,
            split_test_trajectory,
            [{"segment_id": 0, "patch_id": 9, "path_type": "test"}],
            maximum_normal_change_deg=30.0,
        )
    )
    assert np.array_equal(
        split_test_result["segment_ids"],
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    assert split_test_stats["split_transition_counts"]["sharp_source_mesh_crease"] == 1
    assert split_test_stats["split_transition_counts"]["non_adjacent_source_faces"] == 1
    assert split_test_stats["added_stroke_count"] == 2
    assert np.array_equal(
        split_test_result["surface_points"],
        split_test_trajectory["surface_points"],
    )
    assert len(split_test_metadata) == 3

    # Test 24B: exact STEP adjacency overrides display-triangle vertex sharing.
    exact_topology_trajectory = dict(split_test_trajectory)
    exact_topology_result, exact_topology_metadata, exact_topology_stats = (
        _split_strokes_at_surface_discontinuities(
            split_test_mesh,
            exact_topology_trajectory,
            [{"segment_id": 0, "patch_id": 9, "path_type": "exact_topology_test"}],
            source_face_ids=np.asarray([101, 102, 103], dtype=np.int64),
            source_face_adjacency={"101": [102], "102": [101], "103": []},
            maximum_normal_change_deg=180.0,
        )
    )
    assert exact_topology_stats["adjacency_authority"] == "EXACT_STEP_BREP_SHARED_EDGES"
    assert exact_topology_stats["split_transition_counts"]["non_adjacent_source_faces"] == 1
    assert exact_topology_stats["added_stroke_count"] == 1
    assert np.array_equal(
        exact_topology_result["surface_points"],
        exact_topology_trajectory["surface_points"],
    )
    assert len(exact_topology_metadata) == 2

    # --------------------------------------------------------
    # Test 24C: Continuity diagnostics identify exact source samples and faces.
    # --------------------------------------------------------
    continuity_outlier_test = _inspect_generated_trajectory(
        mesh=inspection_test_mesh,
        source_face_indices=np.asarray([0, 1], dtype=np.int64),
        trajectory={
            "face_indices": np.asarray([0, 0, 1], dtype=np.int64),
            "patch_ids": np.asarray([7, 7, 7], dtype=np.int64),
            "nozzle_points": np.asarray([
                [0.0, 0.0, 180.0],
                [10.0, 0.0, 180.0],
                [100.0, 0.0, 180.0],
            ], dtype=np.float64),
            "spray_directions": np.tile([0.0, 0.0, -1.0], (3, 1)),
            "segment_ids": np.asarray([4, 4, 4], dtype=np.int64),
            "segments": [{
                "segment_id": 4,
                "patch_id": 7,
                "path_type": "test_raster",
            }],
        },
        coordinate_unit="mm",
        sample_spacing_mm=10.0,
        track_spacing_mm=20.0,
    )
    assert continuity_outlier_test["continuity"]["jump_outlier_count"] == 1
    outlier_detail = continuity_outlier_test["continuity"]["outlier_details"][0]
    assert outlier_detail["sample_index_from"] == 1
    assert outlier_detail["sample_index_to"] == 2
    assert outlier_detail["segment_id"] == 4
    assert outlier_detail["patch_id_from"] == 7
    assert outlier_detail["face_index_from"] == 0
    assert outlier_detail["face_index_to"] == 1
    assert outlier_detail["shared_source_face_vertices"] == 2
    assert outlier_detail["cause_classification"] == (
        "large_step_between_vertex_adjacent_faces"
    )
    assert inspection_result["spray_control"]["spray_off_transition_count"] == 1
    assert inspection_result["spray_control"]["spray_on_sample_count"] == 3

    # --------------------------------------------------------
    # Test 24G: Exact STEP surface type overrides planar display facets.
    # A flat tessellation of a cylinder must not be mistaken for a planar CAD face.
    # --------------------------------------------------------
    surface_type_test_mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0, 20.0, 0.0],
            [0.0, 20.0, 0.0],
        ], dtype=np.float64),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    surface_type_test_mesh._step_brep_context = {
        "shape": object(),
        "faces": (object(),),
        "face_records": ({"surface_type": "CYLINDER"},),
        "triangle_face_ids": np.asarray([1, 1], dtype=np.int64),
        "triangle_uvs": np.zeros((2, 3, 2), dtype=np.float64),
        "triangle_uv_valid": np.ones(2, dtype=bool),
        "face_adjacency": {"1": []},
        "schema_identifier": "AUTOMOTIVE_DESIGN",
    }
    force_curved, exact_types = _step_patch_requires_curved_raster(
        surface_type_test_mesh,
        np.asarray([0, 1], dtype=np.int64),
    )
    assert force_curved is True
    assert exact_types == ("CYLINDER",)

    # Exercise the complete dispatcher. STEP evaluation is isolated because
    # this synthetic test validates raster selection, not Open CASCADE geometry.
    original_step_refiner = globals()["_refine_trajectory_with_step_brep"]
    try:
        def _test_skip_step_refinement(mesh, trajectory, standoff_coordinate_units, coordinate_unit):
            return dict(trajectory), {"status": "NOT_APPLICABLE"}

        globals()["_refine_trajectory_with_step_brep"] = _test_skip_step_refinement
        test_face_normals = np.asarray(surface_type_test_mesh.face_normals, dtype=np.float64)
        test_surface_points = np.asarray(surface_type_test_mesh.triangles_center, dtype=np.float64)
        test_raster_result = _order_trajectory_with_planar_raster(
            mesh=surface_type_test_mesh,
            face_normals=test_face_normals,
            surface_points=test_surface_points,
            surface_normals=test_face_normals.copy(),
            nozzle_points=test_surface_points + 100.0 * test_face_normals,
            spray_directions=-test_face_normals,
            nozzle_frames=_build_nozzle_frames(-test_face_normals),
            face_indices=np.asarray([0, 1], dtype=np.int64),
            patch_ids=np.asarray([7, 7], dtype=np.int64),
            standoff_coordinate_units=100.0,
            sampling_spacing_coordinate_units=5.0,
            track_spacing_coordinate_units=5.0,
            coordinate_unit="mm",
        )
    finally:
        globals()["_refine_trajectory_with_step_brep"] = original_step_refiner
    assert test_raster_result["rasterized_curved_patch_ids"] == [7]
    assert test_raster_result["rasterized_planar_patch_ids"] == []
    assert test_raster_result["path_statistics"]["curved_sample_count"] > 0

    # An exact STEP plane remains eligible for the planar raster.
    surface_type_test_mesh._step_brep_context["face_records"] = (
        {"surface_type": "PLANE"},
    )
    force_curved_plane, plane_types = _step_patch_requires_curved_raster(
        surface_type_test_mesh,
        np.asarray([0, 1], dtype=np.int64),
    )
    assert force_curved_plane is False
    assert plane_types == ("PLANE",)

    # --------------------------------------------------------
    # Test 24H: A single non-degenerate source triangle is a valid
    # curved-raster domain; it must not be rejected for having fewer
    # than two projected triangles.
    # --------------------------------------------------------
    single_triangle_mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
        ], dtype=np.float64),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    single_triangle_normals = np.asarray(
        single_triangle_mesh.face_normals, dtype=np.float64
    )
    single_triangle_raster = _generate_surface_following_raster_for_patch(
        mesh=single_triangle_mesh,
        face_indices=np.asarray([0], dtype=np.int64),
        normals=single_triangle_normals,
        standoff_coordinate_units=100.0,
        sample_spacing_coordinate_units=10.0,
        track_spacing_coordinate_units=20.0,
        diagnostics={},
    )
    assert single_triangle_raster is not None
    assert len(single_triangle_raster["surface_points"]) >= 2
    assert np.all(single_triangle_raster["face_indices"] == 0)
    assert len(single_triangle_raster["surface_points"]) == len(
        single_triangle_raster["segment_ids"]
    )
    source_triangle = single_triangle_mesh.vertices[
        single_triangle_mesh.faces[0]
    ]
    for raster_point in single_triangle_raster["surface_points"]:
        weights = _barycentric_coordinates_on_triangle(
            raster_point, source_triangle, tolerance=1e-7
        )
        assert weights is not None
        assert np.allclose(weights @ source_triangle, raster_point, atol=1e-8)
    assert np.allclose(
        single_triangle_raster["surface_normals"],
        np.tile(single_triangle_normals[0], (len(single_triangle_raster["surface_points"]), 1)),
    )

    return {
        "status": "PASS",
        "tests": {
            "sample_count": "PASS",
            "physical_standoff": "PASS",
            "unit_normals": "PASS",
            "unit_spray_directions": "PASS",
            "spray_direction": "PASS",
            "nozzle_position_formula": "PASS",
            "open_mesh_orientation": "PASS",
            "stage1_advanced_geometry_integration": "PASS",
            "inspection_faces_excluded": "PASS",
            "legacy_compatibility": "PASS",
            "json_serialization": "PASS",
            "unknown_unit_rejected": "PASS",
            "physical_standoff_all_units": "PASS",
            "physical_sampling_spacing_all_units": "PASS",
            "source_mesh_unchanged": "PASS",
            "npy_16_column_compatibility": "PASS",
            "sprayer_spray_envelope_and_overlap": "PASS",
            "product_pose_world_transform": "PASS",
            "nozzle_frame_orthonormality": "PASS",
            "invalid_product_transform_rejected": "PASS",
            "planar_raster_spacing_and_clipping": "PASS",
            "planar_raster_standoff_preserved": "PASS",
            "trajectory_coverage_continuity_orientation_and_spray_boundaries": "PASS",
            "inter_segment_spray_off_connector_geometry": "PASS",
            "greedy_stroke_travel_ordering": "PASS",
            "exact_continuity_outlier_diagnostics": "PASS",
            "curved_raster_projection_jump_splitting": "PASS",
            "curved_normal_smoothing_with_8deg_deviation_limit": "PASS",
            "parallel_transport_nozzle_frame_continuity": "PASS",
            "full_nozzle_frame_roll_inspection": "PASS",
            "surface_discontinuity_spray_off_splitting": "PASS",
            "step_brep_refinement_failure_diagnostics": "PASS",
            "step_surface_type_aware_raster_selection": "PASS",
            "single_triangle_surface_raster": "PASS",
        },
        "sample_count": int(sample_count),
        "unit_standoff_tests": unit_test_results,
        "unit_sampling_tests": sampling_test_results,
        "product_pose_world_frame": "PASS",
        "nozzle_frame_convention": "PROVISIONAL",
        "invalid_product_transform_rejected": invalid_transform_rejected,
        "advanced_geometry": advanced["summary"],
        "engineering_assessment": {
            "geometric_calculation": "PASS",
            "normal_orientation": "REVIEW",
            "robot_readiness": "NOT_CHECKED",
        },
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("OPENARMX - STAGE 3 SPRAY TRAJECTORY NUMERICAL TESTS")
    print("=" * 70)

    test_result = run_stage3_numerical_tests()

    print(json.dumps(test_result, indent=4))

    print("=" * 70)
    print("ALL STAGE 3 TESTS PASSED")
    print("=" * 70)
