
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from core.result_manager import ResultManager


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
DEFAULT_SAMPLING_SPACING_MM = 20.0
DEFAULT_SPRAY_ANGLE_DEG = 0.0

MIN_STANDOFF_MM = 50.0
MAX_STANDOFF_MM = 500.0

MIN_SAMPLING_SPACING_MM = 1.0
MAX_SAMPLING_SPACING_MM = 100.0

MAX_TOTAL_SAMPLES = 250_000

NORMAL_EPSILON = 1e-12
SUPPORTED_SPRAY_ANGLE_DEG = 0.0

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


def _validate_parameters(
    standoff_mm: float,
    sampling_spacing_mm: float,
    spray_angle_deg: float,
) -> tuple[float, float, float]:

    standoff = _safe_float(standoff_mm)
    spacing = _safe_float(sampling_spacing_mm)
    angle = _safe_float(spray_angle_deg)

    if standoff is None:
        raise ValueError("Standoff distance must be finite.")

    if spacing is None:
        raise ValueError("Sampling spacing must be finite.")

    if angle is None:
        raise ValueError("Spray angle must be finite.")

    if not MIN_STANDOFF_MM <= standoff <= MAX_STANDOFF_MM:
        raise ValueError(
            f"Standoff must be between {MIN_STANDOFF_MM:.1f} "
            f"and {MAX_STANDOFF_MM:.1f} mm."
        )

    if not MIN_SAMPLING_SPACING_MM <= spacing <= MAX_SAMPLING_SPACING_MM:
        raise ValueError(
            f"Sampling spacing must be between "
            f"{MIN_SAMPLING_SPACING_MM:.1f} and "
            f"{MAX_SAMPLING_SPACING_MM:.1f} mm."
        )

    if abs(angle - SUPPORTED_SPRAY_ANGLE_DEG) > 1e-9:
        raise ValueError(
            "Only a 0-degree spray angle is currently supported. "
            "Nonzero angles require a defined tangent-plane azimuth."
        )

    return standoff, spacing, angle


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

    if triangle.shape != (3, 3):
        raise ValueError("Triangle must have shape (3, 3).")

    if (
        not math.isfinite(spacing_coordinate_units)
        or spacing_coordinate_units <= 0.0
    ):
        raise ValueError("Sampling spacing must be positive and finite.")

    edge_lengths = [
        np.linalg.norm(triangle[1] - triangle[0]),
        np.linalg.norm(triangle[2] - triangle[1]),
        np.linalg.norm(triangle[0] - triangle[2]),
    ]

    longest_edge = max(edge_lengths)

    if not math.isfinite(longest_edge):
        return np.empty((0, 3), dtype=np.float64)

    if longest_edge <= NORMAL_EPSILON:
        return np.empty((0, 3), dtype=np.float64)

    n = max(
        1,
        int(math.ceil(longest_edge / spacing_coordinate_units)),
    )

    samples = np.empty((n * n, 3), dtype=np.float64)

    a, b, c = triangle

    def barycentric_point(i: int, j: int) -> np.ndarray:
        u = i / n
        v = j / n

        return (
            (1.0 - u - v) * a
            + u * b
            + v * c
        )

    sample_index = 0

    for i in range(n):
        for j in range(n - i):

            p0 = barycentric_point(i, j)
            p1 = barycentric_point(i + 1, j)
            p2 = barycentric_point(i, j + 1)

            samples[sample_index] = (p0 + p1 + p2) / 3.0
            sample_index += 1

            if i + j < n - 1:
                p3 = barycentric_point(i + 1, j + 1)

                samples[sample_index] = (p1 + p3 + p2) / 3.0
                sample_index += 1

    return samples[:sample_index]


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

    all_points = []
    all_normals = []
    all_face_indices = []
    all_patch_ids = []

    sample_count = 0
    skipped_invalid_normals = 0
    skipped_invalid_geometry = 0

    for face_index in candidate_faces:

        triangle = vertices[faces[face_index]]
        normal = normals[face_index]

        if not np.all(np.isfinite(normal)):
            skipped_invalid_normals += 1
            continue

        if not np.all(np.isfinite(triangle)):
            skipped_invalid_geometry += 1
            continue

        triangle_area = 0.5 * np.linalg.norm(
            np.cross(
                triangle[1] - triangle[0],
                triangle[2] - triangle[0],
            )
        )

        if (
            not math.isfinite(triangle_area)
            or triangle_area <= NORMAL_EPSILON
        ):
            skipped_invalid_geometry += 1
            continue

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

        all_points.append(points)

        all_normals.append(
            np.repeat(normal[None, :], len(points), axis=0)
        )

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
                "Physical sampling interval converted to STL coordinates. "
                "It does not prove paint coverage."
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
            "meaning": "Number of separate patch paths.",
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
            "Define spray fan width, overlap and process speed.",
            "Validate coating coverage and thickness experimentally.",
            "Optimize travel movements between separate patch segments.",
        ],
    }


# ------------------------------------------------------------
# Main trajectory calculation
# ------------------------------------------------------------

def generate_spray_trajectory(
    mesh: trimesh.Trimesh,
    face_status_labels: np.ndarray,
    patch_labels: np.ndarray,
    standoff_mm: float = DEFAULT_STANDOFF_MM,
    sampling_spacing_mm: float = DEFAULT_SAMPLING_SPACING_MM,
    spray_angle_deg: float = DEFAULT_SPRAY_ANGLE_DEG,
    stage1_report: dict | None = None,
    coordinate_unit: str | None = None,
    sprayer_profile: dict | None = None,
    product_transform: np.ndarray | list | None = None,
) -> dict:

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh object.")

    standoff_mm, sampling_spacing_mm, spray_angle_deg = (
        _validate_parameters(
            standoff_mm,
            sampling_spacing_mm,
            spray_angle_deg,
        )
    )

    product_transform_matrix = _validate_product_transform(product_transform)

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

    spacing_coordinate_units = _mm_to_coordinate_units(
        sampling_spacing_mm,
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

    # 7. Order trajectory points within individual patches.
    ordered = _order_trajectory_by_patch(
        surface_points=surface_points,
        surface_normals=surface_normals,
        nozzle_points=nozzle_points,
        spray_directions=spray_directions,
        nozzle_frames=nozzle_frames,
        face_indices=face_indices,
        patch_ids=patch_ids,
    )

    # 7a. Transform CAD-local samples and nozzle frames into World XYZ.
    # The source mesh and CAD-local arrays remain unchanged.
    world_surface_points = _transform_points(ordered['surface_points'], product_transform_matrix)
    world_surface_normals = _transform_vectors(ordered['surface_normals'], product_transform_matrix)
    world_nozzle_points = _transform_points(ordered['nozzle_points'], product_transform_matrix)
    world_spray_directions = _transform_vectors(ordered['spray_directions'], product_transform_matrix)
    world_nozzle_frames = np.einsum('ij,njk->nik', product_transform_matrix[:3, :3], ordered['nozzle_frames'])

    # 8. Calculate path metrics in both coordinate units and mm.
    path_metrics = _calculate_path_metrics(
        nozzle_points=ordered["nozzle_points"],
        segment_ids=ordered["segment_ids"],
        coordinate_unit=source_unit,
    )

    # 9. Associate Stage 1 geometry features.
    advanced_analysis = _build_advanced_geometry_analysis(
        feature_data=feature_data,
        patch_ids=ordered["patch_ids"],
        coordinate_unit=source_unit,
    )

    # 10. Numerical validation.
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

    # 11. Engineering analysis.
    analysis = _build_analysis(
        sampling_result=sampling_result,
        normal_info=normal_info,
        orientation_info=orientation_info,
        path_metrics=path_metrics,
        standoff_mm=standoff_mm,
        spacing_mm=sampling_spacing_mm,
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
    cone_angle = _safe_float(
        profile_data.get("spray_cone_angle_deg", 0.0)
    ) or 0.0
    cone_angle = max(0.0, min(150.0, cone_angle))
    overlap_percent = _safe_float(
        profile_data.get("overlap_percent", 0.0)
    ) or 0.0
    overlap_percent = max(0.0, min(90.0, overlap_percent))
    spray_width_at_standoff_mm = (
        2.0 * standoff_mm * math.tan(math.radians(cone_angle * 0.5))
        if cone_angle > 0.0 else 0.0
    )
    nominal_track_spacing_mm = (
        spray_width_at_standoff_mm * (1.0 - overlap_percent / 100.0)
    )

    return {
        "parameters": {
            "standoff_mm": standoff_mm,
            "sampling_spacing_mm": sampling_spacing_mm,
            "spray_angle_deg": spray_angle_deg,
            "sprayer_profile": profile_data,
            "spray_width_at_standoff_mm": spray_width_at_standoff_mm,
            "nominal_track_spacing_from_overlap_mm": nominal_track_spacing_mm,
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
        "face_indices": ordered["face_indices"],
        "patch_ids": ordered["patch_ids"],
        "segment_ids": ordered["segment_ids"],
        "segments": ordered["segments"],
        "sampling": {
            "coordinate_unit": source_unit,
            "sampling_spacing_mm": sampling_spacing_mm,
            "sampling_spacing_coordinate_units": spacing_coordinate_units,
            "candidate_face_count": sampling_result["candidate_face_count"],
            "sample_count": sampling_result["sample_count"],
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
        "segments": trajectory["segments"],
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
            "column_compatibility": (
                "The 16-column order is unchanged from the previous "
                "Stage 3 format. Coordinate columns now explicitly "
                "identify their values as STL coordinate units."
            ),
        },
        "engineering_limitations": [
            "Robot reachability is not checked.",
            "Robot collision is not checked.",
            "Fixture and hook clearance are not checked.",
            "Spray coverage is not verified.",
            "Coating thickness is not calculated.",
            "No actual spray-gun CAD or TCP offset is applied.",
            "No robot-base coordinate transformation is applied.",
            "Fan width and overlap are recorded as planning inputs; raster spray lanes are not yet generated.",
            "Travel movements between separate patches are not generated.",
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
    sampling_spacing_mm: float = DEFAULT_SAMPLING_SPACING_MM,
    spray_angle_deg: float = DEFAULT_SPRAY_ANGLE_DEG,
    stage1_report: dict | None = None,
    coordinate_unit: str | None = None,
    sprayer_profile: dict | None = None,
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
        sampling_spacing_mm=sampling_spacing_mm,
        spray_angle_deg=spray_angle_deg,
        stage1_report=stage1_report,
        coordinate_unit=coordinate_unit,
        sprayer_profile=sprayer_profile,
    )

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

    test_mesh = _make_test_mesh(1.0)

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
        sampling_spacing_mm=50.0,
        coordinate_unit="mm",
    )

    surface_points = result["surface_points"]
    normals = result["surface_normals"]
    nozzle_points = result["nozzle_points"]
    directions = result["spray_directions"]

    assert len(surface_points) == 18, (
        f"Expected 18 samples, got {len(surface_points)}"
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
        sampling_spacing_mm=50.0,
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
            sampling_spacing_mm=50.0,
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
            sampling_spacing_mm=50.0,
            coordinate_unit=unit,
        )

        expected_spacing = 50.0 / mm_per_unit

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
            "spacing_mm": 50.0,
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
        sampling_spacing_mm=50.0,
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
        },
        "sample_count": int(sample_count),
        "unit_standoff_tests": unit_test_results,
        "unit_sampling_tests": sampling_test_results,
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
