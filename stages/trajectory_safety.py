"""Conservative spray-gun envelope clearance checks in PRODUCT_CAD coordinates.

This module is deliberately independent from Stage 3 persistence, coverage,
trajectory correction, viewer transforms, and robot execution. It checks a
cylindrical approximation of the gun body. It does not certify robot or tool
safety and reports uncertainty when nozzle outer diameter / closed-mesh inside
classification cannot be established.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


MM_PER_UNIT = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
}


@dataclass(frozen=True)
class SprayGunEnvelopeConfig:
    """Physical tool dimensions; all dimensions are specified in millimeters."""

    gun_body_length_mm: float
    gun_body_diameter_mm: float
    nozzle_length_mm: float
    minimum_surface_clearance_mm: float
    nozzle_outer_diameter_mm: float | None = None
    tcp_to_nozzle_tip_mm: float | None = None
    dimensions_are_estimate: bool = True
    sample_step_mm: float | None = None

    def validate(self) -> None:
        required = (
            self.gun_body_length_mm,
            self.gun_body_diameter_mm,
            self.nozzle_length_mm,
            self.minimum_surface_clearance_mm,
        )
        if not all(np.isfinite(value) for value in required):
            raise ValueError("Spray-gun dimensions and clearance must be finite.")
        if self.gun_body_length_mm <= 0.0:
            raise ValueError("Gun body length must be positive.")
        if self.gun_body_diameter_mm <= 0.0:
            raise ValueError("Gun body diameter must be positive.")
        if self.nozzle_length_mm < 0.0:
            raise ValueError("Nozzle length cannot be negative.")
        if self.minimum_surface_clearance_mm < 0.0:
            raise ValueError("Minimum surface clearance cannot be negative.")
        if self.nozzle_outer_diameter_mm is not None:
            if not np.isfinite(self.nozzle_outer_diameter_mm) or self.nozzle_outer_diameter_mm <= 0.0:
                raise ValueError("Nozzle outer diameter must be positive when supplied.")
        if self.tcp_to_nozzle_tip_mm is not None:
            if not np.isfinite(self.tcp_to_nozzle_tip_mm) or self.tcp_to_nozzle_tip_mm < 0.0:
                raise ValueError("TCP-to-nozzle-tip offset cannot be negative.")
        if self.sample_step_mm is not None:
            if not np.isfinite(self.sample_step_mm) or self.sample_step_mm <= 0.0:
                raise ValueError("Tool-envelope sample step must be positive.")


def _mm_per_coordinate_unit(coordinate_unit: str) -> float:
    key = str(coordinate_unit).strip().lower()
    if key not in MM_PER_UNIT:
        raise ValueError(
            f"Unsupported or unknown CAD coordinate unit: {coordinate_unit!r}"
        )
    return MM_PER_UNIT[key]


def _validate_frames(frames: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float64)
    if values.shape != (count, 3, 3) or not np.all(np.isfinite(values)):
        raise ValueError("nozzle_frames must be a finite (N, 3, 3) array.")
    identity = np.broadcast_to(np.eye(3), values.shape)
    if not np.allclose(np.swapaxes(values, 1, 2) @ values, identity, atol=1e-5):
        raise ValueError("Every nozzle frame must be orthonormal.")
    determinants = np.linalg.det(values)
    if not np.allclose(determinants, 1.0, atol=1e-5):
        raise ValueError("Every nozzle frame must be right-handed.")
    return values


def build_gun_body_centerline(
    nozzle_tip: np.ndarray,
    nozzle_frame: np.ndarray,
    config: SprayGunEnvelopeConfig,
    coordinate_unit: str,
) -> np.ndarray:
    """Sample the gun-body axis behind the nozzle tip.

    Frame column +Z points from the nozzle toward the product. Therefore the
    gun body extends along -Z, away from the product. Nozzle length is the
    axial gap between the nozzle tip and the start of the cylindrical body.
    """
    config.validate()
    mm_per_unit = _mm_per_coordinate_unit(coordinate_unit)
    tip = np.asarray(nozzle_tip, dtype=np.float64)
    frame = np.asarray(nozzle_frame, dtype=np.float64)
    if tip.shape != (3,) or not np.all(np.isfinite(tip)):
        raise ValueError("nozzle_tip must be a finite 3-vector.")
    _validate_frames(frame[None, :, :], 1)

    step_mm = config.sample_step_mm or max(
        1.0, min(config.gun_body_diameter_mm * 0.25, 10.0)
    )
    interval_count = max(1, int(np.ceil(config.gun_body_length_mm / step_mm)))
    axial_distances_mm = (
        config.nozzle_length_mm
        + np.linspace(0.0, config.gun_body_length_mm, interval_count + 1)
    )
    body_axis = -frame[:, 2]
    return tip[None, :] + (
        axial_distances_mm[:, None] / mm_per_unit
    ) * body_axis[None, :]


def evaluate_spray_gun_clearance(
    mesh: trimesh.Trimesh,
    nozzle_points: np.ndarray,
    nozzle_frames: np.ndarray,
    config: SprayGunEnvelopeConfig,
    coordinate_unit: str,
) -> dict[str, Any]:
    """Evaluate sampled gun-body clearance against the complete product mesh.

    The body is approximated as a capsule with radius gun_body_diameter / 2.
    Each sampled centerline point must remain at least
    minimum_surface_clearance + body_radius from the mesh. This is a
    conservative keep-out check, not a full CAD solid collision calculation.

    A missing nozzle outer diameter is explicitly reported as UNCERTAIN because
    the nozzle shaft's radial envelope cannot be certified from orifice size.
    """
    config.validate()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("A non-empty product triangle mesh is required.")

    tips = np.asarray(nozzle_points, dtype=np.float64)
    if tips.ndim != 2 or tips.shape[1] != 3 or len(tips) == 0:
        raise ValueError("nozzle_points must be a non-empty (N, 3) array.")
    if not np.all(np.isfinite(tips)):
        raise ValueError("nozzle_points contain non-finite values.")
    frames = _validate_frames(nozzle_frames, len(tips))
    mm_per_unit = _mm_per_coordinate_unit(coordinate_unit)

    radius_units = 0.5 * config.gun_body_diameter_mm / mm_per_unit
    required_center_distance_units = (
        config.minimum_surface_clearance_mm / mm_per_unit + radius_units
    )
    sample_step_mm = config.sample_step_mm or max(
        1.0, min(config.gun_body_diameter_mm * 0.25, 10.0)
    )

    all_centerline_points = []
    sample_owner = []
    for index, (tip, frame) in enumerate(zip(tips, frames)):
        centerline = build_gun_body_centerline(
            tip, frame, config, coordinate_unit
        )
        all_centerline_points.append(centerline)
        sample_owner.extend([index] * len(centerline))
    query_points = np.vstack(all_centerline_points)

    # Build a centroid KD-tree once. For each query point, the nearest
    # centroid gives an initial upper bound. Any triangle that could be closer
    # must have its centroid within (upper_bound + maximum triangle radius).
    # Querying that conservative radius and evaluating exact point-to-triangle
    # distances avoids the O(query_points * all_faces) naive proximity routine
    # and does not require the optional rtree package.
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    triangle_centers = np.mean(triangles, axis=1)
    triangle_radii = np.max(
        np.linalg.norm(triangles - triangle_centers[:, None, :], axis=2),
        axis=1,
    )
    max_triangle_radius = float(np.max(triangle_radii))
    center_tree = cKDTree(triangle_centers)
    _, nearest_triangle_indices = center_tree.query(query_points, k=1)
    nearest_triangle_indices = np.asarray(
        nearest_triangle_indices, dtype=np.int64
    )
    initial_triangles = triangles[nearest_triangle_indices]
    initial_closest = trimesh.triangles.closest_point(
        initial_triangles, query_points
    )
    initial_upper_bounds = np.linalg.norm(
        initial_closest - query_points, axis=1
    )

    distances = np.empty(len(query_points), dtype=np.float64)
    for query_index, point in enumerate(query_points):
        search_radius = (
            float(initial_upper_bounds[query_index])
            + max_triangle_radius
            + 1e-12
        )
        candidate_indices = center_tree.query_ball_point(
            point, search_radius
        )
        if not candidate_indices:
            candidate_indices = [int(nearest_triangle_indices[query_index])]
        candidate_triangles = triangles[
            np.asarray(candidate_indices, dtype=np.int64)
        ]
        candidate_points = np.repeat(
            point[None, :], len(candidate_triangles), axis=0
        )
        closest_points = trimesh.triangles.closest_point(
            candidate_triangles, candidate_points
        )
        distances[query_index] = float(np.min(
            np.linalg.norm(closest_points - candidate_points, axis=1)
        ))

    owner = np.asarray(sample_owner, dtype=np.int64)
    minimum_by_sample = np.full(len(tips), np.inf, dtype=np.float64)
    np.minimum.at(minimum_by_sample, owner, distances)
    clearance_by_sample_mm = (
        minimum_by_sample - radius_units
    ) * mm_per_unit

    collision_mask = distances < required_center_distance_units
    collision_samples = np.unique(owner[collision_mask]).astype(np.int64)

    inside_test = "NOT_AVAILABLE"
    inside_mask = np.zeros(len(query_points), dtype=bool)
    if bool(getattr(mesh, "is_watertight", False)):
        try:
            inside_mask = np.asarray(mesh.contains(query_points), dtype=bool)
            inside_test = "PASS"
        except Exception:
            inside_test = "UNAVAILABLE"
    else:
        inside_test = "MESH_NOT_WATERTIGHT"

    inside_samples = np.unique(owner[inside_mask]).astype(np.int64)
    if len(inside_samples):
        collision_samples = np.unique(
            np.concatenate((collision_samples, inside_samples))
        ).astype(np.int64)

    if len(collision_samples):
        status = "UNSAFE"
    elif (
        inside_test != "PASS"
        or config.nozzle_outer_diameter_mm is None
        or config.dimensions_are_estimate
    ):
        status = "UNCERTAIN"
    else:
        status = "CLEAR"

    sample_status = np.full(len(tips), "GREEN", dtype=object)
    if status == "UNCERTAIN":
        sample_status[:] = "YELLOW"
    elif status == "UNSAFE":
        sample_status[:] = "YELLOW"
    if len(collision_samples):
        sample_status[collision_samples] = "RED"

    return {
        "status": status,
        "sample_status": sample_status.tolist(),
        "method": "SAMPLED_CYLINDRICAL_GUN_BODY_ENVELOPE",
        "coordinate_frame": "PRODUCT_CAD",
        "coordinate_unit": str(coordinate_unit),
        "sample_count": int(len(tips)),
        "body_centerline_sample_count": int(len(query_points)),
        "body_radius_mm": float(config.gun_body_diameter_mm * 0.5),
        "minimum_surface_clearance_mm": float(
            config.minimum_surface_clearance_mm
        ),
        "minimum_measured_body_clearance_mm": float(
            np.min(clearance_by_sample_mm)
        ),
        "unsafe_sample_indices": collision_samples.tolist(),
        "clearance_by_sample_mm": clearance_by_sample_mm,
        "inside_mesh_test": inside_test,
        "nozzle_outer_diameter_known": (
            config.nozzle_outer_diameter_mm is not None
        ),
        "dimensions_are_estimate": bool(config.dimensions_are_estimate),
        "limitations": [
            "Gun body is represented by a sampled cylindrical/capsule envelope.",
            "Nozzle shaft radial clearance is unknown unless nozzle_outer_diameter_mm is supplied.",
            "Robot links, wrist, fixtures, hoses, and external obstacles are not checked.",
            "This geometric approximation is not a certified robot safety assessment.",
        ],
        "sampling_step_mm": float(sample_step_mm),
    }
