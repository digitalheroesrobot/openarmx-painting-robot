"""Human-like raster planning primitives for Stage 3.

This module deliberately contains no UI, robot, coverage-correction, or persistence
code. It orders already-sampled PRODUCT_CAD surface points into broad serpentine
sweeps and constructs continuous nozzle frames. Stage 3 remains responsible for
sampling, physical-unit conversion, standoff offsets, and the legacy output schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SweepPlanningConfig:
    """Geometric sweep settings, expressed in the mesh coordinate unit."""

    lane_spacing: float
    sample_spacing: float
    max_gap_factor: float = 2.75
    tangent_blend: float = 0.35

    def validate(self) -> None:
        values = (self.lane_spacing, self.sample_spacing, self.max_gap_factor,
                  self.tangent_blend)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Sweep planning parameters must be finite.")
        if self.lane_spacing <= 0.0 or self.sample_spacing <= 0.0:
            raise ValueError("Lane and sample spacing must be positive.")
        if self.max_gap_factor < 1.0:
            raise ValueError("max_gap_factor must be at least 1.")
        if not 0.0 <= self.tangent_blend <= 1.0:
            raise ValueError("tangent_blend must be in [0, 1].")


def _unit_rows(vectors: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= epsilon):
        raise ValueError("Cannot normalize a zero-length vector.")
    return vectors / lengths[:, None]


def _principal_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid and stable major/minor PCA axes for the point cloud."""
    center = np.mean(points, axis=0)
    centered = points - center
    if len(points) < 2 or np.linalg.norm(centered) <= 1e-12:
        raise ValueError("At least two spatially distinct surface samples are required.")
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    major = vh[0]
    if len(singular_values) > 1 and singular_values[1] > 1e-10:
        minor = vh[1]
    else:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(reference, major))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        minor = np.cross(major, reference)
        minor /= np.linalg.norm(minor)
    # Fix the otherwise arbitrary PCA signs for repeatable output.
    dominant = int(np.argmax(np.abs(major)))
    if major[dominant] < 0.0:
        major = -major
    normal = np.cross(major, minor)
    normal /= np.linalg.norm(normal)
    minor = np.cross(normal, major)
    minor /= np.linalg.norm(minor)
    return center, major, minor


def _split_lane(indices: np.ndarray, points: np.ndarray, max_gap: float) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    ordered = indices[np.argsort(points[indices, 0], kind="stable")]
    if len(ordered) == 1:
        return [ordered]
    gaps = np.linalg.norm(np.diff(points[ordered], axis=0), axis=1)
    cuts = np.flatnonzero(gaps > max_gap) + 1
    return [part for part in np.split(ordered, cuts) if len(part)]


def _path_tangents(points: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    tangents = np.zeros_like(points, dtype=float)
    for segment_id in np.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == segment_id)
        if len(idx) == 1:
            tangents[idx[0]] = np.array([1.0, 0.0, 0.0])
            continue
        local = points[idx]
        tangents[idx[0]] = local[1] - local[0]
        tangents[idx[-1]] = local[-1] - local[-2]
        if len(idx) > 2:
            tangents[idx[1:-1]] = local[2:] - local[:-2]
    return _unit_rows(tangents)


def build_nozzle_frames(
    surface_normals: np.ndarray,
    path_tangents: np.ndarray,
    *,
    tangent_blend: float = 0.35,
) -> np.ndarray:
    """Build right-handed frames; local Z points from nozzle toward the surface.

    The path tangent defines the preferred roll axis. Its sign is made continuous
    along the path so serpentine reversals do not create artificial 180-degree
    frame flips. Surface normals remain the primary aiming constraint.
    """
    normals = _unit_rows(np.asarray(surface_normals, dtype=float))
    tangents = _unit_rows(np.asarray(path_tangents, dtype=float))
    if normals.shape != tangents.shape or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("Normals and tangents must both have shape (N, 3).")
    if not 0.0 <= tangent_blend <= 1.0:
        raise ValueError("tangent_blend must be in [0, 1].")

    frames = np.empty((len(normals), 3, 3), dtype=float)
    previous_x = None
    for index, normal in enumerate(normals):
        z_axis = -normal
        tangent = tangents[index] - np.dot(tangents[index], z_axis) * z_axis
        tangent_length = np.linalg.norm(tangent)
        if tangent_length <= 1e-10:
            tangent = previous_x.copy() if previous_x is not None else np.cross(
                z_axis, np.array([0.0, 0.0, 1.0])
            )
            if np.linalg.norm(tangent) <= 1e-10:
                tangent = np.cross(z_axis, np.array([0.0, 1.0, 0.0]))
        tangent /= np.linalg.norm(tangent)

        if previous_x is not None and np.dot(tangent, previous_x) < 0.0:
            tangent = -tangent
        if previous_x is not None and tangent_blend > 0.0:
            blended = (1.0 - tangent_blend) * previous_x + tangent_blend * tangent
            if np.linalg.norm(blended) > 1e-10:
                tangent = blended / np.linalg.norm(blended)

        x_axis = tangent - np.dot(tangent, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        frames[index] = np.column_stack((x_axis, y_axis, z_axis))
        previous_x = x_axis
    return frames


def plan_human_like_sweeps(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    face_indices: np.ndarray,
    patch_ids: np.ndarray,
    config: SweepPlanningConfig,
) -> dict[str, Any]:
    """Order every input sample exactly once into serpentine spatial sweeps.

    Face and patch boundaries are intentionally not used as path boundaries.
    A segment is split only when adjacent ordered samples are spatially
    discontinuous. All returned positions remain in the input CAD coordinates.
    """
    config.validate()
    points = np.asarray(surface_points, dtype=float)
    normals = np.asarray(surface_normals, dtype=float)
    faces = np.asarray(face_indices, dtype=np.int64)
    patches = np.asarray(patch_ids, dtype=np.int64)

    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("surface_points must be a non-empty (N, 3) array.")
    count = len(points)
    if normals.shape != (count, 3) or faces.shape != (count,) or patches.shape != (count,):
        raise ValueError("Sample metadata arrays must have matching lengths.")
    if not all(np.all(np.isfinite(value)) for value in (points, normals)):
        raise ValueError("Surface samples contain non-finite values.")
    normals = _unit_rows(normals)

    center, major, minor = _principal_plane(points)
    centered = points - center
    projected = np.column_stack((centered @ major, centered @ minor))
    lane_index = np.floor(
        (projected[:, 1] - np.min(projected[:, 1])) / config.lane_spacing
    ).astype(np.int64)
    max_gap = config.max_gap_factor * max(
        config.sample_spacing, config.lane_spacing
    )

    ordered_chunks: list[np.ndarray] = []
    lane_ids: list[int] = []
    for lane_number in np.unique(lane_index):
        members = np.flatnonzero(lane_index == lane_number)
        # Sort by the dominant sweep coordinate, then split only physical gaps.
        members = members[np.argsort(projected[members, 0], kind="stable")]
        chunks = _split_lane(members, projected, max_gap)
        for chunk_number, chunk in enumerate(chunks):
            ordered_chunks.append(chunk)
            lane_ids.append(int(lane_number))

    # Alternate direction across successive chunks to create a serpentine path.
    ordered_indices: list[int] = []
    segment_ids: list[int] = []
    for segment_id, chunk in enumerate(ordered_chunks):
        if segment_id % 2:
            chunk = chunk[::-1]
        ordered_indices.extend(int(value) for value in chunk)
        segment_ids.extend([segment_id] * len(chunk))

    order = np.asarray(ordered_indices, dtype=np.int64)
    segments = np.asarray(segment_ids, dtype=np.int64)
    if len(order) != count or not np.array_equal(np.sort(order), np.arange(count)):
        raise RuntimeError("Sweep planner failed to preserve every sample exactly once.")

    ordered_points = points[order].copy()
    ordered_normals = normals[order].copy()
    ordered_tangents = _path_tangents(ordered_points, segments)
    frames = build_nozzle_frames(
        ordered_normals, ordered_tangents, tangent_blend=config.tangent_blend
    )
    directions = frames[:, :, 2].copy()

    # A stable sample sequence index is distinct from the source mesh face index.
    return {
        "sample_order": order,
        "surface_points": ordered_points,
        "surface_normals": ordered_normals,
        "face_indices": faces[order].copy(),
        "patch_ids": patches[order].copy(),
        "segment_ids": segments,
        "path_tangents": ordered_tangents,
        "spray_directions": directions,
        "nozzle_frames": frames,
        "segment_count": int(len(ordered_chunks)),
        "lane_count": int(len(np.unique(lane_index))),
        "algorithm": "PCA_SPATIAL_SERPENTINE_SWEEP",
        "topology_boundary_policy": "NO_SPLIT_ON_FACE_OR_PATCH_ID",
        "gap_split_threshold": float(max_gap),
    }
