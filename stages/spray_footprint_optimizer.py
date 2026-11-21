"""Mesh-sampled spray-cone footprint overlap reduction.

Candidate nozzle cones are evaluated against a denser, area-weighted probe set
sampled on the actual triangulated product surface. A candidate is removed
only when its footprint overlap meets the configured threshold and removing it
leaves every previously covered surface probe covered by another candidate.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def optimize_redundant_spray_points(
    *,
    surface_points: np.ndarray,
    nozzle_points: np.ndarray,
    spray_directions: np.ndarray,
    probe_points: np.ndarray,
    probe_area_weights: np.ndarray,
    standoff_coordinate_units: float,
    cone_angle_deg: float,
    overlap_percent: float,
    sample_spacing_coordinate_units: float,
) -> dict[str, Any]:
    """Remove redundant candidate points while preserving dense sampled coverage.

    For nozzle i and surface probe q, q belongs to the 3-D cone when its axial
    distance is positive and radial distance <= axial_distance*tan(angle/2).
    """
    candidates = np.asarray(surface_points, dtype=np.float64)
    nozzles = np.asarray(nozzle_points, dtype=np.float64)
    directions = np.asarray(spray_directions, dtype=np.float64)
    probes = np.asarray(probe_points, dtype=np.float64)
    weights = np.asarray(probe_area_weights, dtype=np.float64)
    count = len(candidates)
    probe_count = len(probes)

    if candidates.ndim != 2 or candidates.shape[1] != 3 or count == 0:
        raise ValueError("surface_points must be a non-empty (N, 3) array.")
    if nozzles.shape != candidates.shape or directions.shape != candidates.shape:
        raise ValueError("Nozzle points and directions must match surface_points.")
    if probes.ndim != 2 or probes.shape[1] != 3 or probe_count == 0:
        raise ValueError("probe_points must be a non-empty (M, 3) array.")
    if weights.shape != (probe_count,):
        raise ValueError("probe_area_weights must have shape (M,).")
    if not all(np.all(np.isfinite(value)) for value in (candidates, nozzles, directions, probes, weights)):
        raise ValueError("Footprint optimizer inputs must be finite.")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("Probe area weights must have a positive sum.")

    standoff = float(standoff_coordinate_units)
    angle = float(cone_angle_deg)
    threshold = float(overlap_percent)
    spacing = float(sample_spacing_coordinate_units)
    if not math.isfinite(standoff) or standoff <= 0.0:
        raise ValueError("Standoff must be finite and positive.")
    if not math.isfinite(angle) or not 0.0 < angle < 180.0:
        raise ValueError("Cone angle must be between 0 and 180 degrees.")
    if not math.isfinite(threshold) or not 0.0 <= threshold < 100.0:
        raise ValueError("Overlap threshold must be in [0, 100).")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Probe spacing must be finite and positive.")

    direction_lengths = np.linalg.norm(directions, axis=1)
    if np.any(direction_lengths <= 1e-12):
        raise ValueError("Spray directions cannot contain zero vectors.")
    directions = directions / direction_lengths[:, None]

    cone_tangent = math.tan(math.radians(angle * 0.5))
    nominal_radius = standoff * cone_tangent
    search_radius = max(2.5 * nominal_radius, 2.0 * spacing, 1e-9)

    tree = cKDTree(probes)
    neighborhoods = tree.query_ball_point(candidates, r=search_radius)
    footprints: list[np.ndarray] = []
    footprint_area = np.zeros(count, dtype=np.float64)
    for index, neighborhood in enumerate(neighborhoods):
        candidate_indices = np.asarray(neighborhood, dtype=np.int64)
        vectors = probes[candidate_indices] - nozzles[index]
        axial = vectors @ directions[index]
        squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
        radial_squared = np.maximum(squared_lengths - axial * axial, 0.0)
        inside = (
            (axial > 0.0)
            & (radial_squared <= (axial * cone_tangent + 1e-10) ** 2)
        )
        selected = candidate_indices[inside]
        footprints.append(np.unique(selected))
        if len(selected):
            footprint_area[index] = float(np.sum(weights[selected]))

    coverage_count = np.zeros(probe_count, dtype=np.int32)
    for footprint in footprints:
        coverage_count[footprint] += 1

    initial_covered = coverage_count > 0
    initial_overlap = np.zeros(count, dtype=np.float64)
    for index, footprint in enumerate(footprints):
        if len(footprint) == 0 or footprint_area[index] <= 0.0:
            continue
        covered_by_others = coverage_count[footprint] > 1
        initial_overlap[index] = (
            float(np.sum(weights[footprint[covered_by_others]]))
            / footprint_area[index]
        )

    order = np.argsort(-initial_overlap, kind="stable")
    active = np.ones(count, dtype=bool)
    removed_count = 0
    threshold_fraction = threshold / 100.0
    for index_value in order:
        index = int(index_value)
        footprint = footprints[index]
        if len(footprint) == 0 or footprint_area[index] <= 0.0:
            continue
        current_overlap = (
            float(np.sum(weights[footprint[coverage_count[footprint] > 1]]))
            / footprint_area[index]
        )
        if current_overlap + 1e-12 < threshold_fraction:
            continue
        # No sampled surface area may become newly uncovered.
        if np.all(coverage_count[footprint] > 1):
            coverage_count[footprint] -= 1
            active[index] = False
            removed_count += 1

    kept_indices = np.flatnonzero(active)
    final_covered = coverage_count > 0
    coverage_preserved = bool(np.array_equal(initial_covered, final_covered))
    if not coverage_preserved:
        raise RuntimeError("Footprint reduction unexpectedly changed sampled coverage.")

    retained_overlap = []
    for index in kept_indices:
        footprint = footprints[int(index)]
        if len(footprint) == 0 or footprint_area[int(index)] <= 0.0:
            continue
        covered_by_others = coverage_count[footprint] > 1
        retained_overlap.append(
            float(np.sum(weights[footprint[covered_by_others]]))
            / footprint_area[int(index)]
        )
    retained_overlap_array = np.asarray(retained_overlap, dtype=np.float64)
    covered_area = float(np.sum(weights[initial_covered]))
    total_area = float(np.sum(weights))

    return {
        "keep_indices": kept_indices,
        "status": "COMPLETED_MESH_SAMPLED_CONE_OVERLAP_REDUCTION",
        "algorithm": "3D_CONE_MEMBERSHIP_WITH_AREA_WEIGHTED_GREEDY_REDUNDANCY_REMOVAL",
        "overlap_threshold_percent": threshold,
        "cone_angle_deg": angle,
        "standoff_coordinate_units": standoff,
        "nominal_planar_footprint_radius_coordinate_units": nominal_radius,
        "neighborhood_search_radius_coordinate_units": search_radius,
        "input_point_count": int(count),
        "probe_count": int(probe_count),
        "probe_spacing_coordinate_units": spacing,
        "removed_point_count": int(removed_count),
        "retained_point_count": int(len(kept_indices)),
        "coverage_preserved": coverage_preserved,
        "sampled_surface_coverage_percent": (
            100.0 * covered_area / total_area if total_area > 0.0 else 0.0
        ),
        "mean_initial_footprint_overlap_percent": float(np.mean(initial_overlap) * 100.0),
        "mean_retained_footprint_overlap_percent": (
            float(np.mean(retained_overlap_array) * 100.0)
            if len(retained_overlap_array) else 0.0
        ),
        "max_retained_footprint_overlap_percent": (
            float(np.max(retained_overlap_array) * 100.0)
            if len(retained_overlap_array) else 0.0
        ),
        "area_measurement": "DENSE_TRIANGLE_AREA_WEIGHTED_MESH_PROBE_ESTIMATE",
        "coverage_preservation_policy": (
            "NO_PREVIOUSLY_COVERED_SURFACE_PROBE_MAY_BECOME_UNCOVERED"
        ),
        "geometry_limitations": [
            "Cone membership and area are estimated at dense area-weighted mesh probes, not by exact continuous cone-triangle clipping.",
            "The local KD-tree search is capped and may undercount highly grazing or sharply folded intersections.",
            "Occlusion by nearer product geometry is not ray-tested in this version.",
            "This is geometric footprint coverage, not calibrated paint deposition or film-thickness prediction.",
        ],
    }
