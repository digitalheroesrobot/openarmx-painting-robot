"""Mesh-sampled spray-cone footprint overlap reduction.

Cone membership is evaluated in 3-D against area-weighted probes on the
triangulated product surface. A candidate is removed only if its footprint
overlap meets the configured threshold and removing it leaves every previously
covered probe covered by another candidate. This preserves sampled coverage;
it is not a calibrated paint-film or continuous analytic area solver.
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
    sample_area_weights: np.ndarray,
    standoff_coordinate_units: float,
    cone_angle_deg: float,
    overlap_percent: float,
    sample_spacing_coordinate_units: float,
) -> dict[str, Any]:
    """Reduce redundant cone centers while preserving the sampled coverage mask.

    For nozzle i and surface probe q, q is inside its cone when its axial
    distance is positive and radial distance <= axial_distance*tan(angle/2).
    All calculations use the PRODUCT_CAD coordinate frame.
    """
    points = np.asarray(surface_points, dtype=np.float64)
    nozzles = np.asarray(nozzle_points, dtype=np.float64)
    directions = np.asarray(spray_directions, dtype=np.float64)
    weights = np.asarray(sample_area_weights, dtype=np.float64)
    count = len(points)

    if points.ndim != 2 or points.shape[1] != 3 or count == 0:
        raise ValueError("surface_points must be a non-empty (N, 3) array.")
    if nozzles.shape != points.shape or directions.shape != points.shape:
        raise ValueError("Nozzle points and directions must match surface_points.")
    if weights.shape != (count,):
        raise ValueError("sample_area_weights must have shape (N,).")
    if not all(np.all(np.isfinite(value)) for value in (points, nozzles, directions, weights)):
        raise ValueError("Footprint optimizer inputs must be finite.")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("Surface area weights must have a positive sum.")

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
        raise ValueError("Sample spacing must be finite and positive.")

    direction_lengths = np.linalg.norm(directions, axis=1)
    if np.any(direction_lengths <= 1e-12):
        raise ValueError("Spray directions cannot contain zero vectors.")
    directions = directions / direction_lengths[:, None]

    cone_tangent = math.tan(math.radians(angle * 0.5))
    nominal_radius = standoff * cone_tangent
    # The local search is wider than the planar footprint to accommodate
    # moderate curvature without querying the entire mesh for every nozzle.
    search_radius = max(2.5 * nominal_radius, 2.0 * spacing, 1e-9)

    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, r=search_radius)
    footprints: list[np.ndarray] = []
    for index, neighborhood in enumerate(neighborhoods):
        candidate_indices = np.asarray(neighborhood, dtype=np.int64)
        vectors = points[candidate_indices] - nozzles[index]
        axial = vectors @ directions[index]
        squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
        radial_squared = np.maximum(squared_lengths - axial * axial, 0.0)
        inside = (
            (axial > 0.0)
            & (radial_squared <= (axial * cone_tangent + 1e-10) ** 2)
        )
        selected = candidate_indices[inside]
        if index not in selected:
            selected = np.append(selected, index)
        footprints.append(np.unique(selected))

    coverage_count = np.zeros(count, dtype=np.int32)
    footprint_area = np.zeros(count, dtype=np.float64)
    initial_overlap = np.zeros(count, dtype=np.float64)
    for index, footprint in enumerate(footprints):
        coverage_count[footprint] += 1
        footprint_area[index] = float(np.sum(weights[footprint]))
    if np.any(footprint_area <= 0.0):
        raise ValueError("At least one spray point has zero sampled footprint area.")

    for index, footprint in enumerate(footprints):
        covered_by_others = coverage_count[footprint] > 1
        initial_overlap[index] = (
            float(np.sum(weights[footprint[covered_by_others]]))
            / footprint_area[index]
        )

    # Highly redundant points are considered first. Before each deletion,
    # overlap and unique coverage are re-evaluated against the live set.
    order = np.argsort(-initial_overlap, kind="stable")
    active = np.ones(count, dtype=bool)
    removed: list[int] = []
    threshold_fraction = threshold / 100.0
    for index_value in order:
        index = int(index_value)
        footprint = footprints[index]
        current_overlap = (
            float(np.sum(weights[footprint[coverage_count[footprint] > 1]]))
            / footprint_area[index]
        )
        if current_overlap + 1e-12 < threshold_fraction:
            continue
        if np.all(coverage_count[footprint] > 1):
            coverage_count[footprint] -= 1
            active[index] = False
            removed.append(index)

    kept_indices = np.flatnonzero(active)
    baseline_covered = np.zeros(count, dtype=bool)
    for footprint in footprints:
        baseline_covered[footprint] = True
    final_covered = coverage_count > 0
    coverage_preserved = bool(np.array_equal(baseline_covered, final_covered))
    if not coverage_preserved:
        raise RuntimeError("Footprint reduction unexpectedly changed sampled coverage.")

    retained_overlap = []
    for index in kept_indices:
        footprint = footprints[int(index)]
        covered_by_others = coverage_count[footprint] > 1
        retained_overlap.append(
            float(np.sum(weights[footprint[covered_by_others]]))
            / footprint_area[int(index)]
        )
    retained_overlap_array = np.asarray(retained_overlap, dtype=np.float64)
    covered_area = float(np.sum(weights[baseline_covered]))
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
        "removed_point_count": int(len(removed)),
        "retained_point_count": int(len(kept_indices)),
        "removed_indices_in_preoptimized_order": removed,
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
        "area_measurement": "TRIANGLE_AREA_WEIGHTED_MESH_SAMPLE_ESTIMATE",
        "coverage_preservation_policy": (
            "NO_PREVIOUSLY_COVERED_SURFACE_SAMPLE_MAY_BECOME_UNCOVERED"
        ),
        "geometry_limitations": [
            "Cone membership is evaluated at area-weighted surface probes, not by exact continuous cone-triangle clipping.",
            "The local KD-tree search is capped and may undercount highly grazing or sharply folded intersections.",
            "Occlusion by nearer product geometry is not ray-tested in this version.",
            "This is geometric footprint coverage, not calibrated paint deposition or film-thickness prediction.",
        ],
    }
