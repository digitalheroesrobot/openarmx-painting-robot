"""Coverage-driven surface deficit analysis and correction-stroke proposals.

This module is deliberately independent of Qt, VTK and robot kinematics.
It converts per-surface-sample reach/thickness results into auditable deficit
metrics and conservative candidate strokes. Candidate strokes are NOT yet
robot-executable: collision, reachability, TCP calibration and spray physics
must be validated by downstream planning.
"""
from __future__ import annotations

import math
import numpy as np


def _as_vector(values, name, dtype=np.float64):
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return result


def analyze_surface_deficits(
    *,
    sample_area_mm2,
    target_mask,
    reached_mask,
    thickness_um,
    target_thickness_um,
    lower_ratio=0.90,
    upper_ratio=1.10,
    zero_thickness_epsilon_um=1e-6,
):
    """Classify target samples and report area-weighted coverage/thickness.

    The denominator includes every Stage-2 target sample, including samples
    with no geometric reach or zero predicted film. Excluded faces are omitted
    only when target_mask is False.
    """
    area = _as_vector(sample_area_mm2, "sample_area_mm2")
    target = _as_vector(target_mask, "target_mask", dtype=bool)
    reached = _as_vector(reached_mask, "reached_mask", dtype=bool)
    film = _as_vector(thickness_um, "thickness_um")
    if not (area.shape == target.shape == reached.shape == film.shape):
        raise ValueError("All per-sample arrays must have identical shapes.")
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("sample_area_mm2 must contain finite positive values.")
    if not np.all(np.isfinite(film)) or np.any(film < 0.0):
        raise ValueError("thickness_um must contain finite non-negative values.")

    target_dft = float(target_thickness_um)
    low = float(lower_ratio)
    high = float(upper_ratio)
    epsilon = float(zero_thickness_epsilon_um)
    if not math.isfinite(target_dft) or target_dft <= 0.0:
        raise ValueError("target_thickness_um must be finite and positive.")
    if not math.isfinite(low) or not math.isfinite(high) or not 0.0 < low <= 1.0 <= high:
        raise ValueError("Ratios must satisfy 0 < lower_ratio <= 1 <= upper_ratio.")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("zero_thickness_epsilon_um must be finite and non-negative.")

    # Stage-2 exclusion always wins, even if stale viewer arrays contain hits.
    active = target
    active_area = area[active]
    active_film = film[active]
    active_reached = reached[active]
    total_area = float(np.sum(active_area))
    if total_area <= 0.0:
        raise ValueError("At least one Stage-2 target sample is required.")

    lower = target_dft * low
    upper = target_dft * high
    no_film = active_film <= epsilon
    below = active_film < lower
    in_band = (active_film >= lower) & (active_film <= upper)
    over = active_film > upper

    def area_percent(mask):
        return float(np.sum(active_area[mask]) / total_area * 100.0)

    reached_area = float(np.sum(active_area[active_reached]))
    below_area = float(np.sum(active_area[below]))
    in_band_area = float(np.sum(active_area[in_band]))
    over_area = float(np.sum(active_area[over]))
    no_film_area = float(np.sum(active_area[no_film]))
    weighted_mean = float(np.sum(active_area * active_film) / total_area)

    # Preserve the original input indexing so callers can map diagnostics
    # directly back to Stage-2 faces / Viewer samples.
    status = np.full(len(area), "EXCLUDED", dtype=object)
    status[target & (film <= epsilon)] = "NO_FILM"
    status[target & (film > epsilon) & (film < lower)] = "UNDER_TARGET"
    status[target & (film >= lower) & (film <= upper)] = "IN_BAND"
    status[target & (film > upper)] = "OVER_TARGET"

    return {
        "status_by_sample": status,
        "target_sample_count": int(np.count_nonzero(active)),
        "excluded_sample_count": int(len(area) - np.count_nonzero(active)),
        "target_area_mm2": total_area,
        "geometrically_reached_area_mm2": reached_area,
        "geometric_reach_percent": 100.0 * reached_area / total_area,
        "no_film_area_percent": area_percent(no_film),
        "below_target_area_percent": area_percent(below),
        "in_target_band_area_percent": area_percent(in_band),
        "over_target_area_percent": area_percent(over),
        "area_weighted_mean_thickness_um": weighted_mean,
        "target_thickness_um": target_dft,
        "accepted_thickness_range_um": (lower, upper),
        "deficit_sample_indices": np.flatnonzero(target & (film < lower)),
        "unreached_sample_indices": np.flatnonzero(target & ~reached),
        "model_status": "PREDICTED_UNCALIBRATED",
    }


def build_deficit_correction_strokes(
    *,
    sample_points_mm,
    sample_normals,
    deficit_sample_indices,
    patch_ids=None,
    standoff_mm,
    max_gap_mm,
    max_normal_change_deg=30.0,
):
    """Group deficit samples into conservative, surface-following stroke proposals.

    Samples are grouped by Stage-1 patch, ordered along the dominant local
    tangent axis, then split at spatial gaps or abrupt normal changes. No
    interpolation is performed: every returned waypoint corresponds to an
    actual deficit sample. This avoids inventing unsupported surface points.

    Returned nozzle positions use P_nozzle = P_surface + standoff * normal.
    The returned strokes require downstream smoothing, footprint overlap,
    visibility, robot IK and collision validation before execution.
    """
    points = np.asarray(sample_points_mm, dtype=np.float64)
    normals = np.asarray(sample_normals, dtype=np.float64)
    indices = np.asarray(deficit_sample_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("sample_points_mm must have shape (N, 3).")
    if normals.shape != points.shape:
        raise ValueError("sample_normals must have shape (N, 3).")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError("Sample points and normals must be finite.")
    count = len(points)
    if indices.ndim != 1 or (len(indices) and (
        np.any(indices < 0) or np.any(indices >= count)
    )):
        raise ValueError("deficit_sample_indices contains invalid indices.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("deficit_sample_indices must not contain duplicates.")

    standoff = float(standoff_mm)
    gap_limit = float(max_gap_mm)
    angle_limit = float(max_normal_change_deg)
    if not math.isfinite(standoff) or standoff <= 0.0:
        raise ValueError("standoff_mm must be finite and positive.")
    if not math.isfinite(gap_limit) or gap_limit <= 0.0:
        raise ValueError("max_gap_mm must be finite and positive.")
    if not math.isfinite(angle_limit) or not 0.0 < angle_limit < 180.0:
        raise ValueError("max_normal_change_deg must be in (0, 180).")

    if patch_ids is None:
        patches = np.zeros(count, dtype=np.int64)
    else:
        patches = np.asarray(patch_ids)
        if patches.shape != (count,):
            raise ValueError("patch_ids must have shape (N,).")

    if len(indices) == 0:
        return {
            "surface_points_mm": np.empty((0, 3), dtype=np.float64),
            "surface_normals": np.empty((0, 3), dtype=np.float64),
            "nozzle_points_mm": np.empty((0, 3), dtype=np.float64),
            "spray_directions": np.empty((0, 3), dtype=np.float64),
            "source_sample_indices": np.empty(0, dtype=np.int64),
            "stroke_ids": np.empty(0, dtype=np.int64),
            "patch_ids": np.empty(0, dtype=patches.dtype),
            "stroke_ranges": [],
            "candidate_status": "NO_DEFICITS",
        }

    lengths = np.linalg.norm(normals[indices], axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Deficit sample normals must be non-zero.")
    unit_normals = np.zeros_like(normals)
    unit_normals[indices] = normals[indices] / lengths[:, None]
    cos_limit = math.cos(math.radians(angle_limit))

    ordered_groups = []
    for patch in np.unique(patches[indices]):
        group = indices[patches[indices] == patch]
        if len(group) == 0:
            continue
        centered = points[group] - np.mean(points[group], axis=0)
        if len(group) == 1 or float(np.linalg.norm(centered)) <= 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            # SVD eigenvector signs are mathematically arbitrary. Canonicalize
            # the dominant component so waypoint order is deterministic.
            dominant_component = int(np.argmax(np.abs(axis)))
            if axis[dominant_component] < 0.0:
                axis = -axis
        projection = points[group] @ axis
        ordered = group[np.argsort(projection, kind="stable")]

        current = [int(ordered[0])]
        for raw_next in ordered[1:]:
            previous = current[-1]
            nxt = int(raw_next)
            distance = float(np.linalg.norm(points[nxt] - points[previous]))
            normal_dot = float(np.clip(
                np.dot(unit_normals[previous], unit_normals[nxt]), -1.0, 1.0
            ))
            if distance > gap_limit or normal_dot < cos_limit:
                ordered_groups.append((patch, current))
                current = [nxt]
            else:
                current.append(nxt)
        ordered_groups.append((patch, current))

    out_indices = []
    stroke_ids = []
    out_patch_ids = []
    stroke_ranges = []
    for stroke_id, (patch, group) in enumerate(ordered_groups):
        start = len(out_indices)
        out_indices.extend(group)
        stroke_ids.extend([stroke_id] * len(group))
        out_patch_ids.extend([patch] * len(group))
        stroke_ranges.append({
            "stroke_id": stroke_id,
            "patch_id": patch.item() if hasattr(patch, "item") else patch,
            "start": start,
            "stop": start + len(group),
            "waypoint_count": len(group),
        })

    source_ids = np.asarray(out_indices, dtype=np.int64)
    surface = points[source_ids].copy()
    surface_normals = unit_normals[source_ids].copy()
    nozzle = surface + standoff * surface_normals
    spray_directions = -surface_normals
    return {
        "surface_points_mm": surface,
        "surface_normals": surface_normals,
        "nozzle_points_mm": nozzle,
        "spray_directions": spray_directions,
        "source_sample_indices": source_ids,
        "stroke_ids": np.asarray(stroke_ids, dtype=np.int64),
        "patch_ids": np.asarray(out_patch_ids),
        "stroke_ranges": stroke_ranges,
        "candidate_status": "CANDIDATE_REQUIRES_PHYSICAL_VALIDATION",
        "limitations": [
            "Candidate strokes contain only measured deficit sample positions; no surface interpolation is performed.",
            "Track overlap and deposited thickness must be recomputed with the calibrated spray model.",
            "Nozzle visibility, robot reachability, IK, TCP calibration and collision are not checked.",
        ],
    }

def build_coverage_correction_plan(
    *,
    sample_points_mm,
    sample_normals,
    sample_area_mm2,
    target_mask,
    reached_mask,
    thickness_um,
    target_thickness_um,
    patch_ids=None,
    standoff_mm,
    max_gap_mm,
    lower_ratio=0.90,
    upper_ratio=1.10,
    max_normal_change_deg=30.0,
):
    """Combine whole-target deficit analysis with conservative stroke proposals.

    Only samples below the lower accepted thickness threshold are proposed.
    The complete Stage-2 target mask remains the denominator, regardless of
    whether a sample was reached by the original trajectory.
    """
    analysis = analyze_surface_deficits(
        sample_area_mm2=sample_area_mm2,
        target_mask=target_mask,
        reached_mask=reached_mask,
        thickness_um=thickness_um,
        target_thickness_um=target_thickness_um,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
    )
    strokes = build_deficit_correction_strokes(
        sample_points_mm=sample_points_mm,
        sample_normals=sample_normals,
        deficit_sample_indices=analysis["deficit_sample_indices"],
        patch_ids=patch_ids,
        standoff_mm=standoff_mm,
        max_gap_mm=max_gap_mm,
        max_normal_change_deg=max_normal_change_deg,
    )
    return {
        "analysis": analysis,
        "correction_strokes": strokes,
        "status": "PREDICTED_UNCALIBRATED_CANDIDATES_ONLY",
        "execution_ready": False,
    }
