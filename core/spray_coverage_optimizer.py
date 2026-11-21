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


def world_samples_to_product_cad_mm(*, points_world, normals_world, product_transform_4x4, mm_per_unit):
    """Convert Viewer-world surface samples into PRODUCT_CAD millimeters."""
    points = np.asarray(points_world, dtype=np.float64)
    normals = np.asarray(normals_world, dtype=np.float64)
    transform = np.asarray(product_transform_4x4, dtype=np.float64)
    scale = float(mm_per_unit)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3).")
    if normals.shape != points.shape:
        raise ValueError("normals_world must have shape (N, 3).")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("product_transform_4x4 must be a finite 4x4 matrix.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("Product transform must be affine.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Product transform rotation must be orthonormal.")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("mm_per_unit must be finite and positive.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError("World points and normals must be finite.")
    cad_points_mm = ((points - transform[:3, 3]) @ rotation) * scale
    cad_normals = normals @ rotation
    lengths = np.linalg.norm(cad_normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("World normals must be non-zero.")
    cad_normals /= lengths[:, None]
    return cad_points_mm, cad_normals



def append_correction_strokes_to_trajectory(
    *,
    trajectory,
    correction_strokes,
    source_face_ids,
    mm_per_unit,
    base_speed_mm_s,
    minimum_speed_mm_s,
    singleton_duration_s=0.10,
):
    """Append CAD-space correction strokes to a Stage-3 trajectory dictionary.

    Candidate geometry is supplied in PRODUCT_CAD millimeters and converted
    to the source mesh coordinate unit only at this boundary. Every correction
    stroke receives a fresh segment ID. The first waypoint of each stroke is
    spray-OFF by the Viewer segment-boundary convention; singleton strokes
    receive a repeated endpoint so their second sample can deposit paint.

    This helper does not claim visibility, IK, collision, or robot execution
    validity. It returns a new dictionary and never mutates the input.
    """
    required = (
        "surface_points", "surface_normals", "nozzle_points",
        "spray_directions", "segment_ids", "travel_speeds_mm_s",
        "trajectory_time_s",
    )
    missing = [key for key in required if key not in trajectory]
    if missing:
        raise ValueError("Base trajectory is missing: " + ", ".join(missing))

    scale = float(mm_per_unit)
    base_speed = float(base_speed_mm_s)
    min_speed = float(minimum_speed_mm_s)
    singleton_dt = float(singleton_duration_s)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("mm_per_unit must be finite and positive.")
    if not math.isfinite(base_speed) or base_speed <= 0.0:
        raise ValueError("base_speed_mm_s must be finite and positive.")
    if not math.isfinite(min_speed) or min_speed <= 0.0 or min_speed > base_speed:
        raise ValueError("minimum_speed_mm_s must be positive and <= base speed.")
    if not math.isfinite(singleton_dt) or singleton_dt <= 0.0:
        raise ValueError("singleton_duration_s must be finite and positive.")

    points = np.asarray(correction_strokes["surface_points_mm"], dtype=float)
    normals = np.asarray(correction_strokes["surface_normals"], dtype=float)
    nozzle = np.asarray(correction_strokes["nozzle_points_mm"], dtype=float)
    directions = np.asarray(correction_strokes["spray_directions"], dtype=float)
    sample_ids = np.asarray(correction_strokes["source_sample_indices"], dtype=np.int64)
    face_ids = np.asarray(source_face_ids, dtype=np.int64)
    ranges = list(correction_strokes["stroke_ranges"])
    count = len(points)
    if any(array.shape != (count, 3) for array in (normals, nozzle, directions)):
        raise ValueError("Correction geometry arrays must all have shape (N, 3).")
    if sample_ids.shape != (count,) or np.any(sample_ids < 0) or np.any(sample_ids >= len(face_ids)):
        raise ValueError("Correction source sample indices are invalid.")
    if not all(np.all(np.isfinite(array)) for array in (points, normals, nozzle, directions)):
        raise ValueError("Correction geometry must be finite.")
    if count == 0:
        return {key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in trajectory.items()}

    base_count = len(np.asarray(trajectory["nozzle_points"]))
    base_segments = np.asarray(trajectory["segment_ids"], dtype=np.int64)
    base_times = np.asarray(trajectory["trajectory_time_s"], dtype=float)
    if base_segments.shape != (base_count,) or base_times.shape != (base_count,):
        raise ValueError("Base segment/time arrays do not match trajectory length.")
    if not np.all(np.isfinite(base_times)) or (base_count > 1 and np.any(np.diff(base_times) < 0.0)):
        raise ValueError("Base trajectory times must be finite and monotonic.")
    if len(face_ids) == 0 or np.any(face_ids < 0):
        raise ValueError("source_face_ids must map every target sample to a valid mesh face.")
    if not ranges:
        raise ValueError("Non-empty correction geometry requires stroke ranges.")

    # Duplicate singleton samples so the segment's second point is spray-ON
    # under the Viewer convention that the first sample of each segment is OFF.
    expanded = []
    for stroke in ranges:
        start, stop = int(stroke["start"]), int(stroke["stop"])
        if start < 0 or stop <= start or stop > count:
            raise ValueError("Correction stroke range is invalid.")
        ids = list(range(start, stop))
        if len(ids) == 1:
            ids.append(ids[0])
        expanded.append(ids)

    appended_indices = np.asarray([i for group in expanded for i in group], dtype=np.int64)
    out = dict(trajectory)
    total = base_count + len(appended_indices)
    segment_start = int(np.max(base_segments)) + 1 if base_count else 0
    new_segment_ids = []
    new_times = []
    new_speeds = []
    previous_nozzle = (
        np.asarray(trajectory["nozzle_points"], dtype=float)[-1]
        if base_count else None
    )
    current_time = float(base_times[-1]) if base_count else 0.0
    cursor = 0
    for local_stroke_id, group in enumerate(expanded):
        group_points = nozzle[group] / scale
        if previous_nozzle is not None:
            travel_mm = float(np.linalg.norm(group_points[0] - previous_nozzle) * scale)
            current_time += travel_mm / min_speed
        stroke_id = segment_start + local_stroke_id
        for j, local_index in enumerate(group):
            if j > 0:
                distance_mm = float(np.linalg.norm(
                    (nozzle[local_index] - nozzle[group[j - 1]])
                ))
                current_time += distance_mm / base_speed
                if distance_mm <= 1e-12:
                    current_time += singleton_dt
            new_times.append(current_time)
            new_speeds.append(min_speed if j == 0 else base_speed)
            new_segment_ids.append(stroke_id)
        previous_nozzle = group_points[-1]
        cursor += len(group)

    def append_vector(key, values):
        base = np.asarray(trajectory[key])
        extra = np.asarray(values)
        if base.ndim != extra.ndim or base.shape[1:] != extra.shape[1:]:
            raise ValueError(f"Trajectory field {key} has incompatible shape.")
        out[key] = np.concatenate((base, extra), axis=0)

    append_vector("surface_points", points[appended_indices] / scale)
    append_vector("surface_normals", normals[appended_indices])
    append_vector("nozzle_points", nozzle[appended_indices] / scale)
    append_vector("spray_directions", directions[appended_indices])
    append_vector("face_indices", face_ids[sample_ids[appended_indices]])
    patch_values = np.asarray(correction_strokes["patch_ids"])
    append_vector("patch_ids", patch_values[appended_indices])
    append_vector("segment_ids", np.asarray(new_segment_ids, dtype=np.int64))
    append_vector("travel_speeds_mm_s", np.asarray(new_speeds, dtype=float))
    append_vector("trajectory_time_s", np.asarray(new_times, dtype=float))

    # Rebuild right-handed nozzle frames independently per correction stroke.
    frames = np.empty((len(appended_indices), 3, 3), dtype=float)
    offset = 0
    for group in expanded:
        for j, local_index in enumerate(group):
            z = directions[local_index] / np.linalg.norm(directions[local_index])
            if j == 0:
                reference = np.array([1.0, 0.0, 0.0])
                x = reference - np.dot(reference, z) * z
                if np.linalg.norm(x) < 1e-8:
                    reference = np.array([0.0, 1.0, 0.0])
                    x = reference - np.dot(reference, z) * z
                x /= np.linalg.norm(x)
            else:
                x = frames[offset - 1, :, 0].copy()
                x -= np.dot(x, z) * z
                x /= max(np.linalg.norm(x), 1e-12)
            y = np.cross(z, x)
            y /= max(np.linalg.norm(y), 1e-12)
            frames[offset] = np.column_stack((x, y, z))
            offset += 1
    if "nozzle_frames" in trajectory:
        append_vector("nozzle_frames", frames)

    # World arrays are intentionally not generated here: Viewer owns the
    # CAD-to-WORLD display transform and rebuilds these arrays from CAD data.
    for key in (
        "surface_points_world", "surface_normals_world", "nozzle_points_world",
        "spray_directions_world", "nozzle_frames_world",
    ):
        out.pop(key, None)
    out["correction_path_metadata"] = {
        "status": "PREVIEW_ONLY_REQUIRES_VISIBILITY_IK_COLLISION_VALIDATION",
        "added_sample_count": int(len(appended_indices)),
        "added_stroke_count": int(len(expanded)),
        "coordinate_frame": "PRODUCT_CAD",
        "coordinate_unit": "source_mesh_unit",
        "spray_off_at_each_stroke_start": True,
    }
    if len(np.asarray(out["trajectory_time_s"])) != total:
        raise RuntimeError("Correction append produced inconsistent trajectory length.")
    return out


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
