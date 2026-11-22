"""Spatial spray deposition and dry-film thickness prediction.

The Double-Beta kernel is a configurable, uncalibrated spatial distribution
model. It is integrated over surface sample areas; paint that misses the
target surface is not redistributed onto the remaining hit samples.
"""
from __future__ import annotations
import math
from typing import Iterable
import numpy as np

def _validate_beta_shape(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value > 100.0:
        raise ValueError(f"{name} must be finite and in (0, 100].")
    return value

def _beta_pdf_unit_interval(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    alpha = _validate_beta_shape(alpha, "alpha")
    beta = _validate_beta_shape(beta, "beta")
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    inside = (x >= 0.0) & (x <= 1.0)
    if not np.any(inside): return result
    log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    values = x[inside]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_pdf = np.full(values.shape, log_norm, dtype=np.float64)
        if alpha != 1.0: log_pdf += (alpha - 1.0) * np.log(values)
        if beta != 1.0: log_pdf += (beta - 1.0) * np.log1p(-values)
        result[inside] = np.exp(log_pdf)
    result[~np.isfinite(result)] = 0.0
    return result

def double_beta_spatial_density(normalized_xy: np.ndarray, alpha_x: float = 2.0, beta_x: float = 2.0, alpha_y: float = 2.0, beta_y: float = 2.0) -> np.ndarray:
    """Normalized separable Beta density over nozzle-frame coordinates [-1,1]²."""
    points = np.asarray(normalized_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2: raise ValueError("normalized_xy must have shape (N, 2).")
    if not np.all(np.isfinite(points)): raise ValueError("normalized_xy contains non-finite values.")
    x, y = (points[:, 0] + 1.0) * 0.5, (points[:, 1] + 1.0) * 0.5
    return (0.5 * _beta_pdf_unit_interval(x, alpha_x, beta_x)) * (0.5 * _beta_pdf_unit_interval(y, alpha_y, beta_y))


def accumulate_double_beta_deposition_step(
    *,
    normalized_xy: np.ndarray,
    sample_area_mm2: np.ndarray,
    sample_facing_cosine: np.ndarray,
    sample_footprint_area_mm2: np.ndarray,
    duration_s: float,
    fluid_flow_ml_min: float,
    transfer_efficiency_percent: float,
    volume_solids_percent: float,
    number_of_coats: int = 1,
    spray_on: bool = True,
    alpha_x: float = 2.0,
    beta_x: float = 2.0,
    alpha_y: float = 2.0,
    beta_y: float = 2.0,
) -> dict:
    """Calculate one time interval's sparse surface deposition increment.

    All arrays refer only to visible samples returned by the viewer's
    footprint and first-hit ray-casting stage. The caller accumulates the
    returned sample thickness into its persistent per-mesh-sample array.
    """
    xy = np.asarray(normalized_xy, dtype=np.float64)
    areas = np.asarray(sample_area_mm2, dtype=np.float64)
    facing = np.asarray(sample_facing_cosine, dtype=np.float64)
    footprint_areas = np.asarray(sample_footprint_area_mm2, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("normalized_xy must have shape (N, 2).")
    count = len(xy)
    if any(a.shape != (count,) for a in (areas, facing, footprint_areas)):
        raise ValueError("Sparse deposition arrays must have matching sample counts.")
    if not all(np.all(np.isfinite(a)) for a in (xy, areas, facing, footprint_areas)):
        raise ValueError("Sparse deposition arrays must be finite.")
    if np.any(areas <= 0.0) or np.any(footprint_areas <= 0.0):
        raise ValueError("Sample and footprint areas must be positive.")
    dt, flow = float(duration_s), float(fluid_flow_ml_min)
    efficiency = float(transfer_efficiency_percent) / 100.0
    solids = float(volume_solids_percent) / 100.0
    coats = int(number_of_coats)
    if not all(math.isfinite(v) for v in (dt, flow, efficiency, solids)):
        raise ValueError("Process values must be finite.")
    if dt < 0.0 or flow < 0.0:
        raise ValueError("Duration and flow must be non-negative.")
    if not 0.0 <= efficiency <= 1.0 or not 0.0 <= solids <= 1.0:
        raise ValueError("Efficiency and solids must be in [0,100] percent.")
    if coats < 1 or coats > 20 or coats != number_of_coats:
        raise ValueError("number_of_coats must be an integer from 1 to 20.")
    for name, value in (("alpha_x", alpha_x), ("beta_x", beta_x),
                        ("alpha_y", alpha_y), ("beta_y", beta_y)):
        _validate_beta_shape(value, name)
    input_dry = (
        flow * dt / 60.0 * 1000.0 * efficiency * solids * coats
        if spray_on else 0.0
    )
    increments = np.zeros(count, dtype=np.float64)
    deposited = np.zeros(count, dtype=np.float64)
    if input_dry > 0.0 and count:
        density = double_beta_spatial_density(
            xy, alpha_x, beta_x, alpha_y, beta_y
        )
        fractions = np.clip(
            density * 4.0 * areas / footprint_areas
            * np.clip(facing, 0.0, 1.0),
            0.0, 1.0,
        )
        deposited = input_dry * fractions
        total = float(np.sum(deposited))
        if total > input_dry and total > 0.0:
            deposited *= input_dry / total
        increments = deposited / areas * 1000.0
    return {
        "sample_thickness_increment_um": increments,
        "sample_deposited_volume_mm3": deposited,
        "input_dry_volume_mm3": float(input_dry),
        "deposited_volume_mm3": float(np.sum(deposited)),
        "deposition_fraction": float(np.sum(deposited) / input_dry) if input_dry > 0 else 0.0,
    }


def calculate_cone_deposition_step(
    *,
    surface_points_mm: np.ndarray,
    surface_normals: np.ndarray,
    sample_area_mm2: np.ndarray,
    nozzle_position_mm: np.ndarray,
    spray_direction: np.ndarray,
    cone_angle_deg: float,
    duration_s: float,
    fluid_flow_ml_min: float,
    transfer_efficiency_percent: float,
    volume_solids_percent: float,
    number_of_coats: int = 1,
    travel_speed_mm_s: float = 0.0,
    radial_alpha: float = 2.0,
    radial_beta: float = 2.0,
    core_deposition_fraction: float = 0.88,
    overspray_tail_fraction: float = 0.12,
    overspray_tail_sigma_fraction: float = 0.35,
) -> dict:
    """Calculate one geometry-aware spray interval over face centroids.

    The nominal spray footprint uses a normalized double-Beta profile. A
    separate Gaussian far-field tail is added outside the nominal cone so
    weak overspray can reach surfaces beyond the nominal footprint. This
    reflects the literature distinction between finite-range Beta models and
    infinite-range Gaussian/Cauchy models: real spray clouds can have a small
    non-zero deposition outside the nominal cone.

    Distance controls both the nominal footprint and overspray scale, cone
    angle controls the nominal footprint, incidence controls local
    deposition, and duration/travel speed controls deposited volume. The
    distribution fractions are engineering defaults until measured spray
    pattern data are supplied for the actual nozzle.
    """
    points = np.asarray(surface_points_mm, dtype=np.float64)
    normals = np.asarray(surface_normals, dtype=np.float64)
    areas = np.asarray(sample_area_mm2, dtype=np.float64)
    nozzle = np.asarray(nozzle_position_mm, dtype=np.float64)
    direction = np.asarray(spray_direction, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("surface_points_mm must be finite (N,3).")
    if normals.shape != points.shape or not np.all(np.isfinite(normals)):
        raise ValueError("surface_normals must match surface_points_mm.")
    if areas.shape != (len(points),) or not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("sample_area_mm2 must be positive and match surface_points_mm.")
    if nozzle.shape != (3,) or not np.all(np.isfinite(nozzle)):
        raise ValueError("nozzle_position_mm must be finite (3,).")
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("spray_direction must be finite (3,).")
    cone = float(cone_angle_deg)
    dt = float(duration_s)
    speed = float(travel_speed_mm_s)
    flow = float(fluid_flow_ml_min)
    if not math.isfinite(cone) or not 0.0 < cone < 179.0:
        raise ValueError("cone_angle_deg must be in (0,179).")
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("duration_s must be finite and non-negative.")
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("travel_speed_mm_s must be finite and non-negative.")
    if not math.isfinite(flow) or flow < 0.0:
        raise ValueError("fluid_flow_ml_min must be finite and non-negative.")

    core_fraction = float(core_deposition_fraction)
    tail_fraction = float(overspray_tail_fraction)
    tail_sigma_fraction = float(overspray_tail_sigma_fraction)
    if (
        not all(math.isfinite(v) for v in (
            core_fraction, tail_fraction, tail_sigma_fraction
        ))
        or core_fraction < 0.0
        or tail_fraction < 0.0
        or tail_sigma_fraction <= 0.0
    ):
        raise ValueError(
            "Deposition distribution parameters must be finite; fractions "
            "must be non-negative and tail sigma must be positive."
        )
    fraction_total = core_fraction + tail_fraction
    if fraction_total <= 0.0:
        raise ValueError(
            "core_deposition_fraction + overspray_tail_fraction must be positive."
        )
    core_fraction /= fraction_total
    tail_fraction /= fraction_total

    nlen = np.linalg.norm(normals, axis=1)
    if np.any(nlen <= 1e-12):
        raise ValueError("Surface normals cannot be zero.")
    normals = normals / nlen[:, None]
    dlen = float(np.linalg.norm(direction))
    if dlen <= 1e-12:
        raise ValueError("spray_direction cannot be zero.")
    direction = direction / dlen

    delta = points - nozzle[None, :]
    axial = delta @ direction
    radial_vec = delta - axial[:, None] * direction[None, :]
    radial_distance = np.linalg.norm(radial_vec, axis=1)
    radius = np.maximum(axial, 0.0) * math.tan(math.radians(cone * 0.5))
    inside = (axial > 1e-9) & (radial_distance <= radius) & (radius > 1e-9)
    facing = np.clip(normals @ (-direction), 0.0, 1.0)

    # Build a stable orthonormal basis perpendicular to the spray axis.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, direction))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_x = np.cross(direction, reference)
    basis_x /= max(float(np.linalg.norm(basis_x)), 1e-12)
    basis_y = np.cross(direction, basis_x)
    basis_y /= max(float(np.linalg.norm(basis_y)), 1e-12)

    local_x = radial_vec @ basis_x
    local_y = radial_vec @ basis_y
    normalized_xy = np.column_stack(
        (local_x / np.maximum(radius, 1e-12),
         local_y / np.maximum(radius, 1e-12))
    )
    normalized_xy[~inside] = 2.0

    # The cone cross-section grows with standoff². This is the key distance
    # term: identical paint volume spread over a larger footprint produces a
    # lower film thickness.
    footprint_area = math.pi * np.maximum(radius, 1e-9) ** 2

    core_density = double_beta_spatial_density(
        normalized_xy, radial_alpha, radial_beta, radial_alpha, radial_beta
    )
    core_density[~inside] = 0.0

    # Infinite-range overspray tail. It starts at the nominal footprint edge
    # and decays radially with a Gaussian law. The conditional normalization
    # below makes the tail carry exactly its configured mass fraction outside
    # the nominal radius in the ideal planar integral.
    sigma = np.maximum(
        radius * tail_sigma_fraction,
        1e-6,
    )
    radial_sq = np.maximum(radial_distance * radial_distance, 0.0)
    tail_density = np.zeros(len(points), dtype=np.float64)
    outside = (axial > 1e-9) & (radial_distance > radius)
    if np.any(outside):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            edge_survival = np.exp(
                -0.5 * (radius[outside] / sigma[outside]) ** 2
            )
            gaussian_norm = (
                2.0 * math.pi * sigma[outside] ** 2
                * np.maximum(edge_survival, 1e-300)
            )
            tail_density[outside] = (
                np.exp(-0.5 * radial_sq[outside] / sigma[outside] ** 2)
                / np.maximum(gaussian_norm, 1e-300)
            )
        tail_density[~np.isfinite(tail_density)] = 0.0

    # The core density integrates to one over the nominal disk. The tail
    # density integrates to one over the exterior region. Their configured
    # fractions therefore remain interpretable as core vs overspray paint.
    density = core_fraction * core_density + tail_fraction * tail_density

    input_dry = (
        flow * dt / 60.0 * 1000.0
        * (float(transfer_efficiency_percent) / 100.0)
        * (float(volume_solids_percent) / 100.0)
        * int(number_of_coats)
    )
    increments = np.zeros(len(points), dtype=np.float64)
    if input_dry <= 0.0 or not len(points):
        return {
            "thickness_increment_um": increments,
            "deposited_volume_mm3": 0.0,
            "input_dry_volume_mm3": float(max(input_dry, 0.0)),
            "inside_cone": inside,
            "facing_cosine": facing,
            "travel_speed_mm_s": speed,
            "footprint_radius_mm": radius,
        }

    # Integrate over the actual face area. The footprint area grows with
    # standoff, so identical paint flow produces a thinner film farther away.
    weight = density * 4.0 * areas / np.maximum(footprint_area, 1e-12)
    weight *= np.clip(facing, 0.0, 1.0)
    weight[~inside] = 0.0
    total = float(np.sum(weight))
    if total <= 1e-12:
        return {
            "thickness_increment_um": increments,
            "deposited_volume_mm3": 0.0,
            "input_dry_volume_mm3": float(input_dry),
            "inside_cone": inside,
            "facing_cosine": facing,
            "travel_speed_mm_s": speed,
            "footprint_radius_mm": radius,
        }
    deposited = input_dry * weight
    deposited_total = float(np.sum(deposited))
    if deposited_total > input_dry:
        deposited *= input_dry / deposited_total
    return {
        "deposited_volume_mm3": deposited,
        "total_deposited_volume_mm3": float(np.sum(deposited)),
        "input_dry_volume_mm3": float(input_dry),
        "inside_cone": inside,
        "overspray_outside_cone": outside,
        "facing_cosine": facing,
        "travel_speed_mm_s": speed,
        "footprint_radius_mm": radius,
        "overspray_sigma_mm": sigma,
        "normalized_xy": normalized_xy,
        "distribution_model": "BETA_CORE_PLUS_INFINITE_GAUSSIAN_OVERSPRAY_TAIL",
        "core_deposition_fraction": float(core_fraction),
        "overspray_tail_fraction": float(tail_fraction),
        "overspray_tail_sigma_fraction": float(tail_sigma_fraction),
        "calibration_status": "ENGINEERING_DEFAULT_UNCALIBRATED",
    }

def simulate_double_beta_deposition(*, sample_normalized_xy_by_path: np.ndarray, sample_area_mm2: np.ndarray, sample_facing_cosine_by_path: np.ndarray, sample_face_ids: np.ndarray, trajectory_time_s: np.ndarray, spray_on_mask: np.ndarray, hit_sample_indices_by_path: Iterable[np.ndarray], fluid_flow_ml_min: float, transfer_efficiency_percent: float, volume_solids_percent: float, footprint_width_mm: float, footprint_thickness_mm: float, number_of_coats: int = 1, alpha_x: float = 2.0, beta_x: float = 2.0, alpha_y: float = 2.0, beta_y: float = 2.0) -> dict:
    """Accumulate predicted dry film; visible hit IDs must already include occlusion tests.

    The Beta kernel is integrated over projected surface sample area. Uncaptured
    paint remains uncollected; it is never redistributed to other hit samples.
    Coordinates are nozzle-frame x/y normalized by half footprint dimensions.
    """
    xy = np.asarray(sample_normalized_xy_by_path, dtype=np.float64)
    areas = np.asarray(sample_area_mm2, dtype=np.float64)
    facing = np.asarray(sample_facing_cosine_by_path, dtype=np.float64)
    face_ids = np.asarray(sample_face_ids, dtype=np.int64)
    times = np.asarray(trajectory_time_s, dtype=np.float64)
    spray_on = np.asarray(spray_on_mask, dtype=bool)
    if xy.ndim != 3 or xy.shape[2] != 2: raise ValueError("sample_normalized_xy_by_path must have shape (P,S,2).")
    path_count, sample_count, _ = xy.shape
    if areas.shape != (sample_count,) or face_ids.shape != (sample_count,): raise ValueError("sample areas and face IDs must have shape (S,).")
    if facing.shape != (path_count, sample_count): raise ValueError("sample_facing_cosine_by_path must have shape (P,S).")
    if times.shape != (path_count,) or spray_on.shape != (path_count,): raise ValueError("trajectory time and spray mask must have shape (P,).")
    if not all(np.all(np.isfinite(a)) for a in (xy, areas, facing, times)): raise ValueError("Deposition arrays must be finite.")
    if np.any(areas <= 0.0) or np.any(face_ids < 0): raise ValueError("Sample areas must be positive and face IDs non-negative.")
    if path_count > 1 and np.any(np.diff(times) < 0.0): raise ValueError("Trajectory times must be monotonically non-decreasing.")
    flow, efficiency, solids = float(fluid_flow_ml_min), float(transfer_efficiency_percent) / 100.0, float(volume_solids_percent) / 100.0
    width, thickness = float(footprint_width_mm), float(footprint_thickness_mm)
    if not all(math.isfinite(v) for v in (flow, efficiency, solids, width, thickness)): raise ValueError("Process parameters must be finite.")
    if flow < 0.0 or not 0.0 <= efficiency <= 1.0 or not 0.0 <= solids <= 1.0: raise ValueError("Flow must be non-negative; efficiency and solids must be in [0,100] percent.")
    if width <= 0.0 or thickness <= 0.0: raise ValueError("Footprint dimensions must be positive.")
    if int(number_of_coats) != number_of_coats or not 1 <= int(number_of_coats) <= 20: raise ValueError("number_of_coats must be an integer from 1 to 20.")
    for name, value in (("alpha_x", alpha_x), ("beta_x", beta_x), ("alpha_y", alpha_y), ("beta_y", beta_y)): _validate_beta_shape(value, name)
    hit_lists = list(hit_sample_indices_by_path)
    if len(hit_lists) != path_count: raise ValueError("One hit-sample index array is required per path sample.")
    sample_thickness = np.zeros(sample_count, dtype=np.float64)
    sample_volume = np.zeros(sample_count, dtype=np.float64)
    input_by_path = np.zeros(path_count, dtype=np.float64)
    deposited_by_path = np.zeros(path_count, dtype=np.float64)
    area_scale = 4.0 / (width * thickness)
    for p in range(1, path_count):
        dt = float(times[p] - times[p - 1])
        if dt <= 0.0 or not spray_on[p] or flow <= 0.0: continue
        input_dry = flow * dt / 60.0 * 1000.0 * efficiency * solids * int(number_of_coats)
        input_by_path[p] = input_dry
        if input_dry <= 0.0: continue
        ids = np.asarray(hit_lists[p], dtype=np.int64)
        if ids.size == 0: continue
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= sample_count) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid or duplicate hit sample IDs at path sample {p}.")
        density = double_beta_spatial_density(xy[p, ids], alpha_x, beta_x, alpha_y, beta_y)
        incidence = np.clip(facing[p, ids], 0.0, 1.0)
        fraction = np.clip(density * area_scale * areas[ids] * incidence, 0.0, 1.0)
        volume = input_dry * fraction
        # Coarse surface quadrature can over-integrate a narrow kernel.
        # Enforce the physical upper bound: deposited paint cannot exceed
        # the dry paint supplied during this trajectory interval.
        interval_deposited = float(np.sum(volume))
        if interval_deposited > input_dry and interval_deposited > 0.0:
            volume *= input_dry / interval_deposited
        sample_volume[ids] += volume
        sample_thickness[ids] += volume / areas[ids] * 1000.0
        deposited_by_path[p] = float(np.sum(volume))
    face_count = int(np.max(face_ids)) + 1 if len(face_ids) else 0
    face_area = np.bincount(face_ids, weights=areas, minlength=face_count)
    face_volume = np.bincount(face_ids, weights=sample_volume, minlength=face_count)
    face_thickness = np.divide(face_volume * 1000.0, face_area, out=np.zeros(face_count), where=face_area > 0.0)
    total_input, total_deposited = float(np.sum(input_by_path)), float(np.sum(deposited_by_path))
    return {"sample_thickness_um": sample_thickness, "sample_deposited_volume_mm3": sample_volume, "face_thickness_um": face_thickness, "face_deposited_volume_mm3": face_volume, "input_dry_volume_by_path_mm3": input_by_path, "deposited_volume_by_path_mm3": deposited_by_path, "total_input_dry_volume_mm3": total_input, "total_deposited_volume_mm3": total_deposited, "deposition_fraction": total_deposited / total_input if total_input > 0.0 else 0.0, "model_status": "PREDICTED_UNCALIBRATED"}

def calculate_film_thickness_statistics(
    face_thickness_um: np.ndarray,
    face_area_mm2: np.ndarray,
    target_thickness_um: float,
    lower_band_ratio: float = 0.90,
    upper_band_ratio: float = 1.10,
) -> dict:
    """Return area-weighted dry-film statistics over all target faces.

    Faces with zero deposition remain in the distribution. P95 is the
    area-weighted 95th percentile, not the percentile of face count.
    The in-band metric counts only area within the inclusive target-thickness
    interval [lower_band_ratio * target, upper_band_ratio * target].
    """
    thickness = np.asarray(face_thickness_um, dtype=np.float64)
    area = np.asarray(face_area_mm2, dtype=np.float64)
    target = float(target_thickness_um)
    if thickness.ndim != 1 or area.shape != thickness.shape:
        raise ValueError("Thickness and face-area arrays must be matching 1-D arrays.")
    if not np.all(np.isfinite(thickness)) or not np.all(np.isfinite(area)):
        raise ValueError("Thickness and face areas must be finite.")
    if np.any(thickness < 0.0) or np.any(area <= 0.0):
        raise ValueError("Thickness must be non-negative and face areas positive.")
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("Target thickness must be finite and positive.")
    lower_ratio = float(lower_band_ratio)
    upper_ratio = float(upper_band_ratio)
    if (
        not math.isfinite(lower_ratio)
        or not math.isfinite(upper_ratio)
        or lower_ratio < 0.0
        or upper_ratio <= lower_ratio
    ):
        raise ValueError("Film-thickness band ratios must be finite, non-negative, and increasing.")
    if thickness.size == 0:
        raise ValueError("At least one target face is required.")

    total_area = float(np.sum(area))
    mean = float(np.sum(thickness * area) / total_area)
    order = np.argsort(thickness, kind="stable")
    sorted_thickness = thickness[order]
    cumulative_area = np.cumsum(area[order])
    p95_index = int(np.searchsorted(cumulative_area, 0.95 * total_area, side="left"))
    p95 = float(sorted_thickness[min(p95_index, len(sorted_thickness) - 1)])
    at_or_above = float(np.sum(area[thickness >= target]) / total_area * 100.0)
    lower_um = target * lower_ratio
    upper_um = target * upper_ratio
    in_band = (thickness >= lower_um) & (thickness <= upper_um)
    below_band = thickness < lower_um
    above_band = thickness > upper_um
    return {
        "min_um": float(np.min(thickness)),
        "mean_um": mean,
        "p95_um": p95,
        "max_um": float(np.max(thickness)),
        "area_at_or_above_target_percent": at_or_above,
        "area_below_target_percent": 100.0 - at_or_above,
        "area_in_target_band_percent": float(np.sum(area[in_band]) / total_area * 100.0),
        "area_below_target_band_percent": float(np.sum(area[below_band]) / total_area * 100.0),
        "area_above_target_band_percent": float(np.sum(area[above_band]) / total_area * 100.0),
        "lower_band_thickness_um": lower_um,
        "upper_band_thickness_um": upper_um,
        "total_area_mm2": total_area,
    }


def update_face_thickness_from_changed_samples(
    sample_thickness_um: np.ndarray,
    face_thickness_um: np.ndarray,
    target_face_ids: np.ndarray,
    changed_sample_ids: np.ndarray,
    sample_count_per_face: int,
) -> np.ndarray:
    """Update only target faces touched by newly deposited samples.

    Samples are stored contiguously per target face. The returned IDs are
    global mesh face IDs whose area-independent sample mean was refreshed.
    Unchanged face values are deliberately left untouched.
    """
    samples = np.asarray(sample_thickness_um, dtype=np.float64)
    face_values = np.asarray(face_thickness_um, dtype=np.float64)
    target_ids = np.asarray(target_face_ids, dtype=np.int64)
    changed_ids = np.asarray(changed_sample_ids, dtype=np.int64)
    count = int(sample_count_per_face)
    if count <= 0:
        raise ValueError("sample_count_per_face must be positive.")
    if samples.ndim != 1 or samples.size != len(target_ids) * count:
        raise ValueError("Sample thickness must contain a fixed number of samples per target face.")
    if face_values.ndim != 1:
        raise ValueError("Face thickness must be a one-dimensional array.")
    if target_ids.ndim != 1 or np.any(target_ids < 0) or np.any(target_ids >= len(face_values)):
        raise ValueError("Target face IDs are outside the face-thickness array.")
    if changed_ids.ndim != 1 or np.any(changed_ids < 0) or np.any(changed_ids >= len(samples)):
        raise ValueError("Changed sample IDs are outside the sample-thickness array.")
    if changed_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    local_face_ids = np.unique(changed_ids // count)
    global_face_ids = target_ids[local_face_ids]
    per_face_samples = samples.reshape(len(target_ids), count)
    face_values[global_face_ids] = np.mean(per_face_samples[local_face_ids], axis=1)
    return global_face_ids
