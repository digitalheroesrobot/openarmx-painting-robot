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