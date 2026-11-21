"""Uncalibrated geometry-aware spray deposition kernel."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import numpy as np

_EPS = 1e-12

@dataclass(frozen=True)
class SprayDepositionConfig:
    """Parameters for a Gaussian radial approximation of a conical spray."""
    cone_angle_deg: float
    radial_sigma_fraction: float = 0.45
    incidence_exponent: float = 1.0
    minimum_axial_distance: float = 1e-6

    def validate(self) -> None:
        values = (self.cone_angle_deg, self.radial_sigma_fraction,
                  self.incidence_exponent, self.minimum_axial_distance)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Spray deposition parameters must be finite.")
        if not 0.0 < self.cone_angle_deg < 150.0:
            raise ValueError("cone_angle_deg must be between 0 and 150 degrees.")
        if self.radial_sigma_fraction <= 0.0:
            raise ValueError("radial_sigma_fraction must be positive.")
        if self.incidence_exponent < 0.0:
            raise ValueError("incidence_exponent cannot be negative.")
        if self.minimum_axial_distance <= 0.0:
            raise ValueError("minimum_axial_distance must be positive.")

def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    length = float(np.linalg.norm(value))
    if length <= _EPS:
        raise ValueError(f"{name} must have non-zero length.")
    return value / length

def evaluate_pose_deposition(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    nozzle_position: np.ndarray,
    spray_direction: np.ndarray,
    config: SprayDepositionConfig,
) -> dict[str, Any]:
    """Evaluate relative instantaneous deposition from one nozzle pose.

    Spray direction points from nozzle toward target; target normals point
    outward. This is an uncalibrated engineering approximation, not film
    thickness. Occlusion must be checked by a separate geometry stage.
    """
    config.validate()
    points = np.asarray(surface_points, dtype=np.float64)
    normals = np.asarray(surface_normals, dtype=np.float64)
    nozzle = np.asarray(nozzle_position, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("surface_points must be a finite (N, 3) array.")
    if normals.shape != points.shape or not np.all(np.isfinite(normals)):
        raise ValueError("surface_normals must be finite and match surface_points.")
    if nozzle.shape != (3,) or not np.all(np.isfinite(nozzle)):
        raise ValueError("nozzle_position must be a finite 3-vector.")
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths <= _EPS):
        raise ValueError("surface_normals cannot contain zero-length vectors.")
    unit_normals = normals / normal_lengths[:, None]
    axis = _unit_vector(spray_direction, "spray_direction")

    delta = points - nozzle[None, :]
    axial = delta @ axis
    squared_distance = np.einsum("ij,ij->i", delta, delta)
    distance = np.sqrt(squared_distance)
    safe_axial = np.maximum(axial, config.minimum_axial_distance)
    radial_squared = np.maximum(squared_distance - axial * axial, 0.0)
    radial_distance = np.sqrt(radial_squared)
    cone_radius = safe_axial * math.tan(math.radians(config.cone_angle_deg * 0.5))
    sigma = np.maximum(cone_radius * config.radial_sigma_fraction, _EPS)
    radial_weight = np.exp(-0.5 * (radial_distance / sigma) ** 2)
    incidence = np.clip(unit_normals @ (-axis), 0.0, 1.0)
    incidence_weight = incidence ** config.incidence_exponent
    inside_cone = ((axial > config.minimum_axial_distance)
                   & (radial_distance <= cone_radius))
    valid = (inside_cone & (distance > config.minimum_axial_distance)
             & (incidence > _EPS))
    relative_flux = np.zeros(len(points), dtype=np.float64)
    relative_flux[valid] = (
        radial_weight[valid] * incidence_weight[valid]
        / np.maximum(squared_distance[valid], _EPS)
    )
    return {
        "relative_flux": relative_flux,
        "inside_cone": inside_cone,
        "incidence_cosine": incidence,
        "radial_distance": radial_distance,
        "cone_radius": cone_radius,
        "model": "GAUSSIAN_RADIAL_CONE_INVERSE_SQUARE_INCIDENCE",
        "calibration_status": "UNCALIBRATED_RELATIVE_DOSE_ONLY",
        "limitations": [
            "Gaussian radial profile is an approximation, not a measured nozzle profile.",
            "Inverse-square and incidence weighting are simplified engineering assumptions.",
            "Occlusion is not evaluated by this function.",
            "Relative flux cannot be converted to film thickness without calibration.",
        ],
    }

def integrate_spray_path_deposition(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    nozzle_points: np.ndarray,
    spray_directions: np.ndarray,
    pose_durations_s: np.ndarray,
    config: SprayDepositionConfig,
) -> dict[str, Any]:
    """Integrate relative deposition over spray-on poses with explicit durations."""
    config.validate()
    nozzles = np.asarray(nozzle_points, dtype=np.float64)
    directions = np.asarray(spray_directions, dtype=np.float64)
    durations = np.asarray(pose_durations_s, dtype=np.float64)
    if nozzles.ndim != 2 or nozzles.shape[1] != 3 or len(nozzles) == 0:
        raise ValueError("nozzle_points must be a non-empty (M, 3) array.")
    if directions.shape != nozzles.shape:
        raise ValueError("spray_directions must match nozzle_points.")
    if durations.shape != (len(nozzles),):
        raise ValueError("pose_durations_s must contain one duration per pose.")
    if not np.all(np.isfinite(nozzles)) or not np.all(np.isfinite(directions)):
        raise ValueError("Path poses must be finite.")
    if not np.all(np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("Pose durations must be finite and non-negative.")
    accumulated = np.zeros(len(surface_points), dtype=np.float64)
    pose_reports = []
    for index in range(len(nozzles)):
        pose = evaluate_pose_deposition(
            surface_points, surface_normals, nozzles[index],
            directions[index], config,
        )
        contribution = pose["relative_flux"] * durations[index]
        accumulated += contribution
        pose_reports.append({
            "pose_index": int(index),
            "duration_s": float(durations[index]),
            "covered_sample_count": int(np.count_nonzero(pose["inside_cone"])),
            "positive_deposition_sample_count": int(np.count_nonzero(contribution > 0.0)),
        })
    positive = accumulated[accumulated > 0.0]
    mean_positive = float(np.mean(positive)) if len(positive) else 0.0
    return {
        "relative_dose": accumulated,
        "sample_count": int(len(accumulated)),
        "covered_sample_count": int(np.count_nonzero(accumulated > 0.0)),
        "uncovered_sample_count": int(np.count_nonzero(accumulated <= 0.0)),
        "relative_dose_statistics": {
            "minimum_positive": float(np.min(positive)) if len(positive) else None,
            "median_positive": float(np.median(positive)) if len(positive) else None,
            "maximum": float(np.max(accumulated)) if len(accumulated) else 0.0,
            "coefficient_of_variation": (
                float(np.std(positive) / mean_positive)
                if len(positive) and mean_positive > _EPS else None
            ),
        },
        "pose_reports": pose_reports,
        "model": "GAUSSIAN_RADIAL_CONE_INVERSE_SQUARE_INCIDENCE",
        "calibration_status": "UNCALIBRATED_RELATIVE_DOSE_ONLY",
        "physical_film_thickness_available": False,
    }
