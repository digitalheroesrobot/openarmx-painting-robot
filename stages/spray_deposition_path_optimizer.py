"""Patch-local spray path candidates ranked by the relative deposition kernel.

Scores are geometric/relative-dose proxies only. No calibrated film thickness,
occlusion, robot IK, collision checking, or robot execution is performed here.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import trimesh

from stages.spray_deposition_model import (
    SprayDepositionConfig,
    integrate_spray_path_deposition,
)
from stages.spray_coverage_path_planner import generate_patchwise_raster_candidates


def _pose_durations(
    nozzle_points: np.ndarray,
    segment_ids: np.ndarray,
    speed: float,
) -> np.ndarray:
    durations = np.zeros(len(nozzle_points), dtype=np.float64)
    for segment in np.unique(segment_ids):
        ids = np.flatnonzero(segment_ids == segment)
        if len(ids) < 2:
            continue
        steps = np.linalg.norm(np.diff(nozzle_points[ids], axis=0), axis=1)
        durations[ids[:-1]] = steps / speed
        durations[ids[-1]] = durations[ids[-2]] if len(ids) > 1 else 0.0
    return durations


def evaluate_raster_candidate(
    mesh: trimesh.Trimesh,
    candidate: dict[str, Any],
    deposition_config: SprayDepositionConfig,
    spray_speed: float,
    minimum_relative_dose_fraction: float = 0.15,
) -> dict[str, Any]:
    """Evaluate accumulated relative dose at area-weighted face-centroid probes."""
    if not np.isfinite(spray_speed) or spray_speed <= 0:
        raise ValueError("spray_speed must be finite and positive.")
    if not 0.0 <= minimum_relative_dose_fraction <= 1.0:
        raise ValueError("minimum_relative_dose_fraction must be in [0, 1].")
    faces = np.asarray(candidate["face_indices"], dtype=np.int64)
    nozzles = np.asarray(candidate["nozzle_points"], dtype=np.float64)
    directions = np.asarray(candidate["spray_directions"], dtype=np.float64)
    segments = np.asarray(candidate["segment_ids"], dtype=np.int64)
    if len(faces) == 0:
        return {"valid": False, "reason": "empty_candidate"}
    if not (len(faces) == len(nozzles) == len(directions) == len(segments)):
        raise ValueError("Candidate arrays have inconsistent sample counts.")
    # Use all source mesh faces as evaluation probes, not only faces hit by path samples.
    probe_points = np.asarray(mesh.triangles_center, dtype=np.float64)
    probe_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    valid_area = np.isfinite(areas) & (areas > 0)
    durations = _pose_durations(nozzles, segments, spray_speed)
    integrated = integrate_spray_path_deposition(
        probe_points, probe_normals, nozzles, directions, durations,
        deposition_config,
    )
    dose = np.asarray(integrated["relative_dose"], dtype=np.float64)
    maximum = float(np.max(dose)) if len(dose) else 0.0
    threshold = maximum * minimum_relative_dose_fraction
    covered = (dose >= threshold) & (dose > 0.0) & valid_area
    total_area = float(np.sum(areas[valid_area]))
    covered_area = float(np.sum(areas[covered]))
    coverage = covered_area / total_area if total_area > 0 else 0.0
    covered_dose = dose[covered]
    mean = float(np.average(covered_dose, weights=areas[covered])) if np.any(covered) else 0.0
    variance = (
        float(np.average((covered_dose - mean) ** 2, weights=areas[covered]))
        if np.any(covered) else 0.0
    )
    cv = float(np.sqrt(variance) / mean) if mean > 1e-12 else None
    path_length = 0.0
    for segment in np.unique(segments):
        ids = np.flatnonzero(segments == segment)
        if len(ids) > 1:
            path_length += float(np.sum(np.linalg.norm(np.diff(nozzles[ids], axis=0), axis=1)))
    return {
        "valid": True,
        "area_weighted_relative_coverage": float(coverage),
        "covered_area_coordinate_units2": covered_area,
        "total_evaluation_area_coordinate_units2": total_area,
        "relative_dose_threshold": float(threshold),
        "covered_dose_coefficient_of_variation": cv,
        "maximum_relative_dose": maximum,
        "spray_path_length_coordinate_units": path_length,
        "estimated_spray_time_s": float(np.sum(durations)),
        "sample_count": int(len(nozzles)),
        "segment_count": int(len(np.unique(segments))),
        "relative_dose": dose,
        "calibration_status": "UNCALIBRATED_RELATIVE_DOSE_ONLY",
        "limitations": [
            "Coverage threshold is relative to this candidate's maximum dose.",
            "Dose values are not physical film thickness.",
            "Spray occlusion and surface self-shadowing are not evaluated.",
            "Mesh-face centroids approximate area sampling.",
            "No robot kinematic, collision, or transition-time feasibility is evaluated.",
        ],
    }


def select_track_spacing_by_relative_deposition(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    patch_ids: np.ndarray,
    face_normals: np.ndarray,
    standoff: float,
    sample_spacing: float,
    track_spacing_candidates: Sequence[float],
    deposition_config: SprayDepositionConfig,
    spray_speed: float,
    minimum_relative_dose_fraction: float = 0.15,
    minimum_area_coverage: float = 0.95,
    lane_orientation_candidates_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
) -> dict[str, Any]:
    """Choose the shortest raster candidate meeting relative area coverage.

    This compares lane spacings for the raster generator's current local sweep
    direction. It is not yet a global orientation or route-order optimizer.
    """
    spacings = sorted(set(float(x) for x in track_spacing_candidates))
    if not spacings or any(not np.isfinite(x) or x <= 0 for x in spacings):
        raise ValueError("track_spacing_candidates must be finite positive values.")
    if not 0.0 <= minimum_area_coverage <= 1.0:
        raise ValueError("minimum_area_coverage must be in [0, 1].")
    angles = sorted(set(float(x) % 180.0 for x in lane_orientation_candidates_deg))
    if not angles or any(not np.isfinite(x) for x in angles):
        raise ValueError("lane_orientation_candidates_deg must contain finite values.")
    evaluations = []
    for angle in angles:
        for spacing in spacings:
            candidate = generate_patchwise_raster_candidates(
                mesh, face_indices, patch_ids, face_normals,
                standoff, sample_spacing, spacing,
                lane_orientation_angle_deg=angle,
            )
            score = evaluate_raster_candidate(
                mesh, candidate, deposition_config, spray_speed,
                minimum_relative_dose_fraction,
            )
            evaluations.append({
                "track_spacing": spacing,
                "lane_orientation_angle_deg": angle,
                "candidate": candidate,
                "evaluation": score,
                "meets_coverage": bool(
                    score.get("valid") and
                    score["area_weighted_relative_coverage"] >= minimum_area_coverage
                ),
            })
    feasible = [item for item in evaluations if item["meets_coverage"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda item: (
                item["evaluation"]["spray_path_length_coordinate_units"],
                item["evaluation"]["estimated_spray_time_s"],
                item["track_spacing"],
            ),
        )
        selection_status = "SHORTEST_PATH_AMONG_COVERAGE_FEASIBLE_SPACINGS"
    else:
        selected = max(
            evaluations,
            key=lambda item: (
                item["evaluation"].get("area_weighted_relative_coverage", 0.0),
                -item["evaluation"].get("spray_path_length_coordinate_units", float("inf")),
            ),
        )
        selection_status = "NO_SPACING_MET_COVERAGE_REQUIREMENT"
    return {
        "selected_track_spacing": selected["track_spacing"],
        "selected_lane_orientation_angle_deg": selected["lane_orientation_angle_deg"],
        "selected_candidate": selected["candidate"],
        "selected_evaluation": selected["evaluation"],
        "selection_status": selection_status,
        "minimum_area_coverage": float(minimum_area_coverage),
        "evaluated_spacing_count": len(spacings),
        "evaluated_orientation_count": len(angles),
        "evaluated_candidate_count": len(evaluations),
        "spacing_evaluations": [
            {key: value for key, value in item.items() if key != "candidate"}
            for item in evaluations
        ],
        "optimizer": "RELATIVE_DEPOSITION_ORIENTATION_SPACING_SEARCH_V2",
        "calibration_status": "UNCALIBRATED_RELATIVE_DOSE_ONLY",
        "orientation_search_performed": True,
        "orientation_search_scope": "PLANAR_PATCH_PCA_AXIS_ROTATION",
        "route_order_optimization_performed": False,
    }
