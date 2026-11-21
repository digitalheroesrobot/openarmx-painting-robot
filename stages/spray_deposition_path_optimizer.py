"""Patch-local spray path candidates ranked by relative deposition.

All scores are geometric/relative-dose proxies. The evaluator does not claim
physical film thickness and does not yet perform spray occlusion or robot IK.
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
    """Voronoi/trapezoidal time weights whose sum equals stroke length / speed."""
    durations = np.zeros(len(nozzle_points), dtype=np.float64)
    for segment in np.unique(segment_ids):
        ids = np.flatnonzero(segment_ids == segment)
        if len(ids) < 2:
            continue
        steps = np.linalg.norm(np.diff(nozzle_points[ids], axis=0), axis=1)
        durations[ids[0]] = 0.5 * steps[0] / speed
        durations[ids[-1]] = 0.5 * steps[-1] / speed
        if len(ids) > 2:
            durations[ids[1:-1]] = (
                0.5 * (steps[:-1] + steps[1:]) / speed
            )
    return durations


def _trajectory_distance_metrics(
    nozzle_points: np.ndarray,
    segment_ids: np.ndarray,
    travel_speed: float,
) -> dict[str, float]:
    """Measure spray strokes plus ordered spray-off transitions.

    Segment order is preserved from the candidate. This measures transition
    cost but does not solve the separate stroke-ordering problem.
    """
    unique_segments = np.unique(segment_ids)
    stroke_length = 0.0
    transition_length = 0.0
    segment_ends: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in unique_segments:
        ids = np.flatnonzero(segment_ids == segment)
        if len(ids) == 0:
            continue
        if len(ids) > 1:
            stroke_length += float(
                np.sum(np.linalg.norm(np.diff(nozzle_points[ids], axis=0), axis=1))
            )
        segment_ends.append((nozzle_points[ids[0]], nozzle_points[ids[-1]]))
    for previous, following in zip(segment_ends[:-1], segment_ends[1:]):
        transition_length += float(np.linalg.norm(following[0] - previous[1]))
    transition_time = transition_length / travel_speed
    return {
        "spray_path_length_coordinate_units": stroke_length,
        "spray_off_transition_length_coordinate_units": transition_length,
        "total_trajectory_length_coordinate_units": stroke_length + transition_length,
        "spray_off_transition_time_s": transition_time,
    }


def evaluate_raster_candidate(
    mesh: trimesh.Trimesh,
    candidate: dict[str, Any],
    deposition_config: SprayDepositionConfig,
    spray_speed: float,
    minimum_relative_dose_fraction: float = 0.15,
    reference_standoff: float | None = None,
    reference_exposure_s: float = 1.0,
    travel_speed: float | None = None,
) -> dict[str, Any]:
    """Evaluate relative dose using a candidate-independent reference threshold.

    The reference dose is the ideal on-axis, normal-incidence flux at
    reference_standoff, integrated for reference_exposure_s:
        D_ref = reference_exposure_s / reference_standoff**2

    For fair comparisons across candidate paths, callers evaluating a search
    must pass the same reference_standoff and reference_exposure_s to every
    candidate. If reference_standoff is omitted, a documented convenience
    estimate is inferred from the candidate's median nozzle-to-target distance.
    """
    if not np.isfinite(spray_speed) or spray_speed <= 0:
        raise ValueError("spray_speed must be finite and positive.")
    if not 0.0 <= minimum_relative_dose_fraction <= 1.0:
        raise ValueError("minimum_relative_dose_fraction must be in [0, 1].")
    if not np.isfinite(reference_exposure_s) or reference_exposure_s <= 0:
        raise ValueError("reference_exposure_s must be finite and positive.")
    if travel_speed is None:
        travel_speed = spray_speed
    if not np.isfinite(travel_speed) or travel_speed <= 0:
        raise ValueError("travel_speed must be finite and positive.")

    faces = np.asarray(candidate["face_indices"], dtype=np.int64)
    nozzles = np.asarray(candidate["nozzle_points"], dtype=np.float64)
    directions = np.asarray(candidate["spray_directions"], dtype=np.float64)
    segments = np.asarray(candidate["segment_ids"], dtype=np.int64)
    if len(faces) == 0:
        return {"valid": False, "reason": "empty_candidate"}
    if not (len(faces) == len(nozzles) == len(directions) == len(segments)):
        raise ValueError("Candidate arrays have inconsistent sample counts.")
    if np.any(faces < 0) or np.any(faces >= len(mesh.faces)):
        raise ValueError("Candidate face_indices contain out-of-range indices.")
    if nozzles.ndim != 2 or nozzles.shape[1] != 3 or not np.all(np.isfinite(nozzles)):
        raise ValueError("nozzle_points must be a finite (N, 3) array.")
    if directions.shape != nozzles.shape or not np.all(np.isfinite(directions)):
        raise ValueError("spray_directions must be finite and match nozzle_points.")
    if segments.ndim != 1:
        raise ValueError("segment_ids must be one-dimensional.")

    # Evaluate the complete intended paintable region, not the entire mesh
    # (which may include fixtures, hidden backsides, or non-paintable faces).
    # Raster candidates retain their full component face membership; manually
    # constructed candidates fall back to their supplied face_indices.
    component_records = candidate.get("components", [])
    if component_records:
        evaluation_faces = np.unique(np.concatenate([
            np.asarray(record.get("source_face_indices", []), dtype=np.int64)
            for record in component_records
            if len(record.get("source_face_indices", [])) > 0
        ])) if any(
            len(record.get("source_face_indices", [])) > 0
            for record in component_records
        ) else np.unique(faces)
    else:
        evaluation_faces = np.unique(faces)
    if len(evaluation_faces) == 0 or np.any(evaluation_faces < 0) or np.any(
        evaluation_faces >= len(mesh.faces)
    ):
        return {"valid": False, "reason": "empty_or_invalid_evaluation_region"}
    probe_points = np.asarray(mesh.triangles_center[evaluation_faces], dtype=np.float64)
    probe_normals = np.asarray(mesh.face_normals[evaluation_faces], dtype=np.float64)
    areas = np.asarray(mesh.area_faces[evaluation_faces], dtype=np.float64)
    valid_area = np.isfinite(areas) & (areas > 0.0)
    if not np.any(valid_area):
        return {"valid": False, "reason": "mesh_has_no_positive_finite_face_area"}

    if reference_standoff is None:
        corresponding_targets = probe_points[faces]
        inferred = np.linalg.norm(nozzles - corresponding_targets, axis=1)
        reference_standoff = float(np.median(inferred))
        threshold_basis = "INFERRED_MEDIAN_CANDIDATE_STANDOFF"
    else:
        threshold_basis = "FIXED_CALLER_SUPPLIED_REFERENCE_STANDOFF"
    if not np.isfinite(reference_standoff) or reference_standoff <= 0:
        raise ValueError("reference_standoff must be finite and positive.")

    durations = _pose_durations(nozzles, segments, spray_speed)
    integrated = integrate_spray_path_deposition(
        probe_points, probe_normals, nozzles, directions, durations,
        deposition_config,
    )
    dose = np.asarray(integrated["relative_dose"], dtype=np.float64)
    reference_dose = reference_exposure_s / (reference_standoff ** 2)
    threshold = reference_dose * minimum_relative_dose_fraction
    positive = dose > 0.0
    covered = (dose >= threshold) & positive & valid_area
    total_area = float(np.sum(areas[valid_area]))
    covered_area = float(np.sum(areas[covered]))
    coverage = covered_area / total_area if total_area > 0.0 else 0.0

    # CV over the entire valid paintable area penalizes both under-dosed and
    # over-dosed regions; a second CV describes variation among covered area.
    valid_dose = dose[valid_area]
    valid_weights = areas[valid_area]
    area_mean = float(np.average(valid_dose, weights=valid_weights))
    area_variance = float(np.average(
        (valid_dose - area_mean) ** 2, weights=valid_weights
    ))
    area_cv = float(np.sqrt(area_variance) / area_mean) if area_mean > 1e-12 else None
    covered_dose = dose[covered]
    covered_mean = (
        float(np.average(covered_dose, weights=areas[covered]))
        if np.any(covered) else 0.0
    )
    covered_variance = (
        float(np.average((covered_dose - covered_mean) ** 2, weights=areas[covered]))
        if np.any(covered) else 0.0
    )
    covered_cv = (
        float(np.sqrt(covered_variance) / covered_mean)
        if covered_mean > 1e-12 else None
    )

    trajectory = _trajectory_distance_metrics(nozzles, segments, travel_speed)
    spray_time = float(np.sum(durations))
    total_time = spray_time + trajectory["spray_off_transition_time_s"]
    return {
        "valid": True,
        "area_weighted_relative_coverage": float(coverage),
        "covered_area_coordinate_units2": covered_area,
        "total_evaluation_area_coordinate_units2": total_area,
        "relative_dose_threshold": float(threshold),
        "reference_relative_dose": float(reference_dose),
        "reference_standoff_coordinate_units": float(reference_standoff),
        "reference_exposure_s": float(reference_exposure_s),
        "coverage_threshold_basis": threshold_basis,
        "area_weighted_dose_coefficient_of_variation": area_cv,
        "covered_dose_coefficient_of_variation": covered_cv,
        "mean_relative_dose_over_evaluation_area": area_mean,
        "maximum_relative_dose": float(np.max(dose)) if len(dose) else 0.0,
        **trajectory,
        "estimated_spray_time_s": spray_time,
        "estimated_total_process_time_s": total_time,
        "sample_count": int(len(nozzles)),
        "segment_count": int(len(np.unique(segments))),
        "relative_dose": dose,
        "calibration_status": "UNCALIBRATED_RELATIVE_DOSE_ONLY",
        "limitations": [
            "The threshold is tied to a fixed ideal reference dose, not candidate maximum.",
            "A fixed reference_standoff must be supplied for fair cross-candidate comparisons.",
            "Dose values are not physical film thickness.",
            "Spray occlusion and surface self-shadowing are not evaluated.",
            "Mesh-face centroids approximate area sampling.",
            "Spray-off transition cost follows generated segment order; stroke order is not optimized.",
            "No robot kinematic, collision, or orientation-transition feasibility is evaluated.",
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
    maximum_area_weighted_dose_cv: float | None = None,
    travel_speed: float | None = None,
    reference_exposure_s: float = 1.0,
) -> dict[str, Any]:
    """Choose the shortest complete trajectory meeting coverage/dose criteria.

    Track spacing and planar-patch lane rotation are evaluated jointly. The
    orientation search rotates the PCA lane axis for planar patches; curved
    fallback paths retain their native direction. Stroke order is unchanged.
    """
    if not np.isfinite(standoff) or standoff <= 0:
        raise ValueError("standoff must be finite and positive.")
    if maximum_area_weighted_dose_cv is not None and (
        not np.isfinite(maximum_area_weighted_dose_cv)
        or maximum_area_weighted_dose_cv < 0
    ):
        raise ValueError("maximum_area_weighted_dose_cv must be finite and non-negative.")
    if not np.isfinite(reference_exposure_s) or reference_exposure_s <= 0:
        raise ValueError("reference_exposure_s must be finite and positive.")
    if travel_speed is None:
        travel_speed = spray_speed
    if not np.isfinite(travel_speed) or travel_speed <= 0:
        raise ValueError("travel_speed must be finite and positive.")

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
                reference_standoff=standoff,
                reference_exposure_s=reference_exposure_s,
                travel_speed=travel_speed,
            )
            coverage_ok = (
                score.get("valid")
                and score["area_weighted_relative_coverage"] >= minimum_area_coverage
            )
            uniformity_ok = (
                maximum_area_weighted_dose_cv is None
                or (
                    score.get("area_weighted_dose_coefficient_of_variation") is not None
                    and score["area_weighted_dose_coefficient_of_variation"]
                    <= maximum_area_weighted_dose_cv
                )
            )
            evaluations.append({
                "track_spacing": spacing,
                "lane_orientation_angle_deg": angle,
                "candidate": candidate,
                "evaluation": score,
                "meets_coverage": bool(coverage_ok),
                "meets_uniformity": bool(uniformity_ok),
                "meets_requirements": bool(coverage_ok and uniformity_ok),
            })

    feasible = [item for item in evaluations if item["meets_requirements"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda item: (
                item["evaluation"]["total_trajectory_length_coordinate_units"],
                item["evaluation"]["estimated_total_process_time_s"],
                item["evaluation"]["area_weighted_dose_coefficient_of_variation"]
                if item["evaluation"]["area_weighted_dose_coefficient_of_variation"] is not None
                else float("inf"),
                item["track_spacing"],
            ),
        )
        selection_status = "SHORTEST_TOTAL_TRAJECTORY_AMONG_FEASIBLE_CANDIDATES"
    else:
        selected = max(
            evaluations,
            key=lambda item: (
                item["evaluation"].get("area_weighted_relative_coverage", 0.0),
                -(
                    item["evaluation"].get(
                        "area_weighted_dose_coefficient_of_variation"
                    )
                    if item["evaluation"].get(
                        "area_weighted_dose_coefficient_of_variation"
                    ) is not None else float("inf")
                ),
                -item["evaluation"].get(
                    "total_trajectory_length_coordinate_units", float("inf")
                ),
            ),
        )
        selection_status = "NO_CANDIDATE_MET_ALL_REQUIREMENTS"

    return {
        "selected_track_spacing": selected["track_spacing"],
        "selected_lane_orientation_angle_deg": selected["lane_orientation_angle_deg"],
        "selected_candidate": selected["candidate"],
        "selected_evaluation": selected["evaluation"],
        "selection_status": selection_status,
        "minimum_area_coverage": float(minimum_area_coverage),
        "maximum_area_weighted_dose_cv": maximum_area_weighted_dose_cv,
        "evaluated_spacing_count": len(spacings),
        "evaluated_orientation_count": len(angles),
        "evaluated_candidate_count": len(evaluations),
        "spacing_evaluations": [
            {key: value for key, value in item.items() if key != "candidate"}
            for item in evaluations
        ],
        "optimizer": "FIXED_REFERENCE_DOSE_TOTAL_TRAJECTORY_SEARCH_V3",
        "calibration_status": "UNCALIBRATED_RELATIVE_DOSE_ONLY",
        "orientation_search_performed": True,
        "orientation_search_scope": "PLANAR_PATCH_PCA_AXIS_ROTATION",
        "route_order_optimization_performed": False,
        "transition_cost_included": True,
    }
