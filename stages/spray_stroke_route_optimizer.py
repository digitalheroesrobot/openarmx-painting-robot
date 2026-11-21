"""Spray-off stroke routing with endpoint reversal and orientation-aware costs.

This is a standalone geometric routing stage. It does not alter deposition,
coverage, Stage 3, robot kinematics, collision checking, or execution.
"""
from __future__ import annotations

from typing import Any
import numpy as np


_SAMPLE_FIELDS = (
    "surface_points", "surface_normals", "nozzle_points",
    "spray_directions", "nozzle_frames", "face_indices",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Spray direction vectors must be finite and non-zero.")
    return vector / norm


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = _unit(a), _unit(b)
    return float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))


def _stroke_data(candidate: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    nozzles = np.asarray(candidate["nozzle_points"], dtype=np.float64)
    directions = np.asarray(candidate["spray_directions"], dtype=np.float64)
    segments = np.asarray(candidate["segment_ids"], dtype=np.int64)
    if nozzles.ndim != 2 or nozzles.shape[1] != 3 or not np.all(np.isfinite(nozzles)):
        raise ValueError("nozzle_points must be a finite (N, 3) array.")
    if directions.shape != nozzles.shape or not np.all(np.isfinite(directions)):
        raise ValueError("spray_directions must be finite and match nozzle_points.")
    if segments.shape != (len(nozzles),):
        raise ValueError("segment_ids must contain one ID per sample.")
    if len(nozzles) == 0:
        raise ValueError("Cannot route an empty candidate.")
    ids = [np.flatnonzero(segments == sid) for sid in np.unique(segments)]
    if any(len(index) == 0 for index in ids):
        raise ValueError("Each stroke must contain at least one sample.")
    return ids, [nozzles[i] for i in ids], [directions[i] for i in ids]


def _transition_cost(
    end_position: np.ndarray,
    end_direction: np.ndarray,
    start_position: np.ndarray,
    start_direction: np.ndarray,
    orientation_weight: float,
) -> tuple[float, float, float]:
    distance = float(np.linalg.norm(start_position - end_position))
    angle = _angle(end_direction, start_direction)
    return distance + orientation_weight * angle, distance, angle


def _best_orientations_for_order(
    order: list[int],
    positions: list[np.ndarray],
    directions: list[np.ndarray],
    orientation_weight: float,
    start_position: np.ndarray | None,
    start_direction: np.ndarray | None,
) -> tuple[list[bool], float]:
    """Two-state dynamic program: each stroke can run forward or backward."""
    if not order:
        return [], 0.0
    costs = np.full((len(order), 2), np.inf, dtype=np.float64)
    parents = np.full((len(order), 2), -1, dtype=np.int64)

    def endpoints(stroke: int, reverse: int):
        if reverse == 0:
            return positions[stroke][0], directions[stroke][0], positions[stroke][-1], directions[stroke][-1]
        return positions[stroke][-1], directions[stroke][-1], positions[stroke][0], directions[stroke][0]

    first = order[0]
    for state in (0, 1):
        entry, entry_dir, _, _ = endpoints(first, state)
        if start_position is None:
            costs[0, state] = 0.0
        else:
            if start_direction is None:
                costs[0, state] = float(np.linalg.norm(entry - start_position))
            else:
                costs[0, state] = _transition_cost(
                    start_position, start_direction, entry, entry_dir,
                    orientation_weight,
                )[0]

    for index in range(1, len(order)):
        previous, current = order[index - 1], order[index]
        for state in (0, 1):
            entry, entry_dir, _, _ = endpoints(current, state)
            for previous_state in (0, 1):
                _, _, exit_position, exit_dir = endpoints(previous, previous_state)
                transition = _transition_cost(
                    exit_position, exit_dir, entry, entry_dir,
                    orientation_weight,
                )[0]
                total = costs[index - 1, previous_state] + transition
                if total < costs[index, state]:
                    costs[index, state] = total
                    parents[index, state] = previous_state

    state = int(np.argmin(costs[-1]))
    total = float(costs[-1, state])
    states = [False] * len(order)
    for index in range(len(order) - 1, -1, -1):
        states[index] = bool(state)
        if index:
            state = int(parents[index, state])
    return states, total


def _route_cost(
    order: list[int],
    positions: list[np.ndarray],
    directions: list[np.ndarray],
    orientation_weight: float,
    start_position: np.ndarray | None,
    start_direction: np.ndarray | None,
) -> tuple[list[bool], float]:
    return _best_orientations_for_order(
        order, positions, directions, orientation_weight,
        start_position, start_direction,
    )


def optimize_stroke_order(
    candidate: dict[str, Any],
    orientation_weight_coordinate_units_per_rad: float = 0.0,
    start_position: np.ndarray | None = None,
    start_spray_direction: np.ndarray | None = None,
    max_2opt_passes: int = 2,
) -> dict[str, Any]:
    """Reorder independent strokes and choose each stroke's travel direction.

    Strategy:
      1. Multi-start nearest-neighbor construction from each stroke.
      2. Dynamic programming chooses forward/reverse orientation for each
         fixed order, minimizing translation plus weighted spray-axis rotation.
      3. 2-opt reversals improve the stroke order; endpoint orientations are
         re-optimized after each proposed order change.

    The original sample sequence inside every stroke is preserved or reversed;
    no samples are interpolated, deleted, or connected with spray-on motion.
    """
    if not np.isfinite(orientation_weight_coordinate_units_per_rad) or orientation_weight_coordinate_units_per_rad < 0:
        raise ValueError("orientation weight must be finite and non-negative.")
    if not isinstance(max_2opt_passes, int) or max_2opt_passes < 0:
        raise ValueError("max_2opt_passes must be a non-negative integer.")
    ids, positions, directions = _stroke_data(candidate)
    count = len(ids)
    if start_position is not None:
        start_position = np.asarray(start_position, dtype=np.float64)
        if start_position.shape != (3,) or not np.all(np.isfinite(start_position)):
            raise ValueError("start_position must be a finite 3-vector.")
    if start_spray_direction is not None:
        start_spray_direction = _unit(np.asarray(start_spray_direction, dtype=np.float64))
    if start_spray_direction is not None and start_position is None:
        raise ValueError("start_spray_direction requires start_position.")

    weight = float(orientation_weight_coordinate_units_per_rad)

    # Multi-start nearest-neighbor seeds. Evaluate both possible directions
    # for every first stroke to avoid dependence on source face/segment IDs.
    best_order: list[int] | None = None
    best_orientations: list[bool] = []
    best_cost = np.inf
    for first in range(count):
        for first_reverse in (False, True):
            order = [first]
            remaining = set(range(count)) - {first}
            while remaining:
                current = order[-1]
                current_exit = positions[current][0 if first_reverse and len(order) == 1 else -1]
                current_exit_dir = directions[current][0 if first_reverse and len(order) == 1 else -1]
                # The preceding chosen stroke's orientation is tracked explicitly.
                if len(order) > 1:
                    current_reverse = chosen_reversals[-1]
                    current_exit = positions[current][-1 if current_reverse else 0]
                    current_exit_dir = directions[current][-1 if current_reverse else 0]
                choices = []
                for nxt in remaining:
                    for reverse in (False, True):
                        entry = positions[nxt][-1 if reverse else 0]
                        entry_dir = directions[nxt][-1 if reverse else 0]
                        cost = _transition_cost(
                            current_exit, current_exit_dir, entry, entry_dir, weight
                        )[0]
                        choices.append((cost, nxt, reverse))
                _, nxt, reverse = min(choices, key=lambda item: (item[0], item[1], item[2]))
                if len(order) == 1:
                    chosen_reversals = [first_reverse]
                order.append(nxt)
                chosen_reversals.append(reverse)
                remaining.remove(nxt)
            orientations, cost = _route_cost(
                order, positions, directions, weight,
                start_position, start_spray_direction,
            )
            if cost < best_cost:
                best_order, best_orientations, best_cost = order, orientations, cost

    assert best_order is not None
    # 2-opt: reverse any contiguous subsequence of stroke order and re-solve
    # each stroke's endpoint orientation with the two-state dynamic program.
    for _ in range(max_2opt_passes):
        improved = False
        for left in range(count - 1):
            for right in range(left + 1, count):
                trial = best_order[:left] + list(reversed(best_order[left:right + 1])) + best_order[right + 1:]
                trial_orientations, trial_cost = _route_cost(
                    trial, positions, directions, weight,
                    start_position, start_spray_direction,
                )
                if trial_cost + 1e-9 < best_cost:
                    best_order = trial
                    best_orientations = trial_orientations
                    best_cost = trial_cost
                    improved = True
        if not improved:
            break

    ordered_indices = []
    route_records = []
    for route_index, (stroke, reverse) in enumerate(zip(best_order, best_orientations)):
        sample_indices = ids[stroke][::-1] if reverse else ids[stroke]
        ordered_indices.extend(sample_indices.tolist())
        route_records.append({
            "route_index": route_index,
            "source_segment_id": int(np.unique(
                np.asarray(candidate["segment_ids"])[ids[stroke]]
            )[0]),
            "reversed": bool(reverse),
            "sample_count": int(len(sample_indices)),
        })
    ordered_indices_array = np.asarray(ordered_indices, dtype=np.int64)
    output = dict(candidate)
    sample_count = len(np.asarray(candidate["nozzle_points"]))
    for field in _SAMPLE_FIELDS:
        if field in candidate:
            values = np.asarray(candidate[field])
            if len(values) != sample_count:
                raise ValueError(f"{field} must have one row per trajectory sample.")
            output[field] = values[ordered_indices_array].copy()
    output["segment_ids"] = np.concatenate([
        np.full(len(ids[stroke]), route_index, dtype=np.int64)
        for route_index, stroke in enumerate(best_order)
    ])
    output["route_optimization"] = {
        "algorithm": "MULTISTART_NEAREST_NEIGHBOR_DP_ORIENTATION_2OPT",
        "stroke_count": count,
        "route_cost_coordinate_units": float(best_cost),
        "orientation_weight_coordinate_units_per_rad": weight,
        "stroke_order": [int(x) for x in best_order],
        "strokes": route_records,
        "spray_on_connectors_added": False,
        "sample_geometry_modified": False,
        "optimality": "HEURISTIC_LOCAL_SEARCH_NOT_GLOBAL_OPTIMUM",
    }
    return output
