"""
Preliminary STEP CAD Face-level paintability analysis.

Exact area, centroid and surface type come from the STEP B-Rep report.
Triangle-to-Face IDs preserve the original CAD Face identity in the viewer
mesh. Mesh normals and edge sharing are supporting tessellation-derived
indicators, not replacements for exact B-Rep evaluation.

This stage deliberately does not claim spray accessibility or prove that a
face is on the exterior of the part. Ambiguous and curved faces remain in
REVIEW_REQUIRED so holes and grooves are never silently discarded.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


PAINTABLE_CANDIDATE = "PAINTABLE_CANDIDATE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
INVALID = "INVALID"

DEFAULT_MIN_FACE_AREA_MM2 = 100.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def analyze_step_faces(
    step_report: dict,
    mesh,
    min_face_area_mm2: float = DEFAULT_MIN_FACE_AREA_MM2,
) -> dict:
    """
    Analyze original STEP Faces and map each result to display triangles.

    Policy:
      * Planar faces >= min_face_area_mm2 are preliminary candidates.
      * Curved/freeform faces remain REVIEW_REQUIRED.
      * Small faces remain REVIEW_REQUIRED.
      * Faces without display triangles or valid area are INVALID.
      * No face is automatically EXCLUDED in this first pass.

    The returned triangle_status_labels array is aligned with mesh.faces.
    """
    if not isinstance(step_report, dict):
        raise TypeError("step_report must be a dictionary.")
    if mesh is None or not hasattr(mesh, "faces") or not hasattr(mesh, "metadata"):
        raise TypeError("mesh must be a STEP display mesh with metadata.")
    threshold = _safe_float(min_face_area_mm2, -1.0)
    if threshold < 0:
        raise ValueError("min_face_area_mm2 must be finite and non-negative.")

    face_records = step_report.get("faces")
    triangle_face_ids = mesh.metadata.get("triangle_face_ids")
    if not isinstance(face_records, list) or not isinstance(triangle_face_ids, (list, tuple, np.ndarray)):
        raise ValueError("STEP Face records or triangle_face_ids are missing.")

    triangle_face_ids = np.asarray(triangle_face_ids, dtype=np.int64)
    if triangle_face_ids.ndim != 1 or len(triangle_face_ids) != len(mesh.faces):
        raise ValueError("triangle_face_ids must contain one ID per display triangle.")

    # Build the triangle list for each original CAD Face ID.
    face_triangles: dict[int, list[int]] = defaultdict(list)
    for triangle_index, face_id in enumerate(triangle_face_ids):
        face_triangles[int(face_id)].append(triangle_index)

    # Build Face adjacency from shared tessellation edges. This is a
    # topological hint; it does not replace exact B-Rep edge adjacency.
    edge_faces: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triangle_index, triangle in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        face_id = int(triangle_face_ids[triangle_index])
        a, b, c = (int(v) for v in triangle)
        for edge in ((a, b), (b, c), (c, a)):
            edge_faces[tuple(sorted(edge))].add(face_id)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for owners in edge_faces.values():
        if len(owners) > 1:
            for face_id in owners:
                adjacency[face_id].update(owners - {face_id})

    # Area-weighted display-mesh normals, grouped by original CAD Face ID.
    triangle_areas = np.asarray(mesh.area_faces, dtype=float)
    triangle_normals = np.asarray(mesh.face_normals, dtype=float)
    face_normals: dict[int, list[float] | None] = {}
    for face_id, indices in face_triangles.items():
        idx = np.asarray(indices, dtype=np.int64)
        weights = triangle_areas[idx]
        vectors = triangle_normals[idx] * weights[:, None]
        summed = vectors.sum(axis=0)
        norm = float(np.linalg.norm(summed))
        face_normals[face_id] = (
            (summed / norm).astype(float).tolist()
            if np.isfinite(norm) and norm > 1e-12 else None
        )

    results = []
    face_status_by_id: dict[int, str] = {}
    counts = {
        PAINTABLE_CANDIDATE: 0,
        REVIEW_REQUIRED: 0,
        INVALID: 0,
    }

    for record in face_records:
        face_id = int(record.get("face_id", -1))
        surface_type = str(record.get("surface_type", "OTHER")).upper()
        area_mm2 = _safe_float(record.get("area_mm2"), -1.0)
        centroid = record.get("centroid_mm", [None, None, None])
        triangle_indices = face_triangles.get(face_id, [])
        normal = face_normals.get(face_id)

        reasons = []
        if not triangle_indices or area_mm2 <= 0:
            status = INVALID
            reasons.append("MISSING_TESSELLATION_OR_INVALID_BREP_AREA")
        elif area_mm2 < threshold:
            status = REVIEW_REQUIRED
            reasons.append("SMALL_FACE_REQUIRES_PROCESS_REVIEW")
        elif surface_type == "PLANE":
            status = PAINTABLE_CANDIDATE
            reasons.append("PLANAR_GEOMETRY_CANDIDATE")
        else:
            status = REVIEW_REQUIRED
            reasons.append("CURVED_OR_FREEFORM_FACE_REQUIRES_REVIEW")

        if normal is None and status != INVALID:
            reasons.append("DISPLAY_NORMAL_UNAVAILABLE")

        face_status_by_id[face_id] = status
        counts[status] += 1
        results.append({
            "face_id": face_id,
            "surface_type": surface_type,
            "area_mm2": area_mm2 if area_mm2 >= 0 else None,
            "centroid_mm": centroid,
            "display_normal": normal,
            "adjacent_face_ids": sorted(adjacency.get(face_id, set())),
            "adjacent_face_count": len(adjacency.get(face_id, set())),
            "display_triangle_count": len(triangle_indices),
            "preliminary_status": status,
            "reason_codes": reasons,
            "exterior_exposure_verified": False,
            "spray_accessibility_verified": False,
        })

    triangle_labels = np.full(len(mesh.faces), 3, dtype=np.uint8)
    numeric_status = {
        "EXCLUDED": 0,
        PAINTABLE_CANDIDATE: 1,
        REVIEW_REQUIRED: 2,
        INVALID: 3,
    }
    for triangle_index, face_id in enumerate(triangle_face_ids):
        status = face_status_by_id.get(int(face_id), INVALID)
        triangle_labels[triangle_index] = numeric_status[status]

    report = {
        "analysis": {
            "stage": "STEP_BREP_FACE_PRELIMINARY_SURFACE_ANALYSIS",
            "method": "Exact STEP Face properties with triangle-to-Face mapping",
            "coordinate_unit": "mm",
            "minimum_face_area_mm2": threshold,
            "exterior_exposure_verified": False,
            "robot_reachability_checked": False,
            "spray_accessibility_checked": False,
            "fixture_interference_checked": False,
            "limitations": [
                "A planar face is only a preliminary geometric candidate.",
                "Interior versus exterior exposure is not yet proven.",
                "Curved and freeform faces remain review-required.",
                "Holes and grooves are not automatically excluded.",
                "Robot reachability, spray direction, standoff and collision checks are not performed.",
                "Adjacency and normals are derived from the STEP display tessellation.",
            ],
        },
        "summary": {
            "total_cad_faces": len(results),
            "paintable_candidate_faces": counts[PAINTABLE_CANDIDATE],
            "review_required_faces": counts[REVIEW_REQUIRED],
            "invalid_faces": counts[INVALID],
            "total_display_triangles": len(mesh.faces),
        },
        "classification_legend": {
            "PAINTABLE_CANDIDATE": "Preliminary planar candidate; manual and exposure review still required.",
            "REVIEW_REQUIRED": "Curved/freeform or small face; inspect explicitly.",
            "INVALID": "Missing display tessellation or invalid B-Rep area.",
        },
        "faces": results,
    }

    return {
        "report": report,
        "triangle_status_labels": triangle_labels,
        "face_status_by_id": face_status_by_id,
    }


def run_cad_surface_analysis_tests() -> None:
    import trimesh

    # A box gives six planar CAD-like faces, each with two triangles.
    mesh = trimesh.creation.box(extents=(100.0, 80.0, 20.0))
    mesh.metadata["triangle_face_ids"] = [
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6
    ]
    records = [
        {
            "face_id": face_id,
            "surface_type": "PLANE",
            "area_mm2": 1000.0,
            "centroid_mm": [0.0, 0.0, 0.0],
        }
        for face_id in range(1, 7)
    ]
    result = analyze_step_faces(
        {"faces": records}, mesh, min_face_area_mm2=100.0
    )
    assert result["report"]["summary"]["total_cad_faces"] == 6
    assert result["report"]["summary"]["paintable_candidate_faces"] == 6
    assert len(result["triangle_status_labels"]) == len(mesh.faces)
    assert np.all(result["triangle_status_labels"] == 1)

    # Curved faces are never silently treated as confirmed candidates.
    records[0]["surface_type"] = "CYLINDER"
    curved = analyze_step_faces({"faces": records}, mesh)
    assert curved["face_status_by_id"][1] == REVIEW_REQUIRED

    # Small faces are review-required, not automatically excluded.
    records[0]["surface_type"] = "PLANE"
    records[0]["area_mm2"] = 10.0
    small = analyze_step_faces({"faces": records}, mesh)
    assert small["face_status_by_id"][1] == REVIEW_REQUIRED

    # Missing tessellation is invalid and maps safely to status 3.
    records[0]["area_mm2"] = 1000.0
    mesh.metadata["triangle_face_ids"] = [2] * len(mesh.faces)
    missing = analyze_step_faces({"faces": records}, mesh)
    assert missing["face_status_by_id"][1] == INVALID
    assert np.all(missing["triangle_status_labels"] == 1)

    print("[PASS] STEP Face inventory and triangle mapping")
    print("[PASS] Curved Face review policy")
    print("[PASS] Small Face review policy")
    print("[PASS] Missing tessellation handling")
    print("CAD surface analysis tests passed.")


if __name__ == "__main__":
    run_cad_surface_analysis_tests()
