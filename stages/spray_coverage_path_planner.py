"""Patch-local raster path generation for spray coverage planning.

This module is an isolated planning milestone. It partitions paintable mesh
faces into geometrically coherent connected components and rasterizes each
component independently. It deliberately does not modify Stage 3 integration.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import trimesh

from stages.stage3_spray_trajectory import (
    _generate_planar_raster_for_patch,
    _generate_surface_following_raster_for_patch,
)


def connected_paintable_components(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    patch_ids: np.ndarray,
    face_normals: np.ndarray,
    max_normal_angle_deg: float = 35.0,
) -> list[np.ndarray]:
    """Split faces by mesh adjacency, patch identity, and normal continuity."""
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh.")
    selected = np.unique(np.asarray(face_indices, dtype=np.int64))
    patches = np.asarray(patch_ids, dtype=np.int64)
    normals = np.asarray(face_normals, dtype=np.float64)
    if patches.shape != (len(mesh.faces),):
        raise ValueError("patch_ids must contain one value per mesh face.")
    if normals.shape != (len(mesh.faces), 3) or not np.all(np.isfinite(normals)):
        raise ValueError("face_normals must be a finite (F, 3) array.")
    if len(selected) == 0:
        return []
    if np.any(selected < 0) or np.any(selected >= len(mesh.faces)):
        raise ValueError("face_indices contains an invalid face index.")
    if not np.isfinite(max_normal_angle_deg) or not 0.0 <= max_normal_angle_deg <= 180.0:
        raise ValueError("max_normal_angle_deg must be in [0, 180].")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths[selected] <= 1e-12):
        raise ValueError("Selected faces must have non-zero normals.")
    unit = normals / np.maximum(lengths[:, None], 1e-12)
    selected_set = set(int(i) for i in selected)
    adjacency: dict[int, list[int]] = {int(i): [] for i in selected}
    for a, b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        a, b = int(a), int(b)
        if a not in selected_set or b not in selected_set:
            continue
        if patches[a] != patches[b]:
            continue
        cosine = float(np.clip(np.dot(unit[a], unit[b]), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        if angle <= max_normal_angle_deg:
            adjacency[a].append(b)
            adjacency[b].append(a)
    unseen = set(selected_set)
    components = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        queue = deque([root])
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def generate_patchwise_raster_candidates(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    patch_ids: np.ndarray,
    face_normals: np.ndarray,
    standoff: float,
    sample_spacing: float,
    track_spacing: float,
    max_normal_angle_deg: float = 35.0,
    planarity_angle_deg: float = 2.0,
) -> dict[str, Any]:
    """Rasterize each coherent component independently; never spray-connect gaps."""
    values = (standoff, sample_spacing, track_spacing)
    if not all(np.isfinite(v) for v in values) or standoff <= 0 or sample_spacing <= 0 or track_spacing <= 0:
        raise ValueError("Standoff and both spacing values must be finite and positive.")
    components = connected_paintable_components(
        mesh, face_indices, patch_ids, face_normals, max_normal_angle_deg
    )
    fields = (
        "surface_points", "surface_normals", "nozzle_points",
        "spray_directions", "nozzle_frames", "face_indices", "segment_ids",
    )
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in fields}
    component_records = []
    segment_offset = 0
    for component_index, component_faces in enumerate(components):
        raster = _generate_planar_raster_for_patch(
            mesh, component_faces, face_normals, standoff,
            sample_spacing, track_spacing, planarity_angle_deg,
        )
        raster_kind = "planar"
        diagnostics: dict[str, Any] = {}
        if raster is None:
            raster = _generate_surface_following_raster_for_patch(
                mesh, component_faces, face_normals, standoff,
                sample_spacing, track_spacing, diagnostics,
            )
            raster_kind = "curved_surface"
        if raster is None:
            component_records.append({
                "component_id": int(component_index),
                "source_face_indices": component_faces.tolist(),
                "status": "RASTER_REJECTED",
                "diagnostic": diagnostics,
            })
            continue
        local_segments = np.asarray(raster["segment_ids"], dtype=np.int64)
        for key in fields:
            value = np.asarray(raster[key])
            if key == "segment_ids":
                value = value + segment_offset
            chunks[key].append(value)
        segment_count = int(len(np.unique(local_segments)))
        component_records.append({
            "component_id": int(component_index),
            "source_face_indices": component_faces.tolist(),
            "status": "RASTERIZED",
            "raster_type": raster_kind,
            "sample_count": int(len(raster["surface_points"])),
            "segment_count": segment_count,
            "lane_count": int(raster.get("lane_count", segment_count)),
            "source_patch_ids": np.unique(
                np.asarray(patch_ids, dtype=np.int64)[component_faces]
            ).tolist(),
            "diagnostic": diagnostics,
        })
        segment_offset += segment_count
    result = {
        key: (np.concatenate(values, axis=0) if values else
              np.empty((0, 3, 3) if key == "nozzle_frames" else
                       (0, 3) if key in ("surface_points", "surface_normals",
                                         "nozzle_points", "spray_directions") else (0,),
                       dtype=np.float64 if key != "face_indices" and key != "segment_ids" else np.int64))
        for key, values in chunks.items()
    }
    result.update({
        "components": component_records,
        "component_count": int(len(components)),
        "rasterized_component_count": int(sum(
            item["status"] == "RASTERIZED" for item in component_records
        )),
        "rejected_component_count": int(sum(
            item["status"] != "RASTERIZED" for item in component_records
        )),
        "segment_count": int(len(np.unique(result["segment_ids"]))) if len(result["segment_ids"]) else 0,
        "planner": "CONNECTED_PATCH_LOCAL_RASTER_V1",
        "spray_on_connectors_between_components": False,
        "global_pca_ordering_used": False,
        "deposition_optimization_status": "NOT_YET_APPLIED",
    })
    return result
