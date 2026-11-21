"""Stage 2 target-mask and sampled-area coverage utilities.

These functions deliberately describe geometric sample reach, not deposited
paint thickness. They are dependency-light so the numerical contract can be
tested without importing the Qt/PyVista viewer.
"""
from __future__ import annotations

import numpy as np

FACE_EXCLUDED = 0
FACE_PAINTABLE_CANDIDATE = 1
FACE_REQUIRES_INSPECTION = 2
FACE_INVALID = 3


def build_paintable_face_mask(face_status_labels, face_count: int) -> np.ndarray:
    """Return exactly the Stage 2-approved paintable candidate faces."""
    if face_status_labels is None:
        raise ValueError("Stage 2 face-status labels are required for coverage.")
    labels = np.asarray(face_status_labels)
    if labels.ndim != 1 or labels.shape[0] != int(face_count):
        raise ValueError("Stage 2 face-status labels must match the mesh face count.")
    if not np.all(np.isin(labels, (
        FACE_EXCLUDED, FACE_PAINTABLE_CANDIDATE,
        FACE_REQUIRES_INSPECTION, FACE_INVALID,
    ))):
        raise ValueError("Stage 2 face-status labels contain unknown values.")
    return labels == FACE_PAINTABLE_CANDIDATE


def mark_sample_hits(sample_hits, hit_indices, spray_on: bool):
    """Accumulate reached surface samples only while spray is ON."""
    hits = np.asarray(sample_hits)
    if hits.ndim != 1 or hits.dtype != np.bool_:
        raise ValueError("sample_hits must be a one-dimensional boolean array.")
    if not spray_on:
        return hits
    indices = np.asarray(hit_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("hit_indices must be one-dimensional.")
    if len(indices) and (np.any(indices < 0) or np.any(indices >= len(hits))):
        raise IndexError("A spray hit index is outside the surface sample array.")
    hits[indices] = True
    return hits


def calculate_area_weighted_coverage(
    face_areas,
    sample_hits,
    samples_per_face: int,
) -> tuple[np.ndarray, float, float, float]:
    """Return per-face hit fractions, reached area, target area and percent."""
    areas = np.asarray(face_areas, dtype=np.float64)
    hits = np.asarray(sample_hits)
    samples_per_face = int(samples_per_face)
    if areas.ndim != 1 or not np.all(np.isfinite(areas)) or np.any(areas < 0.0):
        raise ValueError("face_areas must be a finite non-negative 1D array.")
    if samples_per_face <= 0:
        raise ValueError("samples_per_face must be positive.")
    if hits.ndim != 1 or hits.dtype != np.bool_:
        raise ValueError("sample_hits must be a one-dimensional boolean array.")
    if len(hits) != len(areas) * samples_per_face:
        raise ValueError("sample_hits length must equal faces * samples_per_face.")

    fractions = hits.reshape(-1, samples_per_face).mean(axis=1)
    target_area = float(np.sum(areas))
    reached_area = float(np.dot(areas, fractions))
    percent = 100.0 * reached_area / target_area if target_area > 1e-12 else 0.0
    return fractions, reached_area, target_area, float(np.clip(percent, 0.0, 100.0))
