"""Geometric helpers for planar spray-path cursor projection."""

import numpy as np


def intersect_ray_with_plane(ray_origin, ray_direction, plane_origin, plane_normal):
    """Return the forward ray/plane intersection, or None when no hit exists."""
    origin = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)
    point = np.asarray(plane_origin, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    if any(value.shape != (3,) for value in (origin, direction, point, normal)):
        raise ValueError("Ray and plane vectors must each have shape (3,).")
    if not all(np.all(np.isfinite(value)) for value in (origin, direction, point, normal)):
        raise ValueError("Ray and plane vectors must be finite.")
    direction_norm = float(np.linalg.norm(direction))
    normal_norm = float(np.linalg.norm(normal))
    if direction_norm <= 1e-12 or normal_norm <= 1e-12:
        raise ValueError("Ray direction and plane normal must be non-zero.")
    direction = direction / direction_norm
    normal = normal / normal_norm
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1e-9:
        return None
    distance = float(np.dot(point - origin, normal) / denominator)
    if not np.isfinite(distance) or distance < 0.0:
        return None
    intersection = origin + distance * direction
    return intersection if np.all(np.isfinite(intersection)) else None
