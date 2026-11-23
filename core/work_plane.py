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


def spray_direction_from_surface_normal(surface_normal, yaw_deg=0.0, pitch_deg=0.0):
    """Return a spray axis aimed into a surface, normal by default.

    The surface normal points outward, so the unmodified spray direction is
    its negative. Optional yaw/pitch offsets preserve the manual angle-control
    behavior while making zero offset exactly perpendicular to the surface.
    """
    normal = np.asarray(surface_normal, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("Surface normal must be a finite 3D vector.")
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValueError("Surface normal must be non-zero.")
    axis = -normal / length
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    if not np.isfinite(yaw) or not np.isfinite(pitch):
        raise ValueError("Spray angle offsets must be finite.")
    if abs(yaw) <= 1e-12 and abs(pitch) <= 1e-12:
        return axis

    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, axis))) > 0.92:
        reference = np.asarray([0.0, 1.0, 0.0])
    yaw_axis = np.cross(reference, axis)
    yaw_axis /= max(float(np.linalg.norm(yaw_axis)), 1e-12)
    pitch_axis = np.cross(axis, yaw_axis)
    pitch_axis /= max(float(np.linalg.norm(pitch_axis)), 1e-12)

    def rotate(vector, rotation_axis, angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        return (
            vector * cosine
            + np.cross(rotation_axis, vector) * sine
            + rotation_axis * float(np.dot(rotation_axis, vector)) * (1.0 - cosine)
        )

    yawed = rotate(axis, yaw_axis, yaw)
    yawed_pitch_axis = rotate(pitch_axis, yaw_axis, yaw)
    result = rotate(yawed, yawed_pitch_axis, pitch)
    result_length = float(np.linalg.norm(result))
    if not np.isfinite(result_length) or result_length <= 1e-12:
        raise ValueError("Could not construct a valid spray direction.")
    return result / result_length
