"""Continuous per-face coating fields in original STEP UV coordinates.

The UV grid is a rendering/field representation. Spray physics remains
computed by the existing deposition kernel; its per-triangle increments are
projected into this face-domain grid with normalized Gaussian interpolation.
"""
from __future__ import annotations

import math
import numpy as np


DEFAULT_UV_TEXTURE_RESOLUTION = 256


def face_uv_layout(face_record, uv_triangles):
    """Return UV bounds and periodic-axis metadata for one original STEP Face."""
    uv = np.asarray(uv_triangles, dtype=np.float64)
    if uv.ndim != 3 or uv.shape[1:] != (3, 2) or not np.isfinite(uv).all():
        raise ValueError("uv_triangles must be a finite (N, 3, 2) array.")
    parameter_bounds = dict(face_record.get("parameter_bounds") or {})
    flattened = uv.reshape(-1, 2)
    bounds = []
    for axis, name in enumerate(("u", "v")):
        interval = parameter_bounds.get(name)
        if interval is None:
            low = float(np.min(flattened[:, axis]))
            high = float(np.max(flattened[:, axis]))
        else:
            low, high = map(float, interval)
        if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1e-12:
            raise ValueError(f"Face UV {name} bounds must have positive finite span.")
        bounds.append((low, high))
    surface_type = str(face_record.get("surface_type", "")).upper()
    periodic_u = surface_type in {"CYLINDER", "CONE", "SPHERE", "TORUS"}
    periodic_v = surface_type == "TORUS"
    return {
        "uv_bounds": bounds,
        "periodic_u": periodic_u,
        "periodic_v": periodic_v,
        "surface_type": surface_type,
    }


def face_uv_triangles_to_texture_coordinates(
    uv_triangles, uv_bounds, periodic_u=False, periodic_v=False
):
    """Normalize per-triangle UV nodes, unwrapping periodic seams locally."""
    uv = np.asarray(uv_triangles, dtype=np.float64).copy()
    bounds = np.asarray(uv_bounds, dtype=np.float64)
    if uv.ndim != 3 or uv.shape[1:] != (3, 2) or bounds.shape != (2, 2):
        raise ValueError("Expected UV triangles (N,3,2) and bounds (2,2).")
    if not np.isfinite(uv).all() or not np.isfinite(bounds).all():
        raise ValueError("UV texture coordinates and bounds must be finite.")
    if np.any(bounds[:, 1] - bounds[:, 0] <= 1e-12):
        raise ValueError("UV bounds must have positive spans.")
    for axis, periodic in enumerate((periodic_u, periodic_v)):
        if not periodic:
            continue
        period = 2.0 * math.pi
        low, high = bounds[axis]
        for triangle in uv:
            reference = float(triangle[0, axis])
            delta = (triangle[:, axis] - reference + 0.5 * period) % period - 0.5 * period
            triangle[:, axis] = reference + delta
            center = float(np.mean(triangle[:, axis]))
            shift = round((0.5 * (low + high) - center) / period) * period
            triangle[:, axis] += shift
    return (uv - bounds[:, 0][None, None, :]) / (
        bounds[:, 1] - bounds[:, 0]
    )[None, None, :]


def surface_metric_to_pixel_covariance(
    surface_metric_mm2_per_uv,
    uv_bounds,
    resolution,
    sample_area_mm2,
    smoothing_factor=0.60,
    min_sigma_px=0.75,
    max_sigma_px=32.0,
):
    """Convert an isotropic physical Gaussian footprint into UV-pixel space.

    surface_metric_mm2_per_uv is the first fundamental form
    [[du·du, du·dv], [du·dv, dv·dv]] evaluated on the original CAD Surface.
    The resulting covariance is generally anisotropic in UV texture pixels,
    which compensates for parameterization scale and skew.
    """
    metric = np.asarray(surface_metric_mm2_per_uv, dtype=np.float64)
    bounds = np.asarray(uv_bounds, dtype=np.float64)
    resolution = int(resolution)
    area = float(sample_area_mm2)
    factor = float(smoothing_factor)
    min_sigma = float(min_sigma_px)
    max_sigma = float(max_sigma_px)
    if metric.shape != (2, 2) or not np.isfinite(metric).all():
        raise ValueError("surface metric must be a finite 2x2 matrix.")
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
        raise ValueError("uv_bounds must be a finite 2x2 array.")
    if resolution < 8 or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Texture resolution and UV bounds are invalid.")
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("sample_area_mm2 must be finite and positive.")
    if (
        not math.isfinite(factor) or factor <= 0.0
        or not math.isfinite(min_sigma) or min_sigma <= 0.0
        or not math.isfinite(max_sigma) or max_sigma < min_sigma
    ):
        raise ValueError("Gaussian smoothing parameters are invalid.")

    metric = 0.5 * (metric + metric.T)
    spans = bounds[:, 1] - bounds[:, 0]
    pixel_to_uv = np.diag(spans / float(resolution - 1))
    metric_per_pixel = pixel_to_uv.T @ metric @ pixel_to_uv
    eigenvalues, eigenvectors = np.linalg.eigh(metric_per_pixel)
    if np.any(eigenvalues <= 1e-14):
        raise ValueError("Surface metric is singular at this UV sample.")

    sigma_mm = factor * math.sqrt(area)
    # Clamp the overall footprint without independently clipping the
    # principal axes. Per-axis clipping distorts the physical isotropy ratio
    # (for example, a 10:1 UV scale ratio can become 7.5:1 at max_sigma_px).
    variances = (sigma_mm * sigma_mm) / eigenvalues
    largest = float(np.max(variances))
    smallest = float(np.min(variances))
    scale = 1.0
    if largest > max_sigma * max_sigma:
        scale = min(scale, (max_sigma * max_sigma) / largest)
    if smallest * scale < min_sigma * min_sigma:
        scale = max(scale, (min_sigma * min_sigma) / smallest)
    variances = variances * scale
    covariance = eigenvectors @ np.diag(variances) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def accumulate_face_uv_thickness(
    thickness_grid_um,
    sample_uv,
    thickness_increment_um,
    uv_bounds,
    sigma_px=3.0,
    periodic_u=False,
    periodic_v=False,
    sample_covariance_px=None,
):
    """Accumulate one deposition interval as a smooth UV-domain thickness field.

    Samples from the same interval are normalized together before being added
    to the persistent field. This avoids multiplying the physical increment
    by the number of tessellation triangles while smoothing triangle-scale
    discontinuities in the rendered result.
    """
    grid = np.asarray(thickness_grid_um)
    samples = np.asarray(sample_uv, dtype=np.float64)
    increments = np.asarray(thickness_increment_um, dtype=np.float64)
    bounds = np.asarray(uv_bounds, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] < 8:
        raise ValueError("thickness_grid_um must be a square grid of at least 8 pixels.")
    if samples.ndim != 2 or samples.shape[1] != 2 or increments.shape != (len(samples),):
        raise ValueError("sample_uv and thickness_increment_um have incompatible shapes.")
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
        raise ValueError("uv_bounds must be a finite (2, 2) array.")
    if not np.isfinite(samples).all() or not np.isfinite(increments).all():
        raise ValueError("UV samples and thickness increments must be finite.")
    if np.any(increments < 0.0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Thickness increments must be nonnegative and UV spans positive.")
    sigma = float(sigma_px)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_px must be finite and positive.")
    if sample_covariance_px is None:
        covariances = np.repeat(
            (np.eye(2, dtype=np.float64) * sigma * sigma)[None, :, :],
            len(samples), axis=0,
        )
    else:
        covariances = np.asarray(sample_covariance_px, dtype=np.float64)
        if covariances.shape != (len(samples), 2, 2) or not np.isfinite(covariances).all():
            raise ValueError("sample_covariance_px must be finite with shape (N,2,2).")
        covariances = 0.5 * (covariances + np.swapaxes(covariances, 1, 2))
        if len(covariances) and np.any(np.linalg.eigvalsh(covariances) <= 0.0):
            raise ValueError("Every sample covariance must be positive definite.")

    size = int(grid.shape[0])
    span = bounds[:, 1] - bounds[:, 0]
    centers = (samples - bounds[:, 0]) / span * (size - 1)
    numerator = np.zeros_like(grid, dtype=np.float64)
    denominator = np.zeros_like(grid, dtype=np.float64)
    periodic = (bool(periodic_u), bool(periodic_v))

    for center, increment, covariance in zip(centers, increments, covariances):
        cx, cy = float(center[0]), float(center[1])
        x_center, y_center = int(round(cx)), int(round(cy))
        radius = max(1, int(math.ceil(3.0 * math.sqrt(
            float(np.max(np.linalg.eigvalsh(covariance)))
        ))))
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        xs = x_center + offsets
        ys = y_center + offsets
        x_valid = np.ones(xs.shape, dtype=bool)
        y_valid = np.ones(ys.shape, dtype=bool)
        if periodic[0]:
            xs = np.mod(xs, size)
        else:
            x_valid = (xs >= 0) & (xs < size)
        if periodic[1]:
            ys = np.mod(ys, size)
        else:
            y_valid = (ys >= 0) & (ys < size)
        if not np.any(x_valid) or not np.any(y_valid):
            continue
        # Distances use unwrapped local offsets; wrapped indices are only
        # addresses into the periodic texture.
        delta_x = offsets.astype(np.float64) + (x_center - cx)
        delta_y = offsets.astype(np.float64) + (y_center - cy)
        dx, dy = np.meshgrid(delta_x, delta_y)
        inverse_covariance = np.linalg.inv(covariance)
        mahalanobis = (
            inverse_covariance[0, 0] * dx * dx
            + 2.0 * inverse_covariance[0, 1] * dx * dy
            + inverse_covariance[1, 1] * dy * dy
        )
        kernel = np.exp(-0.5 * mahalanobis)
        valid_kernel = kernel[np.ix_(y_valid, x_valid)]
        yy = ys[y_valid]
        xx = xs[x_valid]
        numerator[np.ix_(yy, xx)] += valid_kernel * float(increment)
        denominator[np.ix_(yy, xx)] += valid_kernel

    weighted_mean = np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    # Preserve the Gaussian footprint of isolated samples while preventing
    # dense tessellation samples from multiplying the same physical interval.
    confidence = np.minimum(denominator, 1.0)
    event_increment = weighted_mean * confidence
    grid[...] += event_increment.astype(grid.dtype, copy=False)
    return grid


def rasterize_face_uv_grid(
    uv_triangles,
    uv_bounds,
    resolution=DEFAULT_UV_TEXTURE_RESOLUTION,
    periodic_u=False,
    periodic_v=False,
):
    """Rasterize selected source-face UV triangles onto texel centers.

    The returned mask is a conservative domain/visibility mask: only texels
    covered by the supplied triangles are valid. Periodic seam-crossing
    triangles are locally unwrapped and tested against neighboring periods.
    Rows correspond to V and columns correspond to U.
    """
    triangles = np.asarray(uv_triangles, dtype=np.float64)
    bounds = np.asarray(uv_bounds, dtype=np.float64)
    size = int(resolution)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("uv_triangles must have shape (N,3,2).")
    if not np.isfinite(triangles).all():
        raise ValueError("uv_triangles must be finite.")
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
        raise ValueError("uv_bounds must be a finite (2,2) array.")
    if size < 8 or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("UV grid resolution and bounds are invalid.")

    normalized = face_uv_triangles_to_texture_coordinates(
        triangles, bounds, periodic_u=periodic_u, periodic_v=periodic_v
    )
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    uu, vv = np.meshgrid(axis, axis)
    points = np.column_stack((uu.ravel(), vv.ravel()))
    mask = np.zeros(len(points), dtype=bool)
    shifts_u = (-1.0, 0.0, 1.0) if periodic_u else (0.0,)
    shifts_v = (-1.0, 0.0, 1.0) if periodic_v else (0.0,)

    for triangle in normalized:
        a, b, c = triangle
        edge0 = b - a
        edge1 = c - a
        determinant = edge0[0] * edge1[1] - edge1[0] * edge0[1]
        if abs(float(determinant)) <= 1e-14:
            continue
        min_xy = np.minimum(np.min(triangle, axis=0), 1.0)
        max_xy = np.maximum(np.max(triangle, axis=0), 0.0)
        # Restrict work to the texel bounding box, including seam copies.
        for shift_u in shifts_u:
            for shift_v in shifts_v:
                shifted = triangle + np.asarray([shift_u, shift_v])
                low = np.maximum(np.min(shifted, axis=0), 0.0)
                high = np.minimum(np.max(shifted, axis=0), 1.0)
                if np.any(high < low):
                    continue
                x0 = max(0, int(math.floor(low[0] * (size - 1))))
                x1 = min(size - 1, int(math.ceil(high[0] * (size - 1))))
                y0 = max(0, int(math.floor(low[1] * (size - 1))))
                y1 = min(size - 1, int(math.ceil(high[1] * (size - 1))))
                if x1 < x0 or y1 < y0:
                    continue
                local = points.reshape(size, size, 2)[y0:y1 + 1, x0:x1 + 1]
                q = local - shifted[0]
                e0 = shifted[1] - shifted[0]
                e1 = shifted[2] - shifted[0]
                det = e0[0] * e1[1] - e1[0] * e0[1]
                alpha = (q[..., 0] * e1[1] - e1[0] * q[..., 1]) / det
                beta = (e0[0] * q[..., 1] - q[..., 0] * e0[1]) / det
                inside = (
                    (alpha >= -1e-10) & (beta >= -1e-10)
                    & (alpha + beta <= 1.0 + 1e-10)
                )
                view = mask.reshape(size, size)
                view[y0:y1 + 1, x0:x1 + 1] |= inside

    uv_u = bounds[0, 0] + axis * (bounds[0, 1] - bounds[0, 0])
    uv_v = bounds[1, 0] + axis * (bounds[1, 1] - bounds[1, 0])
    grid_u, grid_v = np.meshgrid(uv_u, uv_v)
    return {
        "valid_mask": mask.reshape(size, size),
        "uv_grid": np.stack((grid_u, grid_v), axis=-1),
        "resolution": size,
        "valid_texel_count": int(np.count_nonzero(mask)),
        "rasterization": "SELECTED_TRIANGLE_UV_DOMAIN_MASK",
    }
