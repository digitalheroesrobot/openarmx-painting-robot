import numpy as np

from core.uv_coating_texture import (
    accumulate_face_uv_thickness,
    face_uv_layout,
    face_uv_triangles_to_texture_coordinates,
)


def test_face_uv_layout_uses_original_face_parameter_bounds():
    record = {
        "surface_type": "PLANE",
        "parameter_bounds": {"u": [2.0, 6.0], "v": [-1.0, 3.0]},
    }
    triangles = np.asarray([[[2.0, -1.0], [6.0, -1.0], [2.0, 3.0]]])
    layout = face_uv_layout(record, triangles)
    assert layout["uv_bounds"] == [(2.0, 6.0), (-1.0, 3.0)]
    assert not layout["periodic_u"]
    assert not layout["periodic_v"]


def test_cylindrical_uv_texture_coordinates_unwrap_seam_crossing_triangle():
    record = {
        "surface_type": "CYLINDER",
        "parameter_bounds": {
            "u": [0.0, 2.0 * np.pi],
            "v": [0.0, 10.0],
        },
    }
    triangles = np.asarray([[
        [6.20, 1.0],
        [0.05, 1.0],
        [6.25, 2.0],
    ]])
    layout = face_uv_layout(record, triangles)
    coordinates = face_uv_triangles_to_texture_coordinates(
        triangles,
        layout["uv_bounds"],
        periodic_u=layout["periodic_u"],
        periodic_v=layout["periodic_v"],
    )
    # The short seam-crossing edge must remain short in texture space;
    # repeating texture coordinates may legitimately extend beyond [0, 1].
    assert abs(coordinates[0, 1, 0] - coordinates[0, 0, 0]) < 0.05
    assert layout["periodic_u"]


def test_uv_deposition_creates_smooth_continuous_field_not_triangle_cells():
    grid = np.zeros((64, 64), dtype=np.float32)
    accumulate_face_uv_thickness(
        grid,
        sample_uv=np.asarray([[0.5, 0.5]]),
        thickness_increment_um=np.asarray([20.0]),
        uv_bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        sigma_px=3.0,
    )
    assert grid[32, 32] > 19.0
    assert 0.0 < grid[32, 35] < grid[32, 32]
    assert grid[32, 40] < grid[32, 35]
    assert np.count_nonzero(grid) > 100


def test_uv_deposition_accumulates_repeated_passes_and_wraps_cylinder_seam():
    grid = np.zeros((64, 64), dtype=np.float32)
    kwargs = {
        "uv_bounds": np.asarray([[0.0, 2.0 * np.pi], [0.0, 1.0]]),
        "sigma_px": 2.0,
        "periodic_u": True,
    }
    accumulate_face_uv_thickness(
        grid, np.asarray([[0.01, 0.5]]), np.asarray([10.0]), **kwargs
    )
    first = grid.copy()
    accumulate_face_uv_thickness(
        grid, np.asarray([[0.01, 0.5]]), np.asarray([10.0]), **kwargs
    )
    assert np.isclose(grid[32, 0], 2.0 * first[32, 0], rtol=1e-5)
    assert grid[32, -1] > 0.0


def test_surface_metric_maps_physical_isotropic_blur_to_anisotropic_uv_pixels():
    from core.uv_coating_texture import surface_metric_to_pixel_covariance

    # The same UV distance represents 10x more physical distance along U.
    metric = np.asarray([[100.0, 0.0], [0.0, 1.0]])
    covariance = surface_metric_to_pixel_covariance(
        metric,
        uv_bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        resolution=101,
        sample_area_mm2=0.5,
        smoothing_factor=0.60,
    )
    sigma_u, sigma_v = np.sqrt(np.diag(covariance))
    assert np.isclose(sigma_v / sigma_u, 10.0, rtol=1e-8)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_surface_metric_rejects_singular_surface_parameterization():
    import pytest
    from core.uv_coating_texture import surface_metric_to_pixel_covariance

    with pytest.raises(ValueError, match="singular"):
        surface_metric_to_pixel_covariance(
            np.asarray([[1.0, 0.0], [0.0, 0.0]]),
            uv_bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            resolution=64,
            sample_area_mm2=1.0,
        )


def test_metric_covariance_produces_physical_space_anisotropic_smoothing():
    grid = np.zeros((128, 128), dtype=np.float32)
    accumulate_face_uv_thickness(
        grid,
        sample_uv=np.asarray([[0.5, 0.5]]),
        thickness_increment_um=np.asarray([20.0]),
        uv_bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        sample_covariance_px=np.asarray([[[4.0, 0.0], [0.0, 100.0]]]),
    )
    center = 64
    x_offset = float(grid[center, center + 4])
    y_offset = float(grid[center + 4, center])
    assert y_offset > x_offset
    assert grid[center, center] > 19.0


def test_uv_blur_width_scales_with_local_surface_sample_area():
    from core.uv_coating_texture import surface_metric_to_pixel_covariance

    metric = np.eye(2, dtype=np.float64)
    common = {
        "surface_metric_mm2_per_uv": metric,
        "uv_bounds": np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        "resolution": 128,
        "smoothing_factor": 0.60,
    }
    small = surface_metric_to_pixel_covariance(
        **common, sample_area_mm2=0.01
    )
    large = surface_metric_to_pixel_covariance(
        **common, sample_area_mm2=0.04
    )
    assert np.allclose(large, 4.0 * small, rtol=1e-8, atol=1e-10)
