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
