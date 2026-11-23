import numpy as np
import pytest

from core.work_plane import intersect_ray_with_plane


def test_different_cursor_rays_project_onto_same_fixed_work_plane():
    plane_origin = np.asarray([0.0, 0.0, 10.0])
    plane_normal = np.asarray([0.0, 0.0, 1.0])
    first = intersect_ray_with_plane(
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], plane_origin, plane_normal
    )
    second = intersect_ray_with_plane(
        [3.0, -2.0, 0.0], [0.0, 0.0, 1.0], plane_origin, plane_normal
    )
    np.testing.assert_allclose(first, [0.0, 0.0, 10.0])
    np.testing.assert_allclose(second, [3.0, -2.0, 10.0])
    assert first[2] == pytest.approx(second[2])


def test_parallel_cursor_ray_has_no_work_plane_intersection():
    assert intersect_ray_with_plane(
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 0.0, 10.0], [0.0, 0.0, 1.0],
    ) is None


def test_plane_behind_cursor_ray_is_rejected():
    assert intersect_ray_with_plane(
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        [0.0, 0.0, -10.0], [0.0, 0.0, 1.0],
    ) is None


def test_invalid_ray_vectors_are_rejected():
    with pytest.raises(ValueError):
        intersect_ray_with_plane(
            [0.0, 0.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, 10.0], [0.0, 0.0, 1.0],
        )
