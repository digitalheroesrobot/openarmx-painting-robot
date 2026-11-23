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


def test_default_spray_direction_is_perpendicular_to_surface():
    from core.work_plane import spray_direction_from_surface_normal

    normal = np.asarray([1.0, 2.0, 3.0])
    direction = spray_direction_from_surface_normal(normal)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    np.testing.assert_allclose(direction, -normal / np.linalg.norm(normal))
    np.testing.assert_allclose(
        np.dot(-direction, normal / np.linalg.norm(normal)), 1.0
    )


def test_surface_normal_spray_direction_supports_manual_angle_offsets():
    from core.work_plane import spray_direction_from_surface_normal

    direction = spray_direction_from_surface_normal(
        [0.0, 0.0, 1.0], yaw_deg=15.0, pitch_deg=-10.0
    )
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert np.dot(direction, [0.0, 0.0, -1.0]) < 0.999


def test_surface_normal_spray_direction_rejects_invalid_normal():
    from core.work_plane import spray_direction_from_surface_normal

    with pytest.raises(ValueError):
        spray_direction_from_surface_normal([0.0, 0.0, 0.0])


def test_surface_following_nozzle_keeps_exact_standoff():
    from core.work_plane import nozzle_position_from_surface

    surface = np.asarray([12.0, -4.0, 7.0])
    nozzle = nozzle_position_from_surface(surface, [0.0, 0.0, 1.0], 25.0)
    np.testing.assert_allclose(nozzle, [12.0, -4.0, -18.0])
    assert np.linalg.norm(surface - nozzle) == pytest.approx(25.0)


def test_surface_following_nozzle_normalizes_spray_direction():
    from core.work_plane import nozzle_position_from_surface

    surface = np.asarray([1.0, 2.0, 3.0])
    nozzle = nozzle_position_from_surface(surface, [0.0, 0.0, -4.0], 10.0)
    np.testing.assert_allclose(nozzle, [1.0, 2.0, 13.0])


def test_surface_following_nozzle_rejects_invalid_inputs():
    from core.work_plane import nozzle_position_from_surface

    with pytest.raises(ValueError):
        nozzle_position_from_surface([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 10.0)
    with pytest.raises(ValueError):
        nozzle_position_from_surface([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0)
