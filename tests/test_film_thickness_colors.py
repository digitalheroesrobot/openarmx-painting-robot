import numpy as np
import pytest

from ui.viewer_3d import map_film_thickness_to_rgb


def test_film_thickness_uses_requested_discrete_color_bands():
    target = 100.0
    values = np.array([0.0, 24.9, 25.0, 49.9, 50.0, 74.9, 75.0, 99.9, 100.0, 100.1])
    colors = map_film_thickness_to_rgb(values, target)

    expected = np.array([
        [168, 168, 168],
        [168, 168, 168],
        [45, 105, 220],
        [45, 105, 220],
        [0, 190, 230],
        [0, 190, 230],
        [0, 190, 110],
        [0, 190, 110],
        [255, 205, 45],
        [235, 55, 45],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(colors, expected)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_film_thickness_color_mapping_rejects_invalid_target(target):
    with pytest.raises(ValueError):
        map_film_thickness_to_rgb(np.array([0.0, 10.0]), target)


def test_film_thickness_color_mapping_rejects_non_finite_values():
    with pytest.raises(ValueError):
        map_film_thickness_to_rgb(np.array([0.0, np.nan]), 100.0)
