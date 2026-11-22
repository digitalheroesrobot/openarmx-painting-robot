import numpy as np
import pytest

from ui.viewer_3d import map_film_thickness_to_rgb


def test_film_thickness_uses_continuous_color_ramp():
    target = 100.0
    ratios = np.array([0.0, 0.005, 0.02, 0.05, 0.25, 0.50, 0.75, 1.0, 1.5])
    colors = map_film_thickness_to_rgb(ratios * target, target)

    expected = np.array([
        [168, 168, 168],  # untouched surface
        [145, 155, 170],  # trace wetting
        [105, 130, 175],  # thin initial coat
        [45, 105, 220],   # visible initial coat
        [0, 190, 230],
        [0, 190, 110],
        [125, 205, 55],
        [255, 205, 45],   # target thickness
        [235, 55, 45],    # excessive film
    ], dtype=np.uint8)
    np.testing.assert_array_equal(colors, expected)


def test_first_spray_pass_is_visibly_different_from_unpainted_surface():
    target = 50.0
    values = np.array([0.0, 0.25, 1.0, 2.5, 12.5])
    colors = map_film_thickness_to_rgb(values, target)

    assert np.array_equal(colors[0], [168, 168, 168])
    assert not np.array_equal(colors[1], colors[0])
    assert not np.array_equal(colors[2], colors[0])
    assert not np.array_equal(colors[3], colors[0])
    assert np.all(colors[1:] != colors[0])


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_film_thickness_color_mapping_rejects_invalid_target(target):
    with pytest.raises(ValueError):
        map_film_thickness_to_rgb(np.array([0.0, 10.0]), target)


def test_film_thickness_color_mapping_rejects_non_finite_values():
    with pytest.raises(ValueError):
        map_film_thickness_to_rgb(np.array([0.0, np.nan]), 100.0)
