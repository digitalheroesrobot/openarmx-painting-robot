import numpy as np

from core.coating_deposition import calculate_cone_deposition_step


def _run(nozzle_x, points, dt=0.1, speed=100.0):
    points = np.asarray(points, dtype=float)
    return calculate_cone_deposition_step(
        surface_points_mm=points,
        surface_normals=np.tile([-1.0, 0.0, 0.0], (len(points), 1)),
        sample_area_mm2=np.full(len(points), 100.0),
        nozzle_position_mm=np.array([nozzle_x, 0.0, 0.0]),
        spray_direction=np.array([1.0, 0.0, 0.0]),
        cone_angle_deg=45.0,
        duration_s=dt,
        fluid_flow_ml_min=600.0,
        transfer_efficiency_percent=100.0,
        volume_solids_percent=100.0,
        number_of_coats=1,
        travel_speed_mm_s=speed,
        radial_alpha=2.0,
        radial_beta=2.0,
        core_deposition_fraction=0.88,
        overspray_tail_fraction=0.12,
        overspray_tail_sigma_fraction=0.35,
    )


def test_overspray_tail_deposits_beyond_nominal_cone():
    result = _run(
        0.0,
        [
            [100.0, 0.0, 0.0],
            [100.0, 60.0, 0.0],
        ],
    )

    assert bool(result["inside_cone"][0])
    assert not bool(result["inside_cone"][1])
    assert bool(result["overspray_outside_cone"][1])
    assert result["deposited_volume_mm3"][1] > 0.0
    assert result["deposited_volume_mm3"][0] > result["deposited_volume_mm3"][1]


def test_larger_standoff_spreads_the_same_flow_over_a_larger_area():
    near = _run(0.0, [[100.0, 0.0, 0.0]], dt=0.1)
    far = _run(0.0, [[200.0, 0.0, 0.0]], dt=0.1)

    near_thickness = float(near["deposited_volume_mm3"][0]) / 100.0
    far_thickness = float(far["deposited_volume_mm3"][0]) / 100.0

    assert far["footprint_radius_mm"][0] > near["footprint_radius_mm"][0]
    assert far_thickness < near_thickness


def test_shorter_interval_for_higher_travel_speed_deposits_less_paint():
    slow = _run(0.0, [[100.0, 0.0, 0.0]], dt=0.20, speed=100.0)
    fast = _run(0.0, [[100.0, 0.0, 0.0]], dt=0.10, speed=200.0)

    assert float(fast["deposited_volume_mm3"][0]) < float(
        slow["deposited_volume_mm3"][0]
    )


def test_back_facing_surface_receives_no_deposition():
    result = calculate_cone_deposition_step(
        surface_points_mm=np.array([[100.0, 0.0, 0.0]]),
        surface_normals=np.array([[-1.0, 0.0, 0.0]]),
        sample_area_mm2=np.array([100.0]),
        nozzle_position_mm=np.array([0.0, 0.0, 0.0]),
        spray_direction=np.array([1.0, 0.0, 0.0]),
        cone_angle_deg=45.0,
        duration_s=0.1,
        fluid_flow_ml_min=600.0,
        transfer_efficiency_percent=100.0,
        volume_solids_percent=100.0,
    )

    assert float(result["deposited_volume_mm3"][0]) == 0.0


def test_overspray_tail_does_not_extend_behind_nozzle():
    result = _run(
        0.0,
        [
            [-100.0, 0.0, 0.0],
            [-100.0, 60.0, 0.0],
        ],
    )

    assert not bool(result["inside_cone"][0])
    assert not bool(result["overspray_outside_cone"][0])
    assert float(result["deposited_volume_mm3"][0]) == 0.0
    assert float(result["deposited_volume_mm3"][1]) == 0.0
