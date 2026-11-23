import numpy as np

from core.coating_deposition import calculate_cone_deposition_step


def _run(nozzle_x, points, dt=0.1):
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


def test_recorded_interval_duration_controls_deposited_paint():
    short_interval = _run(0.0, [[100.0, 0.0, 0.0]], dt=0.10)
    long_interval = _run(0.0, [[100.0, 0.0, 0.0]], dt=0.20)

    assert float(long_interval["deposited_volume_mm3"][0]) > float(
        short_interval["deposited_volume_mm3"][0]
    )


def test_back_facing_surface_receives_no_deposition():
    result = calculate_cone_deposition_step(
        surface_points_mm=np.array([[100.0, 0.0, 0.0]]),
        surface_normals=np.array([[1.0, 0.0, 0.0]]),
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

def _planar_rectangle_triangles(subdivisions):
    """Triangulate the same X-constant rectangle with different mesh density."""
    triangles = []
    edges = np.linspace(-20.0, 20.0, subdivisions + 1)
    x = 100.0
    for iy in range(subdivisions):
        for iz in range(subdivisions):
            y0, y1 = edges[iy], edges[iy + 1]
            z0, z1 = edges[iz], edges[iz + 1]
            p00 = np.array([x, y0, z0])
            p10 = np.array([x, y1, z0])
            p11 = np.array([x, y1, z1])
            p01 = np.array([x, y0, z1])
            triangles.extend(([p00, p10, p11], [p00, p11, p01]))
    return np.asarray(triangles, dtype=float)


def _deposit_on_triangle_mesh(triangles):
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = np.linalg.norm(cross, axis=1) * 0.5
    normals = np.tile([-1.0, 0.0, 0.0], (len(triangles), 1))
    return calculate_cone_deposition_step(
        surface_points_mm=np.mean(triangles, axis=1),
        sample_vertices_mm=triangles,
        surface_normals=normals,
        sample_area_mm2=areas,
        nozzle_position_mm=np.array([0.0, 0.0, 0.0]),
        spray_direction=np.array([1.0, 0.0, 0.0]),
        cone_angle_deg=60.0,
        duration_s=0.1,
        fluid_flow_ml_min=600.0,
        transfer_efficiency_percent=100.0,
        volume_solids_percent=100.0,
        number_of_coats=1,
    )


def test_triangle_area_quadrature_reduces_mesh_tessellation_sensitivity():
    coarse = _deposit_on_triangle_mesh(
        _planar_rectangle_triangles(subdivisions=1)
    )
    fine = _deposit_on_triangle_mesh(
        _planar_rectangle_triangles(subdivisions=8)
    )

    coarse_total = float(coarse["total_deposited_volume_mm3"])
    fine_total = float(fine["total_deposited_volume_mm3"])
    relative_difference = abs(coarse_total - fine_total) / max(
        coarse_total, fine_total, 1e-12
    )

    assert coarse_total > 0.0
    assert fine_total > 0.0
    assert relative_difference < 0.03

