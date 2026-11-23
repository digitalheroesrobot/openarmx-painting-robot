import numpy as np

from core.trajectory_export import build_product_pose_trajectory


def _records():
    q = np.asarray([0.0, 0.0, 0.0, 1.0])
    return [
        {
            "nozzle_point_product": np.asarray([0.0, 0.0, 0.0]),
            "quaternion_product_xyzw": q.copy(),
            "stroke_id": 0,
        },
        {
            "nozzle_point_product": np.asarray([10.0, 0.0, 0.0]),
            "quaternion_product_xyzw": q.copy(),
            "stroke_id": 0,
        },
        {
            "nozzle_point_product": np.asarray([30.0, 0.0, 0.0]),
            "quaternion_product_xyzw": q.copy(),
            "stroke_id": 1,
        },
    ]


def test_export_time_is_cumulative_distance_over_robot_speed():
    trajectory, metadata = build_product_pose_trajectory(
        _records(),
        {
            "base_travel_speed_mm_s": 100.0,
            "robot_control_frequency_hz": 50.0,
            "name": "TestRobot_Product",
        },
        mm_per_unit=1.0,
        spray_on_delay_s=0.02,
        spray_off_delay_s=0.03,
    )
    assert trajectory.shape == (5, 9)
    np.testing.assert_allclose(
        trajectory[:, 0], [0.0, 0.1, 0.13, 0.33, 0.35]
    )
    np.testing.assert_allclose(trajectory[:, 1:4], [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
    ])
    np.testing.assert_array_equal(trajectory[:, 8], [1, 1, 0, 0, 1])
    assert metadata["columns"] == [
        "t", "x", "y", "z", "qx", "qy", "qz", "qw", "on_off"
    ]
    assert metadata["segment_spray_on"] == [True, False]
    assert np.isclose(metadata["duration_s"], 0.35)
    assert metadata["quaternion_order"] == "XYZW"
    assert metadata["transitions"][0]["off_state_row"] == 2
    assert metadata["transitions"][0]["off_travel_destination_row"] == 3
    assert metadata["transitions"][0]["on_state_row"] == 4


def test_controller_frequency_does_not_change_spatial_waypoints_or_time():
    settings = {"base_travel_speed_mm_s": 100.0}
    slow_rate, slow_meta = build_product_pose_trajectory(
        _records(), {**settings, "robot_control_frequency_hz": 50.0}, 1.0
    )
    fast_rate, fast_meta = build_product_pose_trajectory(
        _records(), {**settings, "robot_control_frequency_hz": 200.0}, 1.0
    )
    np.testing.assert_allclose(slow_rate, fast_rate)
    assert slow_meta["controller_sample_period_s"] == 0.02
    assert fast_meta["controller_sample_period_s"] == 0.005


def test_export_time_is_monotonic_and_ends_at_last_pose():
    trajectory, metadata = build_product_pose_trajectory(
        _records(),
        {"base_travel_speed_mm_s": 250.0, "robot_control_frequency_hz": 100.0},
        mm_per_unit=0.001,
    )
    assert np.all(np.diff(trajectory[:, 0]) >= 0.0)
    assert trajectory[0, 0] == 0.0
    assert trajectory[-1, 0] == metadata["duration_s"]
    assert trajectory.shape[1] == 9
    assert np.all(np.isin(trajectory[:, 8], [0.0, 1.0]))
