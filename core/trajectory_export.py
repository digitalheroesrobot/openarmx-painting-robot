"""Robot-timed export of product-frame spray poses."""

import numpy as np


def build_product_pose_trajectory(
    records,
    robot_settings,
    mm_per_unit,
    spray_on_delay_s=0.0,
    spray_off_delay_s=0.0,
):
    """Build [t, x, y, z, qx, qy, qz, qw] using path distance and robot speed.

    The spatial records retain their spray-overlap-derived spacing. Robot
    painting speed assigns execution time. Controller frequency is retained
    as downstream interpolation metadata and never changes spatial spacing.
    """
    if len(records) < 2:
        raise ValueError("At least two recorded nozzle poses are required.")
    speed = float(robot_settings.get("base_travel_speed_mm_s", 100.0))
    frequency = float(robot_settings.get("robot_control_frequency_hz", 100.0))
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("Robot painting speed must be positive.")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("Robot control frequency must be positive.")
    mm_per_unit = float(mm_per_unit)
    if not np.isfinite(mm_per_unit) or mm_per_unit <= 0.0:
        raise ValueError("Product coordinate unit conversion is unavailable.")

    points = np.asarray(
        [item["nozzle_point_product"] for item in records], dtype=np.float64
    )
    quaternions = np.asarray(
        [item["quaternion_product_xyzw"] for item in records], dtype=np.float64
    )
    stroke_ids = np.asarray(
        [item["stroke_id"] for item in records], dtype=np.int64
    )
    if (
        points.shape != (len(records), 3)
        or quaternions.shape != (len(records), 4)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(quaternions))
    ):
        raise ValueError("Recorded product-frame poses contain invalid values.")
    q_norms = np.linalg.norm(quaternions, axis=1)
    if np.any(q_norms <= 1e-12):
        raise ValueError("A recorded nozzle quaternion has zero length.")
    quaternions = quaternions / q_norms[:, None]

    times = np.zeros(len(records), dtype=np.float64)
    segment_spray_on = []
    transitions = []
    spray_on_delay_s = max(float(spray_on_delay_s), 0.0)
    spray_off_delay_s = max(float(spray_off_delay_s), 0.0)
    for index in range(1, len(records)):
        distance_mm = float(
            np.linalg.norm(points[index] - points[index - 1]) * mm_per_unit
        )
        same_stroke = bool(stroke_ids[index] == stroke_ids[index - 1])
        duration = distance_mm / speed
        if not same_stroke:
            duration += spray_off_delay_s + spray_on_delay_s
            transitions.append({
                "from_row": index - 1,
                "to_row": index,
                "from_stroke_id": int(stroke_ids[index - 1]),
                "to_stroke_id": int(stroke_ids[index]),
                "spray_off_travel_distance_mm": distance_mm,
                "spray_off_delay_s": spray_off_delay_s,
                "spray_on_delay_s": spray_on_delay_s,
            })
        times[index] = times[index - 1] + duration
        segment_spray_on.append(same_stroke)

    trajectory = np.column_stack((times, points, quaternions))
    metadata = {
        "schema": "OPENARMX_PRODUCT_FRAME_POSE_TRAJECTORY_V1",
        "columns": ["t", "x", "y", "z", "qx", "qy", "qz", "qw"],
        "coordinate_frame": "PRODUCT_CAD",
        "position_unit": "source CAD coordinate unit",
        "millimetres_per_position_unit": mm_per_unit,
        "quaternion_order": "XYZW",
        "time_unit": "seconds",
        "time_origin": "first optimized route pose",
        "time_definition": (
            "Cumulative nozzle path distance divided by robot painting "
            "speed. The same configured speed is used for inter-stroke "
            "OFF connectors until a separate travel speed is introduced. "
            "Configured spray on/off delays are included. Human mouse "
            "recording time is not used."
        ),
        "robot_profile": dict(robot_settings),
        "robot_painting_speed_mm_s": speed,
        "off_connector_speed_mm_s": speed,
        "robot_control_frequency_hz": frequency,
        "controller_sample_period_s": 1.0 / frequency,
        "controller_frequency_role": (
            "Downstream controller interpolation/streaming rate only; "
            "it does not alter overlap-derived spatial waypoint spacing."
        ),
        "spatial_waypoint_spacing_role": (
            "Source poses retain the spray footprint and overlap-derived spacing."
        ),
        "row_count": int(len(trajectory)),
        "stroke_ids_per_row": stroke_ids.tolist(),
        "segment_spray_on": segment_spray_on,
        "transitions": transitions,
        "duration_s": float(times[-1]),
    }
    return trajectory, metadata
