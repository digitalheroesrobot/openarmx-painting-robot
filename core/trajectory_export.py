"""Robot-timed export of product-frame spray poses."""

import numpy as np


def build_product_pose_trajectory(
    records,
    robot_settings,
    mm_per_unit,
    spray_on_delay_s=0.0,
    spray_off_delay_s=0.0,
):
    """Build [t, x, y, z, qx, qy, qz, qw, on_off] product-frame rows.

    The spatial records retain their spray-overlap-derived spacing. Robot
    painting speed assigns execution time. Controller frequency is retained
    as downstream interpolation metadata and never changes spatial spacing.

    At a stroke boundary, explicit rows switch spray OFF at the previous
    endpoint, remain OFF during inter-stroke travel, then switch ON at the
    next stroke's first pose. These state-transition rows may share a
    timestamp when the corresponding configured delay is zero.
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

    on_delay = float(spray_on_delay_s)
    off_delay = float(spray_off_delay_s)
    if not np.isfinite(on_delay) or not np.isfinite(off_delay):
        raise ValueError("Spray on/off delays must be finite.")
    on_delay = max(on_delay, 0.0)
    off_delay = max(off_delay, 0.0)

    # Each row is a controller pose plus the spray state commanded at that
    # pose. Add duplicate-pose rows at stroke boundaries to encode state
    # changes without losing the OFF travel connector.
    rows = []
    row_stroke_ids = []
    transitions = []
    current_time = 0.0

    def append_row(timestamp, point, quaternion, spray_on, stroke_id):
        rows.append([
            float(timestamp),
            *point.tolist(),
            *quaternion.tolist(),
            1 if spray_on else 0,
        ])
        row_stroke_ids.append(int(stroke_id))

    append_row(
        current_time, points[0], quaternions[0], True, stroke_ids[0]
    )
    segment_spray_on = []

    for index in range(1, len(records)):
        distance_mm = float(
            np.linalg.norm(points[index] - points[index - 1]) * mm_per_unit
        )
        same_stroke = bool(stroke_ids[index] == stroke_ids[index - 1])
        travel_duration = distance_mm / speed

        if same_stroke:
            current_time += travel_duration
            append_row(
                current_time, points[index], quaternions[index],
                True, stroke_ids[index],
            )
            segment_spray_on.append(True)
            continue

        # Stop spraying at the previous stroke endpoint before OFF travel.
        current_time += off_delay
        append_row(
            current_time, points[index - 1], quaternions[index - 1],
            False, stroke_ids[index - 1],
        )
        off_row = len(rows) - 1

        # Travel to the next stroke start with the spray explicitly OFF.
        current_time += travel_duration
        append_row(
            current_time, points[index], quaternions[index],
            False, stroke_ids[index],
        )
        travel_row = len(rows) - 1

        # Resume spraying at the next stroke start after the configured delay.
        current_time += on_delay
        append_row(
            current_time, points[index], quaternions[index],
            True, stroke_ids[index],
        )
        on_row = len(rows) - 1

        transitions.append({
            "from_stroke_id": int(stroke_ids[index - 1]),
            "to_stroke_id": int(stroke_ids[index]),
            "off_state_row": off_row,
            "off_travel_destination_row": travel_row,
            "on_state_row": on_row,
            "off_travel_distance_mm": distance_mm,
            "spray_off_delay_s": off_delay,
            "spray_on_delay_s": on_delay,
        })
        segment_spray_on.append(False)

    trajectory = np.asarray(rows, dtype=np.float64)
    metadata = {
        "schema": "OPENARMX_PRODUCT_FRAME_POSE_TRAJECTORY_V2",
        "columns": [
            "t", "x", "y", "z", "qx", "qy", "qz", "qw", "on_off"
        ],
        "coordinate_frame": "PRODUCT_CAD",
        "position_unit": "source CAD coordinate unit",
        "millimetres_per_position_unit": mm_per_unit,
        "quaternion_order": "XYZW",
        "time_unit": "seconds",
        "time_origin": "first optimized route pose",
        "time_definition": (
            "Cumulative nozzle path distance divided by robot painting "
            "speed. The same configured speed is used for inter-stroke OFF "
            "connectors until a separate travel speed is introduced. "
            "Configured spray on/off delays are included. Human mouse "
            "recording time is not used."
        ),
        "on_off_encoding": {
            "1": "SPRAY_ON",
            "0": "SPRAY_OFF",
            "row_semantics": (
                "Spray state commanded at that pose. Duplicate poses at "
                "stroke boundaries encode OFF and ON state transitions."
            ),
        },
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
        "stroke_ids_per_row": row_stroke_ids,
        "segment_spray_on": segment_spray_on,
        "transitions": transitions,
        "duration_s": float(trajectory[-1, 0]),
    }
    return trajectory, metadata
