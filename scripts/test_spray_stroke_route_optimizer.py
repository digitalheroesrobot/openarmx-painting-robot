import unittest
import numpy as np

from stages.spray_stroke_route_optimizer import optimize_stroke_order


class SprayStrokeRouteOptimizerTests(unittest.TestCase):
    def _candidate(self):
        # Deliberately poor source order: stroke 0 -> 1 -> 2.
        points = np.array([
            [0., 0., 0.], [10., 0., 0.],
            [100., 0., 0.], [110., 0., 0.],
            [11., 0., 0.], [21., 0., 0.],
        ])
        directions = np.tile([0., 0., -1.], (6, 1))
        return {
            "surface_points": points + np.array([0., 0., -80.]),
            "surface_normals": np.tile([0., 0., 1.], (6, 1)),
            "nozzle_points": points,
            "spray_directions": directions,
            "nozzle_frames": np.tile(np.eye(3), (6, 1, 1)),
            "face_indices": np.array([0, 0, 1, 1, 2, 2]),
            "segment_ids": np.array([0, 0, 1, 1, 2, 2]),
            "components": [{"source_face_indices": [0, 1, 2]}],
        }

    def test_reduces_spray_off_transition_distance(self):
        candidate = self._candidate()
        result = optimize_stroke_order(candidate)
        self.assertLess(
            result["route_optimization"]["route_cost_coordinate_units"],
            200.,
        )
        self.assertEqual(result["route_optimization"]["stroke_count"], 3)
        self.assertEqual(result["segment_ids"].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertFalse(result["route_optimization"]["spray_on_connectors_added"])
        self.assertIs(result["components"], candidate["components"])

    def test_reorders_all_sample_aligned_fields_together(self):
        candidate = self._candidate()
        result = optimize_stroke_order(candidate)
        for segment in np.unique(result["segment_ids"]):
            ids = np.flatnonzero(result["segment_ids"] == segment)
            faces = np.unique(result["face_indices"][ids])
            self.assertEqual(len(faces), 1)
            np.testing.assert_allclose(
                result["nozzle_points"][ids, 2], [0., 0.]
            )
            np.testing.assert_allclose(
                result["surface_points"][ids] - result["nozzle_points"][ids],
                np.tile([0., 0., -80.], (len(ids), 1)),
            )

    def test_reversed_stroke_preserves_pose_and_sample_correspondence(self):
        candidate = self._candidate()
        result = optimize_stroke_order(candidate)
        records = result["route_optimization"]["strokes"]
        for record in records:
            if record["reversed"]:
                route_id = record["route_index"]
                ids = np.flatnonzero(result["segment_ids"] == route_id)
                original_id = record["source_segment_id"]
                source_ids = np.flatnonzero(candidate["segment_ids"] == original_id)
                np.testing.assert_allclose(
                    result["nozzle_points"][ids],
                    candidate["nozzle_points"][source_ids[::-1]],
                )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            optimize_stroke_order(
                self._candidate(),
                orientation_weight_coordinate_units_per_rad=-1.,
            )
        with self.assertRaises(ValueError):
            optimize_stroke_order(
                self._candidate(),
                start_spray_direction=np.array([0., 0., -1.]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
