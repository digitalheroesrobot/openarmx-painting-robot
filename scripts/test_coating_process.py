#!/usr/bin/env python3
"""Numerical tests for coating process parameters and film-volume conservation."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.coating_process import (
    DEFAULT_COATING_PROCESS, coating_model_readiness,
    load_coating_process, save_coating_process,
    uniform_dry_film_increment_um, validate_coating_process,
)


class CoatingProcessTests(unittest.TestCase):
    def test_default_process_is_complete_but_explicitly_uncalibrated(self):
        status, missing = coating_model_readiness(DEFAULT_COATING_PROCESS)
        self.assertEqual(status, "INPUTS_READY_UNCALIBRATED")
        self.assertEqual(missing, [])
        self.assertEqual(DEFAULT_COATING_PROCESS["target_dry_film_thickness_um"], 50.0)
        self.assertEqual(DEFAULT_COATING_PROCESS["fluid_flow_ml_min"], 300.0)
        self.assertEqual(DEFAULT_COATING_PROCESS["transfer_efficiency_percent"], 50.0)
        self.assertEqual(DEFAULT_COATING_PROCESS["volume_solids_percent"], 50.0)
        self.assertEqual(DEFAULT_COATING_PROCESS["base_travel_speed_mm_s"], 500.0)

    def test_uniform_film_volume_conservation_reference(self):
        # 60 ml/min for 1 second, 50% transfer, 40% solids, over 100,000 mm².
        # Dry volume = 60/60 * 1000 * .5 * .4 = 200 mm³.
        # Thickness = 200/100000 mm = 2 µm.
        result = uniform_dry_film_increment_um(60.0, 1.0, 0.5, 0.4, 100000.0)
        self.assertAlmostEqual(result, 2.0, places=12)

    def test_double_flow_doubles_uniform_thickness(self):
        a = uniform_dry_film_increment_um(30, 2, 0.5, 0.4, 10000)
        b = uniform_dry_film_increment_um(60, 2, 0.5, 0.4, 10000)
        self.assertAlmostEqual(b, 2 * a)

    def test_double_duration_doubles_uniform_thickness(self):
        a = uniform_dry_film_increment_um(30, 1, 0.5, 0.4, 10000)
        b = uniform_dry_film_increment_um(30, 2, 0.5, 0.4, 10000)
        self.assertAlmostEqual(b, 2 * a)

    def test_invalid_efficiency_and_area_are_rejected(self):
        with self.assertRaises(ValueError):
            uniform_dry_film_increment_um(10, 1, 1.1, 0.5, 100)
        with self.assertRaises(ValueError):
            uniform_dry_film_increment_um(10, 1, 0.5, 0.5, 0)

    def test_minimum_speed_cannot_exceed_base_speed(self):
        values = dict(DEFAULT_COATING_PROCESS)
        values["min_travel_speed_mm_s"] = 200
        with self.assertRaises(ValueError):
            validate_coating_process(values)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coating_process.json"
            values = dict(DEFAULT_COATING_PROCESS)
            values.update(fluid_flow_ml_min=240.0, transfer_efficiency_percent=65.0, volume_solids_percent=42.0)
            save_coating_process(path, values)
            loaded = load_coating_process(path)
            self.assertEqual(loaded["fluid_flow_ml_min"], 240.0)
            self.assertEqual(loaded["number_of_coats"], 1)

    def test_legacy_sprayer_settings_are_migrated_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coating_process.json"
            legacy = {"travel_speed_mm_s": 125.0, "min_travel_speed_mm_s": 35.0, "max_travel_acceleration_mm_s2": 400.0, "fluid_flow_ml_min": 180.0}
            loaded = load_coating_process(path, legacy)
            self.assertEqual(loaded["base_travel_speed_mm_s"], 125.0)
            self.assertEqual(loaded["fluid_flow_ml_min"], 180.0)
            # Existing process file is authoritative on subsequent loads.
            again = load_coating_process(path, {"travel_speed_mm_s": 900.0})
            self.assertEqual(again["base_travel_speed_mm_s"], 125.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
