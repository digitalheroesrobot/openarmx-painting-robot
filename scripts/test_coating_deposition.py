#!/usr/bin/env python3
"""Numerical tests for spatial Double-Beta coating deposition."""
import sys, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from core.coating_deposition import (
    accumulate_double_beta_deposition_step,
    double_beta_spatial_density,
    simulate_double_beta_deposition,
)
class DoubleBetaDepositionTests(unittest.TestCase):
    def test_sparse_step_doubles_with_flow_and_duration(self):
        args = {"normalized_xy": np.array([[0.,0.]]), "sample_area_mm2": np.array([1.]),
                "sample_facing_cosine": np.array([1.]), "sample_footprint_area_mm2": np.array([100.]),
                "duration_s": 1., "fluid_flow_ml_min": 60., "transfer_efficiency_percent": 50.,
                "volume_solids_percent": 40.}
        base = accumulate_double_beta_deposition_step(**args)
        twice_flow = accumulate_double_beta_deposition_step(**{**args, "fluid_flow_ml_min": 120.})
        twice_time = accumulate_double_beta_deposition_step(**{**args, "duration_s": 2.})
        np.testing.assert_allclose(twice_flow["sample_thickness_increment_um"], 2 * base["sample_thickness_increment_um"])
        np.testing.assert_allclose(twice_time["sample_thickness_increment_um"], 2 * base["sample_thickness_increment_um"])

    def test_sparse_step_spray_off_and_empty_hits_deposit_zero(self):
        args = {"normalized_xy": np.empty((0,2)), "sample_area_mm2": np.empty(0),
                "sample_facing_cosine": np.empty(0), "sample_footprint_area_mm2": np.empty(0),
                "duration_s": 1., "fluid_flow_ml_min": 60., "transfer_efficiency_percent": 50.,
                "volume_solids_percent": 40.}
        result = accumulate_double_beta_deposition_step(**args)
        self.assertEqual(result["deposited_volume_mm3"], 0.)
        args["normalized_xy"] = np.zeros((1,2)); args["sample_area_mm2"] = np.ones(1)
        args["sample_facing_cosine"] = np.ones(1); args["sample_footprint_area_mm2"] = np.ones(1)*100
        result = accumulate_double_beta_deposition_step(**args, spray_on=False)
        self.assertEqual(result["input_dry_volume_mm3"], 0.)
        self.assertEqual(result["sample_thickness_increment_um"][0], 0.)

    def test_sparse_step_conserves_supplied_dry_volume(self):
        args = {"normalized_xy": np.zeros((10,2)), "sample_area_mm2": np.ones(10)*100.,
                "sample_facing_cosine": np.ones(10), "sample_footprint_area_mm2": np.ones(10)*10.,
                "duration_s": 1., "fluid_flow_ml_min": 60., "transfer_efficiency_percent": 100.,
                "volume_solids_percent": 100.}
        result = accumulate_double_beta_deposition_step(**args)
        self.assertLessEqual(result["deposited_volume_mm3"], result["input_dry_volume_mm3"] + 1e-12)


    def test_area_weighted_film_statistics_include_unreached_faces(self):
        from core.coating_deposition import calculate_film_thickness_statistics
        stats = calculate_film_thickness_statistics(
            np.array([0.0, 50.0, 100.0]),
            np.array([2.0, 1.0, 1.0]),
            50.0,
        )
        self.assertAlmostEqual(stats["min_um"], 0.0)
        self.assertAlmostEqual(stats["mean_um"], 37.5)
        self.assertAlmostEqual(stats["p95_um"], 100.0)
        self.assertAlmostEqual(stats["max_um"], 100.0)
        self.assertAlmostEqual(stats["area_at_or_above_target_percent"], 50.0)
        self.assertAlmostEqual(stats["area_below_target_percent"], 50.0)

    def test_film_statistics_reject_invalid_geometry_and_values(self):
        from core.coating_deposition import calculate_film_thickness_statistics
        with self.assertRaises(ValueError):
            calculate_film_thickness_statistics(np.array([1.0]), np.array([0.0]), 50.0)
        with self.assertRaises(ValueError):
            calculate_film_thickness_statistics(np.array([-1.0]), np.array([1.0]), 50.0)
        with self.assertRaises(ValueError):
            calculate_film_thickness_statistics(np.array([1.0]), np.array([1.0]), 0.0)

    def test_viewer_module_has_valid_python_syntax(self):
        # Catch syntax errors in the GUI integration without importing Qt/VTK.
        viewer_path = ROOT / "ui" / "viewer_3d.py"
        source = viewer_path.read_text(encoding="utf-8")
        compile(source, str(viewer_path), "exec")

    def test_viewer_film_legend_uses_the_render_palette_and_active_target(self):
        # The legend must document the exact stops used by the 3D interpolation.
        viewer_path = ROOT / "ui" / "viewer_3d.py"
        source = viewer_path.read_text(encoding="utf-8")
        self.assertIn(
            "FILM_THICKNESS_COLOR_STOPS = (0.0, 0.5, 0.8, 1.0, 1.5)",
            source,
        )
        self.assertIn(
            'FILM_THICKNESS_COLOR_HEX = ("#23394E", "#0091D2", "#00BE7D", "#FFCD2D", "#EB372D")',
            source,
        )
        self.assertIn("stops = np.asarray(FILM_THICKNESS_COLOR_STOPS", source)
        self.assertIn("self._update_film_thickness_legend(target)", source)
        self.assertIn(
            'coating_process["target_dry_film_thickness_um"]',
            source,
        )
        self.assertIn("playback_layout.addWidget(self.trajectory_thickness_legend_label)", source)
        self.assertIn("150%+ — OVERSPRAY", source)
        self.assertIn("WHEN FILM THICKNESS IS OFF: COVERAGE COLORS SHOW GEOMETRIC REACH", source)
        self.assertNotIn("GEOMETRIC REACH ONLY - NO FILM THICKNESS", source)

    def _args(self):
        return {"sample_normalized_xy_by_path": np.zeros((2,2,2)), "sample_area_mm2": np.array([10.,10.]), "sample_facing_cosine_by_path": np.ones((2,2)), "sample_face_ids": np.array([0,1]), "trajectory_time_s": np.array([0.,1.]), "spray_on_mask": np.array([False,True]), "hit_sample_indices_by_path": [np.array([],dtype=int),np.array([0,1])], "fluid_flow_ml_min":60., "transfer_efficiency_percent":50., "volume_solids_percent":40., "footprint_width_mm":10., "footprint_thickness_mm":10.}
    def test_uniform_kernel_integrates_to_one(self):
        axis=np.linspace(-1,1,401); xx,yy=np.meshgrid(axis,axis)
        d=double_beta_spatial_density(np.column_stack((xx.ravel(),yy.ravel())),1,1,1,1).reshape(xx.shape)
        self.assertAlmostEqual(float(np.trapezoid(np.trapezoid(d,axis,axis=1),axis)),1.0,places=4)
    def test_symmetric_beta_peaks_at_center_and_clips_outside(self):
        d=double_beta_spatial_density(np.array([[0,0],[.8,0],[0,.8],[1.1,0]]),2,2,2,2)
        self.assertGreater(d[0],d[1]); self.assertGreater(d[0],d[2]); self.assertEqual(d[3],0.)
    def test_double_flow_doubles_film(self):
        a=self._args(); one=simulate_double_beta_deposition(**a); a["fluid_flow_ml_min"]*=2; two=simulate_double_beta_deposition(**a)
        np.testing.assert_allclose(two["sample_thickness_um"],2*one["sample_thickness_um"])
    def test_double_duration_doubles_film(self):
        a=self._args(); one=simulate_double_beta_deposition(**a); a["trajectory_time_s"]=np.array([0.,2.]); two=simulate_double_beta_deposition(**a)
        np.testing.assert_allclose(two["sample_thickness_um"],2*one["sample_thickness_um"])
    def test_spray_off_does_not_deposit(self):
        a=self._args(); a["spray_on_mask"]=np.array([False,False]); r=simulate_double_beta_deposition(**a); self.assertEqual(r["total_deposited_volume_mm3"],0.)
    def test_occluded_samples_are_not_deposited(self):
        a=self._args(); a["hit_sample_indices_by_path"]=[np.array([],int),np.array([0])]; r=simulate_double_beta_deposition(**a)
        self.assertGreater(r["sample_thickness_um"][0],0.); self.assertEqual(r["sample_thickness_um"][1],0.)
    def test_oblique_surface_receives_less(self):
        a=self._args(); a["sample_facing_cosine_by_path"]=np.array([[1.,1.],[1.,.5]]); r=simulate_double_beta_deposition(**a)
        self.assertGreater(r["sample_thickness_um"][0],r["sample_thickness_um"][1])
    def test_two_coats_double_film(self):
        a=self._args(); one=simulate_double_beta_deposition(**a); two=simulate_double_beta_deposition(**a,number_of_coats=2); np.testing.assert_allclose(two["sample_thickness_um"],2*one["sample_thickness_um"])
    def test_uncollected_paint_is_reported_not_redistributed(self):
        r=simulate_double_beta_deposition(**self._args()); self.assertGreater(r["total_input_dry_volume_mm3"],r["total_deposited_volume_mm3"]); self.assertLessEqual(r["deposition_fraction"],1.)
    def test_nonmonotonic_time_is_rejected(self):
        a=self._args(); a["trajectory_time_s"]=np.array([1.,0.]);
        with self.assertRaises(ValueError): simulate_double_beta_deposition(**a)
if __name__ == "__main__": unittest.main(verbosity=2)