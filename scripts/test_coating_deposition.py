#!/usr/bin/env python3
"""Numerical tests for spatial Double-Beta coating deposition."""
import sys, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from core.coating_deposition import double_beta_spatial_density, simulate_double_beta_deposition
class DoubleBetaDepositionTests(unittest.TestCase):
    def _args(self):
        return {"sample_normalized_xy_by_path": np.zeros((2,2,2)), "sample_area_mm2": np.array([25.,25.]), "sample_facing_cosine_by_path": np.ones((2,2)), "sample_face_ids": np.array([0,1]), "trajectory_time_s": np.array([0.,1.]), "spray_on_mask": np.array([False,True]), "hit_sample_indices_by_path": [np.array([],dtype=int),np.array([0,1])], "fluid_flow_ml_min":60., "transfer_efficiency_percent":50., "volume_solids_percent":40., "footprint_width_mm":10., "footprint_thickness_mm":10.}
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