#!/usr/bin/env python3
"""Integration checks for VTK BSP-tree spray line-of-sight queries."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkFiltersFlowPaths import vtkModifiedBSPTree

from core.spray_coverage import is_line_of_sight_occluded


def build_triangle_plane(z: float) -> pv.PolyData:
    points = np.asarray([
        [-2.0, -2.0, z],
        [2.0, -2.0, z],
        [0.0, 2.0, z],
    ], dtype=np.float64)
    faces = np.asarray([3, 0, 1, 2], dtype=np.int64)
    return pv.PolyData(points, faces)


def intersect(locator, origin, target):
    points = vtkPoints()
    cell_ids = vtkIdList()
    locator.IntersectWithLine(
        tuple(float(v) for v in origin),
        tuple(float(v) for v in target),
        points,
        cell_ids,
    )
    return np.asarray([
        points.GetPoint(i) for i in range(points.GetNumberOfPoints())
    ], dtype=np.float64).reshape(-1, 3)


def main() -> None:
    origin = np.asarray([0.0, 0.0, 0.0])
    target = np.asarray([0.0, 0.0, 10.0])
    tolerance = 1e-7

    # An open triangle at z=4 must occlude the target at z=10.
    blocker = build_triangle_plane(4.0)
    locator = vtkModifiedBSPTree()
    locator.SetDataSet(blocker)
    locator.BuildLocator()
    hits = intersect(locator, origin, target)
    assert len(hits) >= 1, "BSP tree failed to find the open-triangle intersection"
    assert is_line_of_sight_occluded(hits, origin, target, tolerance)
    
    # A triangle only at the target endpoint is not an earlier occluder.
    endpoint = build_triangle_plane(10.0)
    endpoint_locator = vtkModifiedBSPTree()
    endpoint_locator.SetDataSet(endpoint)
    endpoint_locator.BuildLocator()
    endpoint_hits = intersect(endpoint_locator, origin, target)
    assert not is_line_of_sight_occluded(
        endpoint_hits, origin, target, tolerance
    )

    # An off-axis triangle must not intersect the nozzle-to-target segment.
    off_axis = build_triangle_plane(4.0)
    off_axis.points += np.asarray([5.0, 0.0, 0.0])
    off_axis_locator = vtkModifiedBSPTree()
    off_axis_locator.SetDataSet(off_axis)
    off_axis_locator.BuildLocator()
    off_axis_hits = intersect(off_axis_locator, origin, target)
    assert len(off_axis_hits) == 0
    assert not is_line_of_sight_occluded(
        off_axis_hits, origin, target, tolerance
    )

    print("VTK SPRAY OCCLUSION INTEGRATION: 3/3 PASS")


if __name__ == "__main__":
    main()
