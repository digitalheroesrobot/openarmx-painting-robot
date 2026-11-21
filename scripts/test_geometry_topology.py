#!/usr/bin/env python3
"""Regression tests for geometry topology construction."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.geometry_analyzer import _build_topology_mesh


class TopologyMeshTests(unittest.TestCase):
    def test_exact_duplicate_vertices_are_welded_without_reordering_faces(self):
        vertices = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        ])
        faces = np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64)
        source = SimpleNamespace(vertices=vertices.copy(), faces=faces.copy(), metadata={"tag": "test"})
        topology, info = _build_topology_mesh(source)
        self.assertEqual(len(topology.faces), 2)
        self.assertEqual(info["exact_duplicate_vertices_merged"], 3)
        self.assertTrue(info["face_order_preserved"])
        self.assertFalse(info["source_mesh_modified"])
        np.testing.assert_array_equal(source.vertices, vertices)
        np.testing.assert_array_equal(source.faces, faces)
        self.assertEqual(topology.metadata["tag"], "test")

    def test_invalid_face_indices_receive_private_dummy_vertices(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 1, 9], [-1, 1, 2]], dtype=np.int64)
        source = SimpleNamespace(vertices=vertices, faces=faces, metadata={})
        topology, info = _build_topology_mesh(source)
        self.assertEqual(info["invalid_source_faces"], 2)
        self.assertEqual(len(topology.faces), 3)
        valid_vertices = set(topology.faces[0].tolist())
        for invalid_face in topology.faces[1:]:
            self.assertEqual(len(set(invalid_face.tolist())), 3)
            self.assertTrue(valid_vertices.isdisjoint(invalid_face.tolist()))
        self.assertFalse(np.any(topology.face_adjacency[:, 0] == 1))
        self.assertFalse(np.any(topology.face_adjacency[:, 1] == 1))

    def test_nonfinite_vertex_faces_are_isolated(self):
        vertices = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [np.nan, 0., 0.],
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        source = SimpleNamespace(vertices=vertices, faces=faces, metadata={})
        topology, info = _build_topology_mesh(source)
        self.assertEqual(info["invalid_source_faces"], 1)
        self.assertEqual(len(topology.faces), 2)
        self.assertEqual(len(topology.face_adjacency), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
