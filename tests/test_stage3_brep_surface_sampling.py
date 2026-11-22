import numpy as np
import trimesh

from core import step_loader
from stages import stage3_spray_trajectory as stage3


def test_step_stage3_sampling_uses_exact_brep_points_and_normals(monkeypatch):
    mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        faces=np.asarray([[0, 1, 2]]),
        process=False,
    )
    mesh._step_brep_context = {"faces": (object(),)}

    monkeypatch.setattr(
        step_loader,
        "get_step_brep_context",
        lambda value: value._step_brep_context,
    )

    def fake_exact_sample(value, triangle_index, barycentric):
        weights = np.asarray(barycentric, dtype=np.float64)
        mesh_point = weights @ value.vertices[value.faces[triangle_index]]
        # A deterministic curved-surface offset and normal distinguish this
        # from the planar display triangle while preserving sample identity.
        return {
            "point_mm": mesh_point + np.asarray([0.0, 0.0, 0.25]),
            "normal": np.asarray([0.0, 0.0, 1.0]),
        }

    monkeypatch.setattr(
        step_loader, "evaluate_step_triangle_sample", fake_exact_sample
    )

    sampled = stage3._sample_candidate_faces(
        mesh=mesh,
        face_status_labels=np.asarray(
            [stage3.FACE_PAINTABLE_CANDIDATE], dtype=np.int64
        ),
        patch_labels=np.asarray([7], dtype=np.int64),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        spacing_coordinate_units=2.0,
    )

    assert sampled["surface_geometry_source"] == "ORIGINAL_STEP_BREP_SURFACE"
    assert np.allclose(sampled["surface_points"][:, 2], 0.25)
    assert np.allclose(
        sampled["surface_normals"],
        np.tile([0.0, 0.0, 1.0], (len(sampled["surface_points"]), 1)),
    )
    assert np.all(sampled["face_indices"] == 0)
    assert np.all(sampled["patch_ids"] == 7)


def test_stl_stage3_sampling_keeps_triangle_fallback():
    mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        faces=np.asarray([[0, 1, 2]]),
        process=False,
    )
    sampled = stage3._sample_candidate_faces(
        mesh=mesh,
        face_status_labels=np.asarray(
            [stage3.FACE_PAINTABLE_CANDIDATE], dtype=np.int64
        ),
        patch_labels=np.asarray([3], dtype=np.int64),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        spacing_coordinate_units=2.0,
    )
    assert sampled["surface_geometry_source"] == "DISPLAY_MESH_TRIANGLES"
    assert np.allclose(sampled["surface_points"][:, 2], 0.0)
