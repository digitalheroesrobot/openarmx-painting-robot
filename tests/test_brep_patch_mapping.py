import numpy as np
import trimesh

from core.step_loader import build_step_patch_face_mapping


def _mesh_with_step_context():
    mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [2.0, 0.0, 0.0],
        ]),
        faces=np.asarray([[0, 1, 2], [1, 3, 2], [1, 4, 3]]),
        process=False,
    )
    mesh._step_brep_context = {
        "shape": object(),
        "faces": (object(), object()),
        "face_records": ({}, {}),
        "face_adjacency": {},
        "triangle_face_ids": np.asarray([1, 1, 2], dtype=np.int64),
        "triangle_uvs": np.zeros((3, 3, 2), dtype=np.float64),
        "triangle_uv_valid": np.ones(3, dtype=bool),
        "schema_identifier": "TEST",
    }
    return mesh


def test_patch_to_brep_face_mapping_is_many_to_many_and_lossless():
    mesh = _mesh_with_step_context()
    patches = np.asarray([10, 20, 20], dtype=np.int64)
    statuses = np.asarray([1, 2, 1], dtype=np.uint8)

    result = build_step_patch_face_mapping(mesh, patches, statuses)

    assert result["mapping_type"] == "MANY_TO_MANY"
    assert result["triangle_count"] == 3
    assert result["patch_to_brep_face_ids"] == {
        10: [1],
        20: [1, 2],
    }
    assert result["brep_face_to_patch_ids"] == {
        1: [10, 20],
        2: [20],
    }
    pair = {
        (item["patch_id"], item["brep_face_id"]): item
        for item in result["pairs"]
    }
    assert pair[(10, 1)]["status_triangle_counts"] == {1: 1}
    assert pair[(20, 1)]["status_triangle_counts"] == {2: 1}
    assert pair[(20, 2)]["status_triangle_counts"] == {1: 1}


def test_patch_brep_mapping_rejects_mismatched_triangle_labels():
    mesh = _mesh_with_step_context()
    try:
        build_step_patch_face_mapping(mesh, [0, 1])
    except ValueError as exc:
        assert "one ID per mesh triangle" in str(exc)
    else:
        raise AssertionError("Mismatched Patch labels were accepted.")


def test_patch_brep_mapping_rejects_invalid_brep_face_ids():
    mesh = _mesh_with_step_context()
    mesh._step_brep_context["triangle_face_ids"] = np.asarray(
        [1, 2, 3], dtype=np.int64
    )
    try:
        build_step_patch_face_mapping(mesh, [0, 0, 1])
    except ValueError as exc:
        assert "invalid B-Rep ID" in str(exc)
    else:
        raise AssertionError("Invalid B-Rep Face ID was accepted.")
