
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


# ============================================================
# SUPPORTED SOURCE UNITS
# ============================================================

SUPPORTED_UNITS = {
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",

    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",

    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",

    "in": "in",
    "inch": "in",
    "inches": "in",

    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
}


UNIT_METADATA_KEYS = (
    "units",
    "unit",
    "length_unit",
    "linear_unit",
)


# ============================================================
# UNIT NORMALIZATION
# ============================================================

def _normalize_unit(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    return SUPPORTED_UNITS.get(
        value.strip().lower()
    )


# ============================================================
# EXPLICIT UNIT DETECTION
# ============================================================

def _detect_explicit_unit(
    metadata: dict,
) -> str | None:
    """
    Detect only explicitly declared and recognized units.

    Important:
        Standard STL does not define a universal physical
        length unit.

        Never infer units from:
            - Mesh dimensions
            - Bounding box
            - STL header text
            - File name
            - Object appearance
    """

    for key in UNIT_METADATA_KEYS:

        unit = _normalize_unit(
            metadata.get(key)
        )

        if unit is not None:
            return unit

    return None


# ============================================================
# STL LOADER
# ============================================================

def load_stl(
    file_path: str | Path,
) -> trimesh.Trimesh:
    """
    Load an STL mesh while preserving its source geometry.

    Preservation rules:
        - No coordinate scaling.
        - No coordinate translation.
        - No automatic vertex merging.
        - No face reordering.
        - No triangle removal.
        - Unknown units remain unknown.

    Topology reconstruction is handled separately by
    geometry_analyzer.py using exact coordinate welding.
    """

    # --------------------------------------------------------
    # 1. Validate file path
    # --------------------------------------------------------

    path = (
        Path(file_path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"STL file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The selected path is not a file: {path}"
        )

    if path.suffix.lower() != ".stl":
        raise ValueError(
            "Please select an STL file."
        )

    # --------------------------------------------------------
    # 2. Load STL without processing
    # --------------------------------------------------------

    loaded = trimesh.load(
        path,
        force="mesh",
        process=False,
    )

    if not isinstance(
        loaded,
        trimesh.Trimesh,
    ):
        raise ValueError(
            "The selected file did not produce "
            "a triangular mesh."
        )

    # --------------------------------------------------------
    # 3. Validate metadata
    # --------------------------------------------------------

    if not isinstance(
        loaded.metadata,
        dict,
    ):
        loaded.metadata = {}

    # --------------------------------------------------------
    # 4. Read original geometry arrays
    # --------------------------------------------------------

    vertices = np.asarray(
        loaded.vertices
    )

    faces = np.asarray(
        loaded.faces
    )

    if (
        vertices.size == 0
        or faces.size == 0
    ):
        raise ValueError(
            "The STL file contains no usable mesh geometry."
        )

    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
    ):
        raise ValueError(
            "Invalid vertex array. "
            "Expected shape (N, 3)."
        )

    if (
        faces.ndim != 2
        or faces.shape[1] != 3
    ):
        raise ValueError(
            "Invalid triangle array. "
            "Expected shape (N, 3)."
        )

    # --------------------------------------------------------
    # 5. Validate vertex coordinates
    # --------------------------------------------------------

    if not np.issubdtype(
        vertices.dtype,
        np.number,
    ):
        raise ValueError(
            "Vertex coordinates are not numeric."
        )

    if not np.isfinite(vertices).all():
        raise ValueError(
            "The STL file contains NaN or infinite "
            "vertex coordinates."
        )

    # --------------------------------------------------------
    # 6. Validate triangle indices
    # --------------------------------------------------------

    if not np.issubdtype(
        faces.dtype,
        np.integer,
    ):
        raise ValueError(
            "Triangle indices must be integers."
        )

    if (
        np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError(
            "The STL file contains invalid triangle indices."
        )

    # --------------------------------------------------------
    # 7. Detect explicit source unit
    # --------------------------------------------------------

    detected_unit = _detect_explicit_unit(
        loaded.metadata
    )

    if detected_unit is None:
        source_unit = "unknown"
        unit_detection = (
            "not_available_in_standard_stl"
        )
    else:
        source_unit = detected_unit
        unit_detection = "explicit_metadata"

    # --------------------------------------------------------
    # 8. Record source information
    # --------------------------------------------------------

    loaded.metadata.update({

        "source_file": str(path),

        "source_format": "STL",

        "source_unit": source_unit,

        "unit_detection": unit_detection,

        "source_vertex_count": int(
            len(vertices)
        ),

        "source_face_count": int(
            len(faces)
        ),

        "source_coordinate_bounds": (
            np.asarray(
                loaded.bounds,
                dtype=float,
            ).tolist()
        ),

        "source_geometry_preserved": True,

        "source_coordinates_modified": False,

        "automatic_vertex_merging": False,

        "source_face_order_preserved": True,

    })

    return loaded


# ============================================================
# NUMERICAL TESTS
# ============================================================

def run_stl_loader_tests():

    import tempfile

    print("=" * 65)
    print("STL LOADER NUMERICAL TESTS")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        # ----------------------------------------------------
        # Test 1: Create a box and export STL
        # ----------------------------------------------------

        box = trimesh.creation.box(
            extents=[10.0, 20.0, 30.0]
        )

        stl_path = temp_dir / "test_box.stl"

        box.export(stl_path)

        print("[PASS] Test STL creation")

        # ----------------------------------------------------
        # Test 2: Load STL
        # ----------------------------------------------------

        mesh = load_stl(stl_path)

        assert isinstance(
            mesh,
            trimesh.Trimesh,
        )

        assert len(mesh.faces) == 12

        print("[PASS] STL loading")

        # ----------------------------------------------------
        # Test 3: No automatic vertex merging
        # ----------------------------------------------------

        assert len(mesh.vertices) == 36

        print("[PASS] Original triangle vertex structure")

        # ----------------------------------------------------
        # Test 4: Source unit remains unknown
        # ----------------------------------------------------

        assert (
            mesh.metadata["source_unit"]
            == "unknown"
        )

        print("[PASS] Unknown unit handling")

        # ----------------------------------------------------
        # Test 5: Source counts
        # ----------------------------------------------------

        assert (
            mesh.metadata["source_vertex_count"]
            == 36
        )

        assert (
            mesh.metadata["source_face_count"]
            == 12
        )

        print("[PASS] Source geometry counts")

        # ----------------------------------------------------
        # Test 6: Face order and coordinates preserved
        # ----------------------------------------------------

        vertices_before = mesh.vertices.copy()
        faces_before = mesh.faces.copy()

        _ = _detect_explicit_unit(
            mesh.metadata
        )

        assert np.array_equal(
            vertices_before,
            mesh.vertices,
        )

        assert np.array_equal(
            faces_before,
            mesh.faces,
        )

        print("[PASS] Source geometry unchanged")

        # ----------------------------------------------------
        # Test 7: Coordinate bounds
        # ----------------------------------------------------

        bounds = np.asarray(
            mesh.metadata["source_coordinate_bounds"]
        )

        assert bounds.shape == (2, 3)

        assert np.allclose(
            bounds[1] - bounds[0],
            [10.0, 20.0, 30.0],
            atol=1e-5,
        )

        print("[PASS] Source coordinate bounds")

        # ----------------------------------------------------
        # Test 8: Invalid unit
        # ----------------------------------------------------

        assert _normalize_unit("mm") == "mm"
        assert _normalize_unit("inch") == "in"
        assert _normalize_unit("unknown") is None
        assert _normalize_unit("yard") is None

        print("[PASS] Unit normalization")

        # ----------------------------------------------------
        # Test 9: Invalid file extension
        # ----------------------------------------------------

        invalid_path = temp_dir / "invalid.txt"
        invalid_path.write_text("not an STL file")

        try:
            load_stl(invalid_path)

        except ValueError:
            pass

        else:
            raise AssertionError(
                "Invalid file extension was accepted."
            )

        print("[PASS] Invalid file extension rejection")

    print("=" * 65)
    print("ALL STL LOADER TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_stl_loader_tests()
