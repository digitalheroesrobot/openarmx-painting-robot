"""
STEP B-Rep import and geometric-property extraction.

The STEP B-Rep is retained as the authoritative source of CAD topology.
A tessellated Trimesh is produced only for the existing PyVista viewer.
Lengths are normalized to millimetres by Open CASCADE STEP translation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import numpy as np
import trimesh


def _ocp_imports():
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.Interface import Interface_Static
        from OCP.Bnd import Bnd_Box
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import (
            TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID,
            TopAbs_VERTEX, TopAbs_REVERSED,
        )
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.TopLoc import TopLoc_Location
        from OCP.GeomAbs import (
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
            GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
            GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
            GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
        )
    except ImportError as exc:
        raise RuntimeError(
            "STEP import requires Open CASCADE Python bindings. "
            "Install with: pip install cadquery-ocp"
        ) from exc

    return locals()


def _static_call(owner, name, *args):
    """Support OCP bindings that expose static methods with a _s suffix."""
    method = getattr(owner, name + "_s", None)
    if method is None:
        method = getattr(owner, name, None)
    if method is None:
        raise RuntimeError(f"Open CASCADE method not available: {owner}.{name}")
    return method(*args)


def _explore(shape, topology_type, explorer_type, cast=None):
    explorer = explorer_type(shape, topology_type)
    items = []
    while explorer.More():
        item = explorer.Current()
        items.append(cast(item) if cast is not None else item)
        explorer.Next()
    return items


def _surface_name(adaptor, ocp):
    mapping = (
        ("PLANE", "GeomAbs_Plane"),
        ("CYLINDER", "GeomAbs_Cylinder"),
        ("CONE", "GeomAbs_Cone"),
        ("SPHERE", "GeomAbs_Sphere"),
        ("TORUS", "GeomAbs_Torus"),
        ("BEZIER_SURFACE", "GeomAbs_BezierSurface"),
        ("BSPLINE_SURFACE", "GeomAbs_BSplineSurface"),
        ("REVOLUTION", "GeomAbs_SurfaceOfRevolution"),
        ("EXTRUSION", "GeomAbs_SurfaceOfExtrusion"),
        ("OFFSET", "GeomAbs_OffsetSurface"),
    )
    value = adaptor.GetType()
    for label, enum_name in mapping:
        if value == ocp[enum_name]:
            return label
    return "OTHER"


def _build_brep_face_edge_adjacency(faces, explorer_type, edge_type):
    """Build exact face/edge incidence from the original STEP B-Rep.

    IsSame compares underlying TopoDS edge identity independent of edge
    orientation. A seam may occur more than once on one face; it is reported
    separately rather than incorrectly classified as an open boundary.
    """
    edge_shapes = []
    edge_face_ids = []
    edge_occurrences = []
    face_edge_ids = []

    for face_id, face in enumerate(faces, start=1):
        explorer = explorer_type(face, edge_type)
        local_edge_ids = []
        while explorer.More():
            edge = explorer.Current()
            edge_id = None
            for index, known_edge in enumerate(edge_shapes):
                if edge.IsSame(known_edge):
                    edge_id = index + 1
                    break
            if edge_id is None:
                edge_shapes.append(edge)
                edge_face_ids.append(set())
                edge_occurrences.append({})
                edge_id = len(edge_shapes)
            edge_face_ids[edge_id - 1].add(face_id)
            edge_occurrences[edge_id - 1][face_id] = (
                edge_occurrences[edge_id - 1].get(face_id, 0) + 1
            )
            if edge_id not in local_edge_ids:
                local_edge_ids.append(edge_id)
            explorer.Next()
        face_edge_ids.append(sorted(local_edge_ids))

    edge_records = []
    face_adjacency = {face_id: set() for face_id in range(1, len(faces) + 1)}
    pair_to_edge_ids = {}
    for edge_id, face_ids_set in enumerate(edge_face_ids, start=1):
        face_ids = sorted(face_ids_set)
        for i, face_a in enumerate(face_ids):
            for face_b in face_ids[i + 1:]:
                face_adjacency[face_a].add(face_b)
                face_adjacency[face_b].add(face_a)
                pair_to_edge_ids.setdefault((face_a, face_b), []).append(edge_id)
        occurrences = edge_occurrences[edge_id - 1]
        total_occurrences = int(sum(occurrences.values()))
        if len(face_ids) >= 2:
            classification = "SHARED_MANIFOLD" if len(face_ids) == 2 else "NON_MANIFOLD"
        elif total_occurrences >= 2:
            classification = "SEAM_OR_REPEATED_USE"
        else:
            classification = "BOUNDARY"
        edge_records.append({
            "edge_id": edge_id,
            "adjacent_face_ids": face_ids,
            "adjacent_face_count": len(face_ids),
            "face_occurrence_counts": {
                str(key): int(value) for key, value in sorted(occurrences.items())
            },
            "occurrence_count": total_occurrences,
            "classification": classification,
        })

    face_pairs = [
        {"face_id_a": int(a), "face_id_b": int(b), "shared_edge_ids": ids}
        for (a, b), ids in sorted(pair_to_edge_ids.items())
    ]
    counts = {
        "unique_edges": len(edge_records),
        "shared_manifold_edges": sum(
            item["classification"] == "SHARED_MANIFOLD" for item in edge_records
        ),
        "boundary_edges": sum(
            item["classification"] == "BOUNDARY" for item in edge_records
        ),
        "seam_or_repeated_use_edges": sum(
            item["classification"] == "SEAM_OR_REPEATED_USE" for item in edge_records
        ),
        "non_manifold_edges": sum(
            item["classification"] == "NON_MANIFOLD" for item in edge_records
        ),
        "unique_adjacent_face_pairs": len(face_pairs),
    }
    return {
        "edges": edge_records,
        "face_edge_ids": face_edge_ids,
        "face_adjacency": {
            str(face_id): sorted(neighbors)
            for face_id, neighbors in face_adjacency.items()
        },
        "adjacent_face_pairs": face_pairs,
        "summary": counts,
    }


def _xyz(point):
    return [float(point.X()), float(point.Y()), float(point.Z())]


def load_step(
    file_path: str | Path,
    linear_deflection_mm: float = 0.25,
    angular_deflection_rad: float = 0.15,
) -> trimesh.Trimesh:
    """Load STEP, extract exact B-Rep measurements, and create a display mesh."""
    if not math.isfinite(linear_deflection_mm) or linear_deflection_mm <= 0:
        raise ValueError("linear_deflection_mm must be a positive finite value.")
    if not math.isfinite(angular_deflection_rad) or angular_deflection_rad <= 0:
        raise ValueError("angular_deflection_rad must be a positive finite value.")

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STEP file not found: {path}")
    if path.suffix.lower() not in (".step", ".stp"):
        raise ValueError("Expected a .step or .stp file.")

    ocp = _ocp_imports()
    Interface_Static = ocp["Interface_Static"]
    STEPControl_Reader = ocp["STEPControl_Reader"]
    IFSelect_RetDone = ocp["IFSelect_RetDone"]
    TopoDS = ocp["TopoDS"]
    TopExp_Explorer = ocp["TopExp_Explorer"]

    # OCP 7 exposed TopoDS.Face_s; OCP 8 renamed downcasts to TopoDS.Face.
    # Resolve the available API instead of pinning the project to one OCP major.
    face_cast = getattr(TopoDS, "Face", None)
    if face_cast is None:
        face_cast = getattr(TopoDS, "Face_s", None)
    if face_cast is None:
        raise RuntimeError(
            "This Open CASCADE binding exposes neither TopoDS.Face "
            "nor TopoDS.Face_s."
        )

    # Request millimetres as the STEP translation target unit.
    _static_call(Interface_Static, "SetCVal", "xstep.cascade.unit", "MM")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"Open CASCADE could not read STEP file: {path}")
    if reader.TransferRoots() <= 0:
        raise ValueError("STEP file contains no transferable geometry.")

    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("STEP file produced an empty CAD shape.")

    faces = _explore(
        shape, ocp["TopAbs_FACE"], TopExp_Explorer, face_cast
    )
    edges = _explore(shape, ocp["TopAbs_EDGE"], TopExp_Explorer)
    vertices = _explore(shape, ocp["TopAbs_VERTEX"], TopExp_Explorer)
    shells = _explore(shape, ocp["TopAbs_SHELL"], TopExp_Explorer)
    solids = _explore(shape, ocp["TopAbs_SOLID"], TopExp_Explorer)

    # Exact B-Rep global properties (mm, mm², mm³).
    GProp_GProps = ocp["GProp_GProps"]
    BRepGProp = ocp["BRepGProp"]

    surface_props = GProp_GProps()
    _static_call(BRepGProp, "SurfaceProperties", shape, surface_props)
    total_area = float(surface_props.Mass())
    surface_centroid = _xyz(surface_props.CentreOfMass())

    volume_props = GProp_GProps()
    _static_call(BRepGProp, "VolumeProperties", shape, volume_props)
    signed_volume = float(volume_props.Mass())

    # A STEP assembly can contain several independent solids. Summing the
    # absolute volume of each solid avoids cancellation when solid
    # orientations differ. The volume centroid is weighted by each solid's
    # positive physical volume.
    solid_volume_total = 0.0
    solid_volume_moment = np.zeros(3, dtype=float)
    for solid in solids:
        solid_props = GProp_GProps()
        _static_call(BRepGProp, "VolumeProperties", solid, solid_props)
        solid_signed_volume = float(solid_props.Mass())
        solid_volume = abs(solid_signed_volume)
        if solid_volume > 0.0 and math.isfinite(solid_volume):
            solid_volume_total += solid_volume
            solid_volume_moment += (
                solid_volume * np.asarray(
                    _xyz(solid_props.CentreOfMass()), dtype=float
                )
            )

    if solids:
        volume = float(solid_volume_total)
        volume_centroid = (
            (solid_volume_moment / volume).tolist()
            if volume > 0.0 else None
        )
    else:
        # Surface-only STEP geometry has no solid volume.
        volume = abs(signed_volume)
        volume_centroid = (
            _xyz(volume_props.CentreOfMass()) if volume > 0.0 else None
        )

    linear_props = GProp_GProps()
    _static_call(BRepGProp, "LinearProperties", shape, linear_props)
    total_edge_length = float(linear_props.Mass())

    bbox = ocp["Bnd_Box"]()
    _static_call(ocp["BRepBndLib"], "Add", shape, bbox)

    # OCP 8.0.1 does not bind Bnd_Box::Limits, so bbox.Get() raises
    # TypeError even though the underlying OCCT operation succeeds.
    # CornerMin/CornerMax return ordinary gp_Pnt objects in OCP 7 and 8.
    if hasattr(bbox, "CornerMin") and hasattr(bbox, "CornerMax"):
        corner_min = bbox.CornerMin()
        corner_max = bbox.CornerMax()
        bounds_min = _xyz(corner_min)
        bounds_max = _xyz(corner_max)
    else:
        # Compatibility fallback for older bindings exposing scalar getters.
        scalar_getters = (
            "GetXMin", "GetYMin", "GetZMin",
            "GetXMax", "GetYMax", "GetZMax",
        )
        if all(hasattr(bbox, name) for name in scalar_getters):
            xmin, ymin, zmin, xmax, ymax, zmax = (
                float(getattr(bbox, name)()) for name in scalar_getters
            )
            bounds_min = [xmin, ymin, zmin]
            bounds_max = [xmax, ymax, zmax]
        else:
            raise RuntimeError(
                "This Open CASCADE binding cannot expose finite bounding "
                "box corners (CornerMin/CornerMax or scalar getters)."
            )

    if not all(math.isfinite(value) for value in bounds_min + bounds_max):
        raise ValueError(
            "STEP geometry produced a non-finite or open bounding box."
        )
    dimensions = [
        float(bounds_max[i] - bounds_min[i]) for i in range(3)
    ]

    brep_adjacency = _build_brep_face_edge_adjacency(
        faces, TopExp_Explorer, ocp["TopAbs_EDGE"]
    )
    face_records = []
    surface_counts = Counter()
    for face_id, face in enumerate(faces, start=1):
        props = GProp_GProps()
        _static_call(BRepGProp, "SurfaceProperties", face, props)
        adaptor = ocp["BRepAdaptor_Surface"](face, True)
        surface_type = _surface_name(adaptor, ocp)
        area = float(props.Mass())
        center = _xyz(props.CentreOfMass())
        try:
            u_bounds = [float(adaptor.FirstUParameter()), float(adaptor.LastUParameter())]
            v_bounds = [float(adaptor.FirstVParameter()), float(adaptor.LastVParameter())]
            parameter_bounds = {
                "u": u_bounds if all(math.isfinite(value) for value in u_bounds) else None,
                "v": v_bounds if all(math.isfinite(value) for value in v_bounds) else None,
            }
        except (AttributeError, TypeError, ValueError):
            parameter_bounds = {"u": None, "v": None}
        surface_counts[surface_type] += 1
        face_records.append({
            "face_id": face_id,
            "surface_type": surface_type,
            "area_mm2": area,
            "centroid_mm": center,
            "edge_ids": brep_adjacency["face_edge_ids"][face_id - 1],
            "adjacent_face_ids": brep_adjacency["face_adjacency"].get(str(face_id), []),
            "orientation": (
                "REVERSED"
                if face.Orientation() == ocp["TopAbs_REVERSED"]
                else "FORWARD_OR_OTHER"
            ),
            "parameter_bounds": parameter_bounds,
        })


    # Attach exact B-Rep edge lengths (mm). Rebuild the unique edge list
    # once in the same first-seen order used by the adjacency extractor.
    unique_edge_shapes = []
    for face in faces:
        explorer = TopExp_Explorer(face, ocp["TopAbs_EDGE"])
        while explorer.More():
            candidate = explorer.Current()
            if not any(candidate.IsSame(item) for item in unique_edge_shapes):
                unique_edge_shapes.append(candidate)
            explorer.Next()
    if len(unique_edge_shapes) != len(brep_adjacency["edges"]):
        raise RuntimeError(
            "STEP B-Rep edge inventory changed between topology passes."
        )
    for edge_record, edge_shape in zip(
        brep_adjacency["edges"], unique_edge_shapes
    ):
        edge_props = GProp_GProps()
        _static_call(BRepGProp, "LinearProperties", edge_shape, edge_props)
        edge_record["length_mm"] = float(edge_props.Mass())

    # Tessellate each original CAD face separately and retain Face IDs.
    mesher = ocp["BRepMesh_IncrementalMesh"](
        shape, float(linear_deflection_mm), False,
        float(angular_deflection_rad), True,
    )
    mesher.Perform()
    if hasattr(mesher, "IsDone") and not mesher.IsDone():
        raise RuntimeError("Open CASCADE tessellation failed.")

    mesh_vertices = []
    mesh_faces = []
    triangle_face_ids = []
    vertex_lookup = {}
    merge_tolerance = 1e-8
    BRep_Tool = ocp["BRep_Tool"]
    TopLoc_Location = ocp["TopLoc_Location"]

    def add_vertex(point, location):
        transformed = point.Transformed(location.Transformation())
        xyz = (float(transformed.X()), float(transformed.Y()), float(transformed.Z()))
        key = tuple(int(round(value / merge_tolerance)) for value in xyz)
        index = vertex_lookup.get(key)
        if index is None:
            index = len(mesh_vertices)
            vertex_lookup[key] = index
            mesh_vertices.append(xyz)
        return index

    for face_id, face in enumerate(faces, start=1):
        location = TopLoc_Location()
        triangulation = _static_call(BRep_Tool, "Triangulation", face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            continue

        reversed_face = face.Orientation() == ocp["TopAbs_REVERSED"]
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            n1, n2, n3 = triangulation.Triangle(triangle_index).Get()
            tri = [
                add_vertex(triangulation.Node(n1), location),
                add_vertex(triangulation.Node(n2), location),
                add_vertex(triangulation.Node(n3), location),
            ]
            if reversed_face:
                tri[1], tri[2] = tri[2], tri[1]
            if len(set(tri)) == 3:
                mesh_faces.append(tri)
                triangle_face_ids.append(face_id)

    if not mesh_vertices or not mesh_faces:
        raise ValueError("STEP B-Rep was read, but no display triangles were generated.")

    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_vertices, dtype=float),
        faces=np.asarray(mesh_faces, dtype=np.int64),
        process=False,
    )

    report = {
        "source_file": str(path),
        "file_name": path.name,
        "file_format": "STEP",
        "linear_unit": "mm",
        "unit_conversion_target": "mm",
        "unit_note": (
            "STEP translation target is millimetres. Confirm the source CAD "
            "unit when using measurements for manufacturing decisions."
        ),
        "topology": {
            "faces": len(faces),
            "edges": len(brep_adjacency["edges"]),
            "explored_edge_occurrences": len(edges),
            "vertices": len(vertices), "shells": len(shells),
            "solids": len(solids),
            "face_edge_adjacency": brep_adjacency,
        },
        "mesh": {
            "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
            "linear_deflection_mm": float(linear_deflection_mm),
            "angular_deflection_rad": float(angular_deflection_rad),
        },
        "bounding_box": {
            "minimum_mm": bounds_min, "maximum_mm": bounds_max,
            "dimensions_mm": dimensions,
            "diagonal_mm": float(np.linalg.norm(dimensions)),
        },
        "mass_properties": {
            "surface_area_mm2": total_area,
            "volume_mm3": volume,
            "signed_volume_mm3": signed_volume,
            "surface_centroid_mm": surface_centroid,
            "volume_centroid_mm": volume_centroid,
            "total_edge_length_mm": total_edge_length,
        },
        "surface_types": dict(sorted(surface_counts.items())),
        "faces": face_records,
    }

    mesh.metadata.update({
        "source_file": str(path),
        "source_format": "STEP",
        "source_unit": "mm",
        "unit_detection": "opencascade_target_unit_mm",
        "step_geometry_report": report,
        "triangle_face_ids": triangle_face_ids,
        "source_geometry_preserved": True,
        "source_coordinates_modified": False,
    })

    assert len(triangle_face_ids) == len(mesh.faces)
    assert len(face_records) == len(faces)
    assert np.isfinite(mesh.vertices).all()
    assert math.isfinite(total_area) and total_area >= 0
    assert math.isfinite(volume) and volume >= 0
    return mesh


def convert_step_report_units(report: dict, display_unit: str) -> dict:
    """
    Convert display values from the STEP report's canonical millimetres.

    The report itself is never modified; exact source measurements remain
    stored in mm, mm², and mm³. Only values returned for GUI display change.
    """
    unit_factors_mm = {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "in": 25.4,
        "ft": 304.8,
    }
    unit = str(display_unit).strip().lower()
    if unit not in unit_factors_mm:
        raise ValueError(
            f"Unsupported STEP display unit: {display_unit!r}"
        )

    try:
        dimensions_mm = report["bounding_box"]["dimensions_mm"]
        area_mm2 = float(report["mass_properties"]["surface_area_mm2"])
        volume_mm3 = float(report["mass_properties"]["volume_mm3"])
        if len(dimensions_mm) != 3:
            raise ValueError
        dimensions_mm = [float(value) for value in dimensions_mm]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Invalid STEP geometry report structure.") from exc

    if not all(math.isfinite(value) for value in dimensions_mm):
        raise ValueError("STEP bounding dimensions must be finite.")
    if not math.isfinite(area_mm2) or not math.isfinite(volume_mm3):
        raise ValueError("STEP area and volume must be finite.")

    length_factor = 1.0 / unit_factors_mm[unit]
    return {
        "display_unit": unit,
        "dimensions": [value * length_factor for value in dimensions_mm],
        "surface_area": area_mm2 * length_factor ** 2,
        "volume": volume_mm3 * length_factor ** 3,
    }


def run_step_loader_tests():
    """Run input-validation tests; full B-Rep tests require a STEP fixture."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        bad = Path(folder) / "bad.txt"
        bad.write_text("not CAD", encoding="utf-8")
        try:
            load_step(bad)
        except ValueError as exc:
            assert "Expected a .step or .stp" in str(exc)
        else:
            raise AssertionError("Unsupported extension was not rejected.")

        missing = Path(folder) / "missing.step"
        try:
            load_step(missing)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing-file validation failed.")

    print("[PASS] STEP extension validation")
    print("[PASS] Missing-file validation")

    # Pure topology test: shared B-Rep edge, open boundary, and seam use.
    class _FakeEdge:
        def __init__(self, identity):
            self.identity = identity
        def IsSame(self, other):
            return isinstance(other, _FakeEdge) and self.identity == other.identity

    class _FakeFace:
        def __init__(self, edges):
            self.edges = edges

    class _FakeExplorer:
        def __init__(self, face, _edge_type):
            self.items = list(face.edges)
            self.index = 0
        def More(self):
            return self.index < len(self.items)
        def Current(self):
            return self.items[self.index]
        def Next(self):
            self.index += 1

    fake_faces = [
        _FakeFace([_FakeEdge("shared"), _FakeEdge("boundary-a")]),
        _FakeFace([_FakeEdge("shared"), _FakeEdge("boundary-b")]),
        _FakeFace([_FakeEdge("seam"), _FakeEdge("seam")]),
    ]
    topology = _build_brep_face_edge_adjacency(
        fake_faces, _FakeExplorer, object()
    )
    assert topology["face_adjacency"]["1"] == [2]
    assert topology["face_adjacency"]["2"] == [1]
    assert topology["summary"]["shared_manifold_edges"] == 1
    assert topology["summary"]["boundary_edges"] == 2
    assert topology["summary"]["seam_or_repeated_use_edges"] == 1
    assert topology["summary"]["non_manifold_edges"] == 0
    print("[PASS] Exact B-Rep shared-edge adjacency and seam classification")

    # Display conversion tests: exact source report remains in millimetres.
    sample_report = {
        "bounding_box": {"dimensions_mm": [100.0, 200.0, 300.0]},
        "mass_properties": {
            "surface_area_mm2": 10000.0,
            "volume_mm3": 1000000.0,
        },
    }
    cm_values = convert_step_report_units(sample_report, "cm")
    assert np.allclose(cm_values["dimensions"], [10.0, 20.0, 30.0])
    assert math.isclose(cm_values["surface_area"], 100.0)
    assert math.isclose(cm_values["volume"], 1000.0)
    assert sample_report["bounding_box"]["dimensions_mm"] == [
        100.0, 200.0, 300.0
    ]
    print("[PASS] STEP display-unit conversion (mm -> cm)")
    print("[PASS] Canonical STEP report remains unchanged")

    try:
        convert_step_report_units(sample_report, "yard")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported display unit was accepted.")
    print("[PASS] Unsupported STEP display unit rejection")
    print("STEP loader preflight tests passed.")


if __name__ == "__main__":
    run_step_loader_tests()
