"""
STEP CAD loader for OpenArmX Painting Control System.

The STEP B-Rep is the authoritative geometry source. It is tessellated
only to provide the existing PyVista/Trimesh viewer with display geometry.
All lengths are normalized to millimetres by Open CASCADE before analysis.

Install dependency:
    pip install cadquery-ocp

The returned mesh metadata contains a JSON-serializable B-Rep report and
a triangle-to-CAD-face mapping. Original STEP faces are not discarded.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import math

import numpy as np
import trimesh


def _require_ocp():
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
    except ImportError as exc:
        raise RuntimeError(
            "STEP import requires Open CASCADE Python bindings. "
            "Install them with: pip install cadquery-ocp"
        ) from exc

    return locals()


def _explore(shape, topology_type, explorer_type):
    explorer = explorer_type(shape, topology_type)
    items = []
    while explorer.More():
        items.append(explorer.Current())
        explorer.Next()
    return items


def _surface_name(adaptor, geomabs):
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
        enum_value = getattr(geomabs, enum_name, None)
        if enum_value is not None and value == enum_value:
            return label
    return "OTHER"


def _point_xyz(point):
    return [float(point.X()), float(point.Y()), float(point.Z())]


def load_step(file_path: str | Path, linear_deflection_mm: float = 0.25,
              angular_deflection_rad: float = 0.15) -> trimesh.Trimesh:
    """
    Read STEP B-Rep, calculate exact topology/surface properties, and
    return a display mesh in millimetres.

    Args:
        file_path: STEP or STP path.
        linear_deflection_mm: Tessellation chordal tolerance in mm.
        angular_deflection_rad: Tessellation angular tolerance in radians.

    Returns:
        trimesh.Trimesh whose metadata['step_geometry_report'] contains
        exact B-Rep measurements and whose metadata['triangle_face_ids']
        maps each display triangle to its original 1-based CAD Face ID.
    """
    if linear_deflection_mm <= 0 or angular_deflection_rad <= 0:
        raise ValueError("Tessellation tolerances must be positive.")

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STEP file not found: {path}")
    if path.suffix.lower() not in (".step", ".stp"):
        raise ValueError("Expected a .step or .stp file.")

    ocp = _require_ocp()
    BRep_Tool = ocp["BRep_Tool"]
    BRepAdaptor_Surface = ocp["BRepAdaptor_Surface"]
    BRepBndLib = ocp["BRepBndLib"]
    BRepGProp = ocp["BRepGProp"]
    BRepMesh_IncrementalMesh = ocp["BRepMesh_IncrementalMesh"]
    GProp_GProps = ocp["GProp_GProps"]
    IFSelect_RetDone = ocp["IFSelect_RetDone"]
    Interface_Static = ocp["Interface_Static"]
    Bnd_Box = ocp["Bnd_Box"]
    STEPControl_Reader = ocp["STEPControl_Reader"]
    TopAbs_EDGE = ocp["TopAbs_EDGE"]
    TopAbs_FACE = ocp["TopAbs_FACE"]
    TopAbs_SHELL = ocp["TopAbs_SHELL"]
    TopAbs_SOLID = ocp["TopAbs_SOLID"]
    TopAbs_VERTEX = ocp["TopAbs_VERTEX"]
    TopAbs_REVERSED = ocp["TopAbs_REVERSED"]
    TopExp_Explorer = ocp["TopExp_Explorer"]
    TopoDS = ocp["TopoDS"]
    TopLoc_Location = ocp["TopLoc_Location"]

    # Force a consistent physical output unit. Open CASCADE converts
    # STEP model units to this target unit during STEP translation.
    Interface_Static.SetCVal("xstep.cascade.unit", "MM")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise ValueError(f"Open CASCADE could not read STEP file: {path}")
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise ValueError("STEP file contains no transferable geometry.")

    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("STEP file produced an empty CAD shape.")

    # Exact B-Rep topology.
    faces = _explore(shape, TopAbs_FACE, TopExp_Explorer)
    edges = _explore(shape, TopAbs_EDGE, TopExp_Explorer)
    vertices = _explore(shape, TopAbs_VERTEX, TopExp_Explorer)
    shells = _explore(shape, TopAbs_SHELL, TopExp_Explorer)
    solids = _explore(shape, TopAbs_SOLID, TopExp_Explorer)

    # Exact global properties in mm, mm² and mm³.
    surface_props = GProp_GProps()
    BRepGProp.SurfaceProperties(shape, surface_props)
    total_surface_area = float(surface_props.Mass())
    surface_centroid = _point_xyz(surface_props.CentreOfMass())

    volume_props = GProp_GProps()
    BRepGProp.VolumeProperties(shape, volume_props)
    signed_volume = float(volume_props.Mass())
    volume = abs(signed_volume)
    volume_centroid = _point_xyz(volume_props.CentreOfMass()) if volume > 0 else None

    linear_props = GProp_GProps()
    BRepGProp.LinearProperties(shape, linear_props)
    total_edge_length = float(linear_props.Mass())

    bbox = Bnd_Box()
    BRepBndLib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    bounds_min = [float(xmin), float(ymin), float(zmin)]
    bounds_max = [float(xmax), float(ymax), float(zmax)]
    dimensions = [xmax - xmin, ymax - ymin, zmax - zmin]
    diagonal = float(np.linalg.norm(dimensions))

    # Exact per-face properties and stable 1-based CAD face identifiers.
    face_records = []
    surface_counts = Counter()
    for face_id, face in enumerate(faces, start=1):
        props = GProp_GProps()
        BRepGProp.SurfaceProperties(face, props)
        adaptor = BRepAdaptor_Surface(face, True)
        surface_type = _surface_name(adaptor, ocp.get("GeomAbs", object()))
        # Resolve surface enum names without relying on enum string formatting.
        try:
            from OCP import GeomAbs
            surface_type = _surface_name(adaptor, GeomAbs)
        except ImportError:
            surface_type = str(adaptor.GetType()).split(".")[-1].upper()

        area = float(props.Mass())
        center = _point_xyz(props.CentreOfMass())
        surface_counts[surface_type] += 1
        face_records.append({
            "face_id": face_id,
            "surface_type": surface_type,
            "area_mm2": area,
            "centroid_mm": center,
        })

    # Tessellate for the current PyVista viewer. Each triangle retains its
    # source CAD face ID so later painting patches can map back to B-Rep.
    mesher = BRepMesh_IncrementalMesh(
        shape, float(linear_deflection_mm), False,
        float(angular_deflection_rad), True
    )
    mesher.Perform()
    if hasattr(mesher, "IsDone") and not mesher.IsDone():
        raise RuntimeError("Open CASCADE tessellation did not complete.")

    mesh_vertices = []
    mesh_faces = []
    triangle_face_ids = []
    vertex_lookup = {}
    merge_tolerance = 1e-8

    def add_vertex(point):
        p = point.Transformed(location.Transformation())
        xyz = (float(p.X()), float(p.Y()), float(p.Z()))
        key = tuple(int(round(v / merge_tolerance)) for v in xyz)
        existing = vertex_lookup.get(key)
        if existing is not None:
            return existing
        index = len(mesh_vertices)
        mesh_vertices.append(xyz)
        vertex_lookup[key] = index
        return index

    for face_id, face in enumerate(faces, start=1):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            continue

        reversed_face = face.Orientation() == TopAbs_REVERSED
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            n1, n2, n3 = triangulation.Triangle(triangle_index).Get()
            indices = [
                add_vertex(triangulation.Node(n1)),
                add_vertex(triangulation.Node(n2)),
                add_vertex(triangulation.Node(n3)),
            ]
            if reversed_face:
                indices[1], indices[2] = indices[2], indices[1]
            if len(set(indices)) == 3:
                mesh_faces.append(indices)
                triangle_face_ids.append(face_id)

    if not mesh_vertices or not mesh_faces:
        raise ValueError(
            "STEP B-Rep was read, but no display triangles were generated."
        )

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
            "Open CASCADE STEP translation target configured as mm. "
            "Verify against the originating CAD system for critical metrology."
        ),
        "topology": {
            "faces": len(faces),
            "edges": len(edges),
            "vertices": len(vertices),
            "shells": len(shells),
            "solids": len(solids),
        },
        "mesh": {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces),
            "linear_deflection_mm": float(linear_deflection_mm),
            "angular_deflection_rad": float(angular_deflection_rad),
        },
        "bounding_box": {
            "minimum_mm": bounds_min,
            "maximum_mm": bounds_max,
            "dimensions_mm": [float(v) for v in dimensions],
            "diagonal_mm": diagonal,
        },
        "mass_properties": {
            "surface_area_mm2": total_surface_area,
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

    # Basic numerical consistency checks.
    assert len(triangle_face_ids) == len(mesh.faces)
    assert len(face_records) == len(faces)
    assert np.all(np.isfinite(mesh.vertices))
    assert np.all(np.isfinite(mesh.faces))
    assert math.isfinite(total_surface_area) and total_surface_area >= 0
    assert math.isfinite(volume) and volume >= 0

    return mesh


def run_step_loader_tests():
    """Run dependency-independent checks for report/mesh contract."""
    import tempfile
    from pathlib import Path

    # Validate extension and missing-file errors without requiring OCP.
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
            raise AssertionError("Missing STEP file was not rejected.")

    print("[PASS] STEP extension validation")
    print("[PASS] Missing-file validation")
    print("STEP loader preflight tests passed.")


if __name__ == "__main__":
    run_step_loader_tests()
