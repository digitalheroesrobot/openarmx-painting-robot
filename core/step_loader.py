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
import re

import numpy as np
import trimesh


def _parse_step_ap214_schema(step_text: str) -> str:
    """Return the STEP AP214 schema identifier or reject another protocol."""
    header_match = re.search(
        r"\bHEADER\s*;(.*?)\bENDSEC\s*;",
        step_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if header_match is None:
        raise ValueError(
            "Invalid STEP Part 21 header: HEADER section is missing or incomplete."
        )

    schema_match = re.search(
        r"\bFILE_SCHEMA\s*\(\s*\((.*?)\)\s*\)\s*;",
        header_match.group(1),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if schema_match is None:
        raise ValueError(
            "Invalid STEP header: FILE_SCHEMA declaration is missing."
        )

    schema_identifiers = [
        value.replace("''", "'").strip()
        for value in re.findall(r"'((?:''|[^'])*)'", schema_match.group(1))
    ]
    if len(schema_identifiers) != 1:
        raise ValueError(
            "OpenArmX requires exactly one STEP AP214 FILE_SCHEMA identifier."
        )

    schema_identifier = schema_identifiers[0]
    normalized = re.sub(r"\s*\{.*$", "", schema_identifier).strip().upper()
    if normalized not in {"AUTOMOTIVE_DESIGN", "AUTOMOTIVE_DESIGN_CC2"}:
        raise ValueError(
            "Unsupported STEP application protocol. "
            "OpenArmX accepts STEP AP214 only "
            "(FILE_SCHEMA AUTOMOTIVE_DESIGN or AUTOMOTIVE_DESIGN_CC2); "
            f"received {schema_identifier!r}."
        )

    return schema_identifier


def _validate_step_ap214_file(path: Path) -> str:
    """Validate the STEP Part 21 header before Open CASCADE translation."""
    # STEP headers are small. Bound the read so a malformed file cannot cause
    # an unbounded preflight allocation before the CAD kernel is invoked.
    with path.open("rb") as stream:
        header_bytes = stream.read(256 * 1024)
    step_text = header_bytes.decode("latin-1")
    return _parse_step_ap214_schema(step_text)


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
        from OCP.TopExp import TopExp, TopExp_Explorer
        # OCP 8 moved indexed TopoDS maps from OCP.TopTools to OCP.collections.
        from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher
        from OCP.TopoDS import TopoDS
        from OCP.TopLoc import TopLoc_Location
        from OCP.GeomAbs import (
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
            GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
            GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
            GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
        )
        from OCP.gp import gp_Pnt, gp_Vec
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


def _build_brep_face_edge_adjacency(
    shape, faces, top_exp_type, explorer_type, indexed_map_type, edge_type
):
    """Build exact face/edge incidence from the original STEP B-Rep.

    IsSame compares underlying TopoDS edge identity independent of edge
    orientation. A seam may occur more than once on one face; it is reported
    separately rather than incorrectly classified as an open boundary.
    """
    edge_map = indexed_map_type()
    _static_call(top_exp_type, "MapShapes", shape, edge_type, edge_map)
    edge_shapes = [
        edge_map.FindKey(index)
        for index in range(1, int(edge_map.Extent()) + 1)
    ]
    edge_face_ids = [set() for _ in edge_shapes]
    edge_occurrences = [{} for _ in edge_shapes]
    face_edge_ids = []

    for face_id, face in enumerate(faces, start=1):
        explorer = explorer_type(face, edge_type)
        local_edge_ids = []
        while explorer.More():
            edge = explorer.Current()
            edge_id = int(edge_map.FindIndex(edge))
            if edge_id <= 0:
                raise RuntimeError(
                    "A STEP face edge was not found in the global B-Rep edge map."
                )
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
    }, edge_shapes


def _xyz(point):
    return [float(point.X()), float(point.Y()), float(point.Z())]


def get_step_brep_context(mesh):
    """Return the live STEP B-Rep context retained on an imported mesh.

    The context is runtime-only and intentionally contains Open CASCADE
    objects. Do not put it in JSON reports or attempt to serialize it.
    Face IDs are one-based and match the IDs in step_geometry_report.
    """
    metadata = getattr(mesh, "metadata", None)
    if not isinstance(metadata, dict):
        return None

    # Open CASCADE wrappers are not deepcopy/pickle compatible. Keep the
    # live context as a runtime attribute, never inside Trimesh metadata:
    # Trimesh.copy() deep-copies metadata for viewer/display meshes.
    context = getattr(mesh, "_step_brep_context", None)
    if context is None:
        return None
    if not isinstance(context, dict):
        raise TypeError("Invalid STEP B-Rep runtime context.")

    required = {
        "shape",
        "faces",
        "face_records",
        "triangle_face_ids",
        "triangle_uvs",
        "triangle_uv_valid",
        "face_adjacency",
        "schema_identifier",
    }
    missing = required.difference(context)
    if missing:
        raise ValueError(
            "Incomplete STEP B-Rep runtime context: "
            + ", ".join(sorted(missing))
        )

    face_ids = np.asarray(context["triangle_face_ids"], dtype=np.int64)
    if face_ids.shape != (len(mesh.faces),):
        raise ValueError(
            "STEP B-Rep triangle-to-face mapping does not match the mesh."
        )

    face_count = len(context["faces"])
    if len(context["face_records"]) != face_count:
        raise ValueError(
            "STEP B-Rep face objects and face records are inconsistent."
        )
    if np.any(face_ids < 1) or np.any(face_ids > face_count):
        raise ValueError(
            "STEP B-Rep triangle mapping contains an invalid one-based Face ID."
        )

    triangle_uvs = np.asarray(context["triangle_uvs"], dtype=np.float64)
    triangle_uv_valid = np.asarray(context["triangle_uv_valid"], dtype=bool)
    if triangle_uvs.shape != (len(mesh.faces), 3, 2):
        raise ValueError("STEP triangle UV mapping does not match mesh triangles.")
    if triangle_uv_valid.shape != (len(mesh.faces),):
        raise ValueError("STEP triangle UV validity mapping does not match mesh triangles.")
    if np.any(triangle_uv_valid & ~np.isfinite(triangle_uvs).all(axis=(1, 2))):
        raise ValueError("Valid STEP triangle UV coordinates must be finite.")

    return context


def get_step_triangle_surface_coordinates(
    mesh, triangle_index: int, barycentric: np.ndarray | list
) -> dict:
    """Map a point inside a STEP-derived mesh triangle back to its source Face UV.

    The triangle's UV corners originate from Open CASCADE triangulation.
    Barycentric interpolation is local to that triangle; it is not a global
    parameterization and must not cross triangle or face boundaries.
    """
    context = get_step_brep_context(mesh)
    if context is None:
        raise ValueError("Mesh does not contain a live STEP B-Rep context.")
    triangle_index = int(triangle_index)
    if triangle_index < 0 or triangle_index >= len(mesh.faces):
        raise IndexError("Triangle index is outside the mesh.")
    if not bool(context["triangle_uv_valid"][triangle_index]):
        raise ValueError(
            f"Triangle {triangle_index} has no valid STEP UV-node mapping."
        )

    weights = np.asarray(barycentric, dtype=np.float64)
    if weights.shape != (3,) or not np.isfinite(weights).all():
        raise ValueError("Barycentric coordinates must be three finite values.")
    if np.any(weights < -1e-8) or np.any(weights > 1.0 + 1e-8):
        raise ValueError("Barycentric coordinates lie outside the source triangle.")
    total = float(weights.sum())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise ValueError("Barycentric coordinates must sum to one.")
    weights = weights / total

    uv_corners = np.asarray(
        context["triangle_uvs"][triangle_index], dtype=np.float64
    )
    uv = weights @ uv_corners
    if not np.isfinite(uv).all():
        raise ValueError("Barycentric UV interpolation produced non-finite values.")
    face_id = int(context["triangle_face_ids"][triangle_index])
    return {
        "triangle_index": triangle_index,
        "face_id": face_id,
        "uv": uv,
        "barycentric": weights,
        "mapping_method": "SOURCE_TRIANGULATION_UVNODE_BARYCENTRIC",
    }


def _surface_normal_from_derivatives(du, dv, reversed_face=False):
    """Compute a unit surface normal from exact CAD first derivatives."""
    du = np.asarray(du, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    if du.shape != (3,) or dv.shape != (3,):
        raise ValueError("Surface derivatives must be 3D vectors.")
    if not np.isfinite(du).all() or not np.isfinite(dv).all():
        raise ValueError("Surface derivatives must be finite.")
    normal = np.cross(du, dv)
    magnitude = float(np.linalg.norm(normal))
    scale = float(np.linalg.norm(du) * np.linalg.norm(dv))
    if not math.isfinite(magnitude) or scale <= np.finfo(float).tiny or magnitude <= scale * 1e-12:
        raise ValueError("STEP surface normal is undefined at this UV location.")
    normal /= magnitude
    if reversed_face:
        normal *= -1.0
    return normal


def evaluate_step_face_surface(mesh, face_id: int, u: float, v: float) -> dict:
    """Evaluate an original STEP B-Rep face at exact native (u, v) parameters.

    Coordinates and first derivatives are in the STEP loader's canonical mm
    frame. The returned normal respects the TopoDS_Face orientation.
    This is deliberately a read-only primitive; it does not alter Stage 3
    mesh-based path generation.
    """
    context = get_step_brep_context(mesh)
    if context is None:
        raise ValueError("Mesh does not contain a live STEP B-Rep context.")

    face_id = int(face_id)
    if face_id < 1 or face_id > len(context["faces"]):
        raise ValueError(f"Invalid one-based STEP Face ID: {face_id}")
    u = float(u)
    v = float(v)
    if not math.isfinite(u) or not math.isfinite(v):
        raise ValueError("STEP surface parameters u and v must be finite.")

    record = context["face_records"][face_id - 1]
    bounds = record.get("parameter_bounds", {})
    for parameter, value in (("u", u), ("v", v)):
        interval = bounds.get(parameter)
        if interval is not None:
            lower, upper = map(float, interval)
            tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"Face {face_id} parameter {parameter}={value} is outside "
                    f"the trimmed face range [{lower}, {upper}]."
                )

    ocp = _ocp_imports()
    adaptor = ocp["BRepAdaptor_Surface"](context["faces"][face_id - 1], True)
    point = ocp["gp_Pnt"]()
    derivative_u = ocp["gp_Vec"]()
    derivative_v = ocp["gp_Vec"]()
    try:
        adaptor.D1(u, v, point, derivative_u, derivative_v)
    except Exception as exc:
        raise RuntimeError(
            "Open CASCADE could not evaluate the STEP face first derivatives "
            f"at Face {face_id}, UV=({u}, {v})."
        ) from exc

    du = np.asarray(
        [derivative_u.X(), derivative_u.Y(), derivative_u.Z()],
        dtype=np.float64,
    )
    dv = np.asarray(
        [derivative_v.X(), derivative_v.Y(), derivative_v.Z()],
        dtype=np.float64,
    )
    reversed_face = record.get("orientation") == "REVERSED"
    normal = _surface_normal_from_derivatives(du, dv, reversed_face)
    xyz = np.asarray([point.X(), point.Y(), point.Z()], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("STEP surface evaluation produced a non-finite point.")

    return {
        "face_id": face_id,
        "surface_type": record["surface_type"],
        "uv": [u, v],
        "point_mm": xyz,
        "du_mm_per_u": du,
        "dv_mm_per_v": dv,
        "normal": normal,
        "orientation": record.get("orientation", "FORWARD_OR_OTHER"),
        "evaluation_method": "OPENCASCADE_BREP_ADAPTOR_D1",
    }


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

    schema_identifier = _validate_step_ap214_file(path)

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

    brep_adjacency, unique_edge_shapes = _build_brep_face_edge_adjacency(
        shape, faces, ocp["TopExp"], TopExp_Explorer,
        ocp["IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher"], ocp["TopAbs_EDGE"]
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
        face_record = {
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
        }
        if surface_type == "CYLINDER":
            # Preserve the exact analytic cylinder from the original STEP
            # B-Rep. The mesh is only a display tessellation and must not be
            # used to refit the cylinder axis or radius.
            cylinder = adaptor.Cylinder()
            cylinder_axis = cylinder.Axis()
            cylinder_location = cylinder_axis.Location()
            cylinder_direction = cylinder_axis.Direction()
            face_record["cylinder_geometry_mm"] = {
                "axis_location_mm": _xyz(cylinder_location),
                "axis_direction": [
                    float(cylinder_direction.X()),
                    float(cylinder_direction.Y()),
                    float(cylinder_direction.Z()),
                ],
                "radius_mm": float(cylinder.Radius()),
            }
        face_records.append(face_record)


    # Attach exact B-Rep edge lengths (mm) using the indexed global edge map.
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
    triangle_uvs = []
    triangle_uv_valid = []
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
        has_uv_nodes = (
            bool(triangulation.HasUVNodes())
            if hasattr(triangulation, "HasUVNodes") else False
        )
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            n1, n2, n3 = triangulation.Triangle(triangle_index).Get()
            node_ids = [n1, n2, n3]
            tri = [
                add_vertex(triangulation.Node(node_id), location)
                for node_id in node_ids
            ]
            uv_corners = None
            if has_uv_nodes:
                try:
                    uv_corners = [
                        [
                            float(triangulation.UVNode(node_id).X()),
                            float(triangulation.UVNode(node_id).Y()),
                        ]
                        for node_id in node_ids
                    ]
                    if not np.isfinite(np.asarray(uv_corners)).all():
                        uv_corners = None
                except (AttributeError, TypeError, ValueError):
                    uv_corners = None

            if reversed_face:
                tri[1], tri[2] = tri[2], tri[1]
                if uv_corners is not None:
                    uv_corners[1], uv_corners[2] = uv_corners[2], uv_corners[1]
            if len(set(tri)) == 3:
                mesh_faces.append(tri)
                triangle_face_ids.append(face_id)
                triangle_uvs.append(
                    uv_corners if uv_corners is not None
                    else [[np.nan, np.nan]] * 3
                )
                triangle_uv_valid.append(uv_corners is not None)

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
        "application_protocol": "AP214",
        "schema_identifier": schema_identifier,
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

    topology_summary = brep_adjacency["summary"]
    print("\\n" + "=" * 68)
    print("STEP B-REP TOPOLOGY VALIDATION")
    print("=" * 68)
    print(f"Original STEP Faces:    {len(faces):,}")
    print(f"Original STEP Edges:    {len(brep_adjacency['edges']):,}")
    print(f"Original STEP Vertices: {len(vertices):,}")
    print("Surface Types:")
    for surface_name, surface_count in sorted(surface_counts.items()):
        print(f"  {surface_name}: {surface_count:,}")
    print("Face Adjacency:")
    print(f"  Shared manifold edges: {topology_summary['shared_manifold_edges']:,}")
    print(f"  Boundary edges:        {topology_summary['boundary_edges']:,}")
    print(f"  Seam/repeated edges:   {topology_summary['seam_or_repeated_use_edges']:,}")
    print(f"  Non-manifold edges:    {topology_summary['non_manifold_edges']:,}")
    print(f"  Adjacent face pairs:   {topology_summary['unique_adjacent_face_pairs']:,}")
    print("=" * 68)

    # Keep the live Open CASCADE B-Rep objects attached to this mesh.
    # Stage 1 and Stage 2 pass the same mesh object through to Stage 3, so the
    # exact source surfaces remain available for analytic path generation.
    # This private runtime context is deliberately excluded from JSON reports.
    brep_context = {
        "shape": shape,
        "faces": tuple(faces),
        "face_records": tuple(face_records),
        "face_adjacency": brep_adjacency["face_adjacency"],
        "triangle_face_ids": np.asarray(triangle_face_ids, dtype=np.int64),
        "triangle_uvs": np.asarray(triangle_uvs, dtype=np.float64),
        "triangle_uv_valid": np.asarray(triangle_uv_valid, dtype=bool),
        "schema_identifier": schema_identifier,
        "linear_unit": "mm",
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

    # Store Open CASCADE objects as a non-metadata runtime attribute. This
    # keeps the original mesh available to Stage 3 while allowing Trimesh and
    # the 3D viewer to deepcopy the ordinary serializable metadata safely.
    mesh._step_brep_context = brep_context

    # Validate the runtime mapping before returning the imported mesh.
    assert get_step_brep_context(mesh) is brep_context

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

    # STEP AP214 header variants accepted by common CAD exporters.
    ap214_header = """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
END-ISO-10303-21;
"""
    assert _parse_step_ap214_schema(ap214_header).startswith("AUTOMOTIVE_DESIGN")

    ap214_cc2_header = """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2 { 1 2 10303 214 -1 1 5 4 }'));
ENDSEC;
DATA;
END-ISO-10303-21;
"""
    assert _parse_step_ap214_schema(ap214_cc2_header).startswith("AUTOMOTIVE_DESIGN_CC2")

    for unsupported_header in (
        "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\nENDSEC;",
        "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));\nENDSEC;",
    ):
        try:
            _parse_step_ap214_schema(unsupported_header)
        except ValueError as exc:
            assert "AP214 only" in str(exc)
        else:
            raise AssertionError("A non-AP214 STEP schema was accepted.")

    try:
        _parse_step_ap214_schema("ISO-10303-21;\nHEADER;\nENDSEC;")
    except ValueError as exc:
        assert "FILE_SCHEMA" in str(exc)
    else:
        raise AssertionError("A STEP file without FILE_SCHEMA was accepted.")

    print("[PASS] STEP AP214 schema validation")
    print("[PASS] STEP AP214 legacy CC2 schema validation")
    print("[PASS] STEP AP203/AP242 rejection")
    print("[PASS] Missing FILE_SCHEMA rejection")

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

    # Runtime B-Rep context validation (no Open CASCADE fixture required).
    class _FakeMesh:
        def __init__(self, face_count):
            self.faces = [None] * face_count
            self.metadata = {}

    fake_mesh = _FakeMesh(2)
    fake_shape = object()
    fake_faces = (object(), object())
    fake_context = {
        "shape": fake_shape,
        "faces": fake_faces,
        "face_records": ({"face_id": 1}, {"face_id": 2}),
        "triangle_face_ids": np.asarray([1, 2], dtype=np.int64),
        "triangle_uvs": np.asarray([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
        ], dtype=np.float64),
        "triangle_uv_valid": np.asarray([True, True], dtype=bool),
        "face_adjacency": {"1": [2], "2": [1]},
        "schema_identifier": "AUTOMOTIVE_DESIGN",
    }
    fake_mesh._step_brep_context = fake_context
    assert get_step_brep_context(fake_mesh) is fake_context
    uv_sample = get_step_triangle_surface_coordinates(
        fake_mesh, 0, [0.25, 0.25, 0.50]
    )
    assert uv_sample["face_id"] == 1
    assert np.allclose(uv_sample["uv"], [0.25, 0.50], atol=1e-12)
    print("[PASS] STEP triangle-to-Face UV mapping")
    print("[PASS] Barycentric mesh-sample to CAD UV conversion")

    # The ordinary Trimesh metadata remains deepcopy-safe for the viewer.
    from copy import deepcopy
    copied_metadata = deepcopy(fake_mesh.metadata)
    assert "_step_brep_context" not in copied_metadata

    fake_mesh._step_brep_context["triangle_face_ids"] = np.asarray(
        [1], dtype=np.int64
    )
    try:
        get_step_brep_context(fake_mesh)
    except ValueError as exc:
        assert "does not match the mesh" in str(exc)
    else:
        raise AssertionError("Mismatched B-Rep triangle mapping was accepted.")

    del fake_mesh._step_brep_context
    assert get_step_brep_context(fake_mesh) is None
    print("[PASS] Live STEP B-Rep runtime context validation")
    print("[PASS] B-Rep triangle-to-Face ID mapping validation")
    print("[PASS] Trimesh metadata deepcopy compatibility")

    # Pure topology test: shared B-Rep edge, open boundary, and seam use.
    class _FakeEdge:
        def __init__(self, identity):
            self.identity = identity
        def IsSame(self, other):
            return isinstance(other, _FakeEdge) and self.identity == other.identity

    class _FakeFace:
        def __init__(self, edges):
            self.edges = edges

    class _FakeIndexedMap:
        def __init__(self):
            self.items = []
        def Add(self, edge):
            if not any(edge.IsSame(item) for item in self.items):
                self.items.append(edge)
        def FindKey(self, index):
            return self.items[index - 1]
        def FindIndex(self, edge):
            for index, item in enumerate(self.items, start=1):
                if edge.IsSame(item):
                    return index
            return 0
        def Extent(self):
            return len(self.items)

    class _FakeTopExp:
        @staticmethod
        def MapShapes(shape, _edge_type, edge_map):
            for face in shape:
                for edge in face.edges:
                    edge_map.Add(edge)

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
    topology, _ = _build_brep_face_edge_adjacency(
        fake_faces, fake_faces, _FakeTopExp, _FakeExplorer, _FakeIndexedMap, object()
    )
    assert topology["face_adjacency"]["1"] == [2]
    assert topology["face_adjacency"]["2"] == [1]
    assert topology["summary"]["shared_manifold_edges"] == 1
    assert topology["summary"]["boundary_edges"] == 2
    assert topology["summary"]["seam_or_repeated_use_edges"] == 1
    assert topology["summary"]["non_manifold_edges"] == 0
    print("[PASS] Exact B-Rep shared-edge adjacency and seam classification")

    # Exact surface-normal derivative tests (analytic plane/cylinder tangent bases).
    plane_normal = _surface_normal_from_derivatives(
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    )
    assert np.allclose(plane_normal, [0.0, 0.0, 1.0], atol=1e-12)
    reversed_plane_normal = _surface_normal_from_derivatives(
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], reversed_face=True
    )
    assert np.allclose(reversed_plane_normal, [0.0, 0.0, -1.0], atol=1e-12)

    # Cylinder at angle zero: dS/du points +Y and dS/dv points +Z.
    cylinder_normal = _surface_normal_from_derivatives(
        [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]
    )
    assert np.allclose(cylinder_normal, [1.0, 0.0, 0.0], atol=1e-12)

    try:
        _surface_normal_from_derivatives([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    except ValueError as exc:
        assert "undefined" in str(exc)
    else:
        raise AssertionError("Degenerate surface derivatives were accepted.")
    print("[PASS] Exact surface normal from first derivatives")
    print("[PASS] Reversed STEP face normal orientation")
    print("[PASS] Degenerate surface derivative rejection")

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
