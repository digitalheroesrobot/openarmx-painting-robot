
from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import trimesh


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_NORMAL_ANGLE_DEG = 25.0
SHARP_EDGE_ANGLE_DEG = 30.0

PLANAR_DISTANCE_TOLERANCE_MM = 0.5
CYLINDER_MIN_RADIUS_MM = 0.5
THICKNESS_MIN_DISTANCE_MM = 0.1

CYLINDER_MAX_RELATIVE_ERROR = 0.05
CYLINDER_MIN_FACES = 8
MAX_THICKNESS_SAMPLES = 128

UNKNOWN_PLANAR_TOLERANCE_RATIO = 1e-4
UNKNOWN_MIN_CYLINDER_RADIUS_RATIO = 1e-4
UNKNOWN_MIN_THICKNESS_RATIO = 1e-9

DEGENERATE_AREA_RELATIVE_EPSILON = 1e-14
MINIMUM_NUMERICAL_SCALE = 1e-12

PLANAR_DISTANCE_TOLERANCE = PLANAR_DISTANCE_TOLERANCE_MM
CYLINDER_MIN_RADIUS = CYLINDER_MIN_RADIUS_MM
THICKNESS_MIN_DISTANCE = THICKNESS_MIN_DISTANCE_MM


# ============================================================
# UNIT UTILITIES
# ============================================================

SUPPORTED_UNITS_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "millimetre": 1.0,
    "millimetres": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "centimetre": 10.0,
    "centimetres": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "metre": 1000.0,
    "metres": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
}


def _normalize_unit(unit) -> str:
    if not isinstance(unit, str):
        return "unknown"

    unit = unit.strip().lower()

    if not unit:
        return "unknown"

    return unit if unit in SUPPORTED_UNITS_TO_MM else "unknown"


def _get_coordinate_unit(
    mesh: trimesh.Trimesh,
    coordinate_unit: str | None = None,
) -> str:

    if coordinate_unit is not None:
        return _normalize_unit(coordinate_unit)

    metadata = getattr(mesh, "metadata", {})

    if not isinstance(metadata, dict):
        return "unknown"

    return _normalize_unit(
        metadata.get("source_unit", "unknown")
    )


def _length_unit_name(unit):
    return "coordinate_units" if unit == "unknown" else unit


def _area_unit_name(unit):
    return (
        "coordinate_units_squared"
        if unit == "unknown"
        else f"{unit}^2"
    )


def _volume_unit_name(unit):
    return (
        "coordinate_units_cubed"
        if unit == "unknown"
        else f"{unit}^3"
    )


def _mm_to_coordinate_units(value_mm, coordinate_unit):

    factor = SUPPORTED_UNITS_TO_MM.get(
        _normalize_unit(coordinate_unit)
    )

    if factor is None:
        return None

    return float(value_mm / factor)


def _get_mesh_scale(mesh):

    try:
        extents = np.asarray(
            mesh.extents,
            dtype=np.float64,
        )

        if (
            extents.shape != (3,)
            or not np.all(np.isfinite(extents))
        ):
            return 1.0

        return max(
            float(np.linalg.norm(extents)),
            MINIMUM_NUMERICAL_SCALE,
        )

    except Exception:
        return 1.0


def _build_threshold_configuration(mesh, coordinate_unit):

    scale = _get_mesh_scale(mesh)

    planar_mm = _mm_to_coordinate_units(
        PLANAR_DISTANCE_TOLERANCE_MM,
        coordinate_unit,
    )

    cylinder_mm = _mm_to_coordinate_units(
        CYLINDER_MIN_RADIUS_MM,
        coordinate_unit,
    )

    thickness_mm = _mm_to_coordinate_units(
        THICKNESS_MIN_DISTANCE_MM,
        coordinate_unit,
    )

    if coordinate_unit != "unknown":

        planar_tolerance = planar_mm
        cylinder_min_radius = cylinder_mm
        thickness_min_distance = thickness_mm

        threshold_basis = "physical_mm_converted_to_source_unit"

    else:

        planar_tolerance = max(
            scale * UNKNOWN_PLANAR_TOLERANCE_RATIO,
            MINIMUM_NUMERICAL_SCALE,
        )

        cylinder_min_radius = max(
            scale * UNKNOWN_MIN_CYLINDER_RADIUS_RATIO,
            MINIMUM_NUMERICAL_SCALE,
        )

        thickness_min_distance = max(
            scale * UNKNOWN_MIN_THICKNESS_RATIO,
            MINIMUM_NUMERICAL_SCALE,
        )

        threshold_basis = "relative_to_mesh_bounding_box"

    area_epsilon = max(
        scale * scale * DEGENERATE_AREA_RELATIVE_EPSILON,
        1e-30,
    )

    return {
        "coordinate_unit": coordinate_unit,
        "threshold_basis": threshold_basis,
        "mesh_scale_coordinate_units": scale,

        "planar_distance_tolerance_coordinate_units":
            float(planar_tolerance),

        "planar_distance_tolerance_mm":
            PLANAR_DISTANCE_TOLERANCE_MM
            if coordinate_unit != "unknown"
            else None,

        "cylinder_min_radius_coordinate_units":
            float(cylinder_min_radius),

        "cylinder_min_radius_mm":
            CYLINDER_MIN_RADIUS_MM
            if coordinate_unit != "unknown"
            else None,

        "thickness_min_distance_coordinate_units":
            float(thickness_min_distance),

        "thickness_min_distance_mm":
            THICKNESS_MIN_DISTANCE_MM
            if coordinate_unit != "unknown"
            else None,

        "degenerate_area_epsilon_coordinate_units_squared":
            float(area_epsilon),
    }


# ============================================================
# EXACT COORDINATE WELDING
# ============================================================

def _build_topology_mesh(
    source_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict]:
    """
    Build a separate topology mesh.

    Important:
    - Source mesh is never modified.
    - Only exactly equal finite coordinates are merged.
    - No distance tolerance is used.
    - Triangle count and triangle order are preserved.
    - Face indices in the returned mesh correspond to source
      face indices.
    - Invalid faces remain isolated and are not used for
      adjacency-based analysis.
    """

    source_vertices = np.asarray(
        source_mesh.vertices,
        dtype=np.float64,
    )

    source_faces = np.asarray(
        source_mesh.faces,
        dtype=np.int64,
    )

    vertex_count = len(source_vertices)
    face_count = len(source_faces)

    if vertex_count == 0 or face_count == 0:
        raise ValueError(
            "STL mesh contains no vertices or triangles."
        )

    finite_vertex_mask = np.all(
        np.isfinite(source_vertices),
        axis=1,
    )

    source_to_topology = np.full(
        vertex_count,
        -1,
        dtype=np.int64,
    )

    finite_ids = np.flatnonzero(
        finite_vertex_mask
    )

    finite_coordinates = source_vertices[
        finite_ids
    ]

    if len(finite_ids):

        unique_coordinates, inverse = np.unique(
            finite_coordinates,
            axis=0,
            return_inverse=True,
        )

        topology_vertices = [
            row.copy()
            for row in unique_coordinates
        ]

        source_to_topology[finite_ids] = inverse

    else:

        unique_coordinates = np.empty(
            (0, 3),
            dtype=np.float64,
        )

        topology_vertices = []

    # Invalid vertices are kept separate.
    # They must never accidentally connect valid triangles.
    invalid_ids = np.flatnonzero(
        ~finite_vertex_mask
    )

    for source_id in invalid_ids:

        topology_id = len(topology_vertices)

        source_to_topology[source_id] = topology_id

        topology_vertices.append(
            np.zeros(3, dtype=np.float64)
        )

    topology_vertices = np.asarray(
        topology_vertices,
        dtype=np.float64,
    )

    topology_faces = source_to_topology[
        source_faces
    ]

    # Validate original face indices.
    valid_indices = np.all(
        (source_faces >= 0)
        & (source_faces < vertex_count),
        axis=1,
    )

    valid_coordinates = np.zeros(
        face_count,
        dtype=bool,
    )

    valid_ids = np.flatnonzero(
        valid_indices
    )

    if len(valid_ids):
        valid_coordinates[valid_ids] = np.all(
            finite_vertex_mask[
                source_faces[valid_ids]
            ],
            axis=1,
        )

    valid_source_faces = (
        valid_indices
        & valid_coordinates
    )

    # Give invalid faces unique dummy vertices.
    # This prevents invalid geometry from creating false
    # adjacency with other invalid faces.
    for face_id in np.flatnonzero(
        ~valid_source_faces
    ):

        new_indices = []

        for _ in range(3):

            new_indices.append(
                len(topology_vertices)
            )

            topology_vertices = np.vstack((
                topology_vertices,
                np.zeros((1, 3), dtype=np.float64),
            ))

        topology_faces[face_id] = new_indices

    topology_mesh = trimesh.Trimesh(
        vertices=topology_vertices,
        faces=topology_faces,
        process=False,
        validate=False,
    )

    topology_mesh.metadata = deepcopy(
        getattr(source_mesh, "metadata", {})
    )

    merged_vertex_count = (
        len(finite_ids) - len(unique_coordinates)
    )

    info = {
        "source_vertex_count": int(vertex_count),
        "topology_vertex_count": int(
            len(topology_vertices)
        ),
        "source_triangle_count": int(face_count),
        "topology_triangle_count": int(
            len(topology_mesh.faces)
        ),
        "exact_duplicate_vertices_merged": int(
            merged_vertex_count
        ),
        "invalid_source_faces": int(
            np.count_nonzero(~valid_source_faces)
        ),
        "welding_method": "exact_coordinate_equality",
        "welding_tolerance_coordinate_units": 0.0,
        "source_mesh_modified": False,
        "face_order_preserved": bool(
            len(topology_mesh.faces) == face_count
        ),
    }

    return topology_mesh, info


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(self, size):

        self.parent = np.arange(
            size,
            dtype=np.int64,
        )

        self.rank = np.zeros(
            size,
            dtype=np.int8,
        )

    def find(self, value):

        while self.parent[value] != value:

            self.parent[value] = self.parent[
                self.parent[value]
            ]

            value = int(self.parent[value])

        return value

    def union(self, a, b):

        root_a = self.find(int(a))
        root_b = self.find(int(b))

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


# ============================================================
# BASIC UTILITIES
# ============================================================

def _safe_float(value, default=None):

    try:
        result = float(value)

        return result if np.isfinite(result) else default

    except (TypeError, ValueError):
        return default


def _get_valid_face_mask(mesh, area_epsilon):

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    face_count = len(faces)

    if face_count == 0:
        return np.zeros(0, dtype=bool)

    valid_indices = np.all(
        (faces >= 0)
        & (faces < len(vertices)),
        axis=1,
    )

    finite_coordinates = np.zeros(
        face_count,
        dtype=bool,
    )

    ids = np.flatnonzero(valid_indices)

    if len(ids):

        finite_coordinates[ids] = np.all(
            np.isfinite(vertices[faces[ids]]),
            axis=(1, 2),
        )

    areas = np.asarray(
        mesh.area_faces,
        dtype=np.float64,
    )

    normals = np.asarray(
        mesh.face_normals,
        dtype=np.float64,
    )

    normal_lengths = np.linalg.norm(
        normals,
        axis=1,
    )

    return (
        valid_indices
        & finite_coordinates
        & np.isfinite(areas)
        & (areas > area_epsilon)
        & np.all(np.isfinite(normals), axis=1)
        & (normal_lengths > 1e-12)
    )


# ============================================================
# SURFACE PATCH SEGMENTATION
# ============================================================

def _segment_faces(mesh, valid_face_mask):

    face_count = len(mesh.faces)

    union_find = UnionFind(face_count)

    adjacency = np.asarray(
        mesh.face_adjacency,
        dtype=np.int64,
    )

    if len(adjacency):

        angles = np.degrees(
            mesh.face_adjacency_angles
        )

        valid_adjacency = (
            valid_face_mask[adjacency[:, 0]]
            & valid_face_mask[adjacency[:, 1]]
            & np.isfinite(angles)
            & (angles <= PATCH_NORMAL_ANGLE_DEG)
        )

        for face_a, face_b in adjacency[valid_adjacency]:

            union_find.union(
                int(face_a),
                int(face_b),
            )

    # Invalid faces remain independent.
    for face_id in np.flatnonzero(
        ~valid_face_mask
    ):
        union_find.parent[face_id] = face_id

    root_to_patch = {}

    labels = np.empty(
        face_count,
        dtype=np.int32,
    )

    next_id = 0

    for face_id in range(face_count):

        root = union_find.find(face_id)

        if root not in root_to_patch:

            root_to_patch[root] = next_id
            next_id += 1

        labels[face_id] = root_to_patch[root]

    return labels


# ============================================================
# CYLINDER FITTING
# ============================================================

def _fit_cylinder(
    mesh,
    face_indices,
    valid_face_mask,
    cylinder_min_radius,
):

    if len(face_indices) < CYLINDER_MIN_FACES:
        return {
            "status": "insufficient_faces",
            "confidence": 0.0,
        }

    valid_ids = face_indices[
        valid_face_mask[face_indices]
    ]

    if len(valid_ids) < CYLINDER_MIN_FACES:
        return {
            "status": "insufficient_valid_faces",
            "confidence": 0.0,
        }

    faces = mesh.faces[valid_ids]

    vertex_ids = np.unique(
        faces.reshape(-1)
    )

    points = mesh.vertices[vertex_ids]

    if len(points) < 8:
        return {
            "status": "insufficient_vertices",
            "confidence": 0.0,
        }

    normals = np.asarray(
        mesh.face_normals[valid_ids],
        dtype=np.float64,
    )

    normal_lengths = np.linalg.norm(
        normals,
        axis=1,
    )

    normals /= np.maximum(
        normal_lengths[:, None],
        1e-12,
    )

    covariance = (
        normals.T @ normals / len(normals)
    )

    try:
        _, eigenvectors = np.linalg.eigh(
            covariance
        )

    except np.linalg.LinAlgError:
        return {
            "status": "fit_failed",
            "confidence": 0.0,
        }

    axis = eigenvectors[:, 0]

    axis_length = np.linalg.norm(axis)

    if axis_length < 1e-12:
        return {
            "status": "fit_failed",
            "confidence": 0.0,
        }

    axis /= axis_length

    reference = np.array([
        1.0,
        0.0,
        0.0,
    ])

    if abs(np.dot(reference, axis)) > 0.9:

        reference = np.array([
            0.0,
            1.0,
            0.0,
        ])

    basis_u = np.cross(
        axis,
        reference,
    )

    basis_u /= max(
        np.linalg.norm(basis_u),
        1e-12,
    )

    basis_v = np.cross(
        axis,
        basis_u,
    )

    basis_v /= max(
        np.linalg.norm(basis_v),
        1e-12,
    )

    projected = np.column_stack((
        points @ basis_u,
        points @ basis_v,
    ))

    x = projected[:, 0]
    y = projected[:, 1]

    matrix = np.column_stack((
        x,
        y,
        np.ones_like(x),
    ))

    target = -(x * x + y * y)

    try:

        coefficients, _, rank, _ = np.linalg.lstsq(
            matrix,
            target,
            rcond=None,
        )

    except np.linalg.LinAlgError:

        return {
            "status": "fit_failed",
            "confidence": 0.0,
        }

    if rank < 3:
        return {
            "status": "fit_failed",
            "confidence": 0.0,
        }

    a, b, c = coefficients

    center_2d = np.array([
        -a / 2.0,
        -b / 2.0,
    ])

    radius_squared = (
        center_2d[0] ** 2
        + center_2d[1] ** 2
        - c
    )

    if radius_squared <= 0:
        return {
            "status": "invalid_radius",
            "confidence": 0.0,
        }

    radius = float(
        np.sqrt(radius_squared)
    )

    if radius < cylinder_min_radius:
        return {
            "status": "radius_too_small",
            "confidence": 0.0,
        }

    radial_distances = np.linalg.norm(
        projected - center_2d,
        axis=1,
    )

    residuals = radial_distances - radius

    rms_error = float(
        np.sqrt(np.mean(residuals ** 2))
    )

    relative_error = (
        rms_error / max(radius, 1e-12)
    )

    axis_center = (
        np.mean(points, axis=0)
        + center_2d[0] * basis_u
        + center_2d[1] * basis_v
    )

    face_centers = np.mean(
        mesh.vertices[mesh.faces[valid_ids]],
        axis=1,
    )

    relative_positions = (
        face_centers - axis_center
    )

    relative_positions -= (
        relative_positions @ axis
    )[:, None] * axis[None, :]

    radial_lengths = np.linalg.norm(
        relative_positions,
        axis=1,
    )

    valid_radial = radial_lengths > 1e-9

    if np.any(valid_radial):

        radial_directions = (
            relative_positions[valid_radial]
            / radial_lengths[valid_radial, None]
        )

        normal_alignment = float(
            np.mean(
                np.sum(
                    radial_directions
                    * normals[valid_radial],
                    axis=1,
                )
            )
        )

    else:
        normal_alignment = 0.0

    status = (
        "candidate"
        if relative_error <= CYLINDER_MAX_RELATIVE_ERROR
        else "poor_fit"
    )

    if status == "candidate" and normal_alignment < -0.5:

        surface_type = "internal_cylindrical_surface"

    elif status == "candidate" and normal_alignment > 0.5:

        surface_type = "external_cylindrical_surface"

    else:

        surface_type = (
            "cylindrical_surface_orientation_uncertain"
        )

    confidence = float(
        np.clip(
            1.0
            - relative_error / CYLINDER_MAX_RELATIVE_ERROR,
            0.0,
            1.0,
        )
    )

    return {
        "status": status,
        "surface_type": surface_type,

        "radius_coordinate_units": radius,
        "diameter_coordinate_units": 2.0 * radius,
        "axis_center_coordinate_units": axis_center.tolist(),
        "rms_error_coordinate_units": rms_error,

        # Legacy compatibility fields.
        "radius_mm": radius,
        "diameter_mm": 2.0 * radius,
        "axis_center_mm": axis_center.tolist(),
        "rms_error_mm": rms_error,

        "axis_direction": axis.tolist(),
        "relative_error": relative_error,
        "normal_alignment": normal_alignment,
        "confidence": confidence,
    }


# ============================================================
# PATCH ANALYSIS
# ============================================================

def _analyze_patch(
    mesh,
    face_indices,
    patch_id,
    valid_face_mask,
    planar_tolerance,
    cylinder_min_radius,
):

    faces = mesh.faces[face_indices]

    triangles = mesh.vertices[faces]

    triangle_areas = mesh.area_faces[face_indices]

    total_area = float(
        np.sum(triangle_areas)
    )

    valid = valid_face_mask[face_indices]

    if not np.any(valid):

        return {
            "patch_id": int(patch_id),
            "classification": "invalid",
            "face_count": int(len(face_indices)),
            "area": total_area,
            "centroid": None,
            "average_normal": None,
            "normal_coherence": None,
            "max_plane_deviation": None,
            "max_plane_deviation_coordinate_units": None,
            "cylindrical_fit": {
                "status": "not_analyzed",
                "confidence": 0.0,
            },
        }

    valid_triangles = triangles[valid]

    valid_areas = triangle_areas[valid]

    valid_ids = face_indices[valid]

    triangle_centroids = np.mean(
        valid_triangles,
        axis=1,
    )

    area_sum = float(
        np.sum(valid_areas)
    )

    if area_sum > 0:

        centroid = np.average(
            triangle_centroids,
            axis=0,
            weights=valid_areas,
        )

    else:

        centroid = np.mean(
            triangle_centroids,
            axis=0,
        )

    normals = mesh.face_normals[valid_ids]

    normal_lengths = np.linalg.norm(
        normals,
        axis=1,
    )

    normalized_normals = (
        normals
        / np.maximum(normal_lengths[:, None], 1e-12)
    )

    weighted_normal = np.average(
        normalized_normals,
        axis=0,
        weights=np.maximum(valid_areas, 1e-12),
    )

    average_normal_length = np.linalg.norm(
        weighted_normal
    )

    if average_normal_length > 1e-12:

        average_normal = (
            weighted_normal / average_normal_length
        )

    else:

        average_normal = np.zeros(3)

    coherence = float(
        average_normal_length
    )

    unique_vertices = np.unique(
        faces[valid].reshape(-1)
    )

    points = mesh.vertices[unique_vertices]

    if len(points) >= 3:

        centered = points - np.mean(
            points,
            axis=0,
        )

        try:

            _, _, vh = np.linalg.svd(
                centered,
                full_matrices=False,
            )

            plane_normal = vh[-1]

            distances = np.abs(
                centered @ plane_normal
            )

            max_deviation = float(
                np.max(distances)
            )

        except np.linalg.LinAlgError:

            max_deviation = None

    else:

        max_deviation = 0.0

    if max_deviation is None:

        classification = "uncertain"

    elif max_deviation <= planar_tolerance:

        classification = "planar"

    else:

        classification = "curved"

    cylindrical_fit = {
        "status": "not_applicable",
        "confidence": 0.0,
    }

    if classification == "curved":

        cylindrical_fit = _fit_cylinder(
            mesh,
            face_indices,
            valid_face_mask,
            cylinder_min_radius,
        )

    return {
        "patch_id": int(patch_id),
        "classification": classification,
        "face_count": int(len(face_indices)),
        "area": total_area,
        "centroid": centroid.tolist(),
        "average_normal": average_normal.tolist(),
        "normal_coherence": coherence,
        "max_plane_deviation": max_deviation,
        "max_plane_deviation_coordinate_units": max_deviation,
        "cylindrical_fit": cylindrical_fit,
    }


# ============================================================
# EDGE ANALYSIS
# ============================================================

def _analyze_edges(mesh):

    edges = np.asarray(
        mesh.edges_sorted,
        dtype=np.int64,
    )

    if len(edges) == 0:

        return {
            "boundary_edge_count": 0,
            "sharp_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "sharp_angle_threshold_deg": SHARP_EDGE_ANGLE_DEG,
            "boundary_edges": [],
            "sharp_edges": [],
        }

    unique_edges, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    boundary_edges = unique_edges[
        counts == 1
    ]

    nonmanifold_edges = unique_edges[
        counts > 2
    ]

    adjacency_edges = np.asarray(
        mesh.face_adjacency_edges,
        dtype=np.int64,
    )

    adjacency_angles = np.degrees(
        mesh.face_adjacency_angles
    )

    sharp_mask = (
        np.isfinite(adjacency_angles)
        & (adjacency_angles >= SHARP_EDGE_ANGLE_DEG)
    )

    vertices = mesh.vertices

    def edge_records(edge_array):

        records = []

        for edge in edge_array:

            p0 = vertices[int(edge[0])]
            p1 = vertices[int(edge[1])]

            if not (
                np.all(np.isfinite(p0))
                and np.all(np.isfinite(p1))
            ):
                continue

            length = float(
                np.linalg.norm(p1 - p0)
            )

            records.append({
                "vertex_indices": [
                    int(edge[0]),
                    int(edge[1]),
                ],
                "start_coordinate_units": p0.tolist(),
                "end_coordinate_units": p1.tolist(),
                "length_coordinate_units": length,

                "start_mm": p0.tolist(),
                "end_mm": p1.tolist(),
                "length_mm": length,
            })

        return records

    sharp_records = []

    for index in np.flatnonzero(sharp_mask):

        edge = adjacency_edges[index]

        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]

        if not (
            np.all(np.isfinite(p0))
            and np.all(np.isfinite(p1))
        ):
            continue

        length = float(
            np.linalg.norm(p1 - p0)
        )

        sharp_records.append({
            "vertex_indices": [
                int(edge[0]),
                int(edge[1]),
            ],
            "start_coordinate_units": p0.tolist(),
            "end_coordinate_units": p1.tolist(),
            "length_coordinate_units": length,
            "dihedral_angle_deg": float(
                adjacency_angles[index]
            ),

            "start_mm": p0.tolist(),
            "end_mm": p1.tolist(),
            "length_mm": length,
        })

    return {
        "boundary_edge_count": int(
            len(boundary_edges)
        ),

        "sharp_edge_count": int(
            len(sharp_records)
        ),

        "nonmanifold_edge_count": int(
            len(nonmanifold_edges)
        ),

        "sharp_angle_threshold_deg": SHARP_EDGE_ANGLE_DEG,

        "boundary_edges": edge_records(
            boundary_edges
        ),

        "sharp_edges": sharp_records,
    }


# ============================================================
# BEND CANDIDATE ANALYSIS
# ============================================================

def _analyze_bends(mesh, patch_labels, patches):

    patch_lookup = {
        int(patch["patch_id"]): patch
        for patch in patches
    }

    adjacency = np.asarray(
        mesh.face_adjacency,
        dtype=np.int64,
    )

    if len(adjacency) == 0:

        return {
            "bend_candidate_count": 0,
            "candidates": [],
        }

    angles = np.degrees(
        mesh.face_adjacency_angles
    )

    grouped = {}

    for index, (face_a, face_b) in enumerate(adjacency):

        patch_a = int(patch_labels[face_a])
        patch_b = int(patch_labels[face_b])

        if patch_a == patch_b:
            continue

        angle = float(angles[index])

        if not np.isfinite(angle) or angle < 1.0:
            continue

        first = patch_lookup.get(patch_a, {})
        second = patch_lookup.get(patch_b, {})

        first_type = first.get("classification")
        second_type = second.get("classification")

        item = {
            "patch_a": patch_a,
            "patch_b": patch_b,
            "surface_type_a": first_type,
            "surface_type_b": second_type,
            "dihedral_angle_deg": angle,
            "possible_bend": (
                first_type == "planar"
                or second_type == "planar"
            ),
        }

        key = tuple(sorted((patch_a, patch_b)))

        grouped.setdefault(key, []).append(item)

    candidates = []

    for (patch_a, patch_b), items in grouped.items():

        angles_for_pair = [
            item["dihedral_angle_deg"]
            for item in items
        ]

        first = patch_lookup.get(patch_a, {})
        second = patch_lookup.get(patch_b, {})

        cylinder_a = first.get("cylindrical_fit", {})
        cylinder_b = second.get("cylindrical_fit", {})

        cylinder = None

        if cylinder_a.get("status") == "candidate":

            cylinder = cylinder_a

        elif cylinder_b.get("status") == "candidate":

            cylinder = cylinder_b

        radius = (
            cylinder.get("radius_coordinate_units")
            if cylinder is not None
            else None
        )

        candidates.append({
            "patch_a": int(patch_a),
            "patch_b": int(patch_b),

            "shared_transition_face_pairs": int(
                len(items)
            ),

            "mean_dihedral_angle_deg": float(
                np.mean(angles_for_pair)
            ),

            "min_dihedral_angle_deg": float(
                np.min(angles_for_pair)
            ),

            "max_dihedral_angle_deg": float(
                np.max(angles_for_pair)
            ),

            "possible_bend": any(
                item["possible_bend"]
                for item in items
            ),

            "estimated_bend_radius_coordinate_units": radius,
            "estimated_bend_radius_mm": radius,

            "radius_confidence": (
                cylinder.get("confidence")
                if cylinder is not None
                else 0.0
            ),
        })

    return {
        "bend_candidate_count": int(
            len(candidates)
        ),
        "candidates": candidates,
        "note": (
            "Bend candidates are geometric transitions. "
            "They are not confirmed manufacturing bends."
        ),
    }


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

def _analyze_components(mesh):

    try:

        components = mesh.split(
            only_watertight=False
        )

    except Exception:

        return {
            "component_count": None,
            "status": "component_split_failed",
            "components": [],
        }

    records = []

    for index, component in enumerate(components):

        vertices = np.asarray(
            component.vertices,
            dtype=np.float64,
        )

        finite = (
            len(vertices) > 0
            and np.all(np.isfinite(vertices))
        )

        if finite:

            bounds = np.asarray(
                component.bounds,
                dtype=float,
            )

            extents = np.asarray(
                component.extents,
                dtype=float,
            )

            center = np.mean(
                vertices,
                axis=0,
            )

            bounds_result = bounds.tolist()
            extents_result = extents.tolist()
            center_result = center.tolist()

        else:

            bounds_result = None
            extents_result = None
            center_result = None

        area = _safe_float(
            component.area
        )

        records.append({
            "component_id": int(index),
            "vertex_count": int(
                len(component.vertices)
            ),
            "triangle_count": int(
                len(component.faces)
            ),

            "surface_area_coordinate_units_squared": area,
            "watertight": bool(
                component.is_watertight
            ),

            "bounds_coordinate_units": bounds_result,
            "dimensions_coordinate_units": extents_result,
            "centroid_coordinate_units": center_result,

            "surface_area_mm2": area,
            "bounds_mm": bounds_result,
            "dimensions_mm": extents_result,
            "centroid_mm": center_result,
        })

    return {
        "component_count": int(len(records)),
        "status": "success",
        "components": records,
        "note": (
            "Connected mesh components do not necessarily "
            "correspond to physical assembly parts."
        ),
    }


# ============================================================
# THICKNESS ESTIMATION
# ============================================================

def _ray_triangle_distance(
    origin,
    direction,
    triangles,
    minimum_distance,
):

    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(
        np.broadcast_to(direction, edge2.shape),
        edge2,
    )

    a = np.einsum(
        "ij,ij->i",
        edge1,
        h,
    )

    valid = np.abs(a) > 1e-10

    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]

    s = origin[None, :] - v0

    u = f * np.einsum(
        "ij,ij->i",
        s,
        h,
    )

    q = np.cross(
        s,
        edge1,
    )

    v = f * np.einsum(
        "ij,j->i",
        q,
        direction,
    )

    t = f * np.einsum(
        "ij,ij->i",
        edge2,
        q,
    )

    intersection = (
        valid
        & (u >= -1e-9)
        & (v >= -1e-9)
        & (u + v <= 1.0 + 1e-9)
        & (t > minimum_distance)
    )

    distances = np.full(
        len(triangles),
        np.inf,
    )

    distances[intersection] = t[intersection]

    return distances


def _estimate_thickness(
    mesh,
    valid_face_mask,
    minimum_distance,
):

    if not mesh.is_watertight:

        return {
            "status": "not_available",
            "reason": "Mesh is not watertight.",
            "sample_count": 0,
        }

    if not mesh.is_winding_consistent:

        return {
            "status": "not_available",
            "reason": "Mesh winding is inconsistent.",
            "sample_count": 0,
        }

    valid_ids = np.flatnonzero(
        valid_face_mask
    )

    if len(valid_ids) == 0:

        return {
            "status": "not_available",
            "reason": "No valid triangles.",
            "sample_count": 0,
        }

    sample_count = min(
        MAX_THICKNESS_SAMPLES,
        len(valid_ids),
    )

    positions = np.linspace(
        0,
        len(valid_ids) - 1,
        sample_count,
        dtype=int,
    )

    selected_faces = valid_ids[positions]

    triangles = mesh.vertices[
        mesh.faces
    ]

    try:

        orientation = (
            -1.0 if mesh.volume < 0 else 1.0
        )

    except Exception:

        return {
            "status": "not_available",
            "reason": "Unable to determine mesh orientation.",
            "sample_count": 0,
        }

    scale = _get_mesh_scale(mesh)

    ray_offset = max(
        scale * 1e-7,
        1e-12,
    )

    values = []

    for face_id in selected_faces:

        triangle = triangles[face_id]

        origin = np.mean(
            triangle,
            axis=0,
        )

        normal = (
            mesh.face_normals[face_id]
            * orientation
        )

        normal_length = np.linalg.norm(normal)

        if normal_length < 1e-12:
            continue

        normal /= normal_length

        direction = -normal

        origin = (
            origin + direction * ray_offset
        )

        distances = _ray_triangle_distance(
            origin,
            direction,
            triangles,
            minimum_distance,
        )

        nearest = float(
            np.min(distances)
        )

        if np.isfinite(nearest):
            values.append(nearest)

    if not values:

        return {
            "status": "not_available",
            "reason": (
                "No valid opposite-surface intersections."
            ),
            "sample_count": 0,
        }

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    mean = float(np.mean(values))
    median = float(np.median(values))

    return {
        "status": "estimated",
        "method": "inward_normal_ray_intersection",
        "sample_count": int(len(values)),

        "minimum_coordinate_units": minimum,
        "maximum_coordinate_units": maximum,
        "mean_coordinate_units": mean,
        "median_coordinate_units": median,

        "minimum_mm": minimum,
        "maximum_mm": maximum,
        "mean_mm": mean,
        "median_mm": median,

        "note": (
            "Local ray-based estimates. Results may be unreliable "
            "near edges, cavities, and complex internal geometry."
        ),
    }


# ============================================================
# MAIN GEOMETRY ANALYSIS
# ============================================================

def analyze_geometry(
    mesh: trimesh.Trimesh,
    coordinate_unit: str | None = None,
) -> tuple[dict, np.ndarray]:
    """
    Stage 1 geometry analysis.

    The source mesh is never modified.

    A separate exact-coordinate-welded topology mesh is used
    for adjacency, segmentation, edge, component, and thickness
    analysis.

    patch_labels always follow original source triangle order.
    """

    if not isinstance(mesh, trimesh.Trimesh):

        raise TypeError(
            "mesh must be a trimesh.Trimesh object"
        )

    source_vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    source_faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    # Preserve source data for verification.
    original_vertices = source_vertices.copy()
    original_faces = source_faces.copy()

    if len(source_vertices) == 0 or len(source_faces) == 0:

        raise ValueError(
            "Mesh contains no vertices or triangles."
        )

    resolved_unit = _get_coordinate_unit(
        mesh,
        coordinate_unit,
    )

    length_unit = _length_unit_name(
        resolved_unit
    )

    area_unit = _area_unit_name(
        resolved_unit
    )

    volume_unit = _volume_unit_name(
        resolved_unit
    )

    # --------------------------------------------------------
    # Build separate topology mesh.
    # --------------------------------------------------------

    topology_mesh, topology_info = _build_topology_mesh(
        mesh
    )

    vertices = topology_mesh.vertices
    faces = topology_mesh.faces

    thresholds = _build_threshold_configuration(
        mesh,
        resolved_unit,
    )

    planar_tolerance = thresholds[
        "planar_distance_tolerance_coordinate_units"
    ]

    cylinder_min_radius = thresholds[
        "cylinder_min_radius_coordinate_units"
    ]

    thickness_min_distance = thresholds[
        "thickness_min_distance_coordinate_units"
    ]

    area_epsilon = thresholds[
        "degenerate_area_epsilon_coordinate_units_squared"
    ]

    finite_vertices = bool(
        len(source_vertices) > 0
        and np.all(np.isfinite(source_vertices))
    )

    valid_face_mask = _get_valid_face_mask(
        topology_mesh,
        area_epsilon,
    )

    face_areas = np.asarray(
        topology_mesh.area_faces,
        dtype=np.float64,
    )

    face_normals = np.asarray(
        topology_mesh.face_normals,
        dtype=np.float64,
    )

    normal_lengths = np.linalg.norm(
        face_normals,
        axis=1,
    )

    degenerate_count = int(
        np.count_nonzero(
            ~np.isfinite(face_areas)
            | (face_areas <= area_epsilon)
        )
    )

    zero_normal_count = int(
        np.count_nonzero(
            ~np.isfinite(normal_lengths)
            | (normal_lengths <= 1e-12)
        )
    )

    # --------------------------------------------------------
    # Segmentation.
    # --------------------------------------------------------

    patch_labels = _segment_faces(
        topology_mesh,
        valid_face_mask,
    )

    patches = []

    for patch_id in np.unique(patch_labels):

        face_indices = np.flatnonzero(
            patch_labels == patch_id
        )

        patches.append(
            _analyze_patch(
                topology_mesh,
                face_indices,
                int(patch_id),
                valid_face_mask,
                planar_tolerance,
                cylinder_min_radius,
            )
        )

    planar_count = sum(
        p["classification"] == "planar"
        for p in patches
    )

    curved_count = sum(
        p["classification"] == "curved"
        for p in patches
    )

    invalid_patch_count = sum(
        p["classification"] in (
            "invalid",
            "uncertain",
        )
        for p in patches
    )

    # --------------------------------------------------------
    # Basic geometry.
    # Use source mesh coordinates, not modified coordinates.
    # --------------------------------------------------------

    if finite_vertices:

        bounds = np.asarray(
            mesh.bounds,
            dtype=float,
        )

        extents = np.asarray(
            mesh.extents,
            dtype=float,
        )

        dimensions = {
            "x": float(extents[0]),
            "y": float(extents[1]),
            "z": float(extents[2]),
        }

        bounds_result = bounds.tolist()

    else:

        dimensions = {
            "x": None,
            "y": None,
            "z": None,
        }

        bounds_result = None

    surface_area = _safe_float(
        mesh.area,
        0.0,
    )

    # Topology-derived properties must use the topology mesh.
    is_watertight = bool(
        topology_mesh.is_watertight
    )

    winding_consistent = bool(
        topology_mesh.is_winding_consistent
    )

    is_volume = bool(
        topology_mesh.is_volume
    )

    try:

        volume = (
            float(abs(topology_mesh.volume))
            if is_volume
            else None
        )

    except Exception:

        volume = None

    try:

        center_of_mass = (
            np.asarray(
                topology_mesh.center_mass,
                dtype=float,
            ).tolist()
            if is_volume
            else None
        )

    except Exception:

        center_of_mass = None

    try:

        connected_components = len(
            topology_mesh.split(
                only_watertight=False
            )
        )

    except Exception:

        connected_components = None

    # --------------------------------------------------------
    # Advanced geometry.
    # --------------------------------------------------------

    edge_analysis = _analyze_edges(
        topology_mesh
    )

    component_analysis = _analyze_components(
        topology_mesh
    )

    bend_analysis = _analyze_bends(
        topology_mesh,
        patch_labels,
        patches,
    )

    thickness_analysis = _estimate_thickness(
        topology_mesh,
        valid_face_mask,
        thickness_min_distance,
    )

    cylindrical_features = []
    hole_candidates = []

    for patch in patches:

        cylinder = patch.get(
            "cylindrical_fit",
            {},
        )

        if cylinder.get("status") != "candidate":
            continue

        radius = cylinder.get(
            "radius_coordinate_units"
        )

        diameter = cylinder.get(
            "diameter_coordinate_units"
        )

        axis_center = cylinder.get(
            "axis_center_coordinate_units"
        )

        rms_error = cylinder.get(
            "rms_error_coordinate_units"
        )

        feature = {
            "patch_id": int(patch["patch_id"]),

            "surface_type": cylinder.get(
                "surface_type"
            ),

            "radius_coordinate_units": radius,
            "diameter_coordinate_units": diameter,
            "axis_center_coordinate_units": axis_center,
            "rms_error_coordinate_units": rms_error,

            "radius_mm": radius,
            "diameter_mm": diameter,
            "axis_center_mm": axis_center,
            "rms_error_mm": rms_error,

            "axis_direction": cylinder.get(
                "axis_direction"
            ),

            "confidence": cylinder.get(
                "confidence"
            ),
        }

        cylindrical_features.append(feature)

        if cylinder.get("surface_type") == (
            "internal_cylindrical_surface"
        ):

            hole_candidates.append({
                **feature,

                "feature_type": "hole_candidate",

                "note": (
                    "Internal cylindrical surface candidate. "
                    "This is not yet a confirmed hole. "
                    "Through-hole or blind-hole classification "
                    "requires additional topology analysis."
                ),
            })

    # --------------------------------------------------------
    # Verify original source mesh unchanged.
    # --------------------------------------------------------

    source_unchanged = (
        np.array_equal(
            original_vertices,
            np.asarray(mesh.vertices),
        )
        and np.array_equal(
            original_faces,
            np.asarray(mesh.faces),
        )
    )

    if not source_unchanged:

        raise RuntimeError(
            "Source mesh was unexpectedly modified during analysis."
        )

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    report = {
        "analysis": {
            "stage": 1,
            "name": "Geometry Analysis",

            "coordinate_unit": resolved_unit,
            "length_unit": length_unit,
            "area_unit": area_unit,
            "volume_unit": volume_unit,

            "coordinates_rescaled": False,

            "segmentation_normal_angle_deg":
                PATCH_NORMAL_ANGLE_DEG,

            **thresholds,
        },

        "mesh": {
            "vertex_count": int(
                len(source_vertices)
            ),

            "triangle_count": int(
                len(source_faces)
            ),

            "connected_components": connected_components,

            "finite_vertices": finite_vertices,

            "watertight": is_watertight,
            "winding_consistent": winding_consistent,
            "is_volume": is_volume,

            "positive_volume": (
                bool(topology_mesh.volume > 0)
                if is_volume
                else None
            ),

            "degenerate_faces": degenerate_count,
            "zero_normal_faces": zero_normal_count,

            "valid_faces": int(
                np.count_nonzero(valid_face_mask)
            ),

            "invalid_faces": int(
                len(faces)
                - np.count_nonzero(valid_face_mask)
            ),

            "topology_reconstruction": topology_info,
        },

        "geometry": {
            "dimensions_coordinate_units": dimensions,
            "bounds_coordinate_units": bounds_result,

            "surface_area_coordinate_units_squared":
                surface_area,

            "volume_coordinate_units_cubed": volume,

            "center_of_mass_coordinate_units":
                center_of_mass,

            # Legacy compatibility fields.
            "dimensions_mm": dimensions,
            "bounds_mm": bounds_result,
            "surface_area_mm2": surface_area,
            "volume_mm3": volume,
            "center_of_mass_mm": center_of_mass,
        },

        "surface_segmentation": {
            "patch_count": int(len(patches)),
            "planar_patch_count": int(planar_count),
            "curved_patch_count": int(curved_count),
            "invalid_patch_count": int(invalid_patch_count),
            "patches": patches,
        },

        "advanced_geometry": {
            "edges": edge_analysis,

            "cylindrical_features": {
                "count": int(
                    len(cylindrical_features)
                ),
                "features": cylindrical_features,
            },

            "hole_candidates": {
                "count": int(
                    len(hole_candidates)
                ),
                "features": hole_candidates,
                "note": (
                    "Only internal cylindrical surface candidates "
                    "are detected. A cylindrical surface alone does "
                    "not prove a through-hole or blind-hole."
                ),
            },

            "bends": bend_analysis,

            "components": component_analysis,

            "thickness": thickness_analysis,

            "limitations": [
                "STL does not contain native CAD feature definitions.",

                "Original STL coordinates and triangle order are preserved.",

                "A separate topology mesh is created for adjacency analysis.",

                "Only exactly equal vertex coordinates are welded.",

                "No distance tolerance is used for vertex welding.",

                "Physical units are unknown unless explicitly provided.",

                "Legacy fields ending in mm are compatibility fields "
                "and are not guaranteed to represent millimeters.",

                "A cylindrical surface is not automatically a hole.",

                "Connected components are not necessarily physical parts.",

                "Thickness estimates require a watertight mesh.",

                "Ray-based thickness estimates may be unreliable near "
                "edges, cavities, and complex internal geometry.",

                "Bend candidates are not confirmed manufacturing bends.",

                "Spray accessibility and robot reachability are not "
                "analyzed here.",
            ],
        },
    }


    return report, patch_labels


def _detect_holes_from_step_brep(mesh: trimesh.Trimesh) -> list[dict]:
    """
    Detect cylindrical bore candidates from exact Open CASCADE STEP data.

    The STEP loader stores the analytic cylinder axis/radius per original
    B-Rep face. Faces on the same cylindrical support are merged by coaxial
    geometry; no tessellated-mesh cylinder fitting is involved.
    """
    report = mesh.metadata.get("step_geometry_report", {})
    face_records = report.get("faces", [])
    cylinders = []

    for face in face_records:
        cylinder = face.get("cylinder_geometry_mm")
        if face.get("surface_type") != "CYLINDER" or not isinstance(cylinder, dict):
            continue
        try:
            location = np.asarray(cylinder["axis_location_mm"], dtype=np.float64)
            axis = np.asarray(cylinder["axis_direction"], dtype=np.float64)
            radius = float(cylinder["radius_mm"])
            centroid = np.asarray(face["centroid_mm"], dtype=np.float64)
            if location.shape != (3,) or axis.shape != (3,) or centroid.shape != (3,):
                continue
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-12 or radius <= 0.0:
                continue
            axis /= axis_norm
            if not (np.all(np.isfinite(location)) and np.all(np.isfinite(axis))
                    and np.all(np.isfinite(centroid)) and np.isfinite(radius)):
                continue
            cylinders.append({
                "face_id": int(face["face_id"]),
                "location": location,
                "axis": axis,
                "radius": radius,
                "centroid": centroid,
                "orientation": str(face.get("orientation", "UNKNOWN")),
                "adjacent_face_ids": [int(v) for v in face.get("adjacent_face_ids", [])],
            })
        except (KeyError, TypeError, ValueError):
            continue

    # Group face records sharing the same analytic cylinder. Direction sign
    # is irrelevant; axis-line distance is measured perpendicular to axis.
    groups = []
    for item in cylinders:
        assigned = False
        for group in groups:
            ref = group[0]
            parallel = abs(float(np.dot(item["axis"], ref["axis"])))
            delta = item["location"] - ref["location"]
            line_offset = delta - float(np.dot(delta, ref["axis"])) * ref["axis"]
            same_radius = abs(item["radius"] - ref["radius"]) <= max(
                0.02, ref["radius"] * 1e-5
            )
            if parallel >= 1.0 - 1e-8 and np.linalg.norm(line_offset) <= 0.05 and same_radius:
                group.append(item)
                assigned = True
                break
        if not assigned:
            groups.append([item])

    candidates = []
    for group in groups:
        reversed_faces = [v for v in group if v["orientation"] == "REVERSED"]
        forward_faces = [v for v in group if v["orientation"] != "REVERSED"]
        # For an outward-oriented solid, the bore wall's material-outward
        # normal points toward the cavity, opposite the analytic cylinder
        # normal; Open CASCADE represents this with a reversed face.
        if not reversed_faces or len(reversed_faces) <= len(forward_faces):
            continue

        ref = group[0]
        axis = ref["axis"].copy()
        location = ref["location"].copy()
        axial_positions = [
            float(np.dot(v["centroid"] - location, axis)) for v in group
        ]
        axis_center = location + float(np.mean(axial_positions)) * axis
        closest_axis_point = location - float(np.dot(location, axis)) * axis
        axis_distance = float(np.linalg.norm(closest_axis_point))
        center_distance = float(np.linalg.norm(axis_center))
        if not np.isfinite(axis_distance + center_distance):
            continue

        face_ids = sorted(v["face_id"] for v in group)
        adjacent_ids = sorted({
            neighbor
            for v in group
            for neighbor in v["adjacent_face_ids"]
            if neighbor not in face_ids
        })
        radius = float(np.mean([v["radius"] for v in group]))
        orientation_ratio = len(reversed_faces) / len(group)
        candidates.append({
            "hole_id": len(candidates) + 1,
            "patch_id": None,
            "brep_face_ids": face_ids,
            "adjacent_face_ids": adjacent_ids,
            "cylindrical_face_count": len(group),
            "axis_center_coordinate_units": axis_center.tolist(),
            "axis_direction": axis.tolist(),
            "closest_axis_point_to_origin_coordinate_units": closest_axis_point.tolist(),
            "radius_coordinate_units": radius,
            "diameter_coordinate_units": 2.0 * radius,
            "origin_to_axis_distance_coordinate_units": axis_distance,
            "origin_to_axis_center_distance_coordinate_units": center_distance,
            "coordinate_unit": "mm",
            "confidence": float(orientation_ratio),
            "fit_relative_error": 0.0,
            "classification": "STEP_BREP_INTERNAL_CYLINDER_CANDIDATE",
            "note": "Exact STEP analytic cylinder; through/blind status requires boundary and solid-topology analysis.",
        })
    return candidates


def detect_hole_candidates(
    mesh: trimesh.Trimesh,
    coordinate_unit: str | None = None,
) -> list[dict]:
    """
    Detect STEP hole-axis candidates from exact B-Rep cylinders first.
    The tessellated-mesh fitting path remains as a compatibility fallback
    for meshes that do not contain exact STEP cylinder metadata.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh object.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return []

    step_report = mesh.metadata.get("step_geometry_report")
    if isinstance(step_report, dict) and step_report.get("faces"):
        exact_candidates = _detect_holes_from_step_brep(mesh)
        if exact_candidates:
            return exact_candidates
        # Do not silently fall back to noisy mesh inference for STEP files:
        # return no exact candidates and let the caller display diagnostics.
        return []

    unit = _get_coordinate_unit(mesh, coordinate_unit)
    topology_mesh, _ = _build_topology_mesh(mesh)
    thresholds = _build_threshold_configuration(mesh, unit)
    valid_face_mask = _get_valid_face_mask(
        topology_mesh,
        thresholds["degenerate_area_epsilon_coordinate_units_squared"],
    )
    patch_labels = _segment_faces(topology_mesh, valid_face_mask)
    holes = []

    for patch_id in np.unique(patch_labels):
        face_indices = np.flatnonzero(patch_labels == patch_id)
        if len(face_indices) < CYLINDER_MIN_FACES:
            continue
        fit = _fit_cylinder(
            topology_mesh, face_indices, valid_face_mask,
            thresholds["cylinder_min_radius_coordinate_units"],
        )
        if fit.get("status") != "candidate" or fit.get("surface_type") != "internal_cylindrical_surface":
            continue
        axis = np.asarray(fit["axis_direction"], dtype=np.float64)
        center = np.asarray(fit["axis_center_coordinate_units"], dtype=np.float64)
        axis_length = float(np.linalg.norm(axis))
        if axis.shape != (3,) or center.shape != (3,) or axis_length <= 1e-12:
            continue
        axis /= axis_length
        closest_axis_point = center - float(np.dot(center, axis)) * axis
        axis_distance = float(np.linalg.norm(closest_axis_point))
        center_distance = float(np.linalg.norm(center))
        if not np.isfinite(axis_distance + center_distance):
            continue
        holes.append({
            "hole_id": len(holes) + 1,
            "patch_id": int(patch_id),
            "axis_center_coordinate_units": center.tolist(),
            "axis_direction": axis.tolist(),
            "closest_axis_point_to_origin_coordinate_units": closest_axis_point.tolist(),
            "radius_coordinate_units": float(fit["radius_coordinate_units"]),
            "diameter_coordinate_units": float(fit["diameter_coordinate_units"]),
            "origin_to_axis_distance_coordinate_units": axis_distance,
            "origin_to_axis_center_distance_coordinate_units": center_distance,
            "coordinate_unit": unit,
            "confidence": float(fit.get("confidence", 0.0)),
            "fit_relative_error": float(fit.get("relative_error", 0.0)),
            "classification": "MESH_INTERNAL_CYLINDER_FALLBACK",
            "note": "Mesh-derived candidate; not exact STEP topology.",
        })
    return holes


# ============================================================
# NUMERICAL TESTS
# ============================================================

def run_geometry_analyzer_tests():

    print("=" * 65)
    print("GEOMETRY ANALYZER NUMERICAL TESTS")
    print("=" * 65)

    # --------------------------------------------------------
    # Test 1: Box geometry and topology.
    # --------------------------------------------------------

    box = trimesh.creation.box(
        extents=[100.0, 80.0, 20.0]
    )

    box.metadata["source_unit"] = "unknown"

    original_vertices = box.vertices.copy()
    original_faces = box.faces.copy()

    report, labels = analyze_geometry(box)

    assert len(labels) == len(box.faces)

    assert report["mesh"]["triangle_count"] == 12

    assert np.isclose(
        report["geometry"]["dimensions_coordinate_units"]["x"],
        100.0,
    )

    assert np.isclose(
        report["geometry"]["dimensions_coordinate_units"]["y"],
        80.0,
    )

    assert np.isclose(
        report["geometry"]["dimensions_coordinate_units"]["z"],
        20.0,
    )

    assert report["mesh"]["watertight"] is True

    assert (
        report["advanced_geometry"]["components"]["component_count"]
        == 1
    )

    assert (
        report["advanced_geometry"]["edges"]["sharp_edge_count"]
        > 0
    )

    assert np.array_equal(
        original_vertices,
        box.vertices,
    )

    assert np.array_equal(
        original_faces,
        box.faces,
    )

    print("[PASS] Box dimensions and topology")

    # --------------------------------------------------------
    # Test 2: Disconnected boxes.
    # --------------------------------------------------------

    box_a = trimesh.creation.box(
        extents=[10.0, 10.0, 10.0]
    )

    box_b = trimesh.creation.box(
        extents=[10.0, 10.0, 10.0]
    )

    box_b.apply_translation([
        30.0,
        0.0,
        0.0,
    ])

    combined = trimesh.util.concatenate([
        box_a,
        box_b,
    ])

    report, labels = analyze_geometry(combined)

    assert report["mesh"]["connected_components"] == 2

    assert (
        report["advanced_geometry"]["components"]["component_count"]
        == 2
    )

    print("[PASS] Connected component analysis")

    # --------------------------------------------------------
    # Test 3: Planar patch classification.
    # --------------------------------------------------------

    plane = trimesh.creation.box(
        extents=[100.0, 100.0, 1.0]
    )

    report, labels = analyze_geometry(plane)

    assert report["surface_segmentation"]["patch_count"] > 0

    assert report["surface_segmentation"]["planar_patch_count"] > 0

    print("[PASS] Planar patch classification")

    # --------------------------------------------------------
    # Test 4: Cylinder.
    # --------------------------------------------------------

    cylinder = trimesh.creation.cylinder(
        radius=10.0,
        height=50.0,
        sections=64,
    )

    report, labels = analyze_geometry(cylinder)

    assert len(labels) == len(cylinder.faces)

    assert report["mesh"]["watertight"] is True

    assert (
        report["geometry"]["dimensions_coordinate_units"]["z"]
        > 49.0
    )

    print("[PASS] Cylinder geometry analysis")

    # --------------------------------------------------------
    # Test 5: Thickness.
    # --------------------------------------------------------

    report, labels = analyze_geometry(box)

    thickness = report["advanced_geometry"]["thickness"]

    assert thickness["status"] == "estimated"

    assert thickness["sample_count"] > 0

    assert thickness["minimum_coordinate_units"] > 0

    assert (
        thickness["maximum_coordinate_units"]
        >= thickness["minimum_coordinate_units"]
    )

    print("[PASS] Thickness ray estimation")

    # --------------------------------------------------------
    # Test 6: Unit metadata.
    # --------------------------------------------------------

    test_mesh = trimesh.creation.box(
        extents=[10.0, 20.0, 30.0]
    )

    test_mesh.metadata["source_unit"] = "cm"

    report, labels = analyze_geometry(test_mesh)

    assert report["analysis"]["coordinate_unit"] == "cm"

    assert np.isclose(
        report["analysis"][
            "planar_distance_tolerance_coordinate_units"
        ],
        0.05,
    )

    assert np.isclose(
        report["analysis"][
            "cylinder_min_radius_coordinate_units"
        ],
        0.05,
    )

    assert np.isclose(
        report["analysis"][
            "thickness_min_distance_coordinate_units"
        ],
        0.01,
    )

    print("[PASS] Unit metadata and threshold conversion")

    # --------------------------------------------------------
    # Test 7: Inch conversion.
    # --------------------------------------------------------

    inch_mesh = trimesh.creation.box(
        extents=[1.0, 2.0, 3.0]
    )

    inch_mesh.metadata["source_unit"] = "in"

    report, labels = analyze_geometry(inch_mesh)

    assert np.isclose(
        report["analysis"][
            "planar_distance_tolerance_coordinate_units"
        ],
        0.5 / 25.4,
    )

    print("[PASS] Inch threshold conversion")

    # --------------------------------------------------------
    # Test 8: Unknown unit.
    # --------------------------------------------------------

    unknown_mesh = trimesh.creation.box(
        extents=[100.0, 80.0, 20.0]
    )

    unknown_mesh.metadata["source_unit"] = "unknown"

    report, labels = analyze_geometry(unknown_mesh)

    assert (
        report["analysis"]["threshold_basis"]
        == "relative_to_mesh_bounding_box"
    )

    assert (
        report["analysis"]["planar_distance_tolerance_mm"]
        is None
    )

    print("[PASS] Unknown-unit relative thresholds")

    # --------------------------------------------------------
    # Test 9: Explicit unit override.
    # --------------------------------------------------------

    override_mesh = trimesh.creation.box(
        extents=[10.0, 20.0, 30.0]
    )

    override_mesh.metadata["source_unit"] = "unknown"

    original = override_mesh.vertices.copy()

    report, labels = analyze_geometry(
        override_mesh,
        coordinate_unit="cm",
    )

    assert report["analysis"]["coordinate_unit"] == "cm"

    assert np.isclose(
        report["analysis"][
            "planar_distance_tolerance_coordinate_units"
        ],
        0.05,
    )

    assert np.array_equal(
        original,
        override_mesh.vertices,
    )

    print("[PASS] Explicit unit override")

    # --------------------------------------------------------
    # Test 10: STL-style duplicated triangle vertices.
    # --------------------------------------------------------

    box = trimesh.creation.box(
        extents=[20.0, 30.0, 40.0]
    )

    source_vertices = box.vertices[
        box.faces
    ].reshape(-1, 3).copy()

    source_faces = np.arange(
        len(source_vertices),
        dtype=np.int64,
    ).reshape(-1, 3)

    raw_stl_style = trimesh.Trimesh(
        vertices=source_vertices.copy(),
        faces=source_faces.copy(),
        process=False,
        validate=False,
    )

    raw_stl_style.metadata["source_unit"] = "mm"

    original_vertices = raw_stl_style.vertices.copy()
    original_faces = raw_stl_style.faces.copy()

    # Raw mesh should not have normal shared-index topology.
    raw_adjacency_count = len(
        raw_stl_style.face_adjacency
    )

    report, labels = analyze_geometry(raw_stl_style)

    topology = report["mesh"]["topology_reconstruction"]

    assert len(labels) == len(source_faces)

    assert topology["source_vertex_count"] == 36

    assert topology["source_triangle_count"] == 12

    assert topology["exact_duplicate_vertices_merged"] > 0

    assert topology["topology_triangle_count"] == 12

    assert topology["source_mesh_modified"] is False

    assert topology["face_order_preserved"] is True

    assert report["mesh"]["watertight"] is True

    assert (
        report["advanced_geometry"]["components"]["component_count"]
        == 1
    )

    assert np.array_equal(
        original_vertices,
        raw_stl_style.vertices,
    )

    assert np.array_equal(
        original_faces,
        raw_stl_style.faces,
    )

    print(
        "[PASS] Exact duplicate welding and source mesh preservation"
    )

    print(
        "       Raw adjacency edges:",
        raw_adjacency_count,
    )

    print(
        "       Welded topology vertices:",
        topology["topology_vertex_count"],
    )

    # --------------------------------------------------------
    # Test 11: JSON serialization.
    # --------------------------------------------------------

    json.dumps(
        report,
        allow_nan=False,
    )

    print("[PASS] JSON serialization")

    # --------------------------------------------------------
    # Test 12: Patch labels follow source face order.
    # --------------------------------------------------------

    assert len(labels) == len(raw_stl_style.faces)

    assert labels.dtype == np.int32

    print("[PASS] Original triangle index compatibility")

    print("=" * 65)
    print("ALL GEOMETRY ANALYZER TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_geometry_analyzer_tests()
