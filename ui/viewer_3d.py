
from __future__ import annotations

import copy
import math
import time
import numpy as np
import pyvista as pv

from core.spray_coverage import (
    build_paintable_face_mask,
    calculate_area_weighted_coverage,
    is_line_of_sight_occluded,
    mark_sample_hits,
)
from core.coating_deposition import (
    accumulate_double_beta_deposition_step,
    calculate_cone_deposition_step,
    calculate_film_thickness_statistics,
    update_face_thickness_from_changed_samples,
)
from core.coating_process import coating_model_readiness
from core.coating_process import DEFAULT_COATING_PROCESS
from core.spray_coverage_optimizer import (
    append_correction_strokes_to_trajectory,
    build_coverage_correction_plan,
)
from stages.trajectory_safety import aggregate_sample_safety_by_face

from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersGeneral import vtkOBBTree
from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter

from PySide6.QtCore import QEvent, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QSlider,
    QProgressBar,
    QColorDialog,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser


def _build_spray_footprint_overlay(
    mesh,
    nozzle_points,
    spray_directions,
    cone_angle_deg,
    rays_per_ring=72,
    radial_rings=(
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.0,
    ),
    surface=None,
    locator=None,
):
    """Intersect each spray cone with the actual product mesh.

    The cone interior is sampled with dense concentric angular rings. Only
    first-hit mesh triangles are returned, so each overlay is a filled
    footprint on the product surface, not a free-space cone projection.
    The orange outline follows the boundary of the intersected mesh cells.
    """
    nozzles = np.asarray(nozzle_points, dtype=np.float64)
    directions = np.asarray(spray_directions, dtype=np.float64)
    if (
        nozzles.ndim != 2 or nozzles.shape[1] != 3
        or directions.shape != nozzles.shape or len(nozzles) == 0
    ):
        raise ValueError("Nozzle points and spray directions must be matching (N, 3) arrays.")
    if not np.all(np.isfinite(nozzles)) or not np.all(np.isfinite(directions)):
        raise ValueError("Spray footprint inputs must be finite.")
    if not np.isfinite(cone_angle_deg) or not 0.0 < cone_angle_deg < 179.0:
        raise ValueError("cone_angle_deg must be in (0, 179).")
    if int(rays_per_ring) < 8:
        raise ValueError("rays_per_ring must be at least 8.")

    # Reuse a prepared PyVista surface and VTK locator for interactive
    # cursor updates; building the spatial index on every mouse event is costly.
    if surface is None:
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise TypeError("Product mesh must expose trimesh-style vertices and faces.")
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
        if (
            mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3
            or mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3
            or len(mesh_vertices) == 0 or len(mesh_faces) == 0
        ):
            return []
        if (
            not np.all(np.isfinite(mesh_vertices))
            or np.any(mesh_faces < 0)
            or np.any(mesh_faces >= len(mesh_vertices))
        ):
            raise ValueError("Product mesh contains invalid vertices or triangle indices.")
        packed_faces = np.column_stack((
            np.full(len(mesh_faces), 3, dtype=np.int64),
            mesh_faces,
        )).ravel()
        surface = pv.PolyData(mesh_vertices, packed_faces).clean()
    if surface.n_cells == 0:
        return []
    bounds = np.asarray(surface.bounds, dtype=np.float64).reshape(3, 2)
    diagonal = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
    if not np.isfinite(diagonal) or diagonal <= 1e-9:
        return []

    if locator is None:
        locator = vtkOBBTree()
        locator.SetDataSet(surface)
        locator.BuildLocator()
    half_angle = math.radians(float(cone_angle_deg)) * 0.5
    ray_length = diagonal * 4.0
    ray_count = int(rays_per_ring)
    ring_values = tuple(float(v) for v in radial_rings)
    if any(not 0.0 < value <= 1.0 for value in ring_values):
        raise ValueError("radial_rings values must be in (0, 1].")

    footprints = []
    for nozzle, axis in zip(nozzles, directions):
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1e-12:
            footprints.append((pv.PolyData(), pv.PolyData()))
            continue
        axis = axis / axis_length

        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, axis))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        tangent_u = np.cross(axis, reference)
        tangent_u /= max(float(np.linalg.norm(tangent_u)), 1e-12)
        tangent_v = np.cross(axis, tangent_u)
        tangent_v /= max(float(np.linalg.norm(tangent_v)), 1e-12)

        hit_cell_ids = set()
        ray_directions = [axis]
        for radial_fraction in ring_values:
            theta = half_angle * radial_fraction
            for ray_index in range(ray_count):
                phi = 2.0 * math.pi * ray_index / ray_count
                radial = math.cos(phi) * tangent_u + math.sin(phi) * tangent_v
                ray = math.cos(theta) * axis + math.sin(theta) * radial
                ray_directions.append(ray / np.linalg.norm(ray))

        for ray in ray_directions:
            endpoint = nozzle + ray * ray_length
            intersections = vtkPoints()
            intersected_cells = vtkIdList()
            hit = locator.IntersectWithLine(
                nozzle.tolist(), endpoint.tolist(), intersections, intersected_cells
            )
            if not hit or intersected_cells.GetNumberOfIds() == 0:
                continue
            # Use only the first surface encountered by this cone ray.
            nearest_index = 0
            nearest_distance = float("inf")
            for hit_index in range(intersected_cells.GetNumberOfIds()):
                hit_point = np.asarray(intersections.GetPoint(hit_index), dtype=np.float64)
                distance = float(np.dot(hit_point - nozzle, hit_point - nozzle))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_index = hit_index
            cell_id = int(intersected_cells.GetId(nearest_index))
            if 0 <= cell_id < surface.n_cells:
                hit_cell_ids.add(cell_id)

        if not hit_cell_ids:
            footprints.append((pv.PolyData(), pv.PolyData()))
            continue

        hit_ids = np.asarray(sorted(hit_cell_ids), dtype=np.int64)
        # Preserve the exact source triangle IDs for the persistent coating field.
        patch = surface.extract_cells(hit_ids)
        patch.cell_data["InteractiveOriginalCellId"] = hit_ids
        boundary = patch.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        footprints.append((patch, boundary))

    return footprints

# ============================================================
# UNIT CONFIGURATION
# ============================================================

MM_PER_UNIT = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
}

DISPLAY_UNITS = ("mm", "cm", "m", "in", "ft")

# Interactive spray working-distance limits controlled by the mouse wheel.
# The distance is always represented in millimetres in the UI.
INTERACTIVE_SPRAY_MIN_STANDOFF_MM = 10.0
INTERACTIVE_SPRAY_MAX_STANDOFF_MM = 500.0
INTERACTIVE_SPRAY_WHEEL_STEP_MM = 5.0

FACE_EXCLUDED = 0
FACE_PAINTABLE_CANDIDATE = 1
FACE_REQUIRES_INSPECTION = 2
FACE_INVALID = 3

CLASSIFICATION_COLORS = [
    "#59636B",
    "#00E676",
    "#FFD740",
    "#FF5252",
]

# ============================================================
# BRIGHT PATCH PALETTE
# ============================================================

# High-luminance colors remain distinguishable against the dark viewport.
PATCH_COLORS = [
    "#00E5FF",  # bright cyan
    "#FF4DFF",  # vivid magenta
    "#A8FF00",  # electric lime
    "#FFB000",  # amber
    "#40A9FF",  # bright blue
    "#FF5C8A",  # pink
    "#00F5A0",  # mint green
    "#B388FF",  # violet
    "#FFF04A",  # luminous yellow
    "#FF784F",  # coral
    "#4DFFE1",  # turquoise
    "#E6FF4D",  # chartreuse
]

MAX_DIRECTION_ARROWS = 300

# Continuous film-thickness palette used by the viewport legend.
FILM_THICKNESS_COLOR_HEX = (
    "#A8A8A8", "#2D69DC", "#00BEE6", "#00BE6E", "#FFCD2D", "#EB372D"
)


def map_film_thickness_to_rgb(thickness_um, target_um):
    """Map film thickness continuously so even the first spray pass is visible.

    The previous 25%-wide discrete bands made every deposit below 12.5 um
    look exactly like an unpainted surface for a 50 um target. A continuous
    ramp now exposes early wetting and gradual coat build-up without changing
    the underlying physical thickness field.
    """
    values = np.asarray(thickness_um, dtype=np.float64)
    target = float(target_um)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_um must be finite and positive.")
    if not np.all(np.isfinite(values)):
        raise ValueError("thickness_um must contain only finite values.")

    ratio = np.maximum(values / target, 0.0)
    # Dense low-thickness knots are intentional: the first few micrometres
    # must be visible rather than hidden in a large unpainted color band.
    knots = np.asarray(
        [0.0, 0.005, 0.02, 0.05, 0.25, 0.50, 0.75, 1.0, 1.5],
        dtype=np.float64,
    )
    palette = np.asarray(
        [
            [168, 168, 168],  # untouched CAD surface
            [145, 155, 170],  # trace wetting
            [105, 130, 175],  # thin initial coat
            [45, 105, 220],   # visible blue coat
            [0, 190, 230],
            [0, 190, 110],
            [125, 205, 55],
            [255, 205, 45],
            [235, 55, 45],
        ],
        dtype=np.float64,
    )
    flat = np.clip(ratio.reshape(-1), knots[0], knots[-1])
    channels = [
        np.interp(flat, knots, palette[:, channel])
        for channel in range(3)
    ]
    rgb = np.stack(channels, axis=1).reshape(values.shape + (3,))
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


class ProductViewer(QWidget):

    pose_confirmed = Signal(dict)
    patch_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_mesh = None
        self.patch_labels = None
        self.face_status_labels = None
        self.trajectory_face_patch_labels = None
        self.trajectory_data = None
        self._trajectory_timing_stale = False
        self.trajectory_path_statistics = {}
        self.trajectory_validation = {}
        self._playback_spray_actor = None
        self._playback_spray_polydata = None
        self._playback_spray_sheet_actor = None
        self.face_surface_types = None

        # Currently selected Stage 2 surface patch.
        self.selected_patch_id = None
        self.hole_candidates = []
        self.hole_candidates = []
        self.hook_diameter_mm = 4.5
        self.hole_clearance_mm = 1.0
        self.hole_info_visible = False
        self._hole_overlay_actors = []

        self.display_mode = "product"

        self.source_unit = "unknown"
        self.display_unit = "mm"

        self.source_origin = np.zeros(3, dtype=float)
        self.rotation_pivot = np.zeros(3, dtype=float)
        self.product_center_pivot = np.zeros(3, dtype=float)
        self.hook_length_source_units = 0.0
        self.selected_hook_point_cad = None
        self.hook_world_position = None
        self._hook_pick_active = False

        # Default display rotation is 180 degrees about local Z, centered
        # on the product bounding box; source CAD geometry remains unchanged.
        self._pose_actor = None
        self._world_orientation_widget = None
        self._world_orientation_axes = None
        self._cad_axes_actor = None
        self._attitude_text_actor = None
        # Display-only bounds; operator must set actual work-cell extents.
        self.workspace_dimensions = np.array([1000.0, 1000.0, 1000.0], dtype=float)

        self.source_to_view_transform = np.eye(4, dtype=float)
        self.view_to_source_transform = np.eye(4, dtype=float)

        # Compatibility field: source-coordinate spacing, not always mm.
        self.grid_spacing_mm = 10.0
        self.grid_spacing_coordinate_units = 10.0

        # Stage 3 starts with the product surface only.
        # Surface filters stay enabled so the complete product remains visible;
        # all trajectory overlays are opt-in and can be enabled one at a time.
        self.trajectory_visibility = {
            # Show both surface classes by default; users can toggle either one.
            "planar": True,
            "curved": True,
            "nozzle": False,
            "arrows": False,
            "paths": True,
            "samples": True,
            "footprints": True,
            "frames": False,
            "safety": False,
        }
        self.trajectory_product_transform = np.eye(4, dtype=float)
        self._last_product_transform = np.eye(4, dtype=float)

        # Stage 3 sequential trajectory playback state.
        self._playback_index = 0
        self._playback_elapsed_s = 0.0
        self._playback_wall_elapsed_s = 0.0
        self._playback_wall_started_at = None
        self._playback_finished = False
        self._playback_last_path_render_index = -1
        self._playback_path_actor = None
        self._playback_nozzle_actor = None
        self._playback_gun_actors = []
        self._playback_cache = None
        self._correction_preview_appended = False
        self._trajectory_paint_color_hex = None
        self._trajectory_surface_polydata = None
        self._trajectory_surface_face_mask = None
        self._trajectory_surface_base_rgb = None
        self._trajectory_surface_base_alpha = None
        self.trajectory_safety = {}
        self._trajectory_safety_actor = None
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(100)
        self._playback_timer.timeout.connect(self._advance_trajectory_playback)

        self.plotter = QtInteractor(self)

        # Direct single-click picking of the original mesh triangles.
        # This avoids PyVista's area-selection interaction, which can
        # conflict with the viewer's normal camera mouse controls.
        self._patch_cell_picker = vtkCellPicker()
        self._patch_cell_picker.SetTolerance(0.0005)
        self._patch_mouse_press_position = None
        interactor = self.plotter.iren.interactor
        self._patch_press_observer_id = interactor.AddObserver(
            "LeftButtonPressEvent",
            self._on_patch_mouse_press,
            1.0,
        )
        self._patch_release_observer_id = interactor.AddObserver(
            "LeftButtonReleaseEvent",
            self._on_patch_mouse_release,
            1.0,
        )

        # Interactive spray-path input. Spray input owns the viewport
        # throughout the session; there is no camera/spray mode switch.
        self._interactive_spray_active = False
        self._interactive_spray_mode = "camera"
        self._interactive_spray_recording = False
        self._interactive_spray_standoff = 0.0
        self._interactive_spray_cone_angle_deg = 0.0
        self._interactive_spray_transform = np.eye(4, dtype=float)
        self._interactive_spray_latest_pose = None
        self._interactive_spray_records = []
        self._interactive_spray_display_mode = "thickness"
        self._interactive_recorded_replay_index = 0
        self._interactive_recorded_replay_actor = None
        self._interactive_recorded_replay_timeline = []
        self._interactive_recorded_replay_end_times = np.empty(0, dtype=np.float64)
        self._interactive_recorded_replay_started_at = None
        self._interactive_recorded_replay_timing = None
        self._interactive_spray_path_actor = None
        self._interactive_spray_route_actors = []
        self._interactive_recorded_replay_timer = QTimer(self)
        self._interactive_recorded_replay_timer.setInterval(40)
        self._interactive_recorded_replay_timer.timeout.connect(
            self._advance_interactive_recorded_replay
        )
        self._interactive_spray_stroke_id = -1
        self._interactive_spray_surface = None
        self._interactive_spray_locator = None
        # Persistent coating state: one per-face field, never one coincident
        # transparent VTK actor per mouse sample.
        self._interactive_spray_painted_actors = []
        self._interactive_spray_face_thickness_um = None
        self._interactive_spray_surface_field = None
        # Preserve the saved route coating separately from verification replay.
        # Simulation starts from bare metal, but returning to route editing must
        # restore the coating state that belongs to the recorded route.
        self._interactive_saved_route_face_thickness_um = None
        self._interactive_saved_route_surface_field = None
        self._interactive_spray_texture_actors = {}
        self._interactive_spray_texture_layout = {}
        self._interactive_spray_texture_objects = {}
        self._interactive_spray_excluded_overlay_actor = None
        self._interactive_spray_dirty_face_ids = set()
        self._interactive_spray_base_opacity = None
        self._interactive_spray_face_area_mm2 = None
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_start_timestamp = None
        self._interactive_spray_paint_footprint_angle_deg = 0.0
        self._interactive_spray_gun_actors = []
        self._interactive_spray_gun_collision = False
        self._interactive_spray_collision_filter = None
        self._interactive_spray_paint_target_um = 50.0
        self._interactive_spray_paint_state_dirty = False
        self._interactive_spray_live_actors = []
        self._interactive_spray_path_actor = None
        # The recorded route is also rendered live while the operator holds
        # the left mouse button. This is visualization only; recording and
        # deposition remain full-rate and use the existing trajectory schema.
        self._interactive_spray_live_route_refresh_at = 0.0
        self._interactive_saved_style_enabled = True
        self._interactive_saved_style = None
        self._interactive_spray_input_style = None

        self._interactive_spray_angle_dragging = False
        self._interactive_spray_angle_last_position = None
        self._interactive_spray_angle_x_deg = 0.0
        self._interactive_spray_angle_y_deg = 0.0
        self._interactive_spray_angle_anchor_pose = None
        self._interactive_spray_locked_direction_world = None
        self._interactive_hud_actor = None
        self._interactive_last_record_time = 0.0
        self._interactive_mouse_move_observer_id = interactor.AddObserver(
            "MouseMoveEvent", self._on_interactive_spray_mouse_move, 2.0
        )
        self._interactive_spray_press_observer_id = interactor.AddObserver(
            "LeftButtonPressEvent", self._on_interactive_spray_left_press, 3.0
        )
        self._interactive_spray_release_observer_id = interactor.AddObserver(
            "LeftButtonReleaseEvent", self._on_interactive_spray_left_release, 3.0
        )
        self._interactive_spray_double_click_observer_id = interactor.AddObserver(
            "LeftButtonDoubleClickEvent",
            self._on_interactive_spray_left_double_click,
            4.0,
        )
        # Qt shortcut is more reliable than VTK KeyPressEvent when the
        # camera interactor style has taken keyboard focus.
        self._interactive_spray_space_shortcut = QShortcut(
            QKeySequence(Qt.Key_Space), self
        )
        self._interactive_spray_space_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._interactive_spray_space_shortcut.activated.connect(
            self._return_to_interactive_spray_mode
        )
        self._interactive_spray_middle_press_observer_id = interactor.AddObserver(
            "MiddleButtonPressEvent", self._on_interactive_spray_middle_press, 3.0
        )
        self._interactive_spray_middle_release_observer_id = interactor.AddObserver(
            "MiddleButtonReleaseEvent", self._on_interactive_spray_middle_release, 3.0
        )
        self._interactive_spray_wheel_forward_observer_id = interactor.AddObserver(
            "MouseWheelForwardEvent", self._on_interactive_spray_wheel, 3.0
        )
        self._interactive_spray_wheel_backward_observer_id = interactor.AddObserver(
            "MouseWheelBackwardEvent", self._on_interactive_spray_wheel, 3.0
        )
        self._interactive_spray_leave_observer_id = interactor.AddObserver(
            "LeaveEvent", self._on_interactive_spray_leave, 3.0
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.interaction_mode_panel = self._build_interaction_mode_controls()
        layout.addWidget(self.interaction_mode_panel)
        self.interactive_recorded_path_panel = self._build_interactive_recorded_path_controls()
        layout.addWidget(self.interactive_recorded_path_panel)

        # Keep Hook Control at the very top of the 3D viewer whenever it is visible.
        self.hole_control_panel = self._build_hole_controls()
        layout.addWidget(self.hole_control_panel)
        self.display_control_panel = self._build_trajectory_controls()
        layout.addWidget(self.display_control_panel)
        self.pose_control_panel = self._build_pose_controls()
        self.pose_control_panel.setMinimumHeight(42)
        self.pose_control_panel.setMaximumHeight(48)
        layout.addWidget(self.pose_control_panel)
        layout.addWidget(self.plotter.interactor, 1)

        self.plotter.set_background("#07111D", top="#142B40")

        self.interaction_mode_panel.hide()
        self.display_control_panel.hide()
        self.hole_control_panel.hide()
        self.pose_control_panel.hide()
        self._draw_empty_scene()

    # =========================================================
    # UNIT MANAGEMENT
    # =========================================================

    @staticmethod
    def _normalize_unit(unit):
        if unit is None:
            return "unknown"

        value = str(unit).strip().lower()

        aliases = {
            "millimeter": "mm",
            "millimeters": "mm",
            "centimeter": "cm",
            "centimeters": "cm",
            "meter": "m",
            "meters": "m",
            "inch": "in",
            "inches": "in",
            "foot": "ft",
            "feet": "ft",
        }

        value = aliases.get(value, value)

        if value == "unknown":
            return "unknown"

        if value not in ("mm", "cm", "m", "in", "ft"):
            raise ValueError(
                f"Unsupported STL coordinate unit: {unit}"
            )

        return value

    def _build_hole_controls(self):
        """Build editable hook-diameter and total-clearance inputs."""
        panel = QFrame()
        panel.setObjectName("HookDiameterControls")
        panel.setStyleSheet("""
            QFrame#HookDiameterControls {
                background-color: #0D1822;
                border: 1px solid #36566A;
                border-radius: 5px;
            }
            QFrame#HookDiameterControls QLabel { color: #D8E5ED; }
            QFrame#HookDiameterControls QDoubleSpinBox {
                background: #172A35;
                color: #F3EAF5;
                border: 1px solid #527184;
                border-radius: 3px;
                padding: 3px 5px;
                min-height: 20px;
            }
            QFrame#HookDiameterControls QCheckBox {
                color: #F3EAF5;
                spacing: 8px;
                font-weight: 600;
                padding: 3px 5px;
            }
            QFrame#HookDiameterControls QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #B8CBD6;
                border-radius: 3px;
                background-color: #172A35;
            }
            QFrame#HookDiameterControls QCheckBox::indicator:unchecked {
                background-color: #172A35;
                border: 2px solid #B8CBD6;
            }
            QFrame#HookDiameterControls QCheckBox::indicator:checked {
                background-color: #00A878;
                border: 2px solid #62E6B5;
            }
            QFrame#HookDiameterControls QCheckBox::indicator:hover {
                border: 2px solid #FFFFFF;
            }
        """)
        row = QHBoxLayout(panel)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(8)

        row.addWidget(QLabel("HOOK DIAMETER (mm):"))
        self.hook_diameter_input = QDoubleSpinBox()
        self.hook_diameter_input.setRange(0.1, 1000.0)
        self.hook_diameter_input.setDecimals(2)
        self.hook_diameter_input.setSingleStep(0.5)
        self.hook_diameter_input.setValue(self.hook_diameter_mm)
        self.hook_diameter_input.setFixedWidth(95)
        self.hook_diameter_input.setToolTip(
            "Minimum hole diameter = hook diameter + total clearance."
        )
        self.hook_diameter_input.valueChanged.connect(
            self._on_hook_diameter_changed
        )
        row.addWidget(self.hook_diameter_input)

        row.addWidget(QLabel("TOTAL CLEARANCE (mm):"))
        self.hole_clearance_input = QDoubleSpinBox()
        self.hole_clearance_input.setRange(0.0, 100.0)
        self.hole_clearance_input.setDecimals(2)
        self.hole_clearance_input.setSingleStep(0.1)
        self.hole_clearance_input.setValue(self.hole_clearance_mm)
        self.hole_clearance_input.setFixedWidth(95)
        self.hole_clearance_input.setToolTip(
            "Total diametral clearance added to the hook diameter."
        )
        self.hole_clearance_input.valueChanged.connect(
            self._on_hole_clearance_changed
        )
        row.addWidget(self.hole_clearance_input)

        self.hole_info_checkbox = QCheckBox("SHOW HOLE")
        self.hole_info_checkbox.setChecked(False)
        self.hole_info_checkbox.setToolTip(
            "Show or hide detected hole centers and their D/L/C annotations."
        )
        self.hole_info_checkbox.toggled.connect(
            self._on_hole_info_visibility_changed
        )
        row.addWidget(self.hole_info_checkbox)

        row.addStretch(1)
        self.hook_compatible_count_label = QLabel("0 compatible holes")
        row.addWidget(self.hook_compatible_count_label)
        return panel

    def _on_hook_diameter_changed(self, value):
        self.hook_diameter_mm = float(value)
        if self.current_mesh is not None:
            self._render_product(reset_camera=False)

    def _on_hole_clearance_changed(self, value):
        self.hole_clearance_mm = float(value)
        if self.current_mesh is not None:
            self._render_product(reset_camera=False)

    def _on_hole_info_visibility_changed(self, checked):
        """Toggle hole annotations without rebuilding the scene or stopping playback."""
        self.hole_info_visible = bool(checked)
        if self.current_mesh is not None:
            self._refresh_hole_candidate_overlay()

    def _filtered_hole_candidates(self):
        """
        Keep hook-compatible bore candidates whose centers lie inside
        the product's original source-coordinate bounding box.
        The bounding box is derived from the unposed source mesh, so
        display RPY rotations do not change this geometric test.
        """
        if self.current_mesh is None or not self.hole_candidates:
            return []

        bounds = np.asarray(self.current_mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            return []

        minimum = bounds[0]
        maximum = bounds[1]
        extent = maximum - minimum
        tolerance = max(float(np.linalg.norm(extent)) * 1e-8, 1e-9)

        minimum_diameter = self.hook_diameter_mm + self.hole_clearance_mm
        result = []
        for item in self.hole_candidates:
            try:
                diameter = float(item["diameter_coordinate_units"])
                length = float(item["hole_length_coordinate_units"])
                center = np.asarray(item["opening_center_coordinate_units"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue

            if (not np.isfinite(diameter) or diameter < minimum_diameter
                    or not np.isfinite(length) or length <= 0.0
                    or center.shape != (3,) or not np.all(np.isfinite(center))):
                continue

            # Reject candidate centers outside the product dimensions.
            # Coordinates are checked before the viewer's RPY transform.
            if np.any(center < minimum - tolerance) or np.any(center > maximum + tolerance):
                continue

            result.append(item)
        return result

    def set_hole_candidates(self, candidates):
        """Set exact STEP bore candidates in the original STEP coordinate frame."""
        normalized = []
        for item in candidates or []:
            try:
                center = np.asarray(item["opening_center_coordinate_units"], dtype=float)
                axis = np.asarray(item["axis_direction"], dtype=float)
                diameter = float(item["diameter_coordinate_units"])
                length = float(item["hole_length_coordinate_units"])
                if (center.shape != (3,) or axis.shape != (3,)
                        or not np.all(np.isfinite(center))
                        or not np.all(np.isfinite(axis))
                        or not np.isfinite(diameter) or not np.isfinite(length)):
                    continue
                if np.linalg.norm(axis) <= 1e-12 or diameter <= 0.0 or length <= 0.0:
                    continue
                normalized.append({**dict(item), "_display_center": center.tolist()})
            except (KeyError, TypeError, ValueError):
                continue
        self.hole_candidates = normalized
        self.hole_control_panel.setVisible(bool(normalized))
        if self.current_mesh is not None:
            self._render_product(reset_camera=False)

    def _add_hole_candidate_overlay(self):
        """Display hook-compatible holes with diameter, length and STEP-frame center."""
        if not self.hole_candidates or self._pose_actor is None:
            return
        candidates = self._filtered_hole_candidates()
        self.hook_compatible_count_label.setText(
            f"{len(candidates)} compatible holes"
        )

        # Hole markers and D/L/C annotations are opt-in. The checkbox
        # starts unchecked so the product opens without hole overlays.
        if not self.hole_info_visible or not candidates:
            return
        matrix = self._actor_matrix_array()
        centers_view, labels = [], []
        unit_label = self._display_unit_label()
        for item in candidates:
            center_source = np.asarray(item["_display_center"], dtype=float)
            homogeneous = matrix @ np.append(center_source, 1.0)
            centers_view.append(homogeneous[:3] / homogeneous[3])
            diameter = self._convert_source_length_to_display(float(item["diameter_coordinate_units"]))
            length = self._convert_source_length_to_display(float(item["hole_length_coordinate_units"]))
            xyz = [self._convert_source_length_to_display(float(value)) for value in center_source]
            labels.append(
                f"D={diameter:.2f} {unit_label} | L={length:.2f} {unit_label}\n"
                f"C=({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) {unit_label}"
            )
        points = pv.PolyData(np.asarray(centers_view, dtype=float))
        marker_actor = self.plotter.add_mesh(
            points, color="#FF3DCE", style="points", point_size=17,
            render_points_as_spheres=True, lighting=False, pickable=False,
        )
        label_actor = self.plotter.add_point_labels(
            np.asarray(centers_view, dtype=float), labels,
            font_size=10, text_color="#FFFFFF", shape="rounded_rect",
            shape_color="#35152F", shape_opacity=0.94,
            always_visible=True, show_points=False, margin=4, pickable=False,
        )
        self._hole_overlay_actors.extend(
            actor for actor in (marker_actor, label_actor) if actor is not None
        )

    def _refresh_hole_candidate_overlay(self):
        """Replace only hole marker actors; leave all playback actors and timers intact."""
        for actor in self._hole_overlay_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._hole_overlay_actors = []
        self._add_hole_candidate_overlay()
        self.plotter.render()

    def set_source_unit(self, unit):
        normalized = self._normalize_unit(unit)
        self.source_unit = normalized

        if self.current_mesh is not None:
            self._render_product(reset_camera=False)

    def set_display_unit(self, unit):
        normalized = self._normalize_unit(unit)

        if normalized == "unknown":
            raise ValueError(
                "Display unit cannot be 'unknown'."
            )

        self.display_unit = normalized

        if self.current_mesh is not None:
            self._render_product(reset_camera=False)

    def _get_source_mm_factor(self):
        if self.source_unit == "unknown":
            return None
        return MM_PER_UNIT[self.source_unit]

    def _get_display_mm_factor(self):
        return MM_PER_UNIT[self.display_unit]

    def _get_display_conversion_factor(self):
        source_factor = self._get_source_mm_factor()

        if source_factor is None:
            return None

        return source_factor / self._get_display_mm_factor()

    def _display_unit_label(self):
        if self.source_unit == "unknown":
            return "coordinate units"
        return self.display_unit

    def _source_unit_label(self):
        if self.source_unit == "unknown":
            return "UNKNOWN"
        return self.source_unit

    def _convert_source_length_to_display(self, value):
        value = float(value)
        factor = self._get_display_conversion_factor()

        if factor is None:
            return value

        return value * factor

    def _convert_source_area_to_display(self, value):
        factor = self._get_display_conversion_factor()

        if factor is None:
            return float(value)

        return float(value) * factor ** 2

    def _convert_source_volume_to_display(self, value):
        factor = self._get_display_conversion_factor()

        if factor is None:
            return float(value)

        return float(value) * factor ** 3

    # =========================================================
    # STAGE 3 DISPLAY CHECKBOXES
    # =========================================================

    def _build_interaction_mode_controls(self):
        """Build spray controls; camera input mode is intentionally removed."""
        panel = QFrame()
        panel.setObjectName("ViewerInteractionModeControls")
        panel.setStyleSheet("""
            QFrame#ViewerInteractionModeControls {
                background-color: #0D1822;
                border: 1px solid #36566A;
                border-radius: 5px;
            }
            QFrame#ViewerInteractionModeControls QLabel {
                color: #C9D8E2;
                font-weight: 700;
            }
            QFrame#ViewerInteractionModeControls QPushButton {
                background-color: #183244;
                color: #E2EDF4;
                border: 1px solid #36566A;
                border-radius: 4px;
                padding: 5px 14px;
                font-weight: 700;
            }
        """)
        row = QHBoxLayout(panel)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(8)

        row.addWidget(QLabel("SPRAY CONTROLS"))

        self.working_distance_label = QLabel("WORK DIST")
        self.working_distance_spin = QDoubleSpinBox()
        self.working_distance_spin.setRange(
            INTERACTIVE_SPRAY_MIN_STANDOFF_MM,
            INTERACTIVE_SPRAY_MAX_STANDOFF_MM,
        )
        self.working_distance_spin.setSingleStep(
            INTERACTIVE_SPRAY_WHEEL_STEP_MM
        )
        self.working_distance_spin.setValue(120.0)
        self.working_distance_spin.setSuffix(" mm")
        self.working_distance_spin.setEnabled(False)
        self.working_distance_spin.setToolTip(
            "Spray tip-to-surface working distance. The spray pattern and "
            "deposition are recomputed from this distance."
        )
        self.working_distance_spin.valueChanged.connect(
            self._on_interactive_working_distance_changed
        )
        row.addWidget(self.working_distance_label)
        row.addWidget(self.working_distance_spin)

        self.interaction_mode_status = QLabel("SPRAY INPUT INACTIVE")
        self.interaction_mode_status.setStyleSheet(
            "color: #69C8D5; font-size: 9px; padding-left: 8px;"
        )
        row.addWidget(self.interaction_mode_status)

        self.return_to_paint_button = QPushButton("PAINT MODE")
        self.return_to_paint_button.setToolTip(
            "Return to spray preview. You can also press Space."
        )
        self.return_to_paint_button.clicked.connect(
            self._return_to_interactive_spray_mode
        )
        row.addWidget(self.return_to_paint_button)

        self.spray_perpendicular_status = QLabel("SURFACE ANGLE: --")
        self.spray_perpendicular_status.setMinimumWidth(178)
        self.spray_perpendicular_status.setToolTip(
            "Angle between the spray centerline and the picked surface normal. "
            "0 degrees means perpendicular spraying. This is an indicator only; "
            "it never changes the manually selected spray direction."
        )
        self.spray_perpendicular_status.setStyleSheet(
            "color: #9AAAB5; font-size: 9px; padding-left: 6px;"
        )
        row.addWidget(self.spray_perpendicular_status)

        self.interactive_paint_color_view_button = QPushButton("ACTUAL PAINT COLOR")
        self.interactive_paint_color_view_button.setCheckable(True)
        self.interactive_paint_color_view_button.setToolTip(
            "Toggle between the predicted film-thickness color map and the selected real paint color."
        )
        self.interactive_paint_color_view_button.toggled.connect(
            self._on_interactive_paint_display_mode_changed
        )
        row.addWidget(self.interactive_paint_color_view_button)
        row.addStretch(1)
        return panel

    def _build_interactive_recorded_path_controls(self):
        """Build persistent controls for optimizing and replaying recorded spray paths."""
        panel = QFrame()
        panel.setObjectName("InteractiveRecordedPathControls")
        panel.setStyleSheet("""
            QFrame#InteractiveRecordedPathControls {
                background: #10212C; border: 1px solid #36566A; border-radius: 5px;
            }
            QFrame#InteractiveRecordedPathControls QLabel {
                color: #C9D8E2; font-weight: 700;
            }
            QFrame#InteractiveRecordedPathControls QPushButton {
                background: #183244; color: #E2EDF4; border: 1px solid #36566A;
                border-radius: 4px; padding: 5px 10px; font-weight: 700;
            }
            QFrame#InteractiveRecordedPathControls QPushButton:disabled {
                color: #71818C; background: #15232D;
            }
        """)
        row = QHBoxLayout(panel)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(7)
        row.addWidget(QLabel("RECORDED SPRAY PATH"))
        self.recorded_path_optimize_button = QPushButton("OPTIMIZE ROUTE")
        self.recorded_path_optimize_button.clicked.connect(
            self._optimize_interactive_recorded_path
        )
        row.addWidget(self.recorded_path_optimize_button)
        self.recorded_path_replay_button = QPushButton("▶ SIMULATE")
        self.recorded_path_replay_button.clicked.connect(
            self._start_interactive_recorded_replay
        )
        row.addWidget(self.recorded_path_replay_button)
        self.recorded_path_stop_button = QPushButton("STOP")
        self.recorded_path_stop_button.setEnabled(False)
        self.recorded_path_stop_button.clicked.connect(
            self._stop_interactive_recorded_replay
        )
        row.addWidget(self.recorded_path_stop_button)
        self.recorded_path_status_label = QLabel("No recorded path")
        row.addWidget(self.recorded_path_status_label, 1)
        self.recorded_path_time_label = QLabel("EST. TIME: --")
        self.recorded_path_time_label.setMinimumWidth(220)
        self.recorded_path_time_label.setStyleSheet(
            "color: #79D7A5; font-weight: 800;"
        )
        row.addWidget(self.recorded_path_time_label)
        self.recorded_path_color_view_button = QPushButton("VIEW: FILM")
        self.recorded_path_color_view_button.setCheckable(True)
        self.recorded_path_color_view_button.toggled.connect(
            self._on_interactive_paint_display_mode_changed
        )
        self.recorded_path_color_view_button.setToolTip(
            "Switch the product surface between film-thickness visualization and actual paint color."
        )
        row.addWidget(self.recorded_path_color_view_button)
        panel.hide()
        return panel

    def set_recorded_path_page(self, page):
        """Show route-editing or simulation controls for the active workflow page."""
        page = str(page).lower()
        if page not in ("route", "simulation"):
            raise ValueError("Recorded path page must be 'route' or 'simulation'.")
        route_page = page == "route"

        # Verification replay intentionally uses a temporary bare-product
        # coating field. When the operator returns to route generation, restore
        # the saved route coating so ADD / EDIT starts from the actual recorded
        # state rather than from the replay result or an interrupted replay.
        if route_page and self._interactive_saved_route_face_thickness_um is not None:
            self._interactive_spray_face_thickness_um = (
                self._interactive_saved_route_face_thickness_um.copy()
            )
            self._interactive_spray_surface_field = copy.deepcopy(
                self._interactive_saved_route_surface_field
            )
            self._interactive_spray_dirty_face_ids = set()
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
                self._interactive_saved_route_face_thickness_um = None
            self._interactive_saved_route_surface_field = None
            self._interactive_refresh_paint_visual(render=False)
        self.recorded_path_optimize_button.setVisible(route_page)
        self.recorded_path_replay_button.setVisible(not route_page)
        self.recorded_path_stop_button.setVisible(not route_page)
        self.recorded_path_time_label.setVisible(True)
        has_route = len(getattr(self, "_interactive_spray_records", [])) >= 2
        self.interactive_recorded_path_panel.setVisible(has_route)
        if route_page and has_route:
            self._refresh_interactive_recorded_path_actor()
        self.plotter.render()

    def _on_interactive_paint_display_mode_changed(self, show_paint_color):
        self._interactive_spray_display_mode = (
            "paint_color" if show_paint_color else "thickness"
        )
        for button in (
            getattr(self, "interactive_paint_color_view_button", None),
            getattr(self, "recorded_path_color_view_button", None),
        ):
            if button is not None:
                button.blockSignals(True)
                button.setChecked(bool(show_paint_color))
                button.setText(
                    "VIEW: PAINT COLOR" if button is self.recorded_path_color_view_button
                    else "ACTUAL PAINT COLOR"
                )
                button.blockSignals(False)
        self._interactive_refresh_paint_visual(render=True)

    def _interactive_selected_paint_rgb(self):
        color = str(getattr(self, "_trajectory_paint_color_hex", None) or "#D95B43")
        color = color.lstrip("#")
        try:
            return np.asarray([int(color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)
        except (ValueError, TypeError):
            return np.asarray([217.0, 91.0, 67.0], dtype=np.float64)

    @staticmethod
    def _format_route_time(seconds):
        value = max(float(seconds), 0.0)
        hours = int(value // 3600.0)
        minutes = int((value % 3600.0) // 60.0)
        whole_seconds = int(round(value % 60.0))
        if whole_seconds >= 60:
            whole_seconds = 0
            minutes += 1
        if minutes >= 60:
            minutes = 0
            hours += 1
        return (
            f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
            if hours else f"{minutes:02d}:{whole_seconds:02d}"
        )

    def _build_interactive_route_metrics(self, records):
        """Measure route geometry and replay recorded sample intervals only.

        No robot speed, controller frequency, or planned motion time is inferred.
        """
        if not records:
            return {
                "spray_distance_mm": 0.0,
                "connector_distance_mm": 0.0,
                "recorded_replay_duration_s": 0.0,
                "timeline": [],
            }
        mm_factor = self._get_source_mm_factor()
        if mm_factor is None:
            raise ValueError("A known source coordinate unit is required for route distances.")
        points = np.asarray(
            [item["nozzle_point_product"] for item in records], dtype=np.float64
        )
        segment_ids = np.asarray(
            [item["stroke_id"] for item in records], dtype=np.int64
        )
        timestamps = np.asarray(
            [item.get("timestamp", 0.0) for item in records], dtype=np.float64
        )
        spray_distance = 0.0
        for segment_id in np.unique(segment_ids):
            indices = np.flatnonzero(segment_ids == segment_id)
            if len(indices) > 1:
                spray_distance += float(
                    np.sum(np.linalg.norm(np.diff(points[indices], axis=0), axis=1))
                    * mm_factor
                )
        starts = np.r_[0, np.flatnonzero(np.diff(segment_ids) != 0) + 1]
        ends = np.r_[starts[1:], len(segment_ids)]
        connector_distance = 0.0
        for index in range(len(starts) - 1):
            p0 = points[int(ends[index] - 1)]
            p1 = points[int(starts[index + 1])]
            connector_distance += float(np.linalg.norm(p1 - p0) * mm_factor)

        timeline = []
        elapsed = 0.0
        for index in range(1, len(records)):
            duration = abs(float(timestamps[index] - timestamps[index - 1]))
            if not np.isfinite(duration) or duration <= 1e-6:
                continue
            spray_on = bool(segment_ids[index] == segment_ids[index - 1])
            timeline.append({
                "start_s": elapsed,
                "end_s": elapsed + duration,
                "start_record": records[index - 1],
                "end_record": records[index],
                "spray_on": spray_on,
                "kind": "SPRAY_ON_RECORDED_INTERVAL" if spray_on else "SPRAY_OFF_CONNECTOR",
            })
            elapsed += duration
        return {
            "spray_distance_mm": float(spray_distance),
            "connector_distance_mm": float(connector_distance),
            "recorded_replay_duration_s": float(elapsed),
            "timeline": timeline,
        }

    def _optimize_interactive_recorded_path(self):
        """Optimize stroke order/orientation and report geometry-only distances."""
        records = list(self._interactive_spray_records)
        if not records:
            self.recorded_path_status_label.setText("No recorded path")
            return False
        source_ids = np.asarray([item["stroke_id"] for item in records], dtype=np.int64)
        stroke_ids = np.unique(source_ids)
        before_metrics = self._build_interactive_route_metrics(records)
        if len(stroke_ids) > 1:
            from stages.spray_stroke_route_optimizer import optimize_stroke_order
            candidate = {
                "nozzle_points": np.asarray(
                    [item["nozzle_point_product"] for item in records], dtype=float
                ),
                "spray_directions": np.asarray(
                    [item["spray_direction_product"] for item in records], dtype=float
                ),
                "segment_ids": source_ids,
            }
            optimized = optimize_stroke_order(
                candidate,
                orientation_weight_coordinate_units_per_rad=0.0,
                max_2opt_passes=2 if len(stroke_ids) <= 30 else 0,
            )
            ordered = []
            for stroke in optimized["route_optimization"]["strokes"]:
                indices = np.flatnonzero(source_ids == stroke["source_segment_id"])
                if stroke["reversed"]:
                    indices = indices[::-1]
                ordered.extend(records[int(index)] for index in indices)
            records = ordered
            self._interactive_spray_records = list(records)
            route_info = optimized["route_optimization"]
            self._interactive_spray_route_optimization = route_info
            self.recorded_path_status_label.setText(
                f"Optimized {len(stroke_ids)} strokes · "
                f"{route_info['optimality'].replace('_', ' ')}"
            )
        else:
            self._interactive_spray_records = list(records)
            self._interactive_spray_route_optimization = {
                "algorithm": "SINGLE_STROKE_ORDER_PRESERVED",
                "stroke_count": 1,
                "optimality": "NO_INTER_STROKE_ROUTING_REQUIRED",
            }
            self.recorded_path_status_label.setText(
                f"Single stroke · {len(records)} poses"
            )

        after_metrics = self._build_interactive_route_metrics(records)
        self._interactive_recorded_replay_timing = after_metrics
        self._interactive_route_optimization_metrics = {
            "before_spray_distance_mm": before_metrics["spray_distance_mm"],
            "after_spray_distance_mm": after_metrics["spray_distance_mm"],
            "before_connector_distance_mm": before_metrics["connector_distance_mm"],
            "after_connector_distance_mm": after_metrics["connector_distance_mm"],
            "stroke_count": int(len(stroke_ids)),
        }
        self.recorded_path_time_label.setText(
            f"SPRAY {after_metrics['spray_distance_mm']:.1f} mm  |  "
            f"SPRAY-OFF CONNECTORS {after_metrics['connector_distance_mm']:.1f} mm"
        )
        self._refresh_interactive_recorded_path_actor()
        return True

    def _refresh_interactive_recorded_path_actor(self):
        for actor in getattr(self, "_interactive_spray_route_actors", []):
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._interactive_spray_route_actors = []
        self._interactive_spray_path_actor = None
        records = self._interactive_spray_records
        if not records:
            return

        segment_ids = np.asarray([item["stroke_id"] for item in records], dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(segment_ids) != 0) + 1]
        ends = np.r_[starts[1:], len(records)]
        stroke_endpoints = []
        arrow_points = []
        arrow_vectors = []
        for start_index, end_index in zip(starts, ends):
            points = np.asarray([
                records[index]["nozzle_point_world"]
                for index in range(int(start_index), int(end_index))
            ], dtype=np.float64)
            if len(points) >= 2:
                path = pv.PolyData(points)
                path.lines = np.asarray([len(points), *range(len(points))], dtype=np.int64)
                actor = self.plotter.add_mesh(
                    path, color="#42E887", line_width=4.0,
                    lighting=False, pickable=False,
                )
                self._interactive_spray_route_actors.append(actor)

                # Keep the route line, but also visualize the actual spray
                # direction stored at each trajectory sample. The arrow points
                # from the nozzle toward the coating surface.
                stroke_indices = np.arange(int(start_index), int(end_index))
                if len(stroke_indices) > 1:
                    arrow_indices = stroke_indices[
                        np.linspace(
                            0,
                            len(stroke_indices) - 1,
                            min(len(stroke_indices), MAX_DIRECTION_ARROWS),
                            dtype=int,
                        )
                    ]
                    arrow_points.extend(
                        np.asarray(
                            [records[index]["nozzle_point_world"] for index in arrow_indices],
                            dtype=np.float64,
                        )
                    )
                    arrow_vectors.extend(
                        np.asarray(
                            [records[index]["spray_direction_world"] for index in arrow_indices],
                            dtype=np.float64,
                        )
                    )
            stroke_endpoints.append((points[0], points[-1]))

        if arrow_points:
            arrow_points_array = np.asarray(arrow_points, dtype=np.float64)
            arrow_vectors_array = np.asarray(arrow_vectors, dtype=np.float64)
            vector_lengths = np.linalg.norm(arrow_vectors_array, axis=1)
            valid = vector_lengths > 1e-9
            if np.any(valid):
                arrow_points_array = arrow_points_array[valid]
                arrow_vectors_array = (
                    arrow_vectors_array[valid]
                    / vector_lengths[valid, None]
                )
                path_extent = np.ptp(
                    np.asarray(
                        [item["nozzle_point_world"] for item in records],
                        dtype=np.float64,
                    ),
                    axis=0,
                )
                # Larger, softly colored arrows make spray orientation easy
                # to read without visually overpowering the recorded route.
                arrow_length = max(float(np.max(path_extent)) * 0.050, 1.0)
                arrow_data = pv.PolyData(arrow_points_array)
                arrow_data["spray_direction"] = arrow_vectors_array
                arrow_actor = arrow_data.glyph(
                    orient="spray_direction",
                    scale=False,
                    factor=arrow_length,
                    geom=pv.Arrow(
                        start=(0.0, 0.0, 0.0),
                        direction=(1.0, 0.0, 0.0),
                        tip_length=0.30,
                        tip_radius=0.10,
                        shaft_radius=0.035,
                        tip_resolution=12,
                    ),
                )
                actor = self.plotter.add_mesh(
                    arrow_actor,
                    color="#FFF0B8",
                    opacity=0.72,
                    lighting=False,
                    pickable=False,
                )
                self._interactive_spray_route_actors.append(actor)

        for index in range(len(stroke_endpoints) - 1):
            connector_points = np.asarray([
                stroke_endpoints[index][1], stroke_endpoints[index + 1][0]
            ], dtype=np.float64)
            connector = pv.PolyData(connector_points)
            connector.lines = np.asarray([2, 0, 1], dtype=np.int64)
            actor = self.plotter.add_mesh(
                connector, color="#5BA8FF", line_width=2.5,
                style="wireframe", lighting=False, pickable=False,
            )
            self._interactive_spray_route_actors.append(actor)

        if self._interactive_spray_route_actors:
            self._interactive_spray_path_actor = self._interactive_spray_route_actors[0]
        self.plotter.render()

    @staticmethod
    def _interpolate_replay_record(first, second, fraction):
        fraction = float(np.clip(fraction, 0.0, 1.0))

        def vector(key):
            a = np.asarray(first[key], dtype=np.float64)
            b = np.asarray(second[key], dtype=np.float64)
            value = a + (b - a) * fraction
            if key in ("spray_direction_world", "surface_normal_world"):
                value /= max(float(np.linalg.norm(value)), 1e-12)
            return value

        return {
            "nozzle_point_world": vector("nozzle_point_world"),
            "surface_point_world": vector("surface_point_world"),
            "surface_normal_world": vector("surface_normal_world"),
            "spray_direction_world": vector("spray_direction_world"),
            "cell_id": int(
                first["face_index"] if fraction < 0.5 else second["face_index"]
            ),
        }

    def _interactive_reset_coating_for_replay(self):
        """Start the verification replay from an uncoated, neutral-gray product."""
        if self._interactive_spray_face_thickness_um is not None:
            self._interactive_spray_face_thickness_um.fill(0.0)
        field = self._interactive_spray_surface_field
        if isinstance(field, dict):
            for face_record in field.get("faces", {}).values():
                grid = face_record.get("thickness_grid_um")
                if grid is not None:
                    np.asarray(grid)[...] = 0.0
            self._interactive_spray_dirty_face_ids = {
                int(face_id) for face_id in field.get("faces", {})
            }
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_gun_collision = False
        self._interactive_spray_surface_excluded = False
        self._interactive_refresh_paint_visual(render=False)
        self.plotter.render()

    def _interactive_add_replay_cone_visual(self, pose):
        """Draw a spray cone with its apex at the nozzle and rim toward the product."""
        nozzle = np.asarray(pose["nozzle_point_world"], dtype=np.float64)
        axis = np.asarray(pose["spray_direction_world"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        half_angle = math.radians(self._interactive_spray_cone_angle_deg) * 0.5
        height = max(float(self._interactive_spray_standoff), 1e-6)
        radius = height * math.tan(half_angle)

        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, axis))) > 0.92:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        radial_u = np.cross(axis, reference)
        radial_u /= max(float(np.linalg.norm(radial_u)), 1e-12)
        radial_v = np.cross(axis, radial_u)
        radial_v /= max(float(np.linalg.norm(radial_v)), 1e-12)

        angles = np.linspace(0.0, 2.0 * math.pi, 33)[:-1]
        rim = (
            nozzle[None, :]
            + axis[None, :] * height
            + radius * (
                np.cos(angles)[:, None] * radial_u[None, :]
                + np.sin(angles)[:, None] * radial_v[None, :]
            )
        )
        points = np.vstack((nozzle, rim))
        faces = []
        for index in range(len(rim)):
            next_index = (index + 1) % len(rim)
            faces.extend((3, 0, index + 1, next_index + 1))
        cone = pv.PolyData(
            points, faces=np.asarray(faces, dtype=np.int64)
        )
        actor = self.plotter.add_mesh(
            cone,
            color="#FFE082",
            opacity=0.16,
            style="wireframe",
            line_width=1.4,
            lighting=False,
            pickable=False,
        )
        self._interactive_spray_live_actors.append(actor)

        particle_fractions = np.asarray([0.18, 0.36, 0.54, 0.72, 0.90])
        particle_points = (
            nozzle[None, :]
            + particle_fractions[:, None] * height * axis[None, :]
        )
        cloud = pv.PolyData(particle_points)
        particle_actor = self.plotter.add_mesh(
            cloud,
            color="#FFF0C7",
            opacity=0.48,
            point_size=5.0,
            render_points_as_spheres=True,
            lighting=False,
            pickable=False,
        )
        self._interactive_spray_live_actors.append(particle_actor)

    def _interactive_replay_deposition_step(self, pose, dt):
        """Apply the existing spray deposition kernel to the current replay pose."""
        cell_id = int(pose["cell_id"])
        excluded = self._interactive_spray_excluded_face_mask
        if (
            excluded is None
            or cell_id < 0
            or cell_id >= len(excluded)
            or bool(excluded[cell_id])
        ):
            self._interactive_spray_surface_excluded = True
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
            return False

        self._interactive_spray_surface_excluded = False
        transform = self._interactive_spray_transform
        inverse = np.linalg.inv(transform)
        nozzle_local = (inverse @ np.append(
            pose["nozzle_point_world"], 1.0
        ))[:3]
        direction_local = inverse[:3, :3] @ np.asarray(
            pose["spray_direction_world"], dtype=np.float64
        )
        direction_local /= max(float(np.linalg.norm(direction_local)), 1e-12)
        footprints = _build_spray_footprint_overlay(
            self.current_mesh,
            nozzle_local.reshape(1, 3),
            direction_local.reshape(1, 3),
            self._interactive_spray_paint_footprint_angle_deg,
            rays_per_ring=8,
            radial_rings=(0.40, 0.75, 1.0),
            surface=self._interactive_spray_surface,
            locator=self._interactive_spray_locator,
        )
        if not footprints or footprints[0][0].n_cells == 0:
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
            return False

        patch, _ = footprints[0]
        patch.points = (
            np.asarray(patch.points) @ transform[:3, :3].T
            + transform[:3, 3]
        )
        now = time.monotonic()
        replay_pose = {
            "nozzle_point_world": np.asarray(
                pose["nozzle_point_world"], dtype=np.float64
            ),
            "surface_point_world": np.asarray(
                pose["surface_point_world"], dtype=np.float64
            ),
            "surface_normal_world": np.asarray(
                pose["surface_normal_world"], dtype=np.float64
            ),
            "spray_direction_world": np.asarray(
                pose["spray_direction_world"], dtype=np.float64
            ),
            "cell_id": cell_id,
            "paintable": True,
            "timestamp": now,
        }
        if self._interactive_spray_gun_collision:
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
                return False
        self._interactive_apply_paint_deposition(
            patch, replay_pose, duration_s=float(dt)
        )
        self._interactive_refresh_paint_visual(render=False)
        return True

    def _start_interactive_recorded_replay(self):
        records = self._interactive_spray_records
        if len(records) < 2:
            self.recorded_path_status_label.setText("No replayable recorded path")
            return
        if not self._interactive_recorded_replay_timing:
            self._optimize_interactive_recorded_path()
        timing = self._interactive_recorded_replay_timing
        timeline = timing.get("timeline", []) if timing else []
        if not timeline:
            self.recorded_path_status_label.setText("Route timing unavailable")
            return

        self._interactive_recorded_replay_timer.stop()
        self._interactive_remove_live_actors()

        # Snapshot the actual route coating before verification resets the
        # product to an unpainted state. This snapshot is used only when the
        # operator goes back to route editing.
        self._interactive_saved_route_face_thickness_um = (
            self._interactive_spray_face_thickness_um.copy()
            if self._interactive_spray_face_thickness_um is not None
            else None
        )
        self._interactive_saved_route_surface_field = (
            copy.deepcopy(self._interactive_spray_surface_field)
            if isinstance(self._interactive_spray_surface_field, dict)
            else None
        )
        self._interactive_reset_coating_for_replay()
        self._interactive_recorded_replay_timeline = timeline
        self._interactive_recorded_replay_end_times = np.asarray(
            [edge["end_s"] for edge in timeline], dtype=np.float64
        )
        self._interactive_recorded_replay_started_at = time.monotonic()
        self._interactive_recorded_replay_last_elapsed = 0.0
        self._interactive_recorded_replay_index = 0
        self.recorded_path_replay_button.setEnabled(False)
        self.recorded_path_stop_button.setEnabled(True)
        self._advance_interactive_recorded_replay()
        self._interactive_recorded_replay_timer.start()
        self.plotter.render()

    def _integrate_interactive_replay_interval(self, start_s, end_s):
        """Deposit at fixed spatial samples along the recorded spray route.

        The UI timer is only a clock. Every edge is divided once by distance,
        and those immutable sample times are consumed across timer callbacks.
        """
        timeline = getattr(self, "_interactive_recorded_replay_timeline", [])
        end_times = np.asarray(
            getattr(self, "_interactive_recorded_replay_end_times", []),
            dtype=np.float64,
        )
        if not timeline or not len(end_times):
            return

        cursor = max(float(start_s), 0.0)
        target = min(float(end_s), float(timeline[-1]["end_s"]))
        epsilon = 1e-10
        source_factor = float(self._get_source_mm_factor() or 1.0)
        spacing_mm = max(
            float(self._interactive_spray_sampling_spacing) * source_factor,
            1e-6,
        )

        while cursor < target - epsilon:
            edge_index = int(np.searchsorted(
                end_times, cursor + epsilon, side="right"
            ))
            if edge_index >= len(timeline):
                break
            edge = timeline[edge_index]
            edge_start = float(edge["start_s"])
            edge_end = float(edge["end_s"])
            interval_end = min(target, edge_end)
            if interval_end <= cursor + epsilon:
                cursor = max(cursor, edge_end)
                continue

            if not edge["spray_on"]:
                self._interactive_spray_last_deposition_timestamp = None
                self._interactive_spray_previous_nozzle_world = None
                        cursor = interval_end
                continue

            first_record = edge["start_record"]
            last_record = edge["end_record"]
            p0 = np.asarray(first_record["nozzle_point_world"], dtype=np.float64)
            p1 = np.asarray(last_record["nozzle_point_world"], dtype=np.float64)
            edge_distance_mm = float(np.linalg.norm(p1 - p0)) * source_factor
            spatial_steps = max(1, int(math.ceil(edge_distance_mm / spacing_mm)))
            edge_duration = max(edge_end - edge_start, 1e-12)
            sample_duration = edge_duration / spatial_steps

            if self._interactive_spray_previous_nozzle_world is None:
                self._interactive_spray_previous_nozzle_world = p0.copy()
                self._interactive_spray_last_deposition_timestamp = time.monotonic()

            # Process only fixed sample endpoints whose scheduled times fall
            # inside this UI interval. No sample is shifted by timer jitter.
            first_sample = max(
                1,
                int(math.floor((cursor - edge_start) / sample_duration + 1e-9)) + 1,
            )
            last_sample = min(
                spatial_steps,
                int(math.floor((interval_end - edge_start) / sample_duration + 1e-9)),
            )
            for sample_index in range(first_sample, last_sample + 1):
                fraction = float(sample_index) / float(spatial_steps)
                pose = self._interpolate_replay_record(
                    first_record, last_record, fraction
                )
                self._interactive_remove_live_actors()
                gun_actors, gun_polydata = self._interactive_build_gun_visual(
                    pose["nozzle_point_world"], pose["spray_direction_world"]
                )
                self._interactive_spray_live_actors.extend(gun_actors)
                self._interactive_spray_gun_collision = (
                    self._interactive_check_gun_collision(gun_polydata)
                )
                self._interactive_add_replay_cone_visual(pose)
                self._interactive_replay_deposition_step(
                    pose, float(sample_duration)
                )
            cursor = interval_end

    def _advance_interactive_recorded_replay(self):
        timeline = getattr(self, "_interactive_recorded_replay_timeline", [])
        if not timeline:
            self._stop_interactive_recorded_replay()
            return
        elapsed = max(
            0.0,
            time.monotonic() - float(self._interactive_recorded_replay_started_at),
        )
        total = float(timeline[-1]["end_s"])
        target_elapsed = min(elapsed, total)
        previous_elapsed = float(getattr(
            self, "_interactive_recorded_replay_last_elapsed", 0.0
        ))

        # Process all elapsed timeline pieces before updating the display pose.
        # This includes the final interval when the wall clock reaches the end.
        if target_elapsed > previous_elapsed:
            self._integrate_interactive_replay_interval(
                previous_elapsed, target_elapsed
            )
        self._interactive_recorded_replay_last_elapsed = target_elapsed

        if target_elapsed >= total - 1e-10:
            self._stop_interactive_recorded_replay(completed=True)
            return

        edge_index = int(np.searchsorted(
            self._interactive_recorded_replay_end_times,
            target_elapsed,
            side="right",
        ))
        edge_index = min(edge_index, len(timeline) - 1)
        edge = timeline[edge_index]
        duration = max(float(edge["end_s"] - edge["start_s"]), 1e-9)
        fraction = np.clip(
            (target_elapsed - float(edge["start_s"])) / duration, 0.0, 1.0
        )
        pose = self._interpolate_replay_record(
            edge["start_record"], edge["end_record"], fraction
        )

        self._interactive_remove_live_actors()
        gun_actors, gun_polydata = self._interactive_build_gun_visual(
            pose["nozzle_point_world"], pose["spray_direction_world"]
        )
        self._interactive_spray_live_actors.extend(gun_actors)
        if edge["spray_on"]:
            self._interactive_add_replay_cone_visual(pose)
            self._interactive_spray_gun_collision = (
                self._interactive_check_gun_collision(gun_polydata)
            )
        else:
            self._interactive_spray_gun_collision = False
            self._interactive_spray_surface_excluded = False

        if self._interactive_recorded_replay_actor is not None:
            try:
                self.plotter.remove_actor(self._interactive_recorded_replay_actor, render=False)
            except Exception:
                pass
            self._interactive_recorded_replay_actor = None

        self._interactive_recorded_replay_index = edge_index
        mode = "SPRAY ON" if edge["spray_on"] else "SPRAY OFF"
        if edge["spray_on"] and self._interactive_spray_gun_collision:
            mode = "SPRAY BLOCKED · GUN COLLISION"
        elif edge["spray_on"] and self._interactive_spray_surface_excluded:
            mode = "SPRAY BLOCKED · STAGE 2 EXCLUDED"
        current = self._format_route_time(target_elapsed)
        total_text = self._format_route_time(total)
        self.recorded_path_status_label.setText(
            f"{mode} · {edge['kind']} · {current} / {total_text}"
        )
        self.plotter.render()

    def _stop_interactive_recorded_replay(self, completed=False):
        self._interactive_recorded_replay_timer.stop()
        self.recorded_path_replay_button.setEnabled(bool(self._interactive_spray_records))
        self.recorded_path_stop_button.setEnabled(False)
        self._interactive_remove_live_actors()
        self._interactive_recorded_replay_timeline = []
        self._interactive_recorded_replay_end_times = np.empty(0, dtype=np.float64)
        self._interactive_recorded_replay_started_at = None
        self._interactive_recorded_replay_last_elapsed = 0.0
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_refresh_paint_visual(render=False)
        if completed:
            self.recorded_path_status_label.setText(
                f"Replay complete · {len(self._interactive_spray_records)} poses"
            )
        elif self._interactive_spray_records:
            self.recorded_path_status_label.setText("Replay stopped")
        self.plotter.render()

    def _build_trajectory_controls(self):
        panel = QFrame()
        panel.setObjectName("TrajectoryDisplayControls")
        panel.setStyleSheet("""
            QFrame#TrajectoryDisplayControls {
                background-color: #0D1822;
                border: 1px solid #294858;
                border-radius: 5px;
            }
            QFrame#TrajectoryDisplayControls QLabel {
                color: #C9D8E2;
            }
            QFrame#TrajectoryDisplayControls QPushButton {
                background-color: #183244;
                color: #E2EDF4;
                border: 1px solid #36566A;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QFrame#TrajectoryDisplayControls QPushButton:hover {
                background-color: #24485E;
            }
            QFrame#TrajectoryDisplayControls QSlider::groove:horizontal {
                height: 5px;
                background: #294454;
                border-radius: 2px;
            }
            QFrame#TrajectoryDisplayControls QSlider::handle:horizontal {
                width: 13px;
                margin: -5px 0;
                background: #79B7C8;
                border: 1px solid #B7DCE5;
                border-radius: 6px;
            }
        """)

        root = QVBoxLayout(panel)
        root.setContentsMargins(8, 5, 8, 5)
        root.setSpacing(5)

        display_row = QHBoxLayout()
        display_row.setSpacing(8)
        display_row.addWidget(QLabel("3D DISPLAY:"))
        self.trajectory_checkboxes = {}

        # Show only the three primary spray-trajectory layers.
        checkbox_items = [
            ("planar", "Planar surface"),
            ("curved", "Curved surface"),
            ("samples", "Spray points"),
            ("footprints", "Circular spray footprints"),
            ("paths", "Trajectory path"),
        ]

        for key, label_text in checkbox_items:
            checkbox = QCheckBox(label_text)
            checkbox.setAutoFillBackground(False)
            if key == "planar":
                checkbox.setObjectName("TrajectoryPlanarSurfaceCheckbox")
                checkbox.setStyleSheet("""
                    QCheckBox {
                        color: #EAF4FA;
                        spacing: 7px;
                        background-color: transparent;
                        border: none;
                    }
                    QCheckBox:disabled { color: #DCE8EF; }
                    QCheckBox::indicator {
                        width: 17px; height: 17px;
                        border: 2px solid #8FA7B5;
                        border-radius: 3px;
                        background-color: #263945;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #2388D1;
                        border: 2px solid #72C4FF;
                    }
                    QCheckBox::indicator:disabled {
                        border: 2px solid #A9BBC5;
                        background-color: #344B59;
                    }
                    QCheckBox::indicator:checked:disabled {
                        background-color: #2388D1;
                        border: 2px solid #72C4FF;
                    }
                """)
            elif key == "curved":
                checkbox.setObjectName("TrajectoryCurvedSurfaceCheckbox")
                checkbox.setStyleSheet("""
                    QCheckBox { color: #EAF4FA; spacing: 7px; }
                    QCheckBox:disabled { color: #DCE8EF; }
                    QCheckBox::indicator {
                        width: 17px; height: 17px;
                        border: 2px solid #8FA7B5;
                        border-radius: 3px;
                        background-color: #263945;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #D97732;
                        border: 2px solid #FFC18D;
                    }
                    QCheckBox::indicator:disabled {
                        border: 2px solid #A9BBC5;
                        background-color: #344B59;
                    }
                    QCheckBox::indicator:checked:disabled {
                        background-color: #D97732;
                        border: 2px solid #FFC18D;
                    }
                """)
            else:
                checkbox.setStyleSheet("""
                    QCheckBox {
                        color: #EAF4FA;
                        spacing: 7px;
                        background-color: transparent;
                        border: none;
                    }
                    QCheckBox:disabled { color: #DCE8EF; }
                    QCheckBox::indicator {
                        width: 17px; height: 17px;
                        border: 2px solid #8FA7B5;
                        border-radius: 3px;
                        background-color: #263945;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #00A878;
                        border: 2px solid #62E6B5;
                    }
                    QCheckBox::indicator:disabled {
                        border: 2px solid #A9BBC5;
                        background-color: #344B59;
                    }
                    QCheckBox::indicator:checked:disabled {
                        background-color: #00A878;
                        border: 2px solid #62E6B5;
                    }
                """)
            checkbox.setChecked(self.trajectory_visibility[key])
            checkbox.toggled.connect(
                lambda checked, item=key:
                self._on_trajectory_visibility_changed(item, checked)
            )
            display_row.addWidget(checkbox)
            self.trajectory_checkboxes[key] = checkbox

        display_row.addStretch()
        root.addLayout(display_row)

        self.trajectory_safety_label = QLabel(
            "SAFETY CHECK | No Stage 3 safety data"
        )
        self.trajectory_safety_label.setWordWrap(True)
        self.trajectory_safety_label.setStyleSheet(
            "color: #E6C36A; font-size: 9px; font-weight: 700; padding-left: 4px;"
        )
        self.trajectory_safety_label.setToolTip(
            "RED = configured gun-body clearance violation. YELLOW = incomplete "
            "geometry checks. GREEN = all configured tool geometry checks complete. "
            "This is not a robot-cell safety certification."
        )
        root.addWidget(self.trajectory_safety_label)

        # Spray playback controls are hosted in the right sidebar.
        # Keep only 3D layer visibility controls above the viewport.
        playback_panel = QFrame()
        playback_panel.setObjectName("TrajectoryPlaybackPanel")
        playback_panel.setStyleSheet("""
            QFrame#TrajectoryPlaybackPanel {
                background: #14232D;
                border: 1px solid #344B58;
                border-radius: 6px;
            }
            QFrame#TrajectoryPlaybackPanel QLabel {
                background: transparent;
                border: none;
                color: #D8E4EA;
            }
            QFrame#TrajectoryPlaybackPanel QPushButton {
                background: #203642;
                color: #EAF4F8;
                border: 1px solid #405B69;
                border-radius: 4px;
                padding: 4px 9px;
                min-height: 23px;
                font-weight: 600;
            }
            QFrame#TrajectoryPlaybackPanel QPushButton:hover {
                background: #2A4858;
                border-color: #6D9CB1;
            }
            QFrame#TrajectoryPlaybackPanel QSlider::groove:horizontal {
                height: 4px; background: #3A515D; border-radius: 2px;
            }
            QFrame#TrajectoryPlaybackPanel QSlider::sub-page:horizontal {
                background: #37A6D5; border-radius: 2px;
            }
            QFrame#TrajectoryPlaybackPanel QSlider::handle:horizontal {
                background: #EAF4F8; border: 2px solid #37A6D5;
                width: 12px; margin: -5px 0; border-radius: 7px;
            }
            QFrame#TrajectoryPlaybackPanel QDoubleSpinBox {
                background: #1C303B; color: #EAF4F8;
                border: 1px solid #405B69; border-radius: 4px;
                padding: 2px 4px; min-height: 22px;
            }
            QFrame#TrajectoryPlaybackPanel QProgressBar {
                background: #0E1A22; color: #EAF4F8;
                border: 1px solid #405B69; border-radius: 3px;
                text-align: center; min-width: 76px; max-width: 100%;
                min-height: 15px; max-height: 15px;
            }
            QFrame#TrajectoryPlaybackPanel QProgressBar::chunk {
                background: #37A6D5; border-radius: 2px;
            }
        """)
        self.trajectory_playback_panel = playback_panel
        playback_layout = QVBoxLayout(playback_panel)
        playback_layout.setContentsMargins(8, 7, 8, 7)
        playback_layout.setSpacing(6)

        status_row = QHBoxLayout()
        playback_title = QLabel("PLAYBACK")
        playback_title.setStyleSheet(
            "color: #8EADBC; font-size: 9px; font-weight: 800; letter-spacing: 1px;"
        )
        status_row.addWidget(playback_title)
        status_row.addStretch(1)

        self.trajectory_spray_state_label = QLabel("● SPRAY OFF")
        self.trajectory_spray_state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.trajectory_spray_state_label.setStyleSheet(
            "color: #91A4AE; font-size: 9px; font-weight: 800; padding: 2px 5px;"
        )
        self.trajectory_spray_state_label.setToolTip(
            "SPRAY ON means the nozzle is applying the selected paint color. "
            "SPRAY OFF indicates a repositioning segment."
        )
        status_row.addWidget(self.trajectory_spray_state_label)
        playback_layout.addLayout(status_row)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(5)

        self.trajectory_play_button = QPushButton("▶")
        self.trajectory_play_button.setFixedWidth(38)
        self.trajectory_play_button.setToolTip("Play or pause trajectory playback")
        self.trajectory_play_button.clicked.connect(self._toggle_trajectory_playback)
        transport_row.addWidget(self.trajectory_play_button)

        self.trajectory_step_button = QPushButton("›")
        self.trajectory_step_button.setFixedWidth(32)
        self.trajectory_step_button.setToolTip("Advance one trajectory sample")
        self.trajectory_step_button.clicked.connect(self._step_trajectory_playback)
        transport_row.addWidget(self.trajectory_step_button)

        self.trajectory_reset_button = QPushButton("↺")
        self.trajectory_reset_button.setFixedWidth(32)
        self.trajectory_reset_button.setToolTip("Reset playback to the beginning")
        self.trajectory_reset_button.clicked.connect(self._reset_trajectory_playback)
        transport_row.addWidget(self.trajectory_reset_button)

        self.trajectory_playback_label = QLabel("0 / 0")
        self.trajectory_playback_label.setMinimumWidth(70)
        self.trajectory_playback_label.setAlignment(Qt.AlignCenter)
        self.trajectory_playback_label.setStyleSheet(
            "color: #B9CCD5; font-size: 10px; font-weight: 700;"
        )
        transport_row.addWidget(self.trajectory_playback_label, 1)
        playback_layout.addLayout(transport_row)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(6)

        color_label = QLabel("PAINT COLOR")
        color_label.setStyleSheet(
            "color: #8EADBC; font-size: 9px; font-weight: 700;"
        )
        settings_row.addWidget(color_label)

        self.trajectory_paint_color_button = QPushButton("●")
        self.trajectory_paint_color_button.setFixedSize(32, 28)
        self.trajectory_paint_color_button.setToolTip(
            "Choose the paint color used by the spray and painted-surface preview"
        )
        self.trajectory_paint_color_button.setStyleSheet(
            "QPushButton { color: #D95B43; font-size: 17px; padding: 0px; "
            "background: #1C303B; border: 1px solid #607986; border-radius: 4px; }"
        )
        self.trajectory_paint_color_button.clicked.connect(
            self._choose_trajectory_paint_color
        )
        settings_row.addWidget(self.trajectory_paint_color_button)
        settings_row.addStretch(1)

        speed_label = QLabel("SPEED")
        speed_label.setStyleSheet(
            "color: #8EADBC; font-size: 9px; font-weight: 700;"
        )
        settings_row.addWidget(speed_label)

        self.trajectory_playback_speed = QDoubleSpinBox()
        self.trajectory_playback_speed.setRange(0.25, 10.0)
        self.trajectory_playback_speed.setSingleStep(0.25)
        self.trajectory_playback_speed.setValue(1.0)
        self.trajectory_playback_speed.setSuffix("x")
        self.trajectory_playback_speed.setFixedWidth(70)
        settings_row.addWidget(self.trajectory_playback_speed)
        playback_layout.addLayout(settings_row)

        self.trajectory_playback_slider = QSlider(Qt.Horizontal)
        self.trajectory_playback_slider.setRange(0, 0)
        self.trajectory_playback_slider.setSingleStep(1)
        self.trajectory_playback_slider.setPageStep(10)
        self.trajectory_playback_slider.setMinimumWidth(80)
        self.trajectory_playback_slider.valueChanged.connect(
            self._on_trajectory_playback_slider_changed
        )
        playback_layout.addWidget(self.trajectory_playback_slider)

        self.trajectory_time_label = QLabel(
            "WALL 00:00 | PATH 00:00 / 00:00 | TRAVEL +00:00 | PLAN 00:00"
        )
        self.trajectory_time_label.setMinimumWidth(0)
        self.trajectory_time_label.setStyleSheet(
            "color: #B9CCD5; font-size: 9px; font-weight: 700; padding-left: 4px;"
        )
        self.trajectory_time_label.setToolTip(
            "WALL is real time spent in this preview. PATH is calculated spray-path "
            "time. TRAVEL is the straight-line spray-OFF transfer estimate between "
            "strokes; transfers are not animated and are not collision checked. "
            "PLAN is PATH plus estimated TRAVEL; it excludes valve delays, robot "
            "IK/collision planning, and other cell-cycle overhead."
        )
        playback_layout.addWidget(self.trajectory_time_label)

        coverage_row = QHBoxLayout()
        coverage_title = QLabel("COVERAGE")
        coverage_title.setStyleSheet(
            "color: #8EADBC; font-size: 9px; font-weight: 700;"
        )
        coverage_row.addWidget(coverage_title)

        self.trajectory_coverage_bar = QProgressBar()
        self.trajectory_coverage_bar.setRange(0, 1000)
        self.trajectory_coverage_bar.setValue(0)
        self.trajectory_coverage_bar.setFormat("%p%")
        self.trajectory_coverage_bar.setFixedHeight(16)
        coverage_row.addWidget(self.trajectory_coverage_bar, 1)

        self.trajectory_coverage_label = QLabel("0.0% | 0.000 / 0.000 area")
        self.trajectory_coverage_label.setMinimumWidth(0)
        self.trajectory_coverage_label.setStyleSheet(
            "color: #AFC2CB; font-size: 9px;"
        )
        self.trajectory_coverage_label.setToolTip(
            "Estimated geometric area reached by the spray envelope; "
            "not a measured coating-thickness or paint-mass result."
        )
        coverage_row.addWidget(self.trajectory_coverage_label)

        self.trajectory_thickness_checkbox = QCheckBox("SHOW FILM THICKNESS")
        self.trajectory_thickness_checkbox.setChecked(False)
        self.trajectory_thickness_checkbox.setToolTip(
            "Display the predicted dry-film thickness accumulated from the "
            "time-based spray model. Results remain uncalibrated predictions."
        )
        self.trajectory_thickness_checkbox.toggled.connect(
            self._on_film_thickness_visibility_changed
        )
        coverage_row.addWidget(self.trajectory_thickness_checkbox)
        playback_layout.addLayout(coverage_row)

        self.trajectory_thickness_legend_label = QLabel()
        self.trajectory_thickness_legend_label.setWordWrap(True)
        self.trajectory_thickness_legend_label.setMinimumHeight(34)
        self.trajectory_thickness_legend_label.setStyleSheet(
            "color: #D7E3EA; font-size: 9px; padding-left: 4px;"
        )
        self.trajectory_thickness_legend_label.setToolTip(
            "The 3D surface uses discrete bands at 25%, 50%, 75%, and 100% "
            "of the configured target dry-film thickness; red indicates over-target."
        )
        self._update_film_thickness_legend(
            DEFAULT_COATING_PROCESS["target_dry_film_thickness_um"]
        )
        playback_layout.addWidget(self.trajectory_thickness_legend_label)

        self.trajectory_thickness_label = QLabel(
            "PREDICTED DRY FILM THICKNESS | N/A"
        )
        self.trajectory_thickness_label.setMinimumHeight(18)
        self.trajectory_thickness_label.setWordWrap(True)
        self.trajectory_thickness_label.setStyleSheet(
            "color: #E8C98A; font-size: 9px; font-weight: 700;"
        )
        self.trajectory_thickness_label.setToolTip(
            "Uncalibrated Double-Beta dry-film estimate. Spray-valve delays, "
            "fluid atomization losses, evaporation dynamics and measured coupon "
            "calibration are not yet included."
        )
        playback_layout.addWidget(self.trajectory_thickness_label)

        self.trajectory_correction_label = QLabel(
            "COVERAGE CORRECTION | RUN PLAYBACK TO ANALYZE THE COMPLETE TARGET"
        )
        self.trajectory_correction_label.setMinimumHeight(30)
        self.trajectory_correction_label.setWordWrap(True)
        self.trajectory_correction_label.setStyleSheet(
            "color: #AFC2CB; font-size: 9px; font-weight: 700;"
        )
        self.trajectory_correction_label.setToolTip(
            "After full playback, this reports predicted surface deficits and "
            "candidate correction strokes. Candidates are not robot executable: "
            "visibility, IK, collision, nozzle calibration and measured coating "
            "validation are still required."
        )
        playback_layout.addWidget(self.trajectory_correction_label)

        self.trajectory_correction_apply_button = QPushButton(
            "APPEND CORRECTION STROKES TO PREVIEW"
        )
        self.trajectory_correction_apply_button.setEnabled(False)
        self.trajectory_correction_apply_button.setToolTip(
            "Append CAD-space deficit candidates to the preview trajectory and "
            "reset playback. Preview only: visibility, IK, collision and calibrated "
            "deposition are not validated."
        )
        self.trajectory_correction_apply_button.clicked.connect(
            self._apply_coverage_correction_preview
        )
        playback_layout.addWidget(self.trajectory_correction_apply_button)

        self.trajectory_path_stats_label = QLabel(
            "PATH SAMPLES | Planar: —  Curved: —  Fallback: —  Total: —"
        )
        self.trajectory_path_stats_label.setMinimumHeight(18)
        self.trajectory_path_stats_label.setWordWrap(True)
        self.trajectory_path_stats_label.setStyleSheet(
            "color: #AFC2CB; font-size: 9px;"
        )
        self.trajectory_path_stats_label.setToolTip(
            "Counts are based on the final generated playback samples and "
            "the path generator assigned to each surface patch."
        )

        self.trajectory_validation_label = QLabel(
            "PATH CHECKS | Coverage: —  Continuity: —  Orientation: —  Spray: —"
        )
        self.trajectory_validation_label.setMinimumHeight(18)
        self.trajectory_validation_label.setWordWrap(True)
        self.trajectory_validation_label.setStyleSheet(
            "color: #AFC2CB; font-size: 9px;"
        )
        self.trajectory_validation_label.setToolTip(
            "Geometric inspection results. These checks do not certify paint deposition or robot reachability."
        )
        self.trajectory_safety_label.hide()
        playback_panel.hide()
        return panel

    def _update_trajectory_safety_label(self):
        safety = self.trajectory_safety if isinstance(self.trajectory_safety, dict) else {}
        statuses = np.asarray(safety.get("sample_status", []), dtype=str)
        if statuses.ndim != 1 or len(statuses) == 0:
            self.trajectory_safety_label.setText(
                "SAFETY CHECK | NOT AVAILABLE — regenerate Stage 3"
            )
            return
        normalized = np.char.upper(np.char.strip(statuses))
        red = int(np.count_nonzero(normalized == "RED"))
        yellow = int(np.count_nonzero(normalized == "YELLOW"))
        green = int(np.count_nonzero(normalized == "GREEN"))
        status = str(safety.get("status", "UNKNOWN")).upper()
        self.trajectory_safety_label.setText(
            f"SAFETY CHECK | {status} | RED {red:,} · YELLOW {yellow:,} · GREEN {green:,}"
        )
        limitations = safety.get("limitations", [])
        self.trajectory_safety_label.setToolTip(
            "RED = configured gun-body clearance violation. YELLOW = incomplete "
            "geometry checks. GREEN = all configured tool geometry checks complete.\n"
            + "\n".join(str(item) for item in limitations)
        )

    def _update_trajectory_validation_label(self):
        validation = self.trajectory_validation
        if not hasattr(self, "trajectory_validation_label"):
            return
        if not validation:
            self.trajectory_validation_label.setText(
                "PATH CHECKS | Diagnostics unavailable"
            )
            return
        continuity = validation.get("continuity", {})
        orientation = validation.get("orientation", {})
        spray = validation.get("spray_control", {})
        coverage_status = validation.get("coverage_status", "UNKNOWN")
        self.trajectory_validation_label.setText(
            f"PATH CHECKS | Coverage: {coverage_status} "
            f"{float(validation.get('surface_face_representation_percent', 0.0)):.1f}%  "
            f"Continuity: {continuity.get('status', 'UNKNOWN')} "
            f"({int(continuity.get('jump_outlier_count', 0))} jumps)  "
            f"Orientation: {orientation.get('status', 'UNKNOWN')} "
            f"(max {float(orientation.get('maximum_change_deg', 0.0)):.1f}°)  "
            f"Spray: {int(spray.get('stroke_segment_count', 0))} strokes"
        )
        self.trajectory_validation_label.setToolTip(
            "\n".join([
                str(validation.get("note", "")),
                f"Continuity max/P95: {float(continuity.get('maximum_step_mm', 0.0)):.2f} / "
                f"{float(continuity.get('p95_step_mm', 0.0)):.2f} mm; "
                f"limit {float(continuity.get('jump_limit_mm', 0.0)):.2f} mm",
                f"Nozzle orientation max/P95: "
                f"{float(orientation.get('maximum_change_deg', 0.0)):.2f} / "
                f"{float(orientation.get('p95_change_deg', 0.0)):.2f} deg",
                f"Spray ON samples: {int(spray.get('spray_on_sample_count', 0))}; "
                f"OFF boundaries: {int(spray.get('spray_off_transition_count', 0))}",
                str(spray.get("policy", "")),
            ])
        )

    def _on_trajectory_visibility_changed(self, key, checked):
        self.trajectory_visibility[key] = bool(checked)

        if self.display_mode != "trajectory" or self.current_mesh is None:
            return

        if key in ("planar", "curved"):
            self._update_trajectory_surface_visibility()
        elif key == "paths":
            if self._playback_path_actor is not None:
                self._playback_path_actor.SetVisibility(bool(checked))
            self.plotter.render()
        elif key == "safety":
            if self._trajectory_safety_actor is not None:
                self._trajectory_safety_actor.SetVisibility(bool(checked))
            self.plotter.render()

    def _set_trajectory_controls_visible(self, visible):
        self.display_control_panel.setVisible(bool(visible))

    def _update_trajectory_path_statistics_label(self):
        stats = self.trajectory_path_statistics
        if not hasattr(self, "trajectory_path_stats_label"):
            return
        if not stats:
            self.trajectory_path_stats_label.setText(
                "PATH SAMPLES | Breakdown unavailable"
            )
            self.trajectory_path_stats_label.setToolTip(
                "Regenerate Stage 3 with the updated trajectory diagnostics."
            )
            return

        planar = int(stats.get("planar_sample_count", 0))
        curved = int(stats.get("curved_sample_count", 0))
        fallback = int(stats.get("fallback_sample_count", 0))
        total = int(stats.get("playback_sample_count", 0))
        source = int(stats.get("source_candidate_sample_count", 0))
        planar_segments = int(stats.get("planar_segment_count", 0))
        curved_segments = int(stats.get("curved_segment_count", 0))
        fallback_segments = int(stats.get("fallback_segment_count", 0))

        self.trajectory_path_stats_label.setText(
            f"PATH SAMPLES | Planar: {planar:,}  Curved: {curved:,}  "
            f"Fallback: {fallback:,}  Total: {total:,}"
        )
        per_patch = stats.get("per_patch", [])
        patch_lines = [
            f"Patch {int(item.get('patch_id', -1))}: "
            f"{item.get('path_type', 'unknown')} | "
            f"{int(item.get('sample_count', 0)):,} samples | "
            f"{int(item.get('segment_count', 0))} segments"
            for item in per_patch
        ]
        tooltip_lines = [
            f"Pre-raster candidate samples: {source:,}",
            f"Final playback samples: {total:,}",
            f"Segments — planar: {planar_segments}, curved: {curved_segments}, "
            f"fallback: {fallback_segments}",
            "Per-patch breakdown:",
            *patch_lines,
            str(stats.get("note", "")),
        ]
        travel_plan = getattr(self, "trajectory_data", {}).get(
            "inter_segment_travel_plan", {}
        ) if isinstance(getattr(self, "trajectory_data", None), dict) else {}
        if travel_plan:
            tooltip_lines.extend([
                "",
                "SPRAY-OFF INTER-STROKE TRAVEL (straight-line estimate only):",
                f"Transitions: {int(travel_plan.get('transition_count', 0))}",
                f"Total travel distance: {float(travel_plan.get('total_distance_mm', 0.0)):.1f} mm",
                f"Estimated travel time: {float(travel_plan.get('estimated_time_s', 0.0)):.2f} s",
                f"Safety: {travel_plan.get('safety_status', 'NOT_CHECKED')}",
                str(travel_plan.get("note", "")),
            ])
        self.trajectory_path_stats_label.setToolTip("\n".join(tooltip_lines))

    # =========================================================
    # MANUAL REFERENCE POSE CONTROLS
    # =========================================================

    def _build_pose_controls(self):
        """Simple native Qt numeric selectors for absolute Roll/Pitch/Yaw."""
        panel = QFrame()
        panel.setObjectName("PoseReferenceControls")
        panel.setStyleSheet("""
            QFrame#PoseReferenceControls {
                background-color: #0D1822;
                border: 1px solid #294858;
                border-radius: 5px;
            }
            QLabel { color: #D8E5ED; padding: 2px; }
        """)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(8)

        attitude_label = QLabel("ATTITUDE RPY (°):")
        attitude_label.setStyleSheet(
            "QLabel { color: #19DDF2; font-weight: 700; }"
        )
        layout.addWidget(attitude_label)

        self.rpy_inputs = []
        for label in ("ROLL", "PITCH", "YAW"):
            field = QDoubleSpinBox()
            field.setRange(-3600.0, 3600.0)
            field.setDecimals(2)
            field.setSingleStep(0.01)
            field.setSuffix(f"° {label}")
            field.setFixedWidth(112)
            field.setToolTip(
                f"Absolute {label.lower()} angle in the World XYZ frame. "
                "The arrow increment is controlled by ANGLE STEP."
            )
            field.valueChanged.connect(self._apply_rpy_inputs)
            field.editingFinished.connect(self._apply_rpy_inputs)
            layout.addWidget(field)
            self.rpy_inputs.append(field)

        step_label = QLabel("ANGLE STEP (°):")
        step_label.setStyleSheet(
            "QLabel { color: #FFB84D; font-weight: 700; }"
        )
        layout.addWidget(step_label)

        self.rpy_step_input = QDoubleSpinBox()
        # Editable native number selector; its value drives all RPY arrow increments.
        self.rpy_step_input.setRange(0.01, 90.0)
        self.rpy_step_input.setDecimals(2)
        self.rpy_step_input.setSingleStep(0.1)
        self.rpy_step_input.setValue(1.0)
        self.rpy_step_input.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.rpy_step_input.setKeyboardTracking(False)
        self.rpy_step_input.setFixedWidth(92)
        self.rpy_step_input.setToolTip(
            "Set the angular increment used by the Roll, Pitch and Yaw number selectors."
        )
        self.rpy_step_input.valueChanged.connect(self._on_rpy_step_changed)
        layout.addWidget(self.rpy_step_input)
        self._on_rpy_step_changed(self.rpy_step_input.value())

        self.reset_rpy_button = QPushButton("RESET RPY")
        self.reset_rpy_button.setToolTip(
            "Reset product Roll, Pitch and Yaw to 0.00 degrees."
        )
        self.reset_rpy_button.clicked.connect(self._reset_rpy_angles)
        layout.addWidget(self.reset_rpy_button)
        layout.addStretch(1)

        # Hidden compatibility controls used by the existing pose workflow.
        self.world_position_inputs = [QDoubleSpinBox(panel) for _ in range(3)]
        self.workspace_size_inputs = [QDoubleSpinBox(panel) for _ in range(3)]
        self.apply_position_button = QPushButton(panel)
        self.apply_workspace_button = QPushButton(panel)
        self.hook_length_input = QDoubleSpinBox(panel)
        self.pick_hook_point_button = QPushButton(panel)
        self.pose_status_label = QLabel(panel)
        self.confirm_pose_button = QPushButton(panel)
        self.apply_rpy_button = QPushButton(panel)
        self.reset_pose_button = QPushButton(panel)
        self.apply_position_button.clicked.connect(self._apply_world_position)
        self.apply_workspace_button.clicked.connect(self._apply_workspace_bounds)
        self.hook_length_input.setRange(0.0, 10000000.0)
        self.hook_length_input.setValue(self.hook_length_source_units)
        self.hook_length_input.valueChanged.connect(self._on_hook_length_changed)
        self.pick_hook_point_button.clicked.connect(self._start_hook_point_picking)
        self.confirm_pose_button.clicked.connect(self._confirm_reference_pose)
        self.apply_rpy_button.clicked.connect(self._apply_rpy_inputs)
        self.reset_pose_button.clicked.connect(self._reset_reference_pose)
        for widget in (
            *self.world_position_inputs, *self.workspace_size_inputs,
            self.apply_position_button, self.apply_workspace_button,
            self.hook_length_input, self.pick_hook_point_button,
            self.pose_status_label, self.confirm_pose_button,
            self.apply_rpy_button, self.reset_pose_button,
        ):
            widget.hide()
        return panel

    def _on_rpy_step_changed(self, value):
        """Apply the selected angular increment to all three RPY selectors."""
        increment = max(float(value), 0.01)
        for field in self.rpy_inputs:
            field.setSingleStep(increment)

    def _reset_rpy_angles(self):
        """Reset orientation only; preserve product position and scale."""
        if self._pose_actor is None:
            return
        for field in self.rpy_inputs:
            field.blockSignals(True)
            field.setValue(0.0)
            field.blockSignals(False)
        self._apply_rpy_inputs()
        self.pose_status_label.setText(
            "ATTITUDE RESET  |  Roll, Pitch and Yaw set to 0.00°."
        )

    def _apply_product_transform_to_display_actors(self, matrix):
        """Apply one CAD-to-World transform to every visible product actor."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Product transform must be a finite 4x4 matrix.")

        self._last_product_transform = matrix.copy()
        vtk_matrix = self._vtk_matrix_from_numpy(matrix)

        # Use the full matrix as the single source of truth. Reset the
        # vtkProp3D component transforms first to avoid composing the same
        # rotation twice with an existing UserMatrix.
        actors = []
        if self._pose_actor is not None:
            actors.append(self._pose_actor)
        actors.extend(
            actor for actor in getattr(self, "_interactive_spray_texture_actors", {}).values()
            if actor is not None
        )

        seen = set()
        for actor in actors:
            if id(actor) in seen:
                continue
            seen.add(id(actor))
            actor.SetOrigin(0.0, 0.0, 0.0)
            actor.SetPosition(0.0, 0.0, 0.0)
            actor.SetOrientation(0.0, 0.0, 0.0)
            actor.SetScale(1.0, 1.0, 1.0)
            actor.SetUserMatrix(vtk_matrix)

        if self._cad_axes_actor is not None:
            self._cad_axes_actor.SetUserMatrix(vtk_matrix)

    def _apply_rpy_inputs(self):
        """Set absolute World-frame ZYX Roll/Pitch/Yaw orientation."""
        if self._pose_actor is None:
            return
        roll, pitch, yaw = np.radians([field.value() for field in self.rpy_inputs])
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)

        # Preserve the current CAD-origin world translation while replacing
        # its absolute orientation with R = Rz(yaw) * Ry(pitch) * Rx(roll).
        matrix = self._actor_matrix_array()
        matrix[:3, :3] = rz @ ry @ rx
        self._apply_product_transform_to_display_actors(matrix)
        self._update_pose_readout()
        self.pose_status_label.setText(
            "ATTITUDE UPDATED  |  Product geometry rotated in the World frame."
        )
        self.plotter.render()

    def _apply_world_position(self):
        if self._pose_actor is None:
            return
        position = np.asarray([field.value() for field in self.world_position_inputs], dtype=float)
        if not np.all(np.isfinite(position)):
            self.pose_status_label.setText("Invalid World XYZ position.")
            return
        self._set_cad_origin_world_position(position)
        self.plotter.render()
        self._update_pose_readout()

    def _set_cad_origin_world_position(self, desired_world_position):
        """Place CAD coordinate origin at a requested World XYZ position."""
        desired = np.asarray(desired_world_position, dtype=float)
        if desired.shape != (3,) or not np.all(np.isfinite(desired)):
            raise ValueError("CAD origin World position must be a finite XYZ vector.")
        current = self._actor_matrix_array()[:3, 3]
        delta = desired - current
        actor_position = np.asarray(self._pose_actor.GetPosition(), dtype=float)
        self._pose_actor.SetPosition(*(actor_position + delta).tolist())

    def _on_hook_length_changed(self, value):
        self.hook_length_source_units = float(value)
        if self._pose_actor is None or self.selected_hook_point_cad is None:
            return
        matrix = self._actor_matrix_array()
        point_world = (matrix @ np.append(self.selected_hook_point_cad, 1.0))[:3]
        target = np.array([0.0, 0.0, self.hook_length_source_units], dtype=float)
        delta = target - point_world
        actor_position = np.asarray(self._pose_actor.GetPosition(), dtype=float)
        self._pose_actor.SetPosition(*(actor_position + delta).tolist())
        self.hook_world_position = target
        self._sync_world_position_inputs()
        self._render_product(reset_camera=False)

    def _start_hook_point_picking(self):
        if self._pose_actor is None:
            self.pose_status_label.setText("Load a STEP/STL product before selecting its hook point.")
            return
        self.hook_length_source_units = float(self.hook_length_input.value())
        if self.hook_length_source_units <= 0.0:
            self.pose_status_label.setText("Enter a positive Hook Length before selecting the product hook point.")
            return
        self._hook_pick_active = True
        self.pose_status_label.setText("Click the product surface at the exact point where the hook will engage.")
        try:
            self.plotter.enable_surface_point_picking(
                callback=self._on_hook_point_picked,
                show_message=True,
                left_clicking=True,
                use_picker=True,
            )
        except Exception:
            try:
                self.plotter.enable_point_picking(
                    callback=self._on_hook_point_picked,
                    show_message=True,
                    left_clicking=True,
                    use_picker=True,
                )
            except Exception as exc:
                self._hook_pick_active = False
                self.pose_status_label.setText(f"Point picking could not be enabled: {exc}")

    def _on_hook_point_picked(self, picked_point, picker=None):
        if not self._hook_pick_active or self._pose_actor is None:
            return
        point = np.asarray(picked_point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            self.pose_status_label.setText("Invalid picked point. Click directly on the product surface.")
            return
        if picker is not None and hasattr(picker, "GetActor"):
            picked_actor = picker.GetActor()
            if picked_actor is not None and picked_actor != self._pose_actor:
                self.pose_status_label.setText("The selected point is not on the product. Please click the product surface.")
                return

        matrix = self._actor_matrix_array()
        try:
            cad_point = np.linalg.inv(matrix) @ np.append(point, 1.0)
        except np.linalg.LinAlgError:
            self.pose_status_label.setText("Product transform is singular; hook point cannot be calculated.")
            return
        cad_point = cad_point[:3] / cad_point[3]

        # Conveyor start point is World origin. The hook engagement point is
        # directly below it by the user-entered hook suspension length (+Z).
        target = np.array([0.0, 0.0, self.hook_length_source_units], dtype=float)
        delta = target - point
        actor_position = np.asarray(self._pose_actor.GetPosition(), dtype=float)
        self._pose_actor.SetPosition(*(actor_position + delta).tolist())

        self.selected_hook_point_cad = cad_point.copy()
        self.hook_world_position = target.copy()
        # Preserve the CAD origin's World position; product rotation
        # remains centered on the STEP coordinate origin.
        cad_origin_world = self._actor_matrix_array()[:3, 3].copy()
        # Keep the STEP CAD origin as the rotation pivot even when a
        # separate hook engagement point has been selected.
        self.rotation_pivot = np.zeros(3, dtype=float)
        self._pose_actor.SetOrigin(0.0, 0.0, 0.0)
        self._set_cad_origin_world_position(cad_origin_world)
        self._hook_pick_active = False
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        self._sync_world_position_inputs()
        self._render_product(reset_camera=False)
        self.pose_status_label.setText(
            f"Hook point aligned at World (0, 0, {target[2]:.2f}) {self._source_unit_label()}. "
            "Choose product orientation, then confirm."
        )

    def _apply_workspace_bounds(self):
        dimensions = np.asarray([field.value() for field in self.workspace_size_inputs], dtype=float)
        if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
            self.pose_status_label.setText("Box dimensions must be positive.")
            return
        self.workspace_dimensions = dimensions
        self._render_product(reset_camera=False)
        self.pose_status_label.setText("World display box updated; extents are configurable visualization bounds.")

    @staticmethod
    def _same_vtk_object(left, right):
        """Compare VTK objects by their underlying C++ address, not wrapper identity."""
        if left is None or right is None:
            return False
        try:
            left_address = left.GetAddressAsString("")
            right_address = right.GetAddressAsString("")
            if left_address and right_address:
                return left_address == right_address
        except Exception:
            pass
        try:
            return bool(left == right)
        except Exception:
            return False

    def _sync_world_position_inputs(self):
        if self._pose_actor is None:
            return
        position = self._actor_matrix_array()[:3, 3]
        for field, value in zip(self.world_position_inputs, position):
            field.blockSignals(True)
            field.setValue(float(value))
            field.blockSignals(False)

    def _actor_matrix_array(self):
        if self._pose_actor is None:
            return self._last_product_transform.copy()
        matrix = self._pose_actor.GetMatrix()
        values = np.array(
            [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)],
            dtype=float,
        )
        self._last_product_transform = values.copy()
        return values

    def get_product_transform(self):
        """Return the current CAD-to-World actor transform as a detached 4x4 array."""
        return self._actor_matrix_array().copy()

    @staticmethod
    def _rotation_to_rpy(rotation):
        # World-axis ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
        sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        if sy >= 1e-8:
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            pitch = math.atan2(-rotation[2, 0], sy)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            roll = math.atan2(-rotation[1, 2], rotation[1, 1])
            pitch = math.atan2(-rotation[2, 0], sy)
            yaw = 0.0
        return np.degrees([roll, pitch, yaw])

    def _current_pose_record(self):
        matrix = self._actor_matrix_array()
        u, _, vt = np.linalg.svd(matrix[:3, :3])
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return {
            "world_frame": "WORLD_XYZ",
            "world_axis_definition": {
                "origin": "WORK_CELL_BOX_GEOMETRIC_CENTER",
                "X": "BOX_WIDTH_CENTERLINE / OPERATOR_RIGHT",
                "Y": "CONVEYOR_TRAVEL_DIRECTION / INTO_WORKCELL",
                "Z": "VERTICALLY_DOWN",
            },
            "hook_length_source_units": self.hook_length_source_units,
            "selected_hook_point_cad": None if self.selected_hook_point_cad is None else self.selected_hook_point_cad.tolist(),
            "hook_engagement_point_world_xyz": None if self.hook_world_position is None else self.hook_world_position.tolist(),
            "product_frame": "PRODUCT_CAD",
            "rotation_pivot": "STEP_CAD_ORIGIN",
            "rotation_pivot_cad": [0.0, 0.0, 0.0],
            "position_world_xyz_source_units": matrix[:3, 3].tolist(),
            "roll_pitch_yaw_world_xyz_deg": self._rotation_to_rpy(rotation).tolist(),
            "rotation_matrix_3x3": rotation.tolist(),
            "transform_matrix_4x4": matrix.tolist(),
            "workspace_display_dimensions_source_units": self.workspace_dimensions.tolist(),
            "workspace_dimensions_are_measured": False,
        }

    def _update_pose_readout(self):
        if self._pose_actor is None:
            return
        rpy = self._current_pose_record()["roll_pitch_yaw_world_xyz_deg"]
        for field, value in zip(self.rpy_inputs, rpy):
            field.blockSignals(True)
            field.setValue(float(value))
            field.blockSignals(False)
        # Keep the Product coordinate frame synchronized with the displayed
        # CAD transform; the fixed World frame is deliberately untouched.
        if self._cad_axes_actor is not None:
            self._cad_axes_actor.SetUserMatrix(
                self._vtk_matrix_from_numpy(self._actor_matrix_array())
            )

    @staticmethod
    def _vtk_matrix_from_numpy(values):
        from vtkmodules.vtkCommonMath import vtkMatrix4x4
        matrix = vtkMatrix4x4()
        for row in range(4):
            for col in range(4):
                matrix.SetElement(row, col, float(values[row, col]))
        return matrix

    def _confirm_reference_pose(self):
        if self._pose_actor is None:
            return
        record = self._current_pose_record()
        record["coordinate_unit"] = self.source_unit
        record["display_unit"] = self.display_unit
        self.pose_confirmed.emit(record)
        self.pose_status_label.setText("Reference pose sent for local result storage.")

    def _reset_reference_pose(self):
        if self._pose_actor is not None:
            self._pose_actor.SetPosition(0.0, 0.0, 0.0)
            self._pose_actor.SetOrientation(0.0, 0.0, 0.0)
            self._pose_actor.SetScale(1.0, 1.0, 1.0)
            for field in getattr(self, "rpy_inputs", []):
                field.setValue(0.0)
            self.rotation_pivot = np.zeros(3, dtype=float)
            self._pose_actor.SetOrigin(0.0, 0.0, 0.0)
            self._set_cad_origin_world_position(np.zeros(3, dtype=float))
            self.selected_hook_point_cad = None
            self.hook_world_position = None
            self._sync_world_position_inputs()
            self.plotter.render()
        self._update_pose_readout()

    # =========================================================
    # PRODUCT COORDINATE SYSTEM
    # =========================================================

    @staticmethod
    def _calculate_cad_origin(mesh):
        bounds = np.asarray(mesh.bounds, dtype=float)

        if bounds.shape != (2, 3):
            raise ValueError(
                f"Unexpected mesh bounds shape: {bounds.shape}"
            )

        if not np.all(np.isfinite(bounds)):
            raise ValueError("Mesh contains invalid bounds.")

        return np.zeros(3, dtype=float)

    def _prepare_product_mesh(self, mesh):
        # The viewer treats the source mesh as read-only. Copying a finely
        # tessellated STEP mesh duplicates vertices, triangles and large
        # STEP metadata arrays without changing the displayed geometry.
        # Keep one shared mesh object; transforms are applied to VTK actors.
        self.source_origin = self._calculate_cad_origin(mesh)
        bounds = np.asarray(mesh.bounds, dtype=float)
        self.product_center_pivot = (bounds[0] + bounds[1]) * 0.5
        # Rotation pivot is the STEP CAD coordinate origin, not the mesh center.
        self.rotation_pivot = np.zeros(3, dtype=float)

        self.source_to_view_transform = np.eye(4, dtype=float)
        self.view_to_source_transform = np.eye(4, dtype=float)

        return mesh

    def source_to_view(self, point):
        point = np.asarray(point, dtype=float)

        if point.shape != (3,):
            raise ValueError("Point must contain 3 coordinates.")

        point_h = np.append(point, 1.0)
        return (self.source_to_view_transform @ point_h)[:3]

    def view_to_source(self, point):
        point = np.asarray(point, dtype=float)

        if point.shape != (3,):
            raise ValueError("Point must contain 3 coordinates.")

        point_h = np.append(point, 1.0)
        return (self.view_to_source_transform @ point_h)[:3]

    # =========================================================
    # MESH CONVERSION
    # =========================================================

    @staticmethod
    def _to_polydata(mesh, face_mask=None):
        vertices = np.asarray(mesh.vertices)
        source_faces = np.asarray(mesh.faces)
        if len(vertices) <= np.iinfo(np.int32).max:
            faces = np.asarray(source_faces, dtype=np.int32)
            index_dtype = np.int32
        else:
            faces = np.asarray(source_faces, dtype=np.int64)
            index_dtype = np.int64

        if face_mask is not None:
            face_mask = np.asarray(face_mask, dtype=bool)
            if len(face_mask) != len(faces):
                raise ValueError(
                    "Face visibility mask does not match mesh faces."
                )
            faces = faces[face_mask]

        if len(faces) == 0:
            return pv.PolyData()

        # Build the VTK packed-cell array in one allocation. The previous
        # hstack path temporarily allocated a second full-size faces block.
        packed_faces = np.empty(
            (len(faces), 4), dtype=index_dtype
        )
        packed_faces[:, 0] = 3
        packed_faces[:, 1:] = faces

        return pv.PolyData(
            vertices,
            packed_faces.reshape(-1),
        )

    def _build_offset_patch_shell(self, mesh, face_mask, offset_distance):
        """
        Build exactly one offset copy of the selected patch.

        Normals are accumulated only from triangles belonging to this patch.
        This prevents neighboring faces elsewhere on the product from bending
        the membrane inward and creating unwanted circular/ring-like layers.
        """
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        face_mask = np.asarray(face_mask, dtype=bool)

        if len(face_mask) != len(faces):
            raise ValueError("Face mask does not match mesh faces.")

        selected_faces = faces[face_mask]
        if len(selected_faces) == 0:
            return pv.PolyData()

        selected_triangles = vertices[selected_faces]
        edge_a = selected_triangles[:, 1] - selected_triangles[:, 0]
        edge_b = selected_triangles[:, 2] - selected_triangles[:, 0]
        weighted_normals = np.cross(edge_a, edge_b)

        vertex_normals = np.zeros_like(vertices, dtype=float)
        for corner in range(3):
            np.add.at(
                vertex_normals,
                selected_faces[:, corner],
                weighted_normals,
            )

        lengths = np.linalg.norm(vertex_normals, axis=1)
        used_vertices = np.unique(selected_faces)
        if np.any(lengths[used_vertices] <= 1e-12):
            raise ValueError(
                "Selected patch contains vertices with undefined local normals."
            )

        normalized = np.zeros_like(vertex_normals)
        valid = lengths > 1e-12
        normalized[valid] = (
            vertex_normals[valid] / lengths[valid, None]
        )

        shell = self._to_polydata(mesh, face_mask=face_mask)
        shell.points = vertices + normalized * float(offset_distance)
        return shell

    @staticmethod
    def _build_patch_light_beams(
        mesh, face_mask, max_beams=48, radius_scale=0.055
    ):
        """
        Build a small white visual halo around the selected patch.

        Ordinary patches use a thin, nearly surface-following white shell.
        Cylindrical/hole patches keep a short one-way axial light column so
        the opening direction remains easy to identify.
        """
        face_mask = np.asarray(face_mask, dtype=bool)
        if len(face_mask) != len(mesh.faces):
            raise ValueError("Selected patch mask does not match mesh triangles.")

        face_indices = np.flatnonzero(face_mask)
        if face_indices.size == 0:
            return pv.PolyData()

        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)[face_indices]
        vertex_ids = np.unique(faces)
        points = vertices[vertex_ids]
        normals = np.asarray(mesh.face_normals, dtype=float)[face_indices]
        normals = normals[
            np.all(np.isfinite(normals), axis=1)
            & (np.linalg.norm(normals, axis=1) > 1e-12)
        ]
        if len(points) < 3 or len(normals) == 0:
            return pv.PolyData()

        dimensions = np.ptp(vertices, axis=0)
        model_diagonal = float(np.linalg.norm(dimensions))
        if not np.isfinite(model_diagonal) or model_diagonal <= 0.0:
            return pv.PolyData()

        bounds = np.asarray(mesh.bounds, dtype=float)
        model_center = (bounds[0] + bounds[1]) * 0.5
        center = points.mean(axis=0)
        centered = points - center
        point_covariance = centered.T @ centered / max(len(centered), 1)
        point_eigenvalues, point_eigenvectors = np.linalg.eigh(point_covariance)

        unit_normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        normal_mean = unit_normals.mean(axis=0)
        normal_coherence = float(np.linalg.norm(normal_mean))
        normal_covariance = unit_normals.T @ unit_normals / len(unit_normals)
        normal_eigenvalues, normal_eigenvectors = np.linalg.eigh(normal_covariance)

        cylindrical = (
            float(normal_eigenvalues[1]) > 0.12
            and float(normal_eigenvalues[0]) < 0.18
            and normal_coherence < 0.80
        )

        if cylindrical:
            # Hole/cylindrical patch: retain a short one-way axial indicator.
            axis = normal_eigenvectors[:, 0]
            axis /= max(float(np.linalg.norm(axis)), 1e-12)

            axial_offset = float(np.dot(center - model_center, axis))
            if axial_offset < -1e-9:
                axis = -axis
            elif abs(axial_offset) <= 1e-9:
                dominant = int(np.argmax(np.abs(axis)))
                if axis[dominant] < 0.0:
                    axis = -axis

            axial = centered @ axis
            radial_vectors = centered - axial[:, None] * axis[None, :]
            radial_distances = np.linalg.norm(radial_vectors, axis=1)
            radial_radius = float(np.median(radial_distances))

            beam_length = model_diagonal * 0.18
            radius = max(
                radial_radius * 0.62,
                model_diagonal * float(radius_scale) * 0.35,
            )
            beam_center = center + axis * (beam_length * 0.5)

            # Exactly one cylinder: no concentric inner cylinder/ring.
            return pv.Cylinder(
                center=beam_center,
                direction=axis,
                radius=radius,
                height=beam_length,
                resolution=48,
                capping=False,
            )

        # Ordinary planar/curved patches use the parallel white membrane
        # rendered by _build_offset_patch_shell; do not add a beam here.
        return pv.PolyData()

    # =========================================================
    # EMPTY SCENE
    # =========================================================

    def _draw_empty_scene(self):
        if self._world_orientation_widget is not None:
            try:
                self._world_orientation_widget.SetEnabled(0)
            except Exception:
                pass
        self._world_orientation_widget = None
        self._world_orientation_axes = None
        self.plotter.clear()
        self._pose_actor = None
        self._cad_axes_actor = None
        self.plotter.set_background("#07111D", top="#142B40")

        self.plotter.render()

    # =========================================================
    # PRODUCT DISPLAY MODES
    # =========================================================

    def show_product(self, mesh):
        self._last_product_transform = np.eye(4, dtype=float)
        self.trajectory_product_transform = np.eye(4, dtype=float)
        self.current_mesh = self._prepare_product_mesh(mesh)

        self.patch_labels = None
        self.face_status_labels = None
        self.trajectory_data = None
        self.face_surface_types = None
        self.selected_patch_id = None

        self.display_mode = "product"
        self._set_trajectory_controls_visible(False)

        # Show the CAD product together with the fixed World frame.
        # Product pose manipulation is enabled; World reference geometry
        # remains locked and cannot be selected or dragged.
        self.pose_control_panel.show()
        self.confirm_pose_button.setEnabled(True)
        self.workspace_dimensions = np.maximum(
            np.asarray(self.current_mesh.extents, dtype=float) * 4.0,
            100.0,
        )
        for field, value in zip(self.workspace_size_inputs, self.workspace_dimensions):
            field.setValue(float(value))
        # World origin is the conveyor travel start point. The CAD origin
        # initially remains at World origin until the product hook point is
        # selected and aligned with (0, 0, hook_length) along +Z.
        self.selected_hook_point_cad = None
        self.hook_world_position = None
        self.hook_length_source_units = float(self.hook_length_input.value())
        initial_world_position = np.zeros(3, dtype=float)
        for field, value in zip(self.world_position_inputs, initial_world_position):
            field.setValue(float(value))

        self._render_product(reset_camera=True)

    def show_surface_patches(self, mesh, labels):
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError(
                "Surface patch labels must be one-dimensional."
            )

        if len(labels) != len(mesh.faces):
            raise ValueError(
                "Surface patch label count does not match "
                "the mesh triangle count."
            )

        if not np.all(np.isfinite(labels)):
            raise ValueError("Surface patch labels contain invalid values.")

        if not np.all(labels == np.floor(labels)):
            raise ValueError("Surface patch labels must be integers.")

        self.current_mesh = self._prepare_product_mesh(mesh)

        self.patch_labels = labels.astype(np.int32, copy=True)
        
        self.face_status_labels = None
        self.trajectory_data = None
        self.face_surface_types = None

        # Reset selection when a new patch dataset is loaded.
        self.selected_patch_id = None

        self.display_mode = "patches"
        self._set_trajectory_controls_visible(False)

        # Keep the reference-pose controls available after Stage 1.
        # The current CAD actor matrix and camera are preserved by the
        # non-resetting render path; Stage 1 must not reset the user's RPY.
        self._render_product(reset_camera=False)

    def set_selected_patch(self, patch_id):
        """
        Select and highlight one surface patch.

        Args:
            patch_id:
                Integer Patch ID, or None to clear the selection.

        The original patch colors are retained.
        The current camera position is preserved.
        """
        if self.display_mode != "patches" or self.patch_labels is None:
            raise RuntimeError(
                "Surface patches have not been loaded."
            )

        if patch_id is None:
            self.selected_patch_id = None

        else:
            try:
                patch_id = int(patch_id)
            except (TypeError, ValueError):
                raise ValueError(
                    "Patch ID must be an integer or None."
                )

            if not np.any(self.patch_labels == patch_id):
                raise ValueError(
                    f"Surface patch ID {patch_id} does not exist."
                )

            self.selected_patch_id = patch_id
            mask = self.patch_labels == patch_id

            print("\n========== SELECTED PATCH ==========")
            print("Patch ID:", patch_id)
            print("Face count:", int(np.sum(mask)))

            patch_faces = self.current_mesh.faces[mask]

            patch_vertices = np.unique(patch_faces)

            patch_points = self.current_mesh.vertices[patch_vertices]

            print("Bounding box minimum:", patch_points.min(axis=0))
            print("Bounding box maximum:", patch_points.max(axis=0))
            print("Center:", patch_points.mean(axis=0))
            print("===================================\n")

        self._render_product(reset_camera=False)

    def start_interactive_spray_path(
        self, standoff, cone_angle_deg, sampling_spacing=None,
        sprayer_profile=None, coating_process=None,
        excluded_face_mask=None,
        resume_existing=False,
    ):
        """Start a spray session, optionally continuing the existing coated route.

        When resume_existing is true, the previously recorded route and
        accumulated coating field are preserved. The operator can inspect the
        already-painted surface and add new strokes without starting over.
        """
        if self.current_mesh is None:
            raise RuntimeError("Load a product before starting an interactive spray path.")
        if not np.isfinite(standoff) or standoff <= 0.0:
            raise ValueError("Spray standoff must be a positive finite value.")
        if not np.isfinite(cone_angle_deg) or not 0.0 < cone_angle_deg < 179.0:
            raise ValueError("Spray cone angle must be in (0, 179) degrees.")
        if sampling_spacing is None:
            sampling_spacing = max(float(standoff) * 0.025, 1e-3)
        if not np.isfinite(sampling_spacing) or sampling_spacing <= 0.0:
            raise ValueError("Spray sampling spacing must be a positive finite value.")
        sampling_spacing = float(sampling_spacing)
        if self._interactive_spray_active:
            return

        # Route editing is additive. Preserve the optimized route and the
        # accumulated wet-film state so the operator can see what has already
        # been sprayed and add only the missing coverage.
        preserve_existing = bool(resume_existing)
        existing_records = list(self._interactive_spray_records) if preserve_existing else []
        existing_thickness = (
            self._interactive_spray_face_thickness_um.copy()
            if preserve_existing
            and self._interactive_spray_face_thickness_um is not None
            and len(self._interactive_spray_face_thickness_um) == len(self.current_mesh.faces)
            else None
        )
        existing_surface_field = (
            copy.deepcopy(self._interactive_spray_surface_field)
            if preserve_existing
            and isinstance(self._interactive_spray_surface_field, dict)
            else None
        )

        self.display_mode = "spray_point_preview"
        self.trajectory_data = None
        self.patch_labels = None
        self.face_status_labels = None
        self.face_surface_types = None
        if excluded_face_mask is None:
            # Fail closed: Stage 3 must never silently treat an unclassified
            # product as fully paintable.
            self._interactive_spray_excluded_face_mask = np.ones(
                len(self.current_mesh.faces), dtype=bool
            )
        else:
            excluded_mask = np.asarray(excluded_face_mask, dtype=bool).reshape(-1)
            if excluded_mask.size != len(self.current_mesh.faces):
                raise ValueError(
                    "Stage 2 excluded-face mask must match the product triangle count."
                )
            self._interactive_spray_excluded_face_mask = excluded_mask.copy()

        self._interactive_spray_face_thickness_um = (
            existing_thickness
            if existing_thickness is not None
            else np.zeros(len(self.current_mesh.faces), dtype=np.float64)
        )
        self._interactive_spray_surface_field = (
            existing_surface_field
            if existing_surface_field is not None
            else {
            "schema": "STEP_BREP_UV_FILM_THICKNESS_FIELD_V3",
            "coordinate_frame": "ORIGINAL_STEP_FACE_UV",
            "field_representation": "CONTINUOUS_UV_TEXTURE_GRID",
            "interpolation_method": "CAD_SURFACE_METRIC_AWARE_GAUSSIAN",
            "texture_resolution": 256,
            "faces": {},
            }
        )
        self._interactive_spray_texture_actors = {}
        self._interactive_spray_texture_layout = {}
        self._interactive_spray_texture_objects = {}
        self._interactive_spray_dirty_face_ids = set()
        self._interactive_spray_base_opacity = None
        self._interactive_spray_face_area_mm2 = None
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_start_timestamp = None
        self._interactive_spray_paint_footprint_angle_deg = float(cone_angle_deg)
        self._interactive_spray_gun_collision = False
        self._interactive_spray_surface_excluded = False
        self._interactive_spray_collision_filter = None
        self._interactive_spray_paint_state_dirty = False
        self._interactive_coating_process = dict(coating_process or {})
        try:
            self._interactive_spray_paint_target_um = max(
                float(self._interactive_coating_process.get(
                    "target_dry_film_thickness_um",
                    DEFAULT_COATING_PROCESS["target_dry_film_thickness_um"],
                )),
                1e-9,
            )
        except (TypeError, ValueError):
            self._interactive_spray_paint_target_um = DEFAULT_COATING_PROCESS[
                "target_dry_film_thickness_um"
            ]
        self._render_product(reset_camera=False)
        # STEP models render coating as per-Face UV textures. The original
        # mesh actor remains available for cell picking and occlusion.
        self._interactive_build_brep_texture_actors()
        self._interactive_build_excluded_face_overlay()

        self._interactive_spray_standoff = float(standoff)
        self._interactive_spray_cone_angle_deg = float(cone_angle_deg)
        self._interactive_spray_sampling_spacing = float(sampling_spacing)
        self._interactive_spray_sampling_previous_pose = None
        self._interactive_spray_sampling_distance_accumulator = 0.0
        self._interactive_sprayer_profile = dict(sprayer_profile or {})
        try:
            tail_sigma_fraction = max(
                float(self._interactive_sprayer_profile.get(
                    "overspray_tail_sigma_fraction", 0.35
                )),
                0.05,
            )
        except (TypeError, ValueError):
            tail_sigma_fraction = 0.35
        overspray_extent_factor = 1.0 + 3.0 * tail_sigma_fraction
        self._interactive_spray_paint_footprint_angle_deg = math.degrees(
            2.0 * math.atan(
                math.tan(math.radians(self._interactive_spray_cone_angle_deg * 0.5))
                * overspray_extent_factor
            )
        )
        self._interactive_recorded_replay_timer.stop()
        if self._interactive_recorded_replay_actor is not None:
            try:
                self.plotter.remove_actor(self._interactive_recorded_replay_actor, render=False)
            except Exception:
                pass
            self._interactive_recorded_replay_actor = None
        if not preserve_existing:
            self.interactive_recorded_path_panel.hide()
        else:
            self.interactive_recorded_path_panel.show()
            self.recorded_path_replay_button.setEnabled(len(existing_records) >= 2)
        self._interactive_spray_transform = self._actor_matrix_array().copy()
        self._interactive_spray_surface = self._to_polydata(self.current_mesh).clean()
        self._interactive_spray_locator = vtkOBBTree()
        self._interactive_spray_locator.SetDataSet(self._interactive_spray_surface)
        self._interactive_spray_locator.BuildLocator()
        self._interactive_spray_latest_pose = None
        self._interactive_spray_records = list(existing_records)
        if existing_records:
            # Continue stroke IDs after the existing route.
            existing_stroke_ids = [
                int(item.get("stroke_id", -1)) for item in existing_records
            ]
            self._interactive_spray_stroke_id = max(existing_stroke_ids)

            # Exclude the idle time between editing sessions from the
            # recording elapsed-time metric.
            latest_timestamp = max(
                float(item.get("timestamp", 0.0))
                for item in existing_records
            )
            timestamp_shift = time.monotonic() - latest_timestamp
            if np.isfinite(timestamp_shift):
                for item in existing_records:
                    item["timestamp"] = (
                        float(item.get("timestamp", 0.0)) + timestamp_shift
                    )
            self._interactive_spray_start_timestamp = max(
                float(item.get("timestamp", 0.0))
                for item in existing_records
            )
        else:
            self._interactive_spray_stroke_id = -1
            self._interactive_spray_start_timestamp = None
        self._interactive_spray_painted_actors = []
        self._interactive_spray_live_actors = []
        self._interactive_spray_active = True

        # The route editor must reopen with the accumulated wet-film state.
        if preserve_existing and existing_thickness is not None:
            self._interactive_refresh_paint_visual(render=False)
            self._refresh_interactive_recorded_path_actor()

        # Cursor motion only previews the nozzle. Paint deposition and pose
        # recording are gated by the left mouse button.
        self._interactive_spray_recording = False
        # Preserve the existing stroke counter in additive edit mode so every
        # new stroke gets a unique ID after the existing route.
        if not existing_records:
            self._interactive_spray_stroke_id = -1
        self._interactive_last_record_time = 0.0

        interactor = self.plotter.iren.interactor
        style = interactor.GetInteractorStyle()
        self._interactive_saved_style = style
        if style is not None:
            self._interactive_saved_style_enabled = bool(style.GetEnabled())
        self._interactive_spray_input_style = vtkInteractorStyleUser()
        self.interaction_mode_panel.show()
        self.working_distance_spin.setEnabled(True)
        factor = self._get_source_mm_factor()
        if factor is not None:
            self.working_distance_spin.blockSignals(True)
            self.working_distance_spin.setValue(float(standoff) * float(factor))
            self.working_distance_spin.blockSignals(False)
        # Stage entry starts in normal camera navigation. Space explicitly
        # switches the viewport to operator-controlled spray input.
        self._interactive_spray_mode = "camera"
        self.interaction_mode_status.setText(
            "CAMERA VIEW · PRESS SPACE FOR SPRAY MODE"
        )

        self._interactive_hud_actor = self.plotter.add_text(
            (
                f"SPRAY CURSOR ACTIVE  |  WORK DIST {self._interactive_spray_standoff * (self._get_source_mm_factor() or 1.0):.1f} mm "
                f"| CONE {self._interactive_spray_cone_angle_deg:.1f} DEG  |  "
                "CAMERA VIEW  |  NAVIGATE: MOUSE  |  SPACE: SPRAY MODE"
            ),
            position="upper_left",
            font_size=10,
            color="#00E5FF",
            name="interactive_spray_hud",
        )
        self.plotter.render()

    def _interactive_build_excluded_face_overlay(self):
        """Show Stage 2 non-paintable triangles as a persistent red overlay."""
        mask = self._interactive_spray_excluded_face_mask
        if mask is None or not np.any(mask) or self.current_mesh is None:
            return False
        excluded_ids = np.flatnonzero(mask)
        if not len(excluded_ids):
            return False
        try:
            source = self._to_polydata(self.current_mesh)
            excluded_surface = source.extract_cells(excluded_ids).extract_surface().triangulate()
            if excluded_surface.n_cells == 0:
                return False
            excluded_surface = excluded_surface.compute_normals(
                point_normals=True,
                cell_normals=False,
                auto_orient_normals=False,
                inplace=False,
            )
            points = np.asarray(excluded_surface.points, dtype=np.float64)
            normals = np.asarray(
                excluded_surface.point_data.get("Normals", np.zeros_like(points)),
                dtype=np.float64,
            )
            diagonal = float(np.linalg.norm(
                np.asarray(self.current_mesh.bounds[1], dtype=np.float64)
                - np.asarray(self.current_mesh.bounds[0], dtype=np.float64)
            ))
            offset = max(diagonal * 3e-6, 1e-8)
            if normals.shape == points.shape:
                excluded_surface.points = points + normals * offset
            actor = self.plotter.add_mesh(
                excluded_surface,
                color="#FF3030",
                opacity=0.42,
                lighting=False,
                show_edges=False,
                pickable=False,
                name="stage2_excluded_face_overlay",
            )
            actor.SetOrigin(0.0, 0.0, 0.0)
            actor.SetUserMatrix(
                self._vtk_matrix_from_numpy(self._actor_matrix_array())
            )
            self._interactive_spray_excluded_overlay_actor = actor
            return True
        except Exception:
            self._interactive_spray_excluded_overlay_actor = None
            return False

    def _interactive_build_brep_texture_actors(self):
        """Create one UV-textured render surface per original STEP B-Rep Face.

        The source mesh actor remains the authoritative picker/occlusion
        geometry. Textured actors are non-pickable display surfaces only.
        """
        if self.current_mesh is None or self._pose_actor is None:
            return False
        try:
            from core.step_loader import get_step_brep_context
            from core.uv_coating_texture import (
                DEFAULT_UV_TEXTURE_RESOLUTION,
                face_uv_layout,
                face_uv_triangles_to_texture_coordinates,
            )
            context = get_step_brep_context(self.current_mesh)
            if context is None:
                return False
            face_ids = np.asarray(context["triangle_face_ids"], dtype=np.int64)
            uv_triangles_all = np.asarray(context["triangle_uvs"], dtype=np.float64)
            uv_valid = np.asarray(context["triangle_uv_valid"], dtype=bool)
            if not np.any(uv_valid):
                return False

            resolution = int(DEFAULT_UV_TEXTURE_RESOLUTION)
            diagonal = float(np.linalg.norm(
                np.asarray(self.current_mesh.bounds[1])
                - np.asarray(self.current_mesh.bounds[0])
            ))
            offset = max(diagonal * 1e-6, 1e-8)
            built_triangle_count = 0

            for face_id in sorted(int(value) for value in np.unique(face_ids[uv_valid])):
                triangle_ids = np.flatnonzero((face_ids == face_id) & uv_valid)
                if triangle_ids.size == 0:
                    continue
                uv_triangles = uv_triangles_all[triangle_ids]
                face_record = context["face_records"][face_id - 1]
                layout = face_uv_layout(face_record, uv_triangles)
                texture_coordinates = face_uv_triangles_to_texture_coordinates(
                    uv_triangles,
                    layout["uv_bounds"],
                    periodic_u=layout["periodic_u"],
                    periodic_v=layout["periodic_v"],
                )

                source_triangles = np.asarray(
                    self.current_mesh.faces[triangle_ids], dtype=np.int64
                )
                points = np.asarray(
                    self.current_mesh.vertices[source_triangles], dtype=np.float64
                ).reshape(-1, 3)
                triangle_normals = np.asarray(
                    self.current_mesh.face_normals[triangle_ids], dtype=np.float64
                )
                points += np.repeat(triangle_normals, 3, axis=0) * offset

                packed_faces = np.empty((len(triangle_ids), 4), dtype=np.int64)
                packed_faces[:, 0] = 3
                packed_faces[:, 1:] = np.arange(
                    len(triangle_ids) * 3, dtype=np.int64
                ).reshape(-1, 3)
                textured_mesh = pv.PolyData(points, packed_faces.reshape(-1))
                textured_mesh.active_texture_coordinates = (
                    texture_coordinates.reshape(-1, 2).astype(np.float32)
                )

                # Reuse the original mesh's vertex normals on the UV render
                # surface. Vertices are duplicated per triangle for Face-local
                # UV seams, but matching point normals keep the CAD surface
                # visually smooth across tessellation boundaries.
                source_vertex_normals = np.asarray(
                    self.current_mesh.vertex_normals, dtype=np.float64
                )
                if (
                    source_vertex_normals.shape
                    == (len(self.current_mesh.vertices), 3)
                    and np.isfinite(source_vertex_normals).all()
                ):
                    point_normals = source_vertex_normals[source_triangles].reshape(-1, 3)
                    normal_lengths = np.linalg.norm(point_normals, axis=1)
                    point_normals /= np.maximum(normal_lengths[:, None], 1e-12)
                    textured_mesh.point_data["Normals"] = point_normals.astype(np.float32)

                base_image = np.full(
                    (resolution, resolution, 3), 168, dtype=np.uint8
                )
                texture = pv.Texture(
                    base_image,
                    interpolate=True,
                    repeat=bool(layout["periodic_u"] or layout["periodic_v"]),
                )
                actor = self.plotter.add_mesh(
                    textured_mesh,
                    texture=texture,
                    lighting=True,
                    smooth_shading=False,
                    ambient=0.22,
                    diffuse=0.78,
                    specular=0.28,
                    specular_power=28.0,
                    pickable=False,
                    name=f"interactive_brep_coating_face_{face_id}",
                )
                actor.GetProperty().SetInterpolationToPhong()
                actor.SetOrigin(0.0, 0.0, 0.0)
                actor.SetUserMatrix(
                    self._vtk_matrix_from_numpy(self._actor_matrix_array())
                )
                self._interactive_spray_texture_actors[face_id] = actor
                self._interactive_spray_texture_layout[face_id] = {
                    **layout,
                    "resolution": resolution,
                    "triangle_count": int(len(triangle_ids)),
                    "triangle_ids": triangle_ids,
                }
                self._interactive_spray_texture_objects[face_id] = texture
                built_triangle_count += int(len(triangle_ids))

            # When every source triangle has exact Face UVs, the texture
            # surfaces fully replace the base mesh visually. Keep that base
            # actor pickable; vtkCellPicker still uses it for global cell IDs.
            if built_triangle_count == len(self.current_mesh.faces):
                self._interactive_spray_base_opacity = float(
                    self._pose_actor.GetProperty().GetOpacity()
                )
                self._pose_actor.GetProperty().SetOpacity(0.001)
            if self._interactive_spray_texture_actors:
                self.plotter.render()
                return True
        except Exception as exc:
            print(
                "[B-REP UV TEXTURE WARNING] Falling back to mesh coating colors: "
                f"{type(exc).__name__}: {exc}"
            )
        return False

    def _interactive_refresh_brep_texture_images(self):
        """Update only the Face textures whose continuous UV fields changed."""
        if not self._interactive_spray_texture_actors:
            return
        faces = self._interactive_spray_surface_field.get("faces", {})
        target = max(float(self._interactive_spray_paint_target_um), 1e-9)
        dirty_ids = set(self._interactive_spray_dirty_face_ids)
        if not dirty_ids:
            dirty_ids = {int(value) for value in faces}
        for face_id in sorted(dirty_ids):
            face_record = faces.get(face_id, faces.get(str(face_id)))
            if face_record is None:
                continue
            face_id = int(face_id)
            actor = self._interactive_spray_texture_actors.get(face_id)
            grid = face_record.get("thickness_grid_um")
            if actor is None or grid is None:
                continue
            thickness_values = np.asarray(grid, dtype=np.float64)
            if self._interactive_spray_display_mode == "paint_color":
                paint_rgb = self._interactive_selected_paint_rgb()
                coverage = np.clip(
                    thickness_values / max(target * 0.02, 0.25), 0.0, 1.0
                )
                rgb = np.rint(
                    168.0 * (1.0 - coverage[..., None])
                    + paint_rgb[None, None, :] * coverage[..., None]
                ).astype(np.uint8)
            else:
                rgb = map_film_thickness_to_rgb(thickness_values, target)

            # A face becomes progressively glossier as its deposited film
            # builds. The UV texture carries the spatial color/thickness
            # pattern; the material response adds the wet-coating highlight.
            positive = thickness_values[thickness_values > 1e-6]
            wetness = (
                float(np.clip(np.percentile(positive, 90) / target, 0.0, 1.0))
                if positive.size else 0.0
            )
            prop = actor.GetProperty()
            prop.SetInterpolationToPhong()
            prop.SetAmbient(0.18 + 0.08 * wetness)
            prop.SetDiffuse(0.82 - 0.08 * wetness)
            prop.SetSpecular(0.18 + 0.62 * wetness)
            prop.SetSpecularPower(18.0 + 42.0 * wetness)

            # VTK texture image rows are bottom-up relative to UV v.
            image = np.ascontiguousarray(np.flipud(rgb), dtype=np.uint8)
            texture = pv.Texture(
                image,
                interpolate=True,
                repeat=bool(
                    face_record.get("periodic_u", False)
                    or face_record.get("periodic_v", False)
                ),
            )
            actor.SetTexture(texture)
            actor.GetMapper().Modified()
            self._interactive_spray_texture_objects[face_id] = texture
        self._interactive_spray_dirty_face_ids.clear()
        self._interactive_spray_paint_state_dirty = False

    def _interactive_export_surface_field(self):
        """Return a serialization-friendly copy of the per-Face UV thickness grids."""
        source = self._interactive_spray_surface_field or {}
        exported = {
            key: value
            for key, value in source.items()
            if key != "faces"
        }
        exported["faces"] = {}
        for face_id, record in source.get("faces", {}).items():
            copied = {
                key: value
                for key, value in record.items()
                if key != "thickness_grid_um"
            }
            grid = record.get("thickness_grid_um")
            if grid is not None:
                copied["thickness_grid_um"] = np.asarray(
                    grid, dtype=np.float32
                ).round(4).tolist()
            exported["faces"][int(face_id)] = copied
        return exported

    def _interactive_paint_rgb(self):
        """Build stable per-face RGB values from accumulated coating thickness."""
        if self.current_mesh is None:
            return None
        thickness = self._interactive_spray_face_thickness_um
        if thickness is None or len(thickness) != len(self.current_mesh.faces):
            return None
        target = max(float(self._interactive_spray_paint_target_um), 1e-9)
        if self._interactive_spray_display_mode == "paint_color":
            paint_rgb = self._interactive_selected_paint_rgb()
            coverage = np.clip(
                np.asarray(thickness, dtype=np.float64) / max(target * 0.02, 0.25),
                0.0, 1.0,
            )
            return np.rint(
                168.0 * (1.0 - coverage[:, None])
                + paint_rgb[None, :] * coverage[:, None]
            ).astype(np.uint8)
        return map_film_thickness_to_rgb(thickness, target)

    def _interactive_refresh_paint_visual(self, render=True):
        """Refresh the single persistent product actor after coating changes."""
        if self._pose_actor is None:
            return
        if self._interactive_spray_texture_actors:
            try:
                self._interactive_refresh_brep_texture_images()
                if render:
                    self.plotter.render()
                return
            except Exception as exc:
                print(
                    "[B-REP UV TEXTURE UPDATE WARNING] "
                    f"{type(exc).__name__}: {exc}"
                )
                self._interactive_spray_paint_state_dirty = True
                return
        rgb = self._interactive_paint_rgb()
        if rgb is None:
            return
        try:
            polydata = self._pose_actor.GetMapper().GetInput()
            if polydata is None or polydata.GetNumberOfCells() != len(rgb):
                return
            polydata.cell_data["Interactive Paint RGB"] = rgb
            polydata.Modified()
            self._pose_actor.GetMapper().Modified()
            if render:
                self.plotter.render()
        except Exception:
            self._interactive_spray_paint_state_dirty = True

    def _interactive_refresh_recorded_paint_color(self):
        """Recolor accumulated paint without replacing any geometry."""
        self._interactive_refresh_paint_visual(render=True)

    def _interactive_remove_live_actors(self):
        for actor in self._interactive_spray_live_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._interactive_spray_live_actors = []

    def _on_interactive_working_distance_changed(self, value_mm):
        """Change the spray tip-to-surface working distance live."""
        if not self._interactive_spray_active:
            return
        factor = self._get_source_mm_factor()
        if factor is None or not np.isfinite(value_mm):
            return
        distance_source = float(value_mm) / float(factor)
        if distance_source <= 0.0:
            return

        self._interactive_spray_standoff = distance_source

        # Changing the working distance is a discontinuous nozzle motion.
        if self._interactive_spray_recording:
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
    
        pose = self._interactive_spray_latest_pose
        if pose is None:
            return

        surface_point = np.asarray(pose["surface_point_world"], dtype=np.float64)
        normal = np.asarray(pose["surface_normal_world"], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)

        direction = np.asarray(
            pose["spray_direction_world"], dtype=np.float64
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        updated = dict(pose)
        # Working distance is measured along the current cone centerline,
        # preserving its surface intersection even when the cone is tilted.
        updated["nozzle_point_world"] = surface_point - direction * distance_source
        updated["timestamp"] = time.monotonic()
        self._interactive_update_cursor(updated)

    @staticmethod
    def _interactive_rotate_spray_direction(direction, yaw_deg, pitch_deg):
        """Rotate the spray axis without changing the nozzle or surface position."""
        axis = np.asarray(direction, dtype=np.float64).copy()
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        yaw, pitch = math.radians(float(yaw_deg)), math.radians(float(pitch_deg))
        if abs(yaw) < 1e-12 and abs(pitch) < 1e-12:
            return axis
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, axis))) > 0.92:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        yaw_axis = np.cross(reference, axis)
        yaw_axis /= max(float(np.linalg.norm(yaw_axis)), 1e-12)
        pitch_axis = np.cross(axis, yaw_axis)
        pitch_axis /= max(float(np.linalg.norm(pitch_axis)), 1e-12)
        def rotate(vector, rotation_axis, angle):
            cosine, sine = math.cos(angle), math.sin(angle)
            return (vector*cosine + np.cross(rotation_axis, vector)*sine
                    + rotation_axis*float(np.dot(rotation_axis, vector))*(1.0-cosine))
        yawed = rotate(axis, yaw_axis, yaw)
        yawed_pitch = rotate(pitch_axis, yaw_axis, yaw)
        result = rotate(yawed, yawed_pitch, pitch)
        return result / max(float(np.linalg.norm(result)), 1e-12)

    def _interactive_pick_surface_pose(self, x, y):
        if self._pose_actor is None or self.current_mesh is None:
            return None
        picker = self._patch_cell_picker
        picker.InitializePickList()
        picker.AddPickList(self._pose_actor)
        picker.PickFromListOn()
        if not picker.Pick(int(x), int(y), 0, self.plotter.renderer):
            return None
        if picker.GetActor() != self._pose_actor:
            return None

        cell_id = int(picker.GetCellId())
        face_normals = np.asarray(self.current_mesh.face_normals, dtype=float)
        if cell_id < 0 or cell_id >= len(face_normals):
            return None
        excluded_mask = self._interactive_spray_excluded_face_mask
        surface_excluded = (
            excluded_mask is not None
            and cell_id < len(excluded_mask)
            and bool(excluded_mask[cell_id])
        )
        normal_local = face_normals[cell_id]
        # Interpolate vertex normals at the picked triangle location so the
        # nozzle attitude does not jump at every triangle boundary.
        try:
            vertex_normals = np.asarray(self.current_mesh.vertex_normals, dtype=float)
            face_vertices = np.asarray(self.current_mesh.faces[cell_id], dtype=np.int64)
            parametric = np.asarray(picker.GetPCoords(), dtype=float)
            if (
                vertex_normals.shape == (len(self.current_mesh.vertices), 3)
                and face_vertices.shape == (3,)
                and parametric.shape == (3,)
            ):
                weights = np.asarray([
                    1.0 - parametric[0] - parametric[1],
                    parametric[0],
                    parametric[1],
                ], dtype=float)
                interpolated = weights @ vertex_normals[face_vertices]
                if np.all(np.isfinite(interpolated)) and np.linalg.norm(interpolated) > 1e-12:
                    normal_local = interpolated
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        normal_length = float(np.linalg.norm(normal_local))
        if not np.isfinite(normal_length) or normal_length <= 1e-12:
            return None
        normal_local = normal_local / normal_length

        transform = self._actor_matrix_array()
        rotation = transform[:3, :3]
        normal_world = rotation @ normal_local
        normal_world_length = float(np.linalg.norm(normal_world))
        if normal_world_length <= 1e-12:
            return None
        normal_world /= normal_world_length

        hit_world = np.asarray(picker.GetPickPosition(), dtype=float)
        if hit_world.shape != (3,) or not np.all(np.isfinite(hit_world)):
            return None

        nozzle_world = hit_world + normal_world * self._interactive_spray_standoff
        spray_direction_world = -normal_world

        spray_direction_world = self._interactive_rotate_spray_direction(
            spray_direction_world, self._interactive_spray_angle_y_deg,
            self._interactive_spray_angle_x_deg,
        )
        return {
            "surface_point_world": hit_world,
            "surface_normal_world": normal_world,
            "nozzle_point_world": nozzle_world,
            "spray_direction_world": spray_direction_world,
            "cell_id": cell_id,
            "paintable": not surface_excluded,
            "timestamp": time.monotonic(),
        }

    @staticmethod
    def _interactive_pose_quaternion_from_direction(direction_product):
        """Return a stable XYZW quaternion for the nozzle tool frame."""
        z_axis = -np.asarray(direction_product, dtype=np.float64)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm <= 1e-12:
            raise ValueError("Cannot build nozzle orientation from a zero direction.")
        z_axis /= z_norm
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, z_axis))) > 0.92:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(reference, z_axis)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        trace = float(np.trace(rotation))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rotation[2, 1] - rotation[1, 2]) / s
            qy = (rotation[0, 2] - rotation[2, 0]) / s
            qz = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
        quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        return quaternion

    def _interactive_pose_to_product_frame(self, pose):
        inverse = np.linalg.inv(self._interactive_spray_transform)
        point_h = np.append(pose["nozzle_point_world"], 1.0)
        surface_h = np.append(pose["surface_point_world"], 1.0)
        direction_product = inverse[:3, :3] @ pose["spray_direction_world"]
        direction_product /= max(float(np.linalg.norm(direction_product)), 1e-12)
        normal_product = inverse[:3, :3] @ np.asarray(
            pose["surface_normal_world"], dtype=np.float64
        )
        normal_product /= max(float(np.linalg.norm(normal_product)), 1e-12)
        return {
            "nozzle_point": (inverse @ point_h)[:3],
            "surface_point": (inverse @ surface_h)[:3],
            "surface_normal": normal_product,
            "spray_direction": direction_product,
            "quaternion_xyzw": self._interactive_pose_quaternion_from_direction(
                direction_product
            ),
        }

    def _interactive_apply_paint_deposition(self, patch, pose, duration_s=None):
        """Accumulate geometry-aware deposition for one sampled spray interval."""
        # Defense in depth: collision must block the deposition calculation,
        # not merely suppress path recording in the cursor event handler.
        if (
            self._interactive_spray_gun_collision
            or self._interactive_spray_surface_excluded
            or not bool(pose.get("paintable", True))
        ):
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
                return
        if self.current_mesh is None:
            return
        ids = patch.cell_data.get("InteractiveOriginalCellId")
        if ids is None:
            return
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        excluded_mask = self._interactive_spray_excluded_face_mask
        if (
            excluded_mask is None
            or np.asarray(excluded_mask).shape != (len(self.current_mesh.faces),)
        ):
            return
        ids = ids[
            (ids >= 0)
            & (ids < len(excluded_mask))
        ]
        if not ids.size:
            return
        ids = np.unique(ids[~excluded_mask[ids]])
        if not ids.size:
            return

        timestamp = float(pose["timestamp"])
        nozzle_world = np.asarray(pose["nozzle_point_world"], dtype=np.float64)
        if self._interactive_spray_last_deposition_timestamp is None:
            self._interactive_spray_last_deposition_timestamp = timestamp
            self._interactive_spray_previous_nozzle_world = nozzle_world.copy()
            return
        raw_dt = timestamp - float(self._interactive_spray_last_deposition_timestamp)
        self._interactive_spray_last_deposition_timestamp = timestamp
        previous_nozzle = self._interactive_spray_previous_nozzle_world
        self._interactive_spray_previous_nozzle_world = nozzle_world.copy()
        if previous_nozzle is None:
            return
        if duration_s is None:
            if not np.isfinite(raw_dt) or raw_dt <= 0.0:
                return
            dt = raw_dt
        else:
            dt = float(duration_s)
            if not np.isfinite(dt) or dt <= 0.0:
                return
            # Keep the exact spatial-sample interval used by the replay clock.
        source_factor = self._get_source_mm_factor()
        if source_factor is None:
            source_factor = 1.0
        process = dict(DEFAULT_COATING_PROCESS)
        process.update(self._interactive_coating_process or {})
        try:
            flow = float(process["fluid_flow_ml_min"])
            efficiency = float(process["transfer_efficiency_percent"])
            solids = float(process["volume_solids_percent"])
            coats = int(process["number_of_coats"])
        except (KeyError, TypeError, ValueError):
            return
        if flow <= 0.0 or efficiency <= 0.0 or solids <= 0.0:
            return
        areas = np.asarray(self.current_mesh.area_faces, dtype=np.float64) * float(source_factor) ** 2
        local_faces = np.asarray(self.current_mesh.faces[ids], dtype=np.int64)
        vertices = np.asarray(self.current_mesh.vertices, dtype=np.float64)
        triangles_local = vertices[local_faces]
        centroids_local = np.mean(triangles_local, axis=1)
        transform = self._actor_matrix_array()
        centroids_world = centroids_local @ transform[:3, :3].T + transform[:3, 3]
        triangles_world = (
            triangles_local @ transform[:3, :3].T + transform[:3, 3]
        )
        triangles_world_mm = triangles_world * float(source_factor)
        normals_local = np.asarray(self.current_mesh.face_normals[ids], dtype=np.float64)

        # STEP deposition uses original B-Rep centroids, normals and surface
        # area Jacobians. Triangle vertices remain the quadrature support so
        # the existing, validated cone kernel and per-triangle state layout
        # are preserved. STL continues to use its native triangle geometry.
        from core.step_loader import (
            get_step_brep_context,
            get_step_triangle_surface_coordinates,
            evaluate_step_face_surface,
            evaluate_step_triangle_sample,
        )
        brep_context = get_step_brep_context(self.current_mesh)
        if brep_context is not None:
            self._interactive_apply_brep_uv_deposition(
                ids=ids, pose=pose, nozzle_world=nozzle_world,
                source_factor=float(source_factor), transform=transform,
                dt=dt,
                flow=flow, efficiency=efficiency, solids=solids,
                coats=coats, brep_context=brep_context,
            )
            self._interactive_spray_paint_state_dirty = True
            return
        if brep_context is not None:
            uv_triangles = np.asarray(
                brep_context["triangle_uvs"], dtype=np.float64
            )
            uv_valid = np.asarray(
                brep_context["triangle_uv_valid"], dtype=bool
            )
            surface_samples_by_triangle = {}
            for local_index, triangle_index in enumerate(ids):
                if not uv_valid[int(triangle_index)]:
                    continue
                center_sample = evaluate_step_triangle_sample(
                    self.current_mesh, int(triangle_index),
                    np.asarray([1.0 / 3.0] * 3, dtype=np.float64),
                )
                uv = uv_triangles[int(triangle_index)]
                surface_samples_by_triangle[int(triangle_index)] = center_sample
                uv_area = 0.5 * abs(
                    (uv[1, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1])
                    - (uv[1, 1] - uv[0, 1]) * (uv[2, 0] - uv[0, 0])
                )
                # Three-point degree-2 quadrature of the CAD surface
                # Jacobian over this source UV triangle.
                area_barycentric = (
                    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
                    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
                    (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
                )
                jacobians = []
                for area_weights in area_barycentric:
                    area_uv = get_step_triangle_surface_coordinates(
                        self.current_mesh, int(triangle_index),
                        np.asarray(area_weights, dtype=np.float64),
                    )
                    area_surface = evaluate_step_face_surface(
                        self.current_mesh,
                        int(area_uv["face_id"]),
                        float(area_uv["uv"][0]),
                        float(area_uv["uv"][1]),
                    )
                    jacobians.append(float(np.linalg.norm(np.cross(
                        area_surface["du_mm_per_u"],
                        area_surface["dv_mm_per_v"],
                    ))))
                exact_area_mm2 = uv_area * float(np.mean(jacobians))
                if np.isfinite(exact_area_mm2) and exact_area_mm2 > 1e-12:
                    areas[int(triangle_index)] = exact_area_mm2
                centroids_local[local_index] = center_sample["point_mm"]
                exact_normal = np.asarray(
                    center_sample["normal"], dtype=np.float64
                )
                if float(np.dot(exact_normal, normals_local[local_index])) < 0.0:
                    exact_normal *= -1.0
                normals_local[local_index] = exact_normal

        normals_world = normals_local @ transform[:3, :3].T
        normals_world /= np.maximum(np.linalg.norm(normals_world, axis=1)[:, None], 1e-12)
        centroids_world = centroids_local @ transform[:3, :3].T + transform[:3, 3]
        result = calculate_cone_deposition_step(
            surface_points_mm=centroids_world * float(source_factor),
            sample_vertices_mm=triangles_world_mm,
            surface_normals=normals_world,
            sample_area_mm2=areas[ids],
            nozzle_position_mm=nozzle_world * float(source_factor),
            spray_direction=pose["spray_direction_world"],
            cone_angle_deg=self._interactive_spray_cone_angle_deg,
            duration_s=dt,
            fluid_flow_ml_min=flow,
            transfer_efficiency_percent=efficiency,
            volume_solids_percent=solids,
            number_of_coats=coats,
            radial_alpha=float(
                self._interactive_sprayer_profile.get("deposition_beta", 2.0)
            ),
            radial_beta=float(
                self._interactive_sprayer_profile.get("deposition_beta", 2.0)
            ),
            core_deposition_fraction=float(
                self._interactive_sprayer_profile.get(
                    "core_deposition_fraction", 0.88
                )
            ),
            overspray_tail_fraction=float(
                self._interactive_sprayer_profile.get(
                    "overspray_tail_fraction", 0.12
                )
            ),
            overspray_tail_sigma_fraction=float(
                self._interactive_sprayer_profile.get(
                    "overspray_tail_sigma_fraction", 0.35
                )
            ),
        )
        deposited = np.asarray(result.get("deposited_volume_mm3", np.zeros(len(ids))), dtype=np.float64)
        if deposited.shape != (len(ids),):
            return
        if self._interactive_spray_face_thickness_um is None:
            self._interactive_spray_face_thickness_um = np.zeros(len(self.current_mesh.faces), dtype=np.float64)
        thickness_increment_um = (
            deposited / np.maximum(areas[ids], 1e-12) * 1000.0
        )
        self._interactive_spray_face_thickness_um[ids] += thickness_increment_um

        # Accumulate a continuous, per-Face UV thickness field. The existing
        # triangle deposition values are spatially interpolated into the
        # texture grid; triangle IDs remain only as diagnostic provenance.
        if brep_context is not None:
            from core.uv_coating_texture import (
                DEFAULT_UV_TEXTURE_RESOLUTION,
                accumulate_face_uv_thickness,
                face_uv_layout,
                surface_metric_to_pixel_covariance,
            )
            if not isinstance(self._interactive_spray_surface_field, dict):
                self._interactive_spray_surface_field = {
                    "schema": "STEP_BREP_UV_FILM_THICKNESS_FIELD_V2",
                    "coordinate_frame": "ORIGINAL_STEP_FACE_UV",
                    "field_representation": "CONTINUOUS_UV_TEXTURE_GRID",
                    "interpolation_method": "CAD_SURFACE_METRIC_AWARE_GAUSSIAN",
                    "texture_resolution": DEFAULT_UV_TEXTURE_RESOLUTION,
                    "faces": {},
                }
            triangle_face_ids = np.asarray(
                brep_context["triangle_face_ids"], dtype=np.int64
            )
            face_samples = self._interactive_spray_surface_field["faces"]
            uv_by_face = {}
            increment_by_face = {}
            metric_by_face = {}
            area_by_face = {}
            provenance_by_face = {}
            for local_index, triangle_index in enumerate(ids):
                triangle_index = int(triangle_index)
                if not bool(brep_context["triangle_uv_valid"][triangle_index]):
                    continue
                sample_record = surface_samples_by_triangle.get(triangle_index)
                if sample_record is None:
                    uv_mapping = get_step_triangle_surface_coordinates(
                        self.current_mesh, triangle_index,
                        np.asarray([1.0 / 3.0] * 3, dtype=np.float64),
                    )
                    sample_uv = np.asarray(uv_mapping["uv"], dtype=np.float64)
                    exact_surface = evaluate_step_face_surface(
                        self.current_mesh,
                        int(triangle_face_ids[triangle_index]),
                        float(sample_uv[0]),
                        float(sample_uv[1]),
                    )
                else:
                    sample_uv = np.asarray(sample_record["uv"], dtype=np.float64)
                    exact_surface = sample_record
                face_id = int(triangle_face_ids[triangle_index])
                du = np.asarray(exact_surface["du_mm_per_u"], dtype=np.float64)
                dv = np.asarray(exact_surface["dv_mm_per_v"], dtype=np.float64)
                surface_metric = np.asarray([
                    [float(np.dot(du, du)), float(np.dot(du, dv))],
                    [float(np.dot(du, dv)), float(np.dot(dv, dv))],
                ], dtype=np.float64)
                uv_by_face.setdefault(face_id, []).append(sample_uv)
                increment_by_face.setdefault(face_id, []).append(
                    float(thickness_increment_um[local_index])
                )
                metric_by_face.setdefault(face_id, []).append(surface_metric)
                area_by_face.setdefault(face_id, []).append(
                    float(areas[triangle_index])
                )
                provenance_by_face.setdefault(face_id, []).append({
                    "triangle_id": triangle_index,
                    "uv": np.asarray(sample_uv, dtype=float).tolist(),
                    "thickness_um": float(
                        self._interactive_spray_face_thickness_um[triangle_index]
                    ),
                    "area_mm2": float(areas[triangle_index]),
                    "patch_id": (
                        int(self.patch_labels[triangle_index])
                        if self.patch_labels is not None
                        and len(self.patch_labels) == len(self.current_mesh.faces)
                        else None
                    ),
                    "stage2_status": (
                        int(self.face_status_labels[triangle_index])
                        if self.face_status_labels is not None
                        and len(self.face_status_labels) == len(self.current_mesh.faces)
                        else None
                    ),
                })

            for face_id, uv_samples in uv_by_face.items():
                texture_layout = self._interactive_spray_texture_layout.get(face_id)
                if texture_layout is None:
                    face_triangle_ids = np.flatnonzero(
                        (triangle_face_ids == face_id)
                        & np.asarray(brep_context["triangle_uv_valid"], dtype=bool)
                    )
                    layout = face_uv_layout(
                        brep_context["face_records"][face_id - 1],
                        np.asarray(brep_context["triangle_uvs"], dtype=float)[face_triangle_ids],
                    )
                    texture_layout = {
                        **layout,
                        "resolution": int(DEFAULT_UV_TEXTURE_RESOLUTION),
                        "triangle_count": int(len(face_triangle_ids)),
                    }
                    self._interactive_spray_texture_layout[face_id] = texture_layout
                face_record = face_samples.setdefault(face_id, {
                    "uv_bounds": [
                        list(map(float, interval))
                        for interval in texture_layout["uv_bounds"]
                    ],
                    "surface_type": texture_layout["surface_type"],
                    "periodic_u": bool(texture_layout["periodic_u"]),
                    "periodic_v": bool(texture_layout["periodic_v"]),
                    "texture_resolution": int(texture_layout["resolution"]),
                    "interpolation_method": "CAD_SURFACE_METRIC_AWARE_GAUSSIAN",
                    "thickness_grid_um": np.zeros(
                        (
                            int(texture_layout["resolution"]),
                            int(texture_layout["resolution"]),
                        ),
                        dtype=np.float32,
                    ),
                    "samples_by_triangle": {},
                })
                sample_count = max(int(texture_layout["triangle_count"]), 1)
                sigma_px = max(
                    1.5,
                    min(
                        8.0,
                        0.65 * int(texture_layout["resolution"])
                        / math.sqrt(float(sample_count)),
                    ),
                )
                uv_bounds_array = np.asarray(
                    texture_layout["uv_bounds"], dtype=np.float64
                )
                covariance_samples = []
                for metric, sample_area in zip(
                    metric_by_face[face_id], area_by_face[face_id]
                ):
                    try:
                        covariance_samples.append(
                            surface_metric_to_pixel_covariance(
                                metric,
                                uv_bounds_array,
                                int(texture_layout["resolution"]),
                                sample_area,
                                smoothing_factor=1.05,
                            )
                        )
                    except ValueError:
                        # Surface singularities (for example sphere poles)
                        # use a bounded isotropic fallback rather than
                        # dropping paint deposition at that sample.
                        covariance_samples.append(
                            np.eye(2, dtype=np.float64) * sigma_px * sigma_px
                        )
                accumulate_face_uv_thickness(
                    face_record["thickness_grid_um"],
                    np.asarray(uv_samples, dtype=np.float64),
                    np.asarray(increment_by_face[face_id], dtype=np.float64),
                    uv_bounds_array,
                    sigma_px=sigma_px,
                    periodic_u=bool(texture_layout["periodic_u"]),
                    periodic_v=bool(texture_layout["periodic_v"]),
                    sample_covariance_px=np.asarray(
                        covariance_samples, dtype=np.float64
                    ),
                )
                for provenance in provenance_by_face[face_id]:
                    face_record["samples_by_triangle"][
                        provenance["triangle_id"]
                    ] = provenance
                self._interactive_spray_dirty_face_ids.add(int(face_id))

        # Re-evaluate STEP deposition directly on the original B-Rep UV grid.
        # The mesh remains the visibility/trim support, but it no longer defines
        # the spatial samples used by the persistent STEP coating texture.
        if brep_context is not None:
            from core.uv_coating_texture import rasterize_face_uv_grid

            direct_batches = []
            direct_face_meta = []
            selected_face_ids = np.unique(triangle_face_ids[ids])
            for face_id_value in selected_face_ids:
                face_id = int(face_id_value)
                selected_triangles = ids[
                    (triangle_face_ids[ids] == face_id)
                    & uv_valid[ids]
                ]
                if selected_triangles.size == 0:
                    continue
                layout_record = self._interactive_spray_texture_layout.get(face_id)
                if layout_record is None:
                    all_face_triangles = np.flatnonzero(
                        (triangle_face_ids == face_id) & uv_valid
                    )
                    face_layout = face_uv_layout(
                        brep_context["face_records"][face_id - 1],
                        uv_triangles[all_face_triangles],
                    )
                    layout_record = {
                        **face_layout,
                        "resolution": int(DEFAULT_UV_TEXTURE_RESOLUTION),
                        "triangle_count": int(len(all_face_triangles)),
                    }
                    self._interactive_spray_texture_layout[face_id] = layout_record

                resolution = int(layout_record["resolution"])
                selected_uv = uv_triangles[selected_triangles]
                raster = rasterize_face_uv_grid(
                    selected_uv,
                    np.asarray(layout_record["uv_bounds"], dtype=np.float64),
                    resolution=resolution,
                    periodic_u=bool(layout_record["periodic_u"]),
                    periodic_v=bool(layout_record["periodic_v"]),
                )
                valid_rows, valid_cols = np.nonzero(raster["valid_mask"])
                if not len(valid_rows):
                    continue
                uv_values = raster["uv_grid"][valid_rows, valid_cols]
                face_points = []
                face_normals = []
                face_areas = []
                accepted_rows = []
                accepted_cols = []
                bounds_uv = np.asarray(layout_record["uv_bounds"], dtype=np.float64)
                du_step = (bounds_uv[0, 1] - bounds_uv[0, 0]) / max(resolution - 1, 1)
                dv_step = (bounds_uv[1, 1] - bounds_uv[1, 0]) / max(resolution - 1, 1)
                edge_u = np.ones(len(valid_cols), dtype=np.float64)
                edge_v = np.ones(len(valid_rows), dtype=np.float64)
                edge_u[(valid_cols == 0) | (valid_cols == resolution - 1)] = 0.5
                edge_v[(valid_rows == 0) | (valid_rows == resolution - 1)] = 0.5
                quadrature_area_weights = edge_u * edge_v
                for sample_index, (uv_value, area_weight) in enumerate(
                    zip(uv_values, quadrature_area_weights)
                ):
                    try:
                        surface = evaluate_step_face_surface(
                            self.current_mesh, face_id,
                            float(uv_value[0]), float(uv_value[1]),
                        )
                    except (RuntimeError, ValueError):
                        continue
                    point = np.asarray(surface["point_mm"], dtype=np.float64)
                    normal = np.asarray(surface["normal"], dtype=np.float64)
                    du = np.asarray(surface["du_mm_per_u"], dtype=np.float64)
                    dv = np.asarray(surface["dv_mm_per_v"], dtype=np.float64)
                    jacobian = float(np.linalg.norm(np.cross(du, dv)))
                    sample_area = jacobian * du_step * dv_step * float(area_weight)
                    if (
                        point.shape != (3,) or normal.shape != (3,)
                        or not np.isfinite(point).all()
                        or not np.isfinite(normal).all()
                        or not math.isfinite(sample_area) or sample_area <= 1e-12
                    ):
                        continue
                    face_points.append(point)
                    face_normals.append(normal)
                    face_areas.append(sample_area * float(source_factor) ** 2)
                    accepted_rows.append(int(valid_rows[sample_index]))
                    accepted_cols.append(int(valid_cols[sample_index]))

                if not face_points:
                    continue
                point_array = np.asarray(face_points, dtype=np.float64)
                normal_array = np.asarray(face_normals, dtype=np.float64)
                area_array = np.asarray(face_areas, dtype=np.float64)
                point_world = (
                    point_array @ transform[:3, :3].T + transform[:3, 3]
                ) * float(source_factor)
                normal_world = normal_array @ transform[:3, :3].T
                normal_world /= np.maximum(
                    np.linalg.norm(normal_world, axis=1)[:, None], 1e-12
                )
                direct_batches.append((point_world, normal_world, area_array))
                direct_face_meta.append({
                    "face_id": face_id,
                    "rows": np.asarray(accepted_rows, dtype=np.int64),
                    "cols": np.asarray(accepted_cols, dtype=np.int64),
                    "uv": uv_values,
                    "layout": layout_record,
                    "sample_count": len(face_points),
                })

            if direct_batches:
                direct_points = np.concatenate([item[0] for item in direct_batches], axis=0)
                direct_normals = np.concatenate([item[1] for item in direct_batches], axis=0)
                direct_areas = np.concatenate([item[2] for item in direct_batches], axis=0)
                direct_result = calculate_cone_deposition_step(
                    surface_points_mm=direct_points,
                    surface_normals=direct_normals,
                    sample_area_mm2=direct_areas,
                    nozzle_position_mm=nozzle_world * float(source_factor),
                    spray_direction=pose["spray_direction_world"],
                    cone_angle_deg=self._interactive_spray_cone_angle_deg,
                    duration_s=dt,
                    fluid_flow_ml_min=flow,
                    transfer_efficiency_percent=efficiency,
                    volume_solids_percent=solids,
                    number_of_coats=coats,
                            radial_alpha=float(
                        self._interactive_sprayer_profile.get("deposition_beta", 2.0)
                    ),
                    radial_beta=float(
                        self._interactive_sprayer_profile.get("deposition_beta", 2.0)
                    ),
                    core_deposition_fraction=float(
                        self._interactive_sprayer_profile.get("core_deposition_fraction", 0.88)
                    ),
                    overspray_tail_fraction=float(
                        self._interactive_sprayer_profile.get("overspray_tail_fraction", 0.12)
                    ),
                    overspray_tail_sigma_fraction=float(
                        self._interactive_sprayer_profile.get("overspray_tail_sigma_fraction", 0.35)
                    ),
                )
                direct_deposited = np.asarray(
                    direct_result.get("deposited_volume_mm3", []), dtype=np.float64
                )
                if direct_deposited.shape == (len(direct_areas),):
                    direct_increment = (
                        direct_deposited / np.maximum(direct_areas, 1e-12) * 1000.0
                    )
                    offset = 0
                    for meta in direct_face_meta:
                        count = int(meta["sample_count"])
                        face_increment = direct_increment[offset:offset + count]
                        offset += count
                        face_id = int(meta["face_id"])
                        face_record = face_samples.get(face_id)
                        if face_record is None:
                            continue
                        resolution = int(meta["layout"]["resolution"])
                        direct_grid = np.zeros((resolution, resolution), dtype=np.float32)
                        # Evaluation failures can remove individual UV samples;
                        # retain their exact pixel indices through a matching
                        # validity pass before writing the computed increments.
                        rows = meta["rows"]
                        cols = meta["cols"]
                        if len(face_increment) != len(rows):
                            # A failed CAD evaluation makes this Face update
                            # incomplete; do not misalign UV pixels and values.
                            continue
                        direct_grid[rows, cols] += face_increment.astype(np.float32)
                        face_record["thickness_grid_um"] = direct_grid
                        face_record["interpolation_method"] = "DIRECT_BREP_UV_TEXEL_DEPOSITION"
                        face_record["direct_deposition_sample_count"] = int(len(face_increment))
                        face_record["direct_deposition_volume_mm3"] = float(
                            np.sum(direct_deposited[offset - count:offset])
                        )
                        self._interactive_spray_dirty_face_ids.add(face_id)
                self._interactive_spray_surface_field["field_representation"] = (
                    "DIRECT_ORIGINAL_BREP_UV_TEXEL_FIELD"
                )
                self._interactive_spray_surface_field["interpolation_method"] = (
                    "DIRECT_BREP_UV_TEXEL_DEPOSITION"
                )

        self._interactive_spray_paint_state_dirty = True

    def _interactive_apply_brep_uv_deposition(
        self, *, ids, pose, nozzle_world, source_factor, transform, dt,
        flow, efficiency, solids, coats, brep_context,
    ):
        """Apply sparse triangle-integrated spray physics and bake it to CAD Face UV textures.

        CAD tessellation is used only as a precomputed quadrature/support mesh.
        No OpenCascade Surface evaluation, UV rasterization, or full-face scan
        occurs in the interactive deposition loop.
        """
        from core.uv_coating_texture import (
            DEFAULT_UV_TEXTURE_RESOLUTION,
            accumulate_face_uv_thickness,
            face_uv_layout,
            face_uv_triangles_to_texture_coordinates,
            surface_metric_to_pixel_covariance,
        )

        mesh = self.current_mesh
        face_ids = np.asarray(brep_context["triangle_face_ids"], dtype=np.int64)
        triangle_uvs = np.asarray(brep_context["triangle_uvs"], dtype=np.float64)
        uv_valid = np.asarray(brep_context["triangle_uv_valid"], dtype=bool)
        excluded_mask = np.asarray(
            self._interactive_spray_excluded_face_mask,
            dtype=bool,
        )
        if excluded_mask.shape != (len(mesh.faces),):
            return
        selected = np.unique(np.asarray(ids, dtype=np.int64))
        selected = selected[(selected >= 0) & (selected < len(mesh.faces))]
        # The ray footprint can intersect multiple triangles. Never allow
        # Stage 2 non-paintable triangles into the deposition kernel.
        selected = selected[~excluded_mask[selected]]
        selected = selected[uv_valid[selected]]
        if not selected.size:
            return

        # Build inexpensive, tessellation-derived surface samples once per mesh.
        # The old path evaluated OpenCascade at hundreds/thousands of UV texels
        # during every mouse update; this cache eliminates those calls entirely.
        cache = getattr(self, "_interactive_brep_surface_sample_cache", None)
        cache_key = id(mesh)
        cache_matches = (
            isinstance(cache, dict)
            and cache.get("mesh_id") == cache_key
            and np.array_equal(cache.get("excluded_mask"), excluded_mask)
        )
        if not cache_matches:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            triangles = np.asarray(mesh.faces, dtype=np.int64)
            triangle_points = vertices[triangles]
            sample_count = len(triangles)
            uv_centers = np.full((sample_count, 2), np.nan, dtype=np.float64)
            metrics = np.full((sample_count, 2, 2), np.nan, dtype=np.float64)
            face_layouts = {}
            from core.uv_coating_texture import rasterize_face_uv_grid
            for face_value in np.unique(face_ids[uv_valid]):
                face_id = int(face_value)
                face_tri_ids = np.flatnonzero((face_ids == face_id) & uv_valid)
                if not len(face_tri_ids):
                    continue
                layout = face_uv_layout(
                    brep_context["face_records"][face_id - 1],
                    triangle_uvs[face_tri_ids],
                )
                paintable_tri_ids = face_tri_ids[~excluded_mask[face_tri_ids]]
                if len(paintable_tri_ids):
                    paintable_uv_mask = rasterize_face_uv_grid(
                        triangle_uvs[paintable_tri_ids],
                        layout["uv_bounds"],
                        resolution=int(DEFAULT_UV_TEXTURE_RESOLUTION),
                        periodic_u=layout["periodic_u"],
                        periodic_v=layout["periodic_v"],
                    )["valid_mask"]
                else:
                    paintable_uv_mask = np.zeros(
                        (int(DEFAULT_UV_TEXTURE_RESOLUTION), int(DEFAULT_UV_TEXTURE_RESOLUTION)),
                        dtype=bool,
                    )
                bounds = np.asarray(layout["uv_bounds"], dtype=np.float64)
                normalized = face_uv_triangles_to_texture_coordinates(
                    triangle_uvs[face_tri_ids],
                    bounds,
                    periodic_u=layout["periodic_u"],
                    periodic_v=layout["periodic_v"],
                )
                unwrapped_uv = (
                    bounds[:, 0][None, None, :]
                    + normalized * (bounds[:, 1] - bounds[:, 0])[None, None, :]
                )
                uv_centers[face_tri_ids] = (
                    bounds[:, 0][None, :]
                    + np.mean(normalized, axis=1)
                    * (bounds[:, 1] - bounds[:, 0])[None, :]
                )
                xyz = triangle_points[face_tri_ids]
                delta_uv = unwrapped_uv[:, 1:, :] - unwrapped_uv[:, :1, :]
                delta_xyz = xyz[:, 1:, :] - xyz[:, :1, :]
                determinants = np.linalg.det(delta_uv)
                nonsingular = np.isfinite(determinants) & (np.abs(determinants) > 1e-12)
                if np.any(nonsingular):
                    try:
                        jacobians = np.linalg.solve(
                            delta_uv[nonsingular], delta_xyz[nonsingular]
                        )
                        metric_batch = jacobians @ np.swapaxes(jacobians, 1, 2)
                        metric_valid = (
                            np.isfinite(metric_batch).all(axis=(1, 2))
                            & (np.linalg.det(metric_batch) > 1e-14)
                        )
                        valid_triangle_ids = face_tri_ids[nonsingular][metric_valid]
                        metrics[valid_triangle_ids] = 0.5 * (
                            metric_batch[metric_valid]
                            + np.swapaxes(metric_batch[metric_valid], 1, 2)
                        )
                    except np.linalg.LinAlgError:
                        pass
                face_layouts[face_id] = {
                    **layout,
                    "resolution": int(DEFAULT_UV_TEXTURE_RESOLUTION),
                    "triangle_count": int(len(face_tri_ids)),
                    "triangle_ids": face_tri_ids,
                    "paintable_uv_mask": paintable_uv_mask,
                }
            cache = {
                "mesh_id": cache_key,
                "excluded_mask": excluded_mask.copy(),
                "points_local": np.mean(triangle_points, axis=1),
                "vertices_local": triangle_points,
                "normals_local": np.asarray(mesh.face_normals, dtype=np.float64).copy(),
                "areas_source": np.asarray(mesh.area_faces, dtype=np.float64).copy(),
                "face_ids": face_ids.copy(),
                "uv_valid": uv_valid.copy(),
                "uv_centers": uv_centers,
                "surface_metrics": metrics,
                "face_layouts": face_layouts,
            }
            self._interactive_brep_surface_sample_cache = cache
            self._interactive_spray_texture_layout.update(face_layouts)

        if not isinstance(self._interactive_spray_surface_field, dict):
            self._interactive_spray_surface_field = {
                "schema": "STEP_BREP_UV_FILM_THICKNESS_FIELD_V4",
                "coordinate_frame": "ORIGINAL_STEP_FACE_UV",
                "field_representation": "SPARSE_TRIANGLE_INTEGRATION_BAKED_TO_FACE_UV",
                "interpolation_method": "PRECOMPUTED_SURFACE_SAMPLE_GAUSSIAN_BAKE",
                "texture_resolution": int(DEFAULT_UV_TEXTURE_RESOLUTION),
                "faces": {},
            }
        fields = self._interactive_spray_surface_field["faces"]

        source_to_mm = float(source_factor)
        rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        translation = np.asarray(transform[:3, 3], dtype=np.float64)
        points_world_mm = (
            cache["points_local"][selected] @ rotation.T + translation
        ) * source_to_mm
        vertices_world_mm = (
            cache["vertices_local"][selected] @ rotation.T + translation
        ) * source_to_mm
        normals_world = cache["normals_local"][selected] @ rotation.T
        normals_world /= np.maximum(
            np.linalg.norm(normals_world, axis=1)[:, None], 1e-12
        )
        areas_mm2 = cache["areas_source"][selected] * source_to_mm ** 2
        valid_area = np.isfinite(areas_mm2) & (areas_mm2 > 1e-12)
        if not np.all(valid_area):
            selected = selected[valid_area]
            points_world_mm = points_world_mm[valid_area]
            vertices_world_mm = vertices_world_mm[valid_area]
            normals_world = normals_world[valid_area]
            areas_mm2 = areas_mm2[valid_area]
        if not len(selected):
            return

        # Expand sparse ray-hit triangles to the complete front-facing cone
        # portion of every CAD Face actually touched by the footprint. This is critical
        # for narrow faces: a handful of preview rays must not create isolated
        # painted dots on otherwise uncoated Face regions.
        hit_face_ids = np.unique(cache["face_ids"][selected])
        candidate_ids = np.flatnonzero(
            np.isin(cache["face_ids"], hit_face_ids)
            & cache["uv_valid"]
            & ~excluded_mask
        )
        if len(candidate_ids):
            candidate_points_mm = (
                cache["points_local"][candidate_ids] @ rotation.T + translation
            ) * source_to_mm
            candidate_normals = cache["normals_local"][candidate_ids] @ rotation.T
            candidate_normals /= np.maximum(
                np.linalg.norm(candidate_normals, axis=1)[:, None], 1e-12
            )
            nozzle_mm = np.asarray(nozzle_world, dtype=np.float64) * source_to_mm
            spray_direction = np.asarray(
                pose["spray_direction_world"], dtype=np.float64
            ).copy()
            spray_direction /= max(float(np.linalg.norm(spray_direction)), 1e-12)
            delta = candidate_points_mm - nozzle_mm[None, :]
            axial = delta @ spray_direction
            radial_squared = np.maximum(
                np.einsum("ij,ij->i", delta, delta) - axial * axial, 0.0
            )
            cone_radius = np.maximum(axial, 0.0) * math.tan(
                math.radians(self._interactive_spray_cone_angle_deg * 0.5)
            )
            facing = candidate_normals @ (-spray_direction)
            inside = (
                (axial > 1e-6)
                & (radial_squared <= (cone_radius * 1.08) ** 2)
                & (facing > 0.02)
            )
            expanded = candidate_ids[inside]
            if len(expanded):
                selected = np.unique(np.concatenate((selected, expanded)))

                # Rebuild every deposition input after expanding the selected
                # triangle set. The previous implementation expanded only
                # "selected" while leaving the point/normal/area arrays at
                # their original ray-hit length. The cone kernel then returned
                # a mismatched result which was discarded, so no paint was
                # accumulated on STEP faces.
                points_world_mm = (
                    cache["points_local"][selected] @ rotation.T + translation
                ) * source_to_mm
                vertices_world_mm = (
                    cache["vertices_local"][selected] @ rotation.T + translation
                ) * source_to_mm
                normals_world = cache["normals_local"][selected] @ rotation.T
                normals_world /= np.maximum(
                    np.linalg.norm(normals_world, axis=1)[:, None], 1e-12
                )
                areas_mm2 = cache["areas_source"][selected] * source_to_mm ** 2
                valid_expanded = (
                    np.isfinite(areas_mm2)
                    & (areas_mm2 > 1e-12)
                    & np.isfinite(points_world_mm).all(axis=1)
                    & np.isfinite(vertices_world_mm).all(axis=(1, 2))
                    & np.isfinite(normals_world).all(axis=1)
                )
                selected = selected[valid_expanded]
                points_world_mm = points_world_mm[valid_expanded]
                vertices_world_mm = vertices_world_mm[valid_expanded]
                normals_world = normals_world[valid_expanded]
                areas_mm2 = areas_mm2[valid_expanded]
                if not len(selected):
                    return

        result = calculate_cone_deposition_step(
            surface_points_mm=points_world_mm,
            sample_vertices_mm=vertices_world_mm,
            surface_normals=normals_world,
            sample_area_mm2=areas_mm2,
            nozzle_position_mm=np.asarray(nozzle_world, dtype=np.float64) * source_to_mm,
            spray_direction=pose["spray_direction_world"],
            cone_angle_deg=self._interactive_spray_cone_angle_deg,
            duration_s=dt,
            fluid_flow_ml_min=flow,
            transfer_efficiency_percent=efficiency,
            volume_solids_percent=solids,
            number_of_coats=coats,
            radial_alpha=float(self._interactive_sprayer_profile.get("deposition_beta", 2.0)),
            radial_beta=float(self._interactive_sprayer_profile.get("deposition_beta", 2.0)),
            core_deposition_fraction=float(self._interactive_sprayer_profile.get("core_deposition_fraction", 0.88)),
            overspray_tail_fraction=float(self._interactive_sprayer_profile.get("overspray_tail_fraction", 0.12)),
            overspray_tail_sigma_fraction=float(self._interactive_sprayer_profile.get("overspray_tail_sigma_fraction", 0.35)),
        )
        deposited = np.asarray(
            result.get("deposited_volume_mm3", np.zeros(len(selected))),
            dtype=np.float64,
        )
        if deposited.shape != (len(selected),):
            return
        thickness_increment = deposited / np.maximum(areas_mm2, 1e-12) * 1000.0
        if self._interactive_spray_face_thickness_um is None:
            self._interactive_spray_face_thickness_um = np.zeros(
                len(mesh.faces), dtype=np.float64
            )
        self._interactive_spray_face_thickness_um[selected] += thickness_increment

        # Bake only the affected samples into their owning original CAD Face.
        for face_value in np.unique(cache["face_ids"][selected]):
            face_id = int(face_value)
            local_mask = cache["face_ids"][selected] == face_id
            triangle_ids = selected[local_mask]
            face_layout = cache["face_layouts"].get(face_id)
            if face_layout is None:
                continue
            face_record = fields.setdefault(face_id, {
                "uv_bounds": [
                    list(map(float, interval))
                    for interval in face_layout["uv_bounds"]
                ],
                "surface_type": face_layout["surface_type"],
                "periodic_u": bool(face_layout["periodic_u"]),
                "periodic_v": bool(face_layout["periodic_v"]),
                "texture_resolution": int(face_layout["resolution"]),
                "interpolation_method": "PRECOMPUTED_SURFACE_SAMPLE_GAUSSIAN_BAKE",
                "thickness_grid_um": np.zeros(
                    (int(face_layout["resolution"]), int(face_layout["resolution"])),
                    dtype=np.float32,
                ),
                "samples_by_triangle": {},
            })
            uv_samples = cache["uv_centers"][triangle_ids]
            increments = thickness_increment[local_mask]
            sample_areas = areas_mm2[local_mask]
            metrics = cache["surface_metrics"][triangle_ids]
            covariances = []
            bounds = np.asarray(face_layout["uv_bounds"], dtype=np.float64)
            sigma_fallback = max(
                1.5,
                min(
                    8.0,
                    0.65 * int(face_layout["resolution"])
                    / math.sqrt(max(float(face_layout["triangle_count"]), 1.0)),
                ),
            )

            # The paint footprint is a property of the spray plume, not of
            # the CAD tessellation density. Using triangle area here creates
            # one small Gaussian spot per triangle (the visible "droplet"
            # artifact). Derive the physical Gaussian width from the local
            # nozzle-to-surface cone radius instead, then map it through the
            # exact local CAD surface metric into anisotropic UV pixels.
            face_points_mm = points_world_mm[local_mask]
            nozzle_mm = np.asarray(nozzle_world, dtype=np.float64) * source_to_mm
            spray_axis = np.asarray(
                pose["spray_direction_world"], dtype=np.float64
            )
            spray_axis /= max(float(np.linalg.norm(spray_axis)), 1e-12)
            axial_mm = np.maximum(
                (face_points_mm - nozzle_mm[None, :]) @ spray_axis, 0.0
            )
            cone_radius_mm = axial_mm * math.tan(
                math.radians(self._interactive_spray_cone_angle_deg * 0.5)
            )
            # A broad, overlapping footprint gives a continuous wet film;
            # the 32 px covariance cap prevents a single sample washing out
            # an entire small CAD Face.
            plume_sigma_mm = np.maximum(0.45, cone_radius_mm * 0.18)
            equivalent_sample_areas = (plume_sigma_mm / 0.60) ** 2

            for metric, effective_area in zip(metrics, equivalent_sample_areas):
                try:
                    if not np.isfinite(metric).all():
                        raise ValueError("Degenerate tessellation UV metric.")
                    covariance = surface_metric_to_pixel_covariance(
                        metric * (source_to_mm ** 2),
                        bounds,
                        int(face_layout["resolution"]),
                        float(effective_area),
                        smoothing_factor=0.60,
                        max_sigma_px=32.0,
                    )
                except ValueError:
                    covariance = np.eye(2, dtype=np.float64) * sigma_fallback ** 2
                covariances.append(covariance)

            accumulate_face_uv_thickness(
                face_record["thickness_grid_um"],
                uv_samples,
                increments,
                bounds,
                sigma_px=sigma_fallback,
                periodic_u=bool(face_layout["periodic_u"]),
                periodic_v=bool(face_layout["periodic_v"]),
                sample_covariance_px=np.asarray(covariances, dtype=np.float64),
            )
            # Gaussian splats may cross a Stage 2 boundary on the same CAD
            # B-Rep Face. Clip the texture back to the exact paintable UV
            # domain so excluded triangles cannot receive visible coating.
            paintable_uv_mask = face_layout.get("paintable_uv_mask")
            if paintable_uv_mask is not None:
                face_record["thickness_grid_um"][~paintable_uv_mask] = 0.0
            face_record["direct_deposition_sample_count"] = int(len(triangle_ids))
            face_record["direct_deposition_volume_mm3"] = float(
                np.sum(deposited[local_mask])
            )
            for triangle_id, uv, increment, area in zip(
                triangle_ids, uv_samples, increments, sample_areas
            ):
                face_record["samples_by_triangle"][int(triangle_id)] = {
                    "triangle_id": int(triangle_id),
                    "uv": np.asarray(uv, dtype=np.float64).tolist(),
                    "thickness_um": float(
                        self._interactive_spray_face_thickness_um[int(triangle_id)]
                    ),
                    "area_mm2": float(area),
                    "patch_id": (
                        int(self.patch_labels[int(triangle_id)])
                        if self.patch_labels is not None
                        and len(self.patch_labels) == len(mesh.faces)
                        else None
                    ),
                    "stage2_status": (
                        int(self.face_status_labels[int(triangle_id)])
                        if self.face_status_labels is not None
                        and len(self.face_status_labels) == len(mesh.faces)
                        else None
                    ),
                }
            self._interactive_spray_dirty_face_ids.add(face_id)

        self._interactive_spray_surface_field["field_representation"] = (
            "SPARSE_TRIANGLE_INTEGRATION_BAKED_TO_FACE_UV"
        )
        self._interactive_spray_surface_field["interpolation_method"] = (
            "PRECOMPUTED_SURFACE_SAMPLE_GAUSSIAN_BAKE"
        )

    def _interactive_build_gun_visual(self, nozzle, spray_direction):
        """Build a lightweight sprayer envelope from the active profile."""
        profile = self._interactive_sprayer_profile or {}
        mm_factor = self._get_source_mm_factor() or 1.0

        def positive(key, default):
            try:
                value = float(profile.get(key, default))
            except (TypeError, ValueError):
                value = float(default)
            return max(value, 0.0)

        body_length = positive("gun_body_length_mm", 220.0) / mm_factor
        body_diameter = positive("gun_body_diameter_mm", 45.0) / mm_factor
        nozzle_length = positive("nozzle_length_mm", 25.0) / mm_factor

        nozzle_diameter = max(
            positive("nozzle_orifice_mm", 1.4) * 2.0 / mm_factor,
            body_diameter * 0.18,
        )

        axis = -np.asarray(spray_direction, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        body_center = nozzle + axis * (nozzle_length + 0.5 * body_length)
        nozzle_center = nozzle + axis * (0.5 * nozzle_length)

        parts = []
        if nozzle_length > 1e-6:
            parts.append(
                pv.Cylinder(
                    center=nozzle_center,
                    direction=axis,
                    height=nozzle_length,
                    radius=max(nozzle_diameter * 0.5, 0.2),
                    resolution=16,
                )
            )
        if body_length > 1e-6 and body_diameter > 1e-6:
            parts.append(
                pv.Cylinder(
                    center=body_center,
                    direction=axis,
                    height=body_length,
                    radius=max(body_diameter * 0.5, 0.5),
                    resolution=24,
                )
            )
        if not parts:
            return [], pv.PolyData()

        gun_polydata = pv.merge(parts, merge_points=True)
        actors = []
        for part in parts:
            actor = self.plotter.add_mesh(
                part,
                color="#B9C4CC",
                opacity=0.88,
                smooth_shading=True,
                lighting=True,
                pickable=False,
            )
            actors.append(actor)
        return actors, gun_polydata

    @staticmethod
    def _interactive_matrix4x4(matrix):
        values = np.asarray(matrix, dtype=np.float64)
        vtk_matrix = vtkMatrix4x4()
        for row in range(4):
            for col in range(4):
                vtk_matrix.SetElement(row, col, float(values[row, col]))
        return vtk_matrix

    def _interactive_check_gun_collision(self, gun_polydata):
        """Check the configured sprayer envelope against the product surface."""
        if gun_polydata is None or gun_polydata.n_cells == 0:
            return False
        if self._interactive_spray_surface is None:
            return False
        try:
            profile = self._interactive_sprayer_profile or {}
            try:
                clearance_mm = max(
                    float(profile.get("gun_collision_clearance_mm", 2.0)),
                    0.0,
                )
            except (TypeError, ValueError):
                clearance_mm = 2.0
            clearance_source_units = clearance_mm / max(
                float(self._get_source_mm_factor() or 1.0), 1e-12
            )

            collision = vtkCollisionDetectionFilter()
            collision.SetInputData(0, self._interactive_spray_surface)
            collision.SetInputData(1, gun_polydata)
            collision.SetMatrix(
                0,
                self._interactive_matrix4x4(self._interactive_spray_transform),
            )
            collision.SetMatrix(1, self._interactive_matrix4x4(np.eye(4)))
            collision.SetCollisionModeToFirstContact()
            collision.SetBoxTolerance(max(clearance_source_units, 1e-4))
            collision.SetCellTolerance(max(clearance_source_units, 1e-4))
            collision.Update()
            self._interactive_spray_collision_filter = collision
            return bool(collision.GetNumberOfContacts() > 0)
        except Exception as exc:
            # Fail safe: if the collision query itself is unavailable, do not
            # allow the interactive spray to paint.
            print(
                "[SPRAYER COLLISION WARNING] "
                f"{type(exc).__name__}: {exc}"
            )
            self._interactive_spray_collision_filter = None
            return True

    def _interactive_add_distribution_tail_visual(self, pose):
        """Draw one nominal cone plus a faint, distance-decaying overspray tail."""
        if self._interactive_spray_surface is None or self._interactive_spray_locator is None:
            return

        nozzle_world = np.asarray(pose["nozzle_point_world"], dtype=np.float64)
        direction_world = np.asarray(pose["spray_direction_world"], dtype=np.float64)
        direction_world /= max(float(np.linalg.norm(direction_world)), 1e-12)

        inverse = np.linalg.inv(self._interactive_spray_transform)
        nozzle_local = (inverse @ np.append(nozzle_world, 1.0))[:3]
        direction_local = inverse[:3, :3] @ direction_world
        direction_local /= max(float(np.linalg.norm(direction_local)), 1e-12)

        half_angle = math.radians(self._interactive_spray_cone_angle_deg) * 0.5
        nominal_standoff = float(self._interactive_spray_standoff)
        nominal_radius = nominal_standoff * math.tan(half_angle)

        try:
            sigma_fraction = max(
                float(self._interactive_sprayer_profile.get(
                    "overspray_tail_sigma_fraction", 0.35
                )),
                0.05,
            )
        except (TypeError, ValueError):
            sigma_fraction = 0.35

        # Three-sigma is the visual cutoff for the weak infinite-range tail.
        tail_sigma_mm = max(nominal_radius * sigma_fraction, 1e-6)
        tail_extension = max(3.0 * tail_sigma_mm, nominal_standoff * 0.25)
        maximum_axial_distance = nominal_standoff + tail_extension

        # Render the atomized plume as a deterministic volumetric point cloud,
        # not only as a handful of thin cone-generator lines. The low-cost
        # golden-angle distribution avoids per-frame random-number generation
        # and makes the spray visibly occupy the nozzle-to-surface volume.
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, direction_world))) > 0.92:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        plume_u = np.cross(direction_world, reference)
        plume_u /= max(float(np.linalg.norm(plume_u)), 1e-12)
        plume_v = np.cross(direction_world, plume_u)
        plume_v /= max(float(np.linalg.norm(plume_v)), 1e-12)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        axial_layers = np.linspace(0.12, 1.0, 4)
        particles_per_layer = 8
        particle_groups = [[], [], []]
        for layer_index, axial_fraction in enumerate(axial_layers):
            axial_distance = nominal_standoff * float(axial_fraction)
            cone_radius = axial_distance * math.tan(half_angle)
            tail_radius = tail_sigma_mm * min(
                max((axial_distance - nominal_standoff) / max(tail_extension, 1e-9), 0.0),
                1.0,
            )
            for particle_index in range(particles_per_layer):
                radial_fraction = math.sqrt(
                    (particle_index + 0.5) / particles_per_layer
                )
                phi = particle_index * golden_angle + layer_index * 0.37
                radial = (
                    math.cos(phi) * plume_u + math.sin(phi) * plume_v
                )
                radius = cone_radius * radial_fraction + tail_radius * radial_fraction
                point = nozzle_world + (
                    direction_world * axial_distance + radial * radius
                )
                group_index = min(2, int(3 * axial_fraction))
                particle_groups[group_index].append(point)
        for group, opacity, point_size in zip(
            particle_groups, (0.15, 0.095, 0.055), (4.0, 3.5, 3.0)
        ):
            if not group:
                continue
            cloud = pv.PolyData(np.asarray(group, dtype=np.float64))
            cloud_actor = self.plotter.add_mesh(
                cloud,
                color="#FFE8B8",
                opacity=opacity,
                point_size=point_size,
                render_points_as_spheres=True,
                lighting=False,
                pickable=False,
            )
            self._interactive_spray_live_actors.append(cloud_actor)

        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, direction_local))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis_u = np.cross(direction_local, reference)
        basis_u /= max(float(np.linalg.norm(basis_u)), 1e-12)
        basis_v = np.cross(direction_local, basis_u)
        basis_v /= max(float(np.linalg.norm(basis_v)), 1e-12)

        core_hit_segments = []
        segments_by_band = [[], [], []]
        ray_count = 8
        radial_fractions = (0.50, 1.0)
        band_boundaries = (
            nominal_standoff,
            nominal_standoff + tail_extension / 3.0,
            nominal_standoff + 2.0 * tail_extension / 3.0,
            maximum_axial_distance,
        )

        for radial_fraction in radial_fractions:
            theta = half_angle * radial_fraction
            for ray_index in range(ray_count):
                phi = 2.0 * math.pi * ray_index / ray_count
                radial = math.cos(phi) * basis_u + math.sin(phi) * basis_v
                ray_local = (
                    math.cos(theta) * direction_local
                    + math.sin(theta) * radial
                )
                ray_local /= max(float(np.linalg.norm(ray_local)), 1e-12)

                ray_max_length = maximum_axial_distance / max(
                    abs(float(np.dot(ray_local, direction_local))), 0.25
                )
                endpoint_local = nozzle_local + ray_local * ray_max_length
                intersections = vtkPoints()
                intersected_cells = vtkIdList()
                hit = self._interactive_spray_locator.IntersectWithLine(
                    nozzle_local.tolist(),
                    endpoint_local.tolist(),
                    intersections,
                    intersected_cells,
                )

                end_local = endpoint_local
                hit_distance = None
                if hit and intersections.GetNumberOfPoints() > 0:
                    nearest_distance = float("inf")
                    nearest_point = None
                    for hit_index in range(intersections.GetNumberOfPoints()):
                        point = np.asarray(
                            intersections.GetPoint(hit_index),
                            dtype=np.float64,
                        )
                        distance = float(np.linalg.norm(point - nozzle_local))
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_point = point
                    if nearest_point is not None:
                        end_local = nearest_point
                        hit_distance = nearest_distance

                end_distance = float(np.linalg.norm(end_local - nozzle_local))
                if hit_distance is not None and hit_distance <= nominal_standoff:
                    start_point_world = (
                        self._interactive_spray_transform
                        @ np.append(nozzle_local, 1.0)
                    )[:3]
                    end_point_world = (
                        self._interactive_spray_transform
                        @ np.append(end_local, 1.0)
                    )[:3]
                    core_hit_segments.append(
                        (start_point_world, end_point_world)
                    )
                if end_distance <= 1e-6:
                    continue

                for band_index in range(3):
                    band_start = band_boundaries[band_index]
                    band_end = band_boundaries[band_index + 1]
                    start_fraction = min(
                        max(band_start / max(end_distance, 1e-9), 0.0),
                        1.0,
                    )
                    end_fraction = min(
                        max(band_end / max(end_distance, 1e-9), 0.0),
                        1.0,
                    )
                    if end_fraction <= start_fraction + 1e-6:
                        continue

                    start_point_local = nozzle_local + (
                        end_local - nozzle_local
                    ) * start_fraction
                    end_point_local = nozzle_local + (
                        end_local - nozzle_local
                    ) * end_fraction
                    start_point_world = (
                        self._interactive_spray_transform
                        @ np.append(start_point_local, 1.0)
                    )[:3]
                    end_point_world = (
                        self._interactive_spray_transform
                        @ np.append(end_point_local, 1.0)
                    )[:3]
                    segments_by_band[band_index].append(
                        (start_point_world, end_point_world)
                    )

        if core_hit_segments:
            core_polydata = self._build_line_polydata(core_hit_segments)
            if core_polydata is not None:
                actor = self.plotter.add_mesh(
                    core_polydata,
                    color="#FFE8A6",
                    opacity=0.10,
                    line_width=1.0,
                    lighting=False,
                    pickable=False,
                )
                self._interactive_spray_live_actors.append(actor)

        opacities = (0.075, 0.045, 0.020)
        for band_segments, opacity in zip(segments_by_band, opacities):
            if not band_segments:
                continue
            polydata = self._build_line_polydata(band_segments)
            if polydata is None:
                continue
            actor = self.plotter.add_mesh(
                polydata,
                color="#FFD180",
                opacity=float(opacity),
                line_width=1.0,
                lighting=False,
                pickable=False,
            )
            self._interactive_spray_live_actors.append(actor)

    def _interactive_update_surface_angle_indicator(self, pose):
        """Display spray-to-surface angular error without steering the nozzle."""
        label = getattr(self, "spray_perpendicular_status", None)
        if label is None:
            return
        if pose is None:
            label.setText("SURFACE ANGLE: --")
            label.setStyleSheet(
                "color: #9AAAB5; font-size: 9px; padding-left: 6px;"
            )
            return

        direction = np.asarray(
            pose.get("spray_direction_world"), dtype=np.float64
        )
        normal = np.asarray(
            pose.get("surface_normal_world"), dtype=np.float64
        )
        direction_length = float(np.linalg.norm(direction))
        normal_length = float(np.linalg.norm(normal))
        if (
            direction.shape != (3,)
            or normal.shape != (3,)
            or not np.isfinite(direction_length)
            or not np.isfinite(normal_length)
            or direction_length <= 1e-12
            or normal_length <= 1e-12
        ):
            label.setText("SURFACE ANGLE: --")
            return

        # Spray direction points toward the surface; the outward normal points
        # away from it. Therefore -direction should align with the normal.
        alignment = float(np.clip(
            np.dot(-direction / direction_length, normal / normal_length),
            -1.0, 1.0,
        ))
        angle_deg = math.degrees(math.acos(alignment))
        if angle_deg <= 5.0:
            state, color = "PERPENDICULAR", "#65D68A"
        elif angle_deg <= 15.0:
            state, color = "SLIGHT ANGLE", "#E7C66A"
        else:
            state, color = "OFF-ANGLE", "#FF8A75"
        label.setText(f"SURFACE ANGLE: {angle_deg:.1f}° | {state}")
        label.setStyleSheet(
            f"color: {color}; font-size: 9px; padding-left: 6px; "
            "font-weight: 700;"
        )

    def _interactive_update_cursor(self, pose, force_record=False):
        # Record and deposit from every valid mouse sample. Only the expensive
        # 3D cursor visualization is throttled; applying the old 30 Hz limit to
        # this entire method silently discarded intermediate spray positions.
        self._interactive_update_surface_angle_indicator(pose)
        self._interactive_spray_latest_pose = pose
        nozzle = pose["nozzle_point_world"]
        direction = pose["spray_direction_world"]

        # Evaluate the complete sprayer envelope before allowing this mouse
        # sample to record or deposit paint.
        gun_actors, gun_polydata = self._interactive_build_gun_visual(
            nozzle, direction
        )
        self._interactive_spray_gun_actors = gun_actors
        collision = self._interactive_check_gun_collision(gun_polydata)
        surface_excluded = not bool(pose.get("paintable", True))
        self._interactive_spray_gun_collision = collision
        self._interactive_spray_surface_excluded = surface_excluded
        spray_blocked = collision or surface_excluded
        if spray_blocked:
            # A blocked interval breaks both deposition continuity and the
            # fixed-distance sampler. Never interpolate a route across it.
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
            self._interactive_spray_sampling_previous_pose = None
            self._interactive_spray_sampling_distance_accumulator = 0.0
    
        local_pose = self._interactive_pose_to_product_frame(pose)
        footprints = _build_spray_footprint_overlay(
            self.current_mesh,
            local_pose["nozzle_point"].reshape(1, 3),
            local_pose["spray_direction"].reshape(1, 3),
            self._interactive_spray_paint_footprint_angle_deg,
            rays_per_ring=8,
            radial_rings=(0.40, 0.75, 1.0),
            surface=self._interactive_spray_surface,
            locator=self._interactive_spray_locator,
        )
        recorded_samples = []
        if footprints and footprints[0][0].n_cells:
            patch, boundary = footprints[0]
            patch.points = (
                np.asarray(patch.points) @ self._interactive_spray_transform[:3, :3].T
                + self._interactive_spray_transform[:3, 3]
            )
            # Generate and coat every fixed-distance sample. The cursor event
            # rate is deliberately not used as the trajectory spacing.
            if self._interactive_spray_recording and not spray_blocked:
                recorded_samples = self._interactive_append_record(
                    pose, local_pose, force=bool(force_record)
                )
                for sample_pose, sample_local in recorded_samples:
                    sample_footprints = _build_spray_footprint_overlay(
                        self.current_mesh,
                        sample_local["nozzle_point"].reshape(1, 3),
                        sample_local["spray_direction"].reshape(1, 3),
                        self._interactive_spray_paint_footprint_angle_deg,
                        rays_per_ring=8,
                        radial_rings=(0.40, 0.75, 1.0),
                        surface=self._interactive_spray_surface,
                        locator=self._interactive_spray_locator,
                    )
                    if not sample_footprints or not sample_footprints[0][0].n_cells:
                        # Match replay behavior: a sample with no valid surface
                        # footprint breaks deposition continuity.
                        self._interactive_spray_last_deposition_timestamp = None
                        self._interactive_spray_previous_nozzle_world = None
                                        continue
                    sample_patch = sample_footprints[0][0]
                    sample_patch.points = (
                        np.asarray(sample_patch.points)
                        @ self._interactive_spray_transform[:3, :3].T
                        + self._interactive_spray_transform[:3, 3]
                    )
                    sample_gun_actors, sample_gun_polydata = self._interactive_build_gun_visual(
                        sample_pose["nozzle_point_world"],
                        sample_pose["spray_direction_world"],
                    )
                    self._interactive_spray_gun_collision = (
                        self._interactive_check_gun_collision(sample_gun_polydata)
                    )
                    self._interactive_spray_surface_excluded = not bool(
                        sample_pose.get("paintable", True)
                    )
                    self._interactive_apply_paint_deposition(
                        sample_patch, sample_pose
                    )
                    for sample_actor in sample_gun_actors:
                        try:
                            self.plotter.remove_actor(sample_actor, render=False)
                        except Exception:
                            pass
            recorded_new_pose = bool(recorded_samples)
        else:
            recorded_new_pose = False
            if self._interactive_spray_recording:
                # A missing footprint is a real discontinuity; do not bridge
                # the invalid area with a later interpolated spray segment.
                self._interactive_spray_sampling_previous_pose = None
                self._interactive_spray_sampling_distance_accumulator = 0.0
                self._interactive_spray_last_deposition_timestamp = None
                self._interactive_spray_previous_nozzle_world = None

        # Keep full-rate trajectory/deposition sampling above, but cap only
        # actor rebuilding and rendering to 30 Hz for interactive responsiveness.
        now = time.monotonic()
        previous = getattr(
            self, "_interactive_spray_last_cursor_update_time", None
        )
        if previous is not None and now - previous < (1.0 / 30.0):
            return
        self._interactive_spray_last_cursor_update_time = now
        self._interactive_remove_live_actors()
        self._interactive_spray_live_actors.extend(gun_actors)
        if recorded_new_pose:
            self._interactive_refresh_paint_visual(render=False)
            # Show the exact recorded trajectory during the stroke. The route
            # renderer already draws the green path and the yellow arrows that
            # follow each sample's actual spray direction. Refreshing only at
            # the existing 30 Hz visual cadence keeps the cursor responsive
            # without dropping any trajectory samples.
            self._interactive_spray_live_route_refresh_at = now
            self._refresh_interactive_recorded_path_actor()

        if footprints and footprints[0][0].n_cells:
            boundary = footprints[0][1]
            if boundary.n_cells:
                boundary.points = (
                    np.asarray(boundary.points) @ self._interactive_spray_transform[:3, :3].T
                    + self._interactive_spray_transform[:3, 3]
                )
                actor = self.plotter.add_mesh(
                    boundary, color="#FF9F1C", line_width=2.0,
                    lighting=False, pickable=False,
                )
                self._interactive_spray_live_actors.append(actor)

        nozzle = pose["nozzle_point_world"]
        hit = pose["surface_point_world"]
        direction = pose["spray_direction_world"]

        # Keep exactly one nominal cone. Construct its topology explicitly:
        # point 0 is the nozzle apex; the circular rim is downstream toward
        # the product. This avoids vtk/PyVista cone-axis sign ambiguity.
        half_angle = math.radians(self._interactive_spray_cone_angle_deg) * 0.5
        nominal_radius = self._interactive_spray_standoff * math.tan(half_angle)
        spray_axis = np.asarray(direction, dtype=np.float64)
        spray_axis /= max(float(np.linalg.norm(spray_axis)), 1e-12)
        reference_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference_axis, spray_axis))) > 0.92:
            reference_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        radial_u = np.cross(spray_axis, reference_axis)
        radial_u /= max(float(np.linalg.norm(radial_u)), 1e-12)
        radial_v = np.cross(spray_axis, radial_u)
        radial_v /= max(float(np.linalg.norm(radial_v)), 1e-12)
        ring_angles = np.linspace(0.0, 2.0 * math.pi, 49)[:-1]
        base_center = nozzle + spray_axis * self._interactive_spray_standoff
        rim = (
            base_center[None, :]
            + nominal_radius
            * (
                np.cos(ring_angles)[:, None] * radial_u[None, :]
                + np.sin(ring_angles)[:, None] * radial_v[None, :]
            )
        )
        cone_points = np.vstack((nozzle, rim))
        cone_faces = []
        for i in range(len(rim)):
            j = (i + 1) % len(rim)
            cone_faces.extend((3, 0, i + 1, j + 1))
        cone = pv.PolyData(
            cone_points,
            faces=np.asarray(cone_faces, dtype=np.int64),
        )
        actor = self.plotter.add_mesh(
            cone,
            color="#FFE082",
            opacity=0.16,
            style="wireframe",
            line_width=1.25,
            lighting=False,
            pickable=False,
        )
        self._interactive_spray_live_actors.append(actor)

        # Trace the nominal cone's outer generators against the product.
        # Their first-hit points form the visible contact boundary on the
        # actual surface (rather than a flat circle floating at standoff).
        contact_boundary_segments = []
        if (
            self._interactive_spray_locator is not None
            and self._interactive_spray_surface is not None
        ):
            inverse = np.linalg.inv(self._interactive_spray_transform)
            nozzle_local = (
                inverse @ np.append(nozzle, 1.0)
            )[:3]
            axis_local = inverse[:3, :3] @ spray_axis
            axis_local /= max(float(np.linalg.norm(axis_local)), 1e-12)
            ref_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(ref_local, axis_local))) > 0.92:
                ref_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            edge_u = np.cross(axis_local, ref_local)
            edge_u /= max(float(np.linalg.norm(edge_u)), 1e-12)
            edge_v = np.cross(axis_local, edge_u)
            edge_v /= max(float(np.linalg.norm(edge_v)), 1e-12)

            contact_points_world = []
            contact_hit = []
            edge_ray_count = 12
            edge_ray_length = (
                self._interactive_spray_standoff
                / max(math.cos(half_angle), 1e-6)
            ) * 1.02
            for ray_index in range(edge_ray_count):
                phi = 2.0 * math.pi * ray_index / edge_ray_count
                radial = math.cos(phi) * edge_u + math.sin(phi) * edge_v
                edge_direction = (
                    math.cos(half_angle) * axis_local
                    + math.sin(half_angle) * radial
                )
                edge_direction /= max(
                    float(np.linalg.norm(edge_direction)), 1e-12
                )
                endpoint = nozzle_local + edge_direction * edge_ray_length
                intersections = vtkPoints()
                intersected_cells = vtkIdList()
                hit = self._interactive_spray_locator.IntersectWithLine(
                    nozzle_local.tolist(),
                    endpoint.tolist(),
                    intersections,
                    intersected_cells,
                )
                nearest_point = None
                if hit and intersections.GetNumberOfPoints() > 0:
                    nearest_distance = float("inf")
                    for hit_index in range(intersections.GetNumberOfPoints()):
                        candidate = np.asarray(
                            intersections.GetPoint(hit_index),
                            dtype=np.float64,
                        )
                        candidate_distance = float(
                            np.linalg.norm(candidate - nozzle_local)
                        )
                        if candidate_distance < nearest_distance:
                            nearest_distance = candidate_distance
                            nearest_point = candidate
                contact_hit.append(nearest_point is not None)
                if nearest_point is None:
                    contact_points_world.append(None)
                else:
                    contact_points_world.append(
                        (
                            self._interactive_spray_transform
                            @ np.append(nearest_point, 1.0)
                        )[:3]
                    )

            # Draw only adjacent hit pairs. Missing rays remain gaps instead
            # of creating misleading chords across openings or silhouette gaps.
            for ray_index in range(edge_ray_count):
                next_index = (ray_index + 1) % edge_ray_count
                if contact_hit[ray_index] and contact_hit[next_index]:
                    contact_boundary_segments.append(
                        (
                            contact_points_world[ray_index],
                            contact_points_world[next_index],
                        )
                    )

        if contact_boundary_segments:
            contact_boundary = self._build_line_polydata(
                contact_boundary_segments
            )
            if contact_boundary is not None:
                contact_actor = self.plotter.add_mesh(
                    contact_boundary,
                    color="#FFF4CC",
                    opacity=0.62,
                    line_width=2.0,
                    lighting=False,
                    pickable=False,
                )
                self._interactive_spray_live_actors.append(contact_actor)

        # Extend the spray visually with a weak, distance-decaying distribution.
        # Each ray stops at the first product surface; a ray with no hit fades
        # out at the configured distribution cutoff.
        self._interactive_add_distribution_tail_visual(pose)

        # Faint centerline direction vector: always points from the nozzle
        # toward the painted surface / spray direction.
        centerline_end = nozzle + direction * self._interactive_spray_standoff
        centerline = pv.Arrow(
            start=nozzle,
            direction=direction,
            scale=self._interactive_spray_standoff,
            tip_length=0.10,
            tip_radius=0.018,
            shaft_radius=0.006,
        )
        centerline_actor = self.plotter.add_mesh(
            centerline,
            color="#FFF3C4",
            opacity=0.32,
            lighting=False,
            pickable=False,
        )
        self._interactive_spray_live_actors.append(centerline_actor)

        # Faint nozzle-to-surface direction vector.
        arrow_length = max(self._interactive_spray_standoff * 0.92, 1e-6)
        hit_vector = hit - nozzle
        arrow = pv.Arrow(
            start=nozzle,
            direction=hit_vector,
            tip_length=0.10,
            tip_radius=max(arrow_length * 0.012, 0.1),
            shaft_radius=max(arrow_length * 0.004, 0.04),
            scale=arrow_length / max(float(np.linalg.norm(hit_vector)), 1e-12),
        )
        actor = self.plotter.add_mesh(
            arrow, color="#B8D7E3", opacity=0.30,
            lighting=False, pickable=False,
        )
        self._interactive_spray_live_actors.append(actor)

        if surface_excluded:
            status_text = "STAGE 2 EXCLUDED - SPRAY DISABLED"
        elif collision:
            status_text = "COLLISION - SPRAY BLOCKED"
        else:
            status_text = (
                "SPRAYER CLEAR | READY"
            )

        if self._interactive_hud_actor is not None:
            hud_text = (
                "SPRAY CURSOR | "
                + ("HOLD LEFT MOUSE TO PAINT | " if not spray_blocked else "")
                + status_text
            )
            try:
                # PyVista's CornerAnnotation exposes SetText(), while the
                # underlying vtkCornerAnnotation uses SetText(index, text).
                if hasattr(self._interactive_hud_actor, "SetText"):
                    self._interactive_hud_actor.SetText(2, hud_text)
                elif hasattr(self._interactive_hud_actor, "SetInput"):
                    self._interactive_hud_actor.SetInput(hud_text)
            except (AttributeError, TypeError):
                pass

        marker = pv.Sphere(
            radius=max(self._interactive_spray_standoff * 0.018, 0.5),
            center=nozzle, theta_resolution=16, phi_resolution=16,
        )
        actor = self.plotter.add_mesh(
            marker,
            color="#FF3B30" if spray_blocked else "#00E676",
            smooth_shading=True,
            pickable=False,
        )
        self._interactive_spray_live_actors.append(actor)

        gun_color = "#FF3B30" if spray_blocked else "#B9C4CC"
        for gun_actor in gun_actors:
            try:
                color = QColor(gun_color)
                gun_actor.GetProperty().SetColor(
                    color.redF(), color.greenF(), color.blueF()
                )
            except Exception:
                pass

        self.plotter.render()

    @staticmethod
    def _interactive_interpolate_input_pose(first, second, fraction):
        """Interpolate a picked pose along one cursor-motion segment."""
        fraction = float(np.clip(fraction, 0.0, 1.0))
        result = {}
        vector_keys = (
            "nozzle_point_world", "surface_point_world",
            "surface_normal_world", "spray_direction_world",
        )
        for key in vector_keys:
            a = np.asarray(first[key], dtype=np.float64)
            b = np.asarray(second[key], dtype=np.float64)
            value = a + (b - a) * fraction
            if key in ("surface_normal_world", "spray_direction_world"):
                value /= max(float(np.linalg.norm(value)), 1e-12)
            result[key] = value
        result["timestamp"] = (
            float(first["timestamp"])
            + (float(second["timestamp"]) - float(first["timestamp"])) * fraction
        )
        result["cell_id"] = int(
            first["cell_id"] if fraction < 0.5 else second["cell_id"]
        )
        result["paintable"] = bool(
            first.get("paintable", True) and second.get("paintable", True)
        )
        return result

    @staticmethod
    def _interactive_copy_input_pose(pose):
        return {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in pose.items()
        }

    def _interactive_store_record_sample(self, pose, local_pose):
        if self._interactive_spray_start_timestamp is None:
            self._interactive_spray_start_timestamp = float(pose["timestamp"])
        relative_time = max(
            0.0,
            float(pose["timestamp"]) - float(self._interactive_spray_start_timestamp),
        )
        record = {
            "nozzle_point_world": np.asarray(pose["nozzle_point_world"], dtype=float).copy(),
            "surface_point_world": np.asarray(pose["surface_point_world"], dtype=float).copy(),
            "surface_normal_world": np.asarray(pose["surface_normal_world"], dtype=float).copy(),
            "spray_direction_world": np.asarray(pose["spray_direction_world"], dtype=float).copy(),
            "nozzle_point_product": local_pose["nozzle_point"].copy(),
            "surface_point_product": local_pose["surface_point"].copy(),
            "surface_normal_product": local_pose["surface_normal"].copy(),
            "spray_direction_product": local_pose["spray_direction"].copy(),
            "quaternion_product_xyzw": local_pose["quaternion_xyzw"].copy(),
            "timestamp": float(pose["timestamp"]),
            "t_product_s": relative_time,
            "face_index": int(pose["cell_id"]),
            "stroke_id": int(self._interactive_spray_stroke_id),
        }
        self._interactive_spray_records.append(record)

    def _interactive_append_record(self, pose, local_pose=None, force=False):
        """Record trajectory samples at fixed arc-length spacing, not event spacing.

        The distance accumulator carries the unconsumed arc length between
        mouse events, so fast, slow, and irregular input event rates produce
        the same spatial sampling interval.
        """
        spacing = max(float(self._interactive_spray_sampling_spacing), 1e-6)
        current = self._interactive_copy_input_pose(pose)
        previous = getattr(self, "_interactive_spray_sampling_previous_pose", None)
        emitted = []

        def emit(sample):
            sample_local = self._interactive_pose_to_product_frame(sample)
            self._interactive_store_record_sample(sample, sample_local)
            emitted.append((sample, sample_local))

        if previous is None:
            emit(current)
            self._interactive_spray_sampling_distance_accumulator = 0.0
            self._interactive_spray_sampling_previous_pose = current
            return emitted

        segment_length = float(np.linalg.norm(
            np.asarray(current["nozzle_point_world"], dtype=np.float64)
            - np.asarray(previous["nozzle_point_world"], dtype=np.float64)
        ))
        accumulated = float(getattr(
            self, "_interactive_spray_sampling_distance_accumulator", 0.0
        ))
        if segment_length > 1e-12:
            consumed = 0.0
            remaining = segment_length
            needed = max(spacing - accumulated, 0.0)
            while remaining + 1e-10 >= needed and needed > 1e-10:
                consumed += needed
                fraction = min(consumed / segment_length, 1.0)
                sample = self._interactive_interpolate_input_pose(
                    previous, current, fraction
                )
                emit(sample)
                remaining = max(segment_length - consumed, 0.0)
                accumulated = 0.0
                needed = spacing
            accumulated += remaining
        if force and (
            not self._interactive_spray_records
            or float(np.linalg.norm(
                np.asarray(current["nozzle_point_world"], dtype=np.float64)
                - np.asarray(self._interactive_spray_records[-1]["nozzle_point_world"], dtype=np.float64)
            )) > 1e-8
        ):
            emit(current)
            accumulated = 0.0

        self._interactive_spray_sampling_distance_accumulator = accumulated
        self._interactive_spray_sampling_previous_pose = current
        return emitted

    def _on_interactive_spray_middle_press(self, interactor, event):
        """Begin a middle-button drag to tilt the spray axis."""
        if (
            not self._interactive_spray_active
            or self._interactive_spray_mode != "spray"
        ):
            return
        self._interactive_spray_angle_dragging = True
        self._interactive_spray_angle_last_position = tuple(
            int(value) for value in interactor.GetEventPosition()
        )
        # Capture the pose once. Subsequent middle-drag events must not repick.
        anchor = self._interactive_spray_latest_pose
        if anchor is None:
            anchor = self._interactive_pick_surface_pose(*self._interactive_spray_angle_last_position)
        self._interactive_spray_angle_anchor_pose = None
        if anchor is not None:
            anchor_pose = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in anchor.items()
            }
            nozzle = np.asarray(
                anchor_pose["nozzle_point_world"], dtype=np.float64
            )
            direction = np.asarray(
                anchor_pose["spray_direction_world"], dtype=np.float64
            )
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            plane_point = np.asarray(
                anchor_pose["surface_point_world"], dtype=np.float64
            )
            plane_normal = np.asarray(
                anchor_pose["surface_normal_world"], dtype=np.float64
            )
            plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-12)
            denominator = float(np.dot(direction, plane_normal))
            if abs(denominator) > 1e-8:
                ray_distance = float(
                    np.dot(plane_point - nozzle, plane_normal) / denominator
                )
                if ray_distance > 0.0 and np.isfinite(ray_distance):
                    anchor_pose["angle_pivot_world"] = (
                        nozzle + direction * ray_distance
                    )
                    self._interactive_spray_angle_anchor_pose = anchor_pose
        # Angle adjustment is a positioning gesture, never a paint stroke.
        self._interactive_spray_recording = False
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None

    def _on_interactive_spray_middle_release(self, interactor, event):
        self._interactive_spray_angle_dragging = False
        self._interactive_spray_angle_last_position = None
        self._interactive_spray_angle_anchor_pose = None

        # Lock the adjusted direction after the gesture. Subsequent cursor
        # picks may move the surface target, but must not replace this axis
        # with the normal of whichever face happens to be under the pointer.
        latest = self._interactive_spray_latest_pose
        if latest is not None:
            direction = np.asarray(
                latest["spray_direction_world"], dtype=np.float64
            ).copy()
            length = float(np.linalg.norm(direction))
            if np.isfinite(length) and length > 1e-12:
                self._interactive_spray_locked_direction_world = direction / length

    def _interactive_lock_spray_direction_from_pose(self, pose):
        """Freeze the current nozzle axis for the duration of a paint stroke."""
        if pose is None:
            return
        # A manually adjusted direction already locked by middle-button release
        # takes precedence. Otherwise capture the current preview direction at
        # the instant the operator starts spraying.
        if self._interactive_spray_locked_direction_world is not None:
            return
        direction = np.asarray(
            pose["spray_direction_world"], dtype=np.float64
        ).copy()
        length = float(np.linalg.norm(direction))
        if np.isfinite(length) and length > 1e-12:
            self._interactive_spray_locked_direction_world = direction / length

    def _interactive_apply_locked_spray_direction(self, pose):
        """Keep the chosen spray axis while moving the target over new faces."""
        locked = self._interactive_spray_locked_direction_world
        if pose is None or locked is None:
            return pose
        direction = np.asarray(locked, dtype=np.float64).copy()
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= 1e-12:
            return pose
        direction /= length

        updated = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in pose.items()
        }
        surface_point = np.asarray(
            updated["surface_point_world"], dtype=np.float64
        )
        # Keep the picked surface point on the cone centerline at the current
        # working distance, without changing the locked spray orientation.
        updated["spray_direction_world"] = direction
        updated["nozzle_point_world"] = (
            surface_point - direction * float(self._interactive_spray_standoff)
        )
        updated["timestamp"] = time.monotonic()
        return updated

    def _rotate_spray_axis_from_middle_drag(self, interactor):
        current = tuple(int(value) for value in interactor.GetEventPosition())
        previous = self._interactive_spray_angle_last_position
        self._interactive_spray_angle_last_position = current
        if previous is None:
            return
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        if dx == 0 and dy == 0:
            return

        # Match the screen drag axes to the observed tool motion:
        # horizontal drag changes the left/right tilt; vertical drag changes
        # the up/down tilt. The pose helper maps angle_x to pitch and angle_y
        # to yaw, so these accumulators are intentionally crossed here.
        sensitivity = 0.20
        self._interactive_spray_angle_x_deg = float(np.clip(
            self._interactive_spray_angle_x_deg - dx * sensitivity,
            -45.0, 45.0,
        ))
        self._interactive_spray_angle_y_deg = float(np.clip(
            self._interactive_spray_angle_y_deg - dy * sensitivity,
            -45.0, 45.0,
        ))

        anchor = self._interactive_spray_angle_anchor_pose
        if anchor is None:
            return

        # Rotate about the intersection of the original cone centerline and
        # the picked surface plane. The intersection stays fixed; the nozzle
        # moves on an arc at the configured working distance.
        pivot = np.asarray(
            anchor["angle_pivot_world"], dtype=np.float64
        )
        rotated_direction = self._interactive_rotate_spray_direction(
            anchor["spray_direction_world"],
            self._interactive_spray_angle_y_deg,
            self._interactive_spray_angle_x_deg,
        )
        distance_source = float(self._interactive_spray_standoff)
        pose = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in anchor.items()
            if key != "angle_pivot_world"
        }
        pose["surface_point_world"] = pivot.copy()
        pose["spray_direction_world"] = rotated_direction
        pose["nozzle_point_world"] = pivot - rotated_direction * distance_source
        pose["timestamp"] = time.monotonic()
        self._interactive_update_cursor(pose)

    def _on_interactive_spray_wheel(self, interactor, event):
        """Adjust spray working distance instead of zooming during SPRAY mode.

        Wheel up increases the nozzle-to-surface distance and wheel down
        decreases it. The value is clamped to the configured minimum and
        maximum limits. The current cursor pose is recomputed immediately,
        so the nozzle moves even when the mouse cursor itself is stationary.
        """
        if (
            not self._interactive_spray_active
            or self._interactive_spray_mode != "spray"
        ):
            return

        event_name = str(event)
        if "Forward" in event_name:
            delta_mm = INTERACTIVE_SPRAY_WHEEL_STEP_MM
        elif "Backward" in event_name:
            delta_mm = -INTERACTIVE_SPRAY_WHEEL_STEP_MM
        else:
            return

        current_mm = float(self.working_distance_spin.value())
        target_mm = float(np.clip(
            current_mm + delta_mm,
            INTERACTIVE_SPRAY_MIN_STANDOFF_MM,
            INTERACTIVE_SPRAY_MAX_STANDOFF_MM,
        ))

        if abs(target_mm - current_mm) <= 1e-9:
            return

        self.working_distance_spin.setValue(target_mm)

        # A distance change is a deliberate discontinuity in nozzle position.
        # Do not integrate the interval before/after the wheel movement as one
        # continuous paint-deposition step.
        if self._interactive_spray_recording:
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
    
        pose = self._interactive_spray_latest_pose
        if pose is not None:
            self._interactive_update_cursor(dict(pose))
        else:
            self.plotter.render()

    def _on_interactive_spray_mouse_move(self, interactor, event):
        if (
            not self._interactive_spray_active
            or self._interactive_spray_mode != "spray"
        ):
            return

        # Middle-drag exclusively controls spray direction.
        if self._interactive_spray_angle_dragging:
            if not bool(QApplication.mouseButtons() & Qt.MouseButton.MiddleButton):
                self._interactive_spray_angle_dragging = False
                self._interactive_spray_angle_last_position = None
            else:
                self._rotate_spray_axis_from_middle_drag(interactor)
                return

        # Qt's global button state is the fallback source of truth. Some
        # PyVista/VTK integrations do not reliably deliver VTK button-press
        # observers after the interactor style is replaced. Detect the held
        # left button here as well, so the first drag sample starts a stroke.
        left_button_held = bool(
            QApplication.mouseButtons() & Qt.MouseButton.LeftButton
        )
        if self._interactive_spray_recording and not left_button_held:
            self._interactive_spray_recording = False
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
    
        x, y = interactor.GetEventPosition()
        pose = self._interactive_pick_surface_pose(x, y)
        if pose is None:
            self._interactive_remove_live_actors()
            self._interactive_spray_latest_pose = None
            self._interactive_update_surface_angle_indicator(None)
            self._interactive_spray_recording = False
            self.plotter.render()
            return
        if not bool(pose.get("paintable", True)):
            self._interactive_spray_recording = False
            self._interactive_spray_last_deposition_timestamp = None
            self._interactive_spray_previous_nozzle_world = None
            elif left_button_held and not self._interactive_spray_recording:
            self._interactive_begin_spray_stroke()

        if self._interactive_spray_recording:
            self._interactive_lock_spray_direction_from_pose(pose)
        pose = self._interactive_apply_locked_spray_direction(pose)
        # Once the left button is held, the spray axis is frozen from the
        # stroke-start pose and is independent of every subsequently picked
        # face normal. Only the target point translates with the cursor.
        self._interactive_update_cursor(pose)

    def _interactive_begin_spray_stroke(self):
        """Start a fresh stroke exactly once when the left button goes down."""
        if self._interactive_spray_recording:
            return
        self._interactive_spray_stroke_id += 1
        self._interactive_spray_recording = True
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_start_timestamp = None
        self._interactive_spray_sampling_previous_pose = None
        self._interactive_spray_sampling_distance_accumulator = 0.0

    def _on_interactive_spray_left_press(self, interactor, event):
        if (
            not self._interactive_spray_active
            or self._interactive_spray_mode != "spray"
        ):
            return
        x, y = interactor.GetEventPosition()
        pose = self._interactive_pick_surface_pose(x, y)
        if pose is None:
            return
        if not bool(pose.get("paintable", True)):
            self._interactive_update_cursor(pose)
            return
        self._interactive_lock_spray_direction_from_pose(pose)
        pose = self._interactive_apply_locked_spray_direction(pose)
        self._interactive_begin_spray_stroke()
        self._interactive_update_cursor(pose)

    def _return_to_interactive_spray_mode(self):
        """Restore spray input after VTK has switched to camera navigation."""
        if not self._interactive_spray_active:
            return False

        interactor = self.plotter.iren.interactor
        if self._interactive_spray_input_style is None:
            self._interactive_spray_input_style = vtkInteractorStyleUser()

        # Reinstall the dedicated input style without ending the spray session.
        interactor.SetInteractorStyle(self._interactive_spray_input_style)
        self._interactive_spray_input_style.SetEnabled(1)
        self._interactive_spray_mode = "spray"

        # Return to the standard spray orientation. Keep coating and records.
        self._interactive_spray_recording = False
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_sampling_previous_pose = None
        self._interactive_spray_sampling_distance_accumulator = 0.0
        self._interactive_spray_angle_dragging = False
        self._interactive_spray_angle_last_position = None
        self._interactive_spray_angle_x_deg = 0.0
        self._interactive_spray_angle_y_deg = 0.0
        self._interactive_spray_angle_anchor_pose = None
        self._interactive_spray_locked_direction_world = None

        self.interaction_mode_status.setText(
            "SPRAY MODE · HOLD LEFT TO PAINT"
        )
        if self._interactive_hud_actor is not None:
            hud_text = (
                f"SPRAY MODE  |  WORK DIST "
                f"{self._interactive_spray_standoff * (self._get_source_mm_factor() or 1.0):.1f} mm "
                f"| CONE {self._interactive_spray_cone_angle_deg:.1f} DEG  |  "
                "MOVE: PREVIEW  |  LEFT: PAINT  |  MIDDLE: ANGLE"
            )
            try:
                self._interactive_hud_actor.SetText(2, hud_text)
            except (AttributeError, TypeError):
                pass
        x, y = interactor.GetEventPosition()
        pose = self._interactive_pick_surface_pose(x, y)
        if pose is not None:
            self._interactive_update_cursor(pose)
        else:
            self._interactive_spray_latest_pose = None
            self._interactive_update_surface_angle_indicator(None)
            self.plotter.render()
        return True

    def _on_interactive_spray_left_double_click(self, interactor, event):
        """Keep a double-click from changing the active spray input mode."""
        if (
            not self._interactive_spray_active
            or self._interactive_spray_mode != "spray"
        ):
            return

        # A double-click is a cursor/preview action, never a request to enter
        # camera navigation. Reassert spray input ownership in case the
        # platform's VTK/Qt double-click handling changed the interactor style.
        # Reassert spray-only input ownership. No camera-mode transition
        # exists while a spray session is active.
        interactor.SetInteractorStyle(self._interactive_spray_input_style)
        self._interactive_spray_input_style.SetEnabled(1)

        self._interactive_spray_recording = False
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_sampling_previous_pose = None
        self._interactive_spray_sampling_distance_accumulator = 0.0
        self._interactive_spray_angle_dragging = False
        self._interactive_spray_angle_last_position = None
        self._interactive_spray_angle_x_deg = 0.0
        self._interactive_spray_angle_y_deg = 0.0
        self._interactive_spray_angle_anchor_pose = None
        self._interactive_spray_locked_direction_world = None

        x, y = interactor.GetEventPosition()
        standard_pose = self._interactive_pick_surface_pose(x, y)
        if standard_pose is not None:
            self._interactive_update_cursor(standard_pose)
        else:
            self._interactive_spray_latest_pose = None
            self._interactive_update_surface_angle_indicator(None)
            self.plotter.render()

    def _on_interactive_spray_left_release(self, interactor, event):
        if (
            not self._interactive_spray_active
            or self._interactive_spray_mode != "spray"
        ):
            return
        x, y = interactor.GetEventPosition()
        # Capture the mouse-up location before closing the stroke. Without this
        # final sample, the last cursor movement between the final move event
        # and button release was omitted from the recorded route.
        if self._interactive_spray_recording:
            final_pose = self._interactive_pick_surface_pose(x, y)
            if final_pose is not None and bool(final_pose.get("paintable", True)):
                final_pose = self._interactive_apply_locked_spray_direction(
                    final_pose
                )
                self._interactive_update_cursor(final_pose, force_record=True)

        self._interactive_spray_recording = False
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_sampling_previous_pose = None
        self._interactive_spray_sampling_distance_accumulator = 0.0

        # One angle adjustment applies to exactly one completed paint stroke.
        # Restore the standard cone direction immediately after the stroke.
        self._interactive_spray_angle_x_deg = 0.0
        self._interactive_spray_angle_y_deg = 0.0
        self._interactive_spray_angle_anchor_pose = None
        self._interactive_spray_locked_direction_world = None
        self._interactive_spray_latest_pose = None
        standard_pose = self._interactive_pick_surface_pose(x, y)
        if standard_pose is not None:
            self._interactive_update_cursor(standard_pose)
        else:
            self.plotter.render()

    def _on_interactive_spray_leave(self, interactor, event):
        # Never keep spraying after the pointer leaves the 3D viewport.
        if not self._interactive_spray_active:
            return
        self._interactive_spray_recording = False

        self._interactive_spray_angle_dragging = False
        self._interactive_spray_angle_last_position = None
        self._interactive_spray_angle_x_deg = 0.0
        self._interactive_spray_angle_y_deg = 0.0
        self._interactive_spray_angle_anchor_pose = None
        self._interactive_spray_locked_direction_world = None
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_spray_latest_pose = None
        self._interactive_update_surface_angle_indicator(None)

    def finish_interactive_spray_path(self):
        """Finish input and return the recorded path without changing Stage 3 data."""
        if not self._interactive_spray_active:
            return None
        self._interactive_spray_recording = False

        self._interactive_spray_active = False
        self._interactive_spray_last_deposition_timestamp = None
        self._interactive_spray_previous_nozzle_world = None
        self._interactive_remove_live_actors()
        if self._interactive_spray_excluded_overlay_actor is not None:
            try:
                self.plotter.remove_actor(
                    self._interactive_spray_excluded_overlay_actor,
                    render=False,
                )
            except Exception:
                pass
            self._interactive_spray_excluded_overlay_actor = None

        interactor = self.plotter.iren.interactor
        if self._interactive_saved_style is not None:
            interactor.SetInteractorStyle(self._interactive_saved_style)
            self._interactive_saved_style.SetEnabled(
                1 if self._interactive_saved_style_enabled else 0
            )
        self._interactive_spray_input_style = None
        self._interactive_saved_style = None
        self.interaction_mode_panel.hide()
        self.working_distance_spin.setEnabled(False)
        self.interaction_mode_status.setText("SPRAY INPUT INACTIVE")

        if self._interactive_hud_actor is not None:
            try:
                self.plotter.remove_actor(self._interactive_hud_actor, render=False)
            except Exception:
                pass
            self._interactive_hud_actor = None

        records = list(self._interactive_spray_records)
        route_length = 0.0
        route_optimization = None
        route_timing = None
        if records:
            self._interactive_spray_records = list(records)
            if len(records) >= 2:
                self.interactive_recorded_path_panel.show()
                self.recorded_path_replay_button.setEnabled(True)
                self._optimize_interactive_recorded_path()
                records = list(self._interactive_spray_records)
                route_optimization = getattr(
                    self, "_interactive_spray_route_optimization", None
                )
                route_timing = getattr(
                    self, "_interactive_recorded_replay_timing", None
                )
                if route_timing:
                    mm_factor = float(self._get_source_mm_factor() or 1.0)
                    route_length = (
                        float(route_timing["spray_distance_mm"])
                        + float(route_timing["connector_distance_mm"])
                    ) / mm_factor
            self._refresh_interactive_recorded_path_actor()

        result = {
            "schema": "INTERACTIVE_SURFACE_FOLLOWING_SPRAY_PATH_V2",
            "coordinate_frame": "PRODUCT_CAD_AND_WORLD",
            "standoff_source_units": self._interactive_spray_standoff,
            "cone_angle_deg": self._interactive_spray_cone_angle_deg,
            "sampling_spacing_source_units": self._interactive_spray_sampling_spacing,
            "sprayer_profile": dict(self._interactive_sprayer_profile),
            "coating_process": dict(self._interactive_coating_process),
            "nozzle_points_product": np.asarray(
                [item["nozzle_point_product"] for item in records], dtype=float
            ).reshape(-1, 3),
            "surface_points_product": np.asarray(
                [item["surface_point_product"] for item in records], dtype=float
            ).reshape(-1, 3),
            "spray_directions_product": np.asarray(
                [item["spray_direction_product"] for item in records], dtype=float
            ).reshape(-1, 3),
            "nozzle_points_world": np.asarray(
                [item["nozzle_point_world"] for item in records], dtype=float
            ).reshape(-1, 3),
            "surface_points_world": np.asarray(
                [item["surface_point_world"] for item in records], dtype=float
            ).reshape(-1, 3),
            "spray_directions_world": np.asarray(
                [item["spray_direction_world"] for item in records], dtype=float
            ).reshape(-1, 3),
            "timestamps_monotonic_s": np.asarray(
                [item["timestamp"] for item in records], dtype=float
            ),
            "spray_poses_product": np.asarray(
                [
                    [
                        item["t_product_s"],
                        *item["nozzle_point_product"].tolist(),
                        *item["quaternion_product_xyzw"].tolist(),
                    ]
                    for item in records
                ],
                dtype=float,
            ).reshape(-1, 8),
            "face_indices": np.asarray(
                [item["face_index"] for item in records], dtype=np.int64
            ),
            "stroke_ids": np.asarray(
                [item["stroke_id"] for item in records], dtype=np.int64
            ),
            "route_length_product_units": route_length,
            "spray_length_mm": (
                float(route_timing["spray_distance_mm"]) if route_timing else 0.0
            ),
            "travel_length_mm": (
                float(route_timing["connector_distance_mm"]) if route_timing else 0.0
            ),
            "spray_duration_s": (
                float(route_timing.get("recorded_replay_duration_s", 0.0)) if route_timing else 0.0
            ),
            "travel_duration_s": (
                0.0 if route_timing else 0.0
            ),
            "tool_delay_s": (
                0.0 if route_timing else 0.0
            ),
            "estimated_duration_s": (
                float(route_timing.get("recorded_replay_duration_s", 0.0)) if route_timing else 0.0
            ),
            "route_timing": (
                {key: value for key, value in route_timing.items() if key != "timeline"}
                if route_timing else None
            ),
            "route_optimization": route_optimization,
            "route_optimization_metrics": dict(
                getattr(self, "_interactive_route_optimization_metrics", {})
            ),
            "recording_elapsed_s": (
                float(np.max([item["timestamp"] for item in records])
                      - np.min([item["timestamp"] for item in records]))
                if len(records) >= 2 else 0.0
            ),
            "surface_film_thickness_field": self._interactive_export_surface_field(),
            "face_thickness_um": (
                self._interactive_spray_face_thickness_um.copy()
                if self._interactive_spray_face_thickness_um is not None
                else None
            ),
            "product_transform_4x4": self._interactive_spray_transform.copy(),
        }
        if len(records) >= 2:
            self.interactive_recorded_path_panel.show()
            self.recorded_path_replay_button.setEnabled(True)
            self.recorded_path_status_label.setText(
                f"{len(records)} poses · {len(np.unique([r['stroke_id'] for r in records]))} strokes"
            )
            if route_timing:
                self.recorded_path_time_label.setText(
                    f"SPRAY {route_timing['spray_distance_mm']:.1f} mm  |  "
                    f"CONNECTORS {route_timing['connector_distance_mm']:.1f} mm"
                )
            self._refresh_interactive_recorded_path_actor()
        self.plotter.add_text(
            f"INTERACTIVE SPRAY PATH COMPLETE  |  {len(records)} RECORDED POSES",
            position="upper_left", font_size=10, color="#00E676",
            name="interactive_spray_complete_hud",
        )
        self.plotter.render()
        return result

    def _on_patch_mouse_press(self, interactor, event):
        if self._interactive_spray_active:
            return
        """Remember the mouse-down point; camera drags are not selections."""
        self._patch_mouse_press_position = tuple(
            interactor.GetEventPosition()
        )

    def _on_patch_mouse_release(self, interactor, event):
        if self._interactive_spray_active:
            self._patch_mouse_press_position = None
            return
        """
        Select the exact original triangle under a single left click.

        The picker is restricted to the patch-map actor, and its cell ID
        maps directly to the Stage 1 patch_labels array.
        """
        if (
            self.display_mode != "patches"
            or self.patch_labels is None
            or self._pose_actor is None
            or self._patch_mouse_press_position is None
        ):
            self._patch_mouse_press_position = None
            return

        press_x, press_y = self._patch_mouse_press_position
        self._patch_mouse_press_position = None
        release_x, release_y = interactor.GetEventPosition()

        # Ignore orbit/pan gestures; only a click (within 5 pixels) selects.
        if max(abs(release_x - press_x), abs(release_y - press_y)) > 5:
            return

        try:
            picker = self._patch_cell_picker
            picker.InitializePickList()
            picker.AddPickList(self._pose_actor)
            picker.PickFromListOn()

            renderer = self.plotter.renderer
            picked = picker.Pick(
                release_x,
                release_y,
                0,
                renderer,
            )
            if not picked:
                return

            picked_actor = picker.GetActor()
            if picked_actor != self._pose_actor:
                return

            cell_id = int(picker.GetCellId())
            if cell_id < 0 or cell_id >= len(self.patch_labels):
                return

            patch_id = int(self.patch_labels[cell_id])
            self.set_selected_patch(patch_id)
            self.patch_selected.emit(patch_id)

        except Exception as exc:
            print(
                "[PATCH PICKING WARNING] Direct cell pick failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def select_surface_patch(self, patch_id):
        """
        Compatibility interface for MainWindow.

        Select and highlight a surface patch while
        preserving the current 3D camera position.
        """
        self.set_selected_patch(patch_id)

    def get_surface_patch_ids(self):
        """Return all available surface patch IDs."""
        if self.patch_labels is None:
            return []

        return sorted(
            int(patch_id)
            for patch_id in np.unique(self.patch_labels)
        )

    def show_surface_classification(self, mesh, labels):
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError(
                "Surface classification labels must be one-dimensional."
            )

        if len(labels) != len(mesh.faces):
            raise ValueError(
                "Surface classification label count does not "
                "match the mesh triangle count."
            )

        if not np.all(np.isin(labels, [0, 1, 2, 3])):
            raise ValueError(
                "Surface classification contains unknown values."
            )

        self.current_mesh = self._prepare_product_mesh(mesh)

        self.face_status_labels = labels.astype(
            np.uint8, copy=True
        )

        self.patch_labels = None
        self.trajectory_face_patch_labels = None
        self.trajectory_data = None
        self.face_surface_types = None
        self.selected_patch_id = None

        self.display_mode = "classification"
        self._set_trajectory_controls_visible(False)
        self.pose_control_panel.hide()

        self._render_product(reset_camera=True)

    def show_spray_point_preview(
        self,
        mesh,
        nozzle_points,
        surface_points=None,
        surface_normals=None,
        spray_directions=None,
        standoff=None,
        cone_angle_deg=None,
        product_transform=None,
    ):
        """Show nozzle locations and mesh-intersected spray footprints."""
        points = np.asarray(nozzle_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("nozzle_points must be a non-empty (N, 3) array.")
        if not np.all(np.isfinite(points)):
            raise ValueError("nozzle_points contains NaN or infinite values.")

        transform = np.eye(4, dtype=np.float64) if product_transform is None else np.asarray(
            product_transform, dtype=np.float64
        )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("Product transform must be a finite 4x4 matrix.")
        rotation = transform[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        ):
            raise ValueError("Product transform must be a rigid homogeneous transform.")

        self.current_mesh = self._prepare_product_mesh(mesh)
        self.trajectory_data = None
        self.patch_labels = None
        self.face_status_labels = None
        self.face_surface_types = None
        self.trajectory_safety = {}
        self.display_mode = "spray_point_preview"
        self.trajectory_product_transform = transform.copy()
        self._last_product_transform = transform.copy()
        self._set_trajectory_controls_visible(False)
        self.pose_control_panel.hide()
        self._render_product(reset_camera=False)

        if all(value is not None for value in (
            surface_points, surface_normals, spray_directions, standoff, cone_angle_deg
        )):
            footprints = _build_spray_footprint_overlay(
                self.current_mesh,
                points,
                spray_directions,
                float(cone_angle_deg),
            )
            for footprint_mesh, boundary_mesh in footprints:
                if footprint_mesh.n_cells:
                    footprint_mesh.points = (
                        np.asarray(footprint_mesh.points) @ rotation.T + transform[:3, 3]
                    )
                    self.plotter.add_mesh(
                        footprint_mesh,
                        color="#FFC857",
                        opacity=0.13,
                        smooth_shading=False,
                        lighting=False,
                        show_edges=False,
                        pickable=False,
                    )
                if boundary_mesh.n_cells:
                    boundary_mesh.points = (
                        np.asarray(boundary_mesh.points) @ rotation.T + transform[:3, 3]
                    )
                    self.plotter.add_mesh(
                        boundary_mesh,
                        color="#FF9F1C",
                        opacity=0.95,
                        line_width=2.0,
                        lighting=False,
                        render_lines_as_tubes=False,
                        pickable=False,
                    )

        world_points = points @ rotation.T + transform[:3, 3]
        self.plotter.add_points(
            world_points,
            color="#00E676",
            point_size=8,
            render_points_as_spheres=True,
            pickable=False,
        )
        self.plotter.add_text(
            f"OPTIMIZED SPRAY POINTS | {len(points)} NOZZLE LOCATIONS",
            position="upper_left",
            font_size=10,
            color="#00E676",
            shadow=True,
        )
        self.plotter.render()

    def show_spray_trajectory(
        self,
        mesh,
        trajectory_data,
        face_surface_types=None,
        face_status_labels=None,
        patch_labels=None,
    ):
        if not isinstance(trajectory_data, dict):
            raise TypeError(
                "trajectory_data must be a dictionary."
            )

        required = [
            "surface_points",
            "surface_normals",
            "nozzle_points",
            "spray_directions",
            "face_indices",
            "patch_ids",
            "segment_ids",
        ]

        data = {}

        for key in required:
            if key not in trajectory_data:
                raise ValueError(
                    f"Trajectory data is missing '{key}'."
                )
            data[key] = np.asarray(trajectory_data[key])

        vector_keys = [
            "surface_points",
            "surface_normals",
            "nozzle_points",
            "spray_directions",
        ]

        count = None

        for key in vector_keys:
            array = np.asarray(data[key], dtype=float)

            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(
                    f"{key} must have shape (N, 3)."
                )

            if count is None:
                count = len(array)

            if len(array) != count:
                raise ValueError(
                    "Trajectory vector arrays have different lengths."
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{key} contains NaN or infinite values."
                )

            data[key] = array.copy()

        if count is None or count == 0:
            raise ValueError(
                "Trajectory contains no valid sample points."
            )

        for key in ["face_indices", "patch_ids", "segment_ids"]:
            array = np.asarray(data[key])

            if array.ndim != 1 or len(array) != count:
                raise ValueError(
                    f"{key} must contain exactly {count} values."
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{key} contains invalid values."
                )

            data[key] = array.copy()

        face_indices = data["face_indices"].astype(np.int64)

        if np.any(face_indices < 0) or np.any(
            face_indices >= len(mesh.faces)
        ):
            raise ValueError(
                "Trajectory contains invalid mesh face indices."
            )

        data["face_indices"] = face_indices

        if "nozzle_frames" in trajectory_data:
            frames = np.asarray(trajectory_data["nozzle_frames"], dtype=float)
            if frames.shape != (count, 3, 3) or not np.all(np.isfinite(frames)):
                raise ValueError("nozzle_frames must have finite shape (N, 3, 3).")
            frame_gram = np.einsum("nji,njk->nik", frames, frames)
            if not np.allclose(
                frame_gram,
                np.repeat(np.eye(3)[None, :, :], count, axis=0),
                atol=1e-5,
            ):
                raise ValueError("nozzle_frames must contain orthonormal frames.")
            if not np.allclose(np.linalg.det(frames), 1.0, atol=1e-5):
                raise ValueError("nozzle_frames must be right-handed.")
            data["nozzle_frames"] = frames.copy()

        parameters = trajectory_data.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        transform = np.asarray(
            parameters.get("product_transform_4x4", np.eye(4)),
            dtype=float,
        )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("Trajectory product transform must be finite and 4x4.")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("Trajectory product transform is not homogeneous.")
        rotation = transform[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ):
            raise ValueError("Trajectory product transform must contain a rigid rotation.")
        data["parameters"] = parameters
        safety_data = trajectory_data.get("trajectory_safety", {})
        if not isinstance(safety_data, dict):
            safety_data = {}
        sample_status = safety_data.get("sample_status")
        if sample_status is not None:
            sample_status = np.asarray(sample_status, dtype=str)
            if sample_status.shape != (count,):
                raise ValueError(
                    "trajectory_safety.sample_status must contain one status per trajectory sample."
                )
            safety_data = dict(safety_data)
            safety_data["sample_status"] = sample_status.tolist()
        self.trajectory_safety = dict(safety_data)
        data["trajectory_safety"] = self.trajectory_safety
        self._update_trajectory_safety_label()
        self._trajectory_timing_stale = False
        travel_plan = trajectory_data.get("inter_segment_travel_plan", {})
        data["inter_segment_travel_plan"] = (
            travel_plan if isinstance(travel_plan, dict) else {}
        )
        self.trajectory_path_statistics = trajectory_data.get(
            "path_statistics", {}
        )
        self.trajectory_validation = trajectory_data.get(
            "trajectory_validation", {}
        )
        self._update_trajectory_path_statistics_label()
        self._update_trajectory_validation_label()
        self.trajectory_product_transform = transform.copy()
        self._last_product_transform = transform.copy()

        if "sample_indices" in trajectory_data:
            sample_indices = np.asarray(
                trajectory_data["sample_indices"]
            )

            if (
                sample_indices.ndim != 1
                or len(sample_indices) != count
            ):
                raise ValueError(
                    "sample_indices must contain N values."
                )

            data["sample_indices"] = sample_indices.copy()
        else:
            data["sample_indices"] = np.arange(
                count, dtype=np.int64
            )

        self.current_mesh = self._prepare_product_mesh(mesh)

        self.patch_labels = None
        if patch_labels is None:
            self.trajectory_face_patch_labels = None
        else:
            stage1_patch_labels = np.asarray(patch_labels)
            if (
                stage1_patch_labels.ndim != 1
                or len(stage1_patch_labels) != len(mesh.faces)
                or not np.all(np.isfinite(stage1_patch_labels))
            ):
                raise ValueError(
                    "patch_labels must contain one finite Stage 1 patch ID per mesh face."
                )
            self.trajectory_face_patch_labels = stage1_patch_labels.astype(
                np.int64, copy=True
            )
        if face_status_labels is None:
            self.face_status_labels = None
        else:
            status_labels = np.asarray(face_status_labels)
            if (
                status_labels.ndim != 1
                or len(status_labels) != len(mesh.faces)
                or not np.all(np.isin(status_labels, [0, 1, 2, 3]))
            ):
                raise ValueError(
                    "face_status_labels must contain one valid Stage 2 status per mesh face."
                )
            self.face_status_labels = status_labels.astype(np.uint8, copy=True)
        self.selected_patch_id = None
        self.trajectory_data = data
        self._correction_preview_appended = False
        self.trajectory_correction_apply_button.setEnabled(False)
        self._update_trajectory_path_statistics_label()
        self._stop_trajectory_playback(reset=True)
        self.trajectory_playback_slider.blockSignals(True)
        self.trajectory_playback_slider.setRange(0, max(0, count - 1))
        self.trajectory_playback_slider.setValue(0)
        self.trajectory_playback_slider.blockSignals(False)
        self.trajectory_playback_label.setText(f"0 / {count}")

        if face_surface_types is not None:
            surface_types = np.asarray(
                face_surface_types, dtype=str
            )

            if (
                surface_types.ndim != 1
                or len(surface_types) != len(mesh.faces)
            ):
                raise ValueError(
                    "face_surface_types must contain one value "
                    "for every mesh face."
                )

            self.face_surface_types = np.char.lower(
                surface_types.copy()
            )
        else:
            self.face_surface_types = None

        self.display_mode = "trajectory"
        # Keep the interface at the trajectory-generation endpoint. The 3D
        # trajectory remains visible, but post-generation controls are hidden.
        self._set_trajectory_controls_visible(False)
        self.pose_control_panel.hide()

        self._render_product(reset_camera=True)

    # =========================================================
    # STAGE 3 SURFACE VISIBILITY
    # =========================================================

    def _get_trajectory_surface_mask(self):
        face_count = len(self.current_mesh.faces)

        planar_visible = self.trajectory_visibility["planar"]
        curved_visible = self.trajectory_visibility["curved"]

        if planar_visible and curved_visible:
            return np.ones(face_count, dtype=bool)

        if not planar_visible and not curved_visible:
            return np.zeros(face_count, dtype=bool)

        if self.face_surface_types is None:
            return np.ones(face_count, dtype=bool)

        if planar_visible:
            return self.face_surface_types == "planar"

        return self.face_surface_types == "curved"

    def _update_trajectory_surface_visibility(self):
        """Update planar/curved styling in-place without restarting playback."""
        polydata = self._trajectory_surface_polydata
        if polydata is None:
            self.plotter.render()
            return

        face_count = len(self.current_mesh.faces)
        if self.face_surface_types is None:
            surface_types = np.full(face_count, "unknown", dtype="<U16")
        else:
            surface_types = np.char.lower(
                np.asarray(self.face_surface_types).astype(str)
            )
            if len(surface_types) != face_count:
                return

        base_rgb = np.tile(
            np.asarray([178, 188, 195], dtype=np.uint8),
            (face_count, 1),
        )
        face_opacity = np.ones(face_count, dtype=np.float32)
        planar_mask = surface_types == "planar"
        curved_mask = surface_types == "curved"
        classified_mask = planar_mask | curved_mask
        planar_selected = bool(self.trajectory_visibility["planar"])
        curved_selected = bool(self.trajectory_visibility["curved"])

        # Match the normal Stage 3 appearance rules, changing only face styling.
        if planar_selected and curved_selected:
            pass
        elif planar_selected:
            base_rgb[planar_mask] = [75, 155, 220]
            face_opacity[classified_mask & ~planar_mask] = 0.20
        elif curved_selected:
            base_rgb[curved_mask] = [230, 150, 75]
            face_opacity[classified_mask & ~curved_mask] = 0.20

        if self.face_status_labels is not None:
            status = np.asarray(self.face_status_labels, dtype=np.uint8)
            base_rgb[status == FACE_EXCLUDED] = [89, 99, 107]
            # A paintable candidate remains unpainted until spray samples hit it.
            base_rgb[status == FACE_PAINTABLE_CANDIDATE] = [178, 188, 195]
            base_rgb[status == FACE_REQUIRES_INSPECTION] = [255, 215, 64]
            base_rgb[status == FACE_INVALID] = [255, 82, 82]

        self._trajectory_surface_base_rgb = base_rgb
        self._trajectory_surface_base_alpha = np.clip(
            np.rint(face_opacity * 255.0), 0, 255
        ).astype(np.uint8)

        # Rebuild the current face colors from accumulated coverage so changing
        # the view does not erase paint already applied during the simulation.
        colors = base_rgb.copy()
        cache = self._playback_cache
        if cache is not None and cache.get("coverage_sample_hits") is not None:
            face_coverage = cache["coverage_sample_hits"].reshape(
                -1, cache["sample_count_per_face"]
            ).mean(axis=1)
            face_fraction = np.zeros(face_count, dtype=float)
            valid_display = cache["target_display_indices"] >= 0
            face_fraction[cache["target_display_indices"][valid_display]] = (
                face_coverage[valid_display]
            )
            painted = (
                base_rgb.astype(float) * (1.0 - face_fraction[:, None])
                + cache["paint_rgb"][None, :] * face_fraction[:, None]
            )
            colors = np.clip(np.rint(painted), 0, 255).astype(np.uint8)

        self._set_trajectory_surface_rgb(colors)
        mapper = self._pose_actor.GetMapper() if self._pose_actor is not None else None
        if mapper is not None:
            mapper.Update()
        self.plotter.render()

    # =========================================================
    # STAGE 3 TRAJECTORY RENDERING
    # =========================================================

    def _add_trajectory_safety_overlay(self):
        """Render sample safety states on their original PRODUCT_CAD faces."""
        self._trajectory_safety_actor = None
        data = self.trajectory_data
        safety = self.trajectory_safety
        if not isinstance(data, dict) or not isinstance(safety, dict):
            return
        sample_status = safety.get("sample_status")
        if sample_status is None:
            return

        face_status = aggregate_sample_safety_by_face(
            data["face_indices"], sample_status, len(self.current_mesh.faces)
        )
        display_face_ids = np.flatnonzero(face_status > 0)
        if len(display_face_ids) == 0:
            return

        # Offset the diagnostic membrane slightly along source-face normals to
        # avoid z-fighting. CAD-to-World transform is display-only.
        triangles = np.asarray(self.current_mesh.triangles, dtype=float)[display_face_ids].copy()
        normals = np.asarray(self.current_mesh.face_normals, dtype=float)[display_face_ids]
        diagonal = float(np.linalg.norm(np.ptp(self.current_mesh.vertices, axis=0)))
        offset = max(diagonal * 1e-6, 1e-7)
        triangles += normals[:, None, :] * offset
        transform = self.trajectory_product_transform
        world_triangles = triangles @ transform[:3, :3].T + transform[:3, 3]
        points = world_triangles.reshape(-1, 3)
        cell_count = len(display_face_ids)
        faces = np.column_stack((
            np.full(cell_count, 3, dtype=np.int64),
            np.arange(cell_count, dtype=np.int64) * 3,
            np.arange(cell_count, dtype=np.int64) * 3 + 1,
            np.arange(cell_count, dtype=np.int64) * 3 + 2,
        )).reshape(-1)
        polydata = pv.PolyData(points, faces)
        rgba = np.zeros((cell_count, 4), dtype=np.uint8)
        local_status = face_status[display_face_ids]
        rgba[local_status == 1] = [40, 220, 120, 150]
        rgba[local_status == 2] = [255, 190, 35, 175]
        rgba[local_status == 3] = [255, 55, 55, 210]
        polydata.cell_data["Trajectory Safety RGBA"] = rgba
        self._trajectory_safety_actor = self.plotter.add_mesh(
            polydata,
            scalars="Trajectory Safety RGBA",
            rgba=True,
            preference="cell",
            lighting=False,
            show_edges=False,
            pickable=False,
        )
        self._trajectory_safety_actor.SetVisibility(
            bool(self.trajectory_visibility.get("safety", True))
        )

    def update_spray_standoff_preview(self, standoff_mm):
        """
        Move the displayed nozzle samples to the newly configured stand-off.

        This updates only the viewer's preview data. The generated Stage 3
        result remains unchanged and is correctly marked outdated by the
        owning workflow when its parameters no longer match.
        """
        if self.trajectory_data is None or self.display_mode != "trajectory":
            return False

        distance_mm = float(standoff_mm)
        if not math.isfinite(distance_mm) or distance_mm <= 0.0:
            raise ValueError("Spray working distance must be finite and positive.")

        mm_per_unit = MM_PER_UNIT.get(self.source_unit, 1.0)
        distance_coordinate_units = distance_mm / mm_per_unit
        surface_points = np.asarray(
            self.trajectory_data["surface_points"], dtype=np.float64
        )
        surface_normals = np.asarray(
            self.trajectory_data["surface_normals"], dtype=np.float64
        )
        normal_lengths = np.linalg.norm(surface_normals, axis=1)
        if (
            surface_points.ndim != 2
            or surface_points.shape[1] != 3
            or surface_normals.shape != surface_points.shape
            or not np.all(np.isfinite(surface_points))
            or not np.all(np.isfinite(surface_normals))
            or np.any(normal_lengths <= 1e-12)
        ):
            raise ValueError("Cannot update spray points: invalid surface geometry.")

        unit_normals = surface_normals / normal_lengths[:, None]
        self.trajectory_data["nozzle_points"] = (
            surface_points + distance_coordinate_units * unit_normals
        )

        parameters = dict(self.trajectory_data.get("parameters", {}))
        parameters["standoff_mm"] = distance_mm
        parameters["standoff_coordinate_units"] = distance_coordinate_units
        profile = dict(parameters.get("sprayer_profile", {}))
        profile["standoff_mm"] = distance_mm
        parameters["sprayer_profile"] = profile
        self.trajectory_data["parameters"] = parameters

        # Cached playback geometry is based on the previous nozzle locations.
        self._playback_cache = None
        self._stop_trajectory_playback(reset=True)
        self._render_product(reset_camera=False)
        return True

    def _add_trajectory_visualization(self):
        """Display independent nozzle spray points at the configured stand-off."""
        data = self.trajectory_data
        if data is None:
            return

        transform = self.trajectory_product_transform
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        # Stage 3 already calculates nozzle_points as
        # surface_points + standoff * surface_normals in source CAD units.
        # Display those nozzle locations, not the target surface samples.
        # Keep each point independent: no paths, connectors, footprints,
        # arrows, orientation frames, or gun geometry in this simple preview.
        spray_points = (
            np.asarray(data["nozzle_points"], dtype=float) @ rotation.T
            + translation
        )
        if len(spray_points) == 0:
            return

        self.plotter.add_points(
            spray_points,
            color="#00E676",
            point_size=7,
            render_points_as_spheres=True,
            pickable=False,
        )

    @staticmethod
    def _format_playback_duration(seconds):
        """Format a non-negative duration as MM:SS or HH:MM:SS."""
        value = max(0, int(round(float(seconds))))
        hours, remainder = divmod(value, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _pause_playback_wall_clock(self):
        started_at = self._playback_wall_started_at
        if started_at is not None:
            self._playback_wall_elapsed_s += max(
                0.0, time.monotonic() - started_at
            )
            self._playback_wall_started_at = None

    def _update_trajectory_time_label(self, index=None):
        """Show point progress only; robot timing is intentionally unconfigured."""
        label = getattr(self, "trajectory_time_label", None)
        if label is None:
            return
        total = 0
        if isinstance(self.trajectory_data, dict):
            total = len(self.trajectory_data.get("nozzle_points", []))
        current = self._playback_index if index is None else int(index)
        label.setText(
            f"VISUAL POINT {max(0, current) + 1:,} / {total:,} "
            "| ROBOT TIMING NOT CONFIGURED"
        )

    def is_trajectory_playing(self):
        """Return True only while the trajectory playback timer is running."""
        return bool(
            hasattr(self, "_playback_timer")
            and self._playback_timer.isActive()
        )

    def _notify_playback_state_changed(self):
        """Notify the owning workflow window when playback starts or stops."""
        callback = getattr(self, "playback_state_changed_callback", None)
        if callable(callback):
            callback(self.is_trajectory_playing())

    def _stop_trajectory_playback(self, reset=False):
        if hasattr(self, "_playback_timer"):
            self._playback_timer.stop()
        self._pause_playback_wall_clock()
        self._notify_playback_state_changed()
        if hasattr(self, "trajectory_play_button"):
            self.trajectory_play_button.setText("▶")
        self._notify_playback_state_changed()
        if reset:
            actors = list(self._playback_gun_actors)
            actors.extend((
                self._playback_path_actor,
                self._playback_nozzle_actor,
                self._playback_spray_actor,
                self._playback_spray_sheet_actor,
            ))
            seen = set()
            for actor in actors:
                if actor is None or id(actor) in seen:
                    continue
                seen.add(id(actor))
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
            self._playback_path_actor = None
            self._playback_nozzle_actor = None
            self._playback_gun_actors = []
            self._playback_spray_actor = None
            self._playback_spray_polydata = None
            self._playback_spray_sheet_actor = None
            self._playback_index = 0
            self._playback_elapsed_s = 0.0
            self._playback_wall_elapsed_s = 0.0
            self._playback_wall_started_at = None
            self._playback_finished = False
            self._playback_last_path_render_index = -1
            self._playback_cache = None
            if hasattr(self, "trajectory_coverage_bar"):
                self.trajectory_coverage_bar.setValue(0)
            if hasattr(self, "trajectory_coverage_label"):
                self.trajectory_coverage_label.setText("0.0% | 0.000 / 0.000 area")
            if hasattr(self, "trajectory_thickness_label"):
                self.trajectory_thickness_label.setText(
                    "PREDICTED DRY FILM THICKNESS | 0.00 µm — RESET"
                )
            if hasattr(self, "trajectory_correction_label"):
                self.trajectory_correction_label.setText(
                    "COVERAGE CORRECTION | RESET — RUN PLAYBACK TO ANALYZE"
                )
            if hasattr(self, "trajectory_spray_state_label"):
                self.trajectory_spray_state_label.setText("● SPRAY OFF")
            self._restore_trajectory_surface_base_colors()
            if hasattr(self, "trajectory_playback_slider"):
                self.trajectory_playback_slider.blockSignals(True)
                self.trajectory_playback_slider.setValue(0)
                self.trajectory_playback_slider.blockSignals(False)
            if hasattr(self, "trajectory_playback_label"):
                total = len(self.trajectory_data["nozzle_points"]) if self.trajectory_data is not None else 0
                self.trajectory_playback_label.setText(f"0 / {total}")

    def update_coating_process_preview(self, coating_process, timing_stale=False):
        """Apply saved process settings to the loaded trajectory without restarting the app."""
        if not isinstance(coating_process, dict):
            raise TypeError("coating_process must be a dictionary.")
        if self.trajectory_data is None:
            self._update_film_thickness_legend(
                coating_process.get("target_dry_film_thickness_um", 0.0)
            )
            return
        parameters = dict(self.trajectory_data.get("parameters", {}))
        parameters["coating_process"] = dict(coating_process)
        self.trajectory_data["parameters"] = parameters
        self._trajectory_timing_stale = bool(timing_stale)
        self._stop_trajectory_playback(reset=True)
        self._update_film_thickness_legend(
            coating_process.get("target_dry_film_thickness_um", 0.0)
        )
        self._update_trajectory_playback_visuals(0, update_coverage=False)
        if (
            hasattr(self, "trajectory_thickness_checkbox")
            and self.trajectory_thickness_checkbox.isChecked()
        ):
            self._update_film_thickness_visualization()
        self._update_trajectory_time_label(0)

    def _prepare_trajectory_playback_cache(self):
        if self.trajectory_data is None:
            return None
        if self._playback_cache is not None:
            return self._playback_cache

        data = self.trajectory_data
        transform = self.trajectory_product_transform
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        parameters = data.get("parameters", {})
        profile = parameters.get("sprayer_profile", {})
        if not isinstance(profile, dict):
            profile = {}

        nozzle_points = data["nozzle_points"] @ rotation.T + translation
        spray_directions = data["spray_directions"] @ rotation.T
        direction_lengths = np.linalg.norm(spray_directions, axis=1)
        valid_directions = direction_lengths > 1e-12
        spray_directions[valid_directions] /= direction_lengths[valid_directions, None]

        if "nozzle_frames" in data:
            nozzle_frames = np.einsum(
                "ij,njk->nik", rotation, data["nozzle_frames"]
            )
        else:
            nozzle_frames = np.repeat(np.eye(3)[None, :, :], len(nozzle_points), axis=0)
            for index, direction in enumerate(spray_directions):
                z_axis = direction / max(float(np.linalg.norm(direction)), 1e-12)
                reference = np.array([1.0, 0.0, 0.0])
                x_axis = reference - np.dot(reference, z_axis) * z_axis
                if np.linalg.norm(x_axis) < 1e-8:
                    reference = np.array([0.0, 1.0, 0.0])
                    x_axis = reference - np.dot(reference, z_axis) * z_axis
                x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
                nozzle_frames[index] = np.column_stack((x_axis, y_axis, z_axis))

        face_count = len(self.current_mesh.faces)
        if self.face_status_labels is None:
            # Fail closed: trajectory-referenced faces are NOT a valid substitute
            # for the complete Stage 2-approved target surface.
            target_face_mask = np.zeros(face_count, dtype=bool)
            target_mask_status = "MISSING_STAGE2_FACE_STATUS"
        else:
            target_face_mask = build_paintable_face_mask(
                self.face_status_labels, face_count
            )
            target_mask_status = "READY"
        target_face_ids = np.flatnonzero(target_face_mask)
        coating_process = parameters.get("coating_process", {})
        try:
            deposition_readiness, deposition_missing_inputs = coating_model_readiness(
                coating_process
            )
        except (TypeError, ValueError):
            deposition_readiness, deposition_missing_inputs = (
                "INPUTS_INCOMPLETE", ["valid coating process parameters"]
            )
        faces = np.asarray(self.current_mesh.faces, dtype=np.int64)
        vertices = np.asarray(self.current_mesh.vertices, dtype=float)
        triangles = vertices[faces[target_face_ids]]
        # Four deterministic samples per triangle (three edge midpoints
        # and centroid) replace the former seven-point sampling grid.
        barycentric = np.asarray([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ], dtype=float)
        triangles_world = triangles @ rotation.T + translation
        face_samples = np.einsum("sv,fvc->fsc", barycentric, triangles).reshape(-1, 3)
        face_samples = face_samples @ rotation.T + translation

        face_normals = np.asarray(self.current_mesh.face_normals, dtype=float)[target_face_ids]
        face_normals = face_normals @ rotation.T
        face_normals /= np.maximum(
            np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12
        )
        sample_normals = np.repeat(face_normals, len(barycentric), axis=0)
        face_areas = np.asarray(self.current_mesh.area_faces, dtype=float)[target_face_ids]
        sample_count_per_face = len(barycentric)
        target_area = float(np.sum(face_areas))

        try:
            from scipy.spatial import cKDTree
            surface_tree = cKDTree(face_samples)
        except Exception:
            surface_tree = None

        # Build a spatially accelerated locator over the COMPLETE product
        # mesh. Non-target faces still physically occlude the spray.
        mesh_faces = np.asarray(self.current_mesh.faces, dtype=np.int64)
        mesh_cell_array = np.column_stack((
            np.full(len(mesh_faces), 3, dtype=np.int64),
            mesh_faces,
        )).reshape(-1)
        mesh_vertices_world = vertices @ rotation.T + translation
        occlusion_polydata = pv.PolyData(
            mesh_vertices_world, mesh_cell_array
        )
        occlusion_locator = vtkOBBTree()
        occlusion_locator.SetDataSet(occlusion_polydata)
        occlusion_locator.BuildLocator()
        mesh_bounds = np.asarray(self.current_mesh.bounds, dtype=np.float64)
        mesh_diagonal = float(np.linalg.norm(mesh_bounds[1] - mesh_bounds[0]))
        ray_tolerance = max(mesh_diagonal * 1e-7, 1e-8)

        source_unit = self.source_unit
        mm_per_unit = MM_PER_UNIT.get(source_unit, 1.0)
        standoff_units = float(parameters.get(
            "standoff_coordinate_units",
            float(profile.get("standoff_mm", parameters.get("standoff_mm", 180.0))) / mm_per_unit,
        ))
        cone_angle = float(profile.get("spray_cone_angle_deg", 0.0) or 0.0)
        cone_angle = float(np.clip(cone_angle, 0.0, 150.0))
        half_angle = math.radians(cone_angle * 0.5)
        cone_tangent = math.tan(half_angle)
        cone_cosine = max(math.cos(half_angle), 0.2)
        pattern_type = "round_cone"
        standoff_mm = standoff_units * mm_per_unit
        footprint_width_units = 2.0 * standoff_units * cone_tangent
        footprint_thickness_units = footprint_width_units

        paint_color_text = str(
            self._trajectory_paint_color_hex or profile.get("paint_color", "#D95B43")
        ).strip()
        if not paint_color_text.startswith("#"):
            paint_color_text = "#" + paint_color_text
        try:
            paint_rgb = np.asarray([
                int(paint_color_text.lstrip("#")[offset:offset + 2], 16)
                for offset in (0, 2, 4)
            ], dtype=float)
        except (ValueError, IndexError):
            paint_rgb = np.asarray([217.0, 91.0, 67.0], dtype=float)
            paint_color_text = "#D95B43"
        self._trajectory_paint_color_hex = paint_color_text

        display_face_ids = (
            np.flatnonzero(self._trajectory_surface_face_mask)
            if self._trajectory_surface_face_mask is not None
            else np.empty(0, dtype=np.int64)
        )
        face_to_display_index = np.full(len(self.current_mesh.faces), -1, dtype=np.int64)
        face_to_display_index[display_face_ids] = np.arange(len(display_face_ids), dtype=np.int64)
        target_display_indices = face_to_display_index[target_face_ids]

        segment_ids = np.asarray(data["segment_ids"], dtype=np.int64)
        spray_on_mask = np.ones(len(segment_ids), dtype=bool)
        if len(segment_ids) > 1:
            spray_on_mask[1:] = segment_ids[1:] == segment_ids[:-1]
        # Precompute spray-OFF boundaries once; playback ticks must not scan
        # the complete trajectory to rediscover the same indices.
        spray_off_boundary_indices = np.flatnonzero(~spray_on_mask)

        self._playback_cache = {
            "nozzle_points": nozzle_points,
            "spray_directions": spray_directions,
            "nozzle_frames": nozzle_frames,
            "segment_ids": segment_ids,
            "spray_on_mask": spray_on_mask,
            "spray_off_boundary_indices": spray_off_boundary_indices,
            "face_samples": face_samples,
            "sample_normals": sample_normals,
            "face_areas": face_areas,
            "sample_area_mm2": np.repeat(face_areas / sample_count_per_face, sample_count_per_face) * (mm_per_unit ** 2),
            "target_face_ids": target_face_ids,
            "target_display_indices": target_display_indices,
            "sample_patch_ids": (
                np.repeat(
                    self.trajectory_face_patch_labels[target_face_ids],
                    sample_count_per_face,
                )
                if self.trajectory_face_patch_labels is not None
                else np.full(len(face_samples), -1, dtype=np.int64)
            ),
            "coverage_correction_plan": None,
            "target_area": target_area,
            "target_mask_status": target_mask_status,
            "coating_process": dict(coating_process) if isinstance(coating_process, dict) else {},
            "deposition_readiness": deposition_readiness,
            "deposition_missing_inputs": list(deposition_missing_inputs),
            "deposition_sample_thickness_um": np.zeros(len(face_samples), dtype=np.float64),
            "deposition_sample_volume_mm3": np.zeros(len(face_samples), dtype=np.float64),
            "deposition_face_thickness_um": np.zeros(face_count, dtype=np.float64),
            "deposition_processed_index": -1,
            "deposition_input_volume_mm3": 0.0,
            "deposition_total_volume_mm3": 0.0,
            "area_display_scale": (
                mm_per_unit ** 2 / 1_000_000.0
                if source_unit in MM_PER_UNIT else 1.0
            ),
            "area_display_unit": (
                "m²" if source_unit in MM_PER_UNIT else "coordinate units²"
            ),
            "surface_tree": surface_tree,
            "occlusion_polydata": occlusion_polydata,
            "occlusion_locator": occlusion_locator,
            "ray_intersection_points": vtkPoints(),
            "ray_intersection_cell_ids": vtkIdList(),
            "ray_tolerance": ray_tolerance,
            "last_hit_path_index": -1,
            "last_hit_sample_indices": np.empty(0, dtype=np.int64),
            "target_triangles_world": triangles_world,
            "coverage_sample_hits": np.zeros(len(face_samples), dtype=bool),
            "coverage_processed_index": -1,
            "coverage_render_stride": 4,
            "coverage_last_render_index": -1,
            "path_render_stride": max(1, int(np.ceil(len(nozzle_points) / 500.0))),
            "sample_count_per_face": sample_count_per_face,
            "cone_tangent": cone_tangent,
            "cone_cosine": cone_cosine,
            "standoff_units": max(standoff_units, 1e-9),
            "pattern_type": pattern_type,
            "footprint_width_units": footprint_width_units,
            "footprint_thickness_units": footprint_thickness_units,
            "footprint_width_mm": footprint_width_units * mm_per_unit,
            "footprint_thickness_mm": footprint_thickness_units * mm_per_unit,
            "paint_rgb": paint_rgb,
            "profile": dict(profile),
            "mm_per_unit": mm_per_unit,
        }
        if (
            hasattr(self, "trajectory_thickness_legend_label")
            and deposition_readiness == "INPUTS_READY_UNCALIBRATED"
        ):
            self._update_film_thickness_legend(
                coating_process.get(
                    "target_dry_film_thickness_um",
                    DEFAULT_COATING_PROCESS["target_dry_film_thickness_um"],
                )
            )
        return self._playback_cache

    def _set_trajectory_surface_rgb(self, rgb):
        """Update paint-preview RGB while preserving each face's alpha."""
        polydata = self._trajectory_surface_polydata
        alpha = self._trajectory_surface_base_alpha
        if polydata is None or alpha is None:
            return
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape != (len(alpha), 3):
            return
        rgba = np.column_stack((rgb, alpha)).astype(np.uint8, copy=False)
        polydata.cell_data["Paint Preview RGBA"] = rgba
        array = polydata.GetCellData().GetArray("Paint Preview RGBA")
        if array is not None:
            array.Modified()
        polydata.Modified()

    def _restore_trajectory_surface_base_colors(self):
        base_rgb = self._trajectory_surface_base_rgb
        if base_rgb is None:
            return
        try:
            self._set_trajectory_surface_rgb(base_rgb)
            if self._pose_actor is not None:
                self._pose_actor.GetMapper().Update()
        except Exception:
            pass

    def _get_surface_sample_hits_from_spray(self, path_index):
        """Return visible mesh samples inside the spray footprint.

        The footprint test is only the geometric envelope. A VTK OBB-tree
        ray query then rejects samples whose nozzle-to-surface segment
        intersects any nearer product triangle, including non-paintable faces.
        """
        cache = self._prepare_trajectory_playback_cache()
        if cache is None:
            return np.empty(0, dtype=np.int64)
        if cache["last_hit_path_index"] == int(path_index):
            return cache["last_hit_sample_indices"].copy()
        if cache["cone_tangent"] <= 0.0:
            return np.empty(0, dtype=np.int64)
        nozzle = cache["nozzle_points"][path_index]
        direction = cache["spray_directions"][path_index]
        if not np.all(np.isfinite(nozzle)) or np.linalg.norm(direction) <= 1e-12:
            return np.empty(0, dtype=np.int64)

        max_axial = cache["standoff_units"] / cache["cone_cosine"] * 1.15
        search_radius = max_axial / cache["cone_cosine"] * 1.02
        tree = cache["surface_tree"]
        if tree is not None:
            candidates = np.asarray(tree.query_ball_point(nozzle, search_radius), dtype=np.int64)
        else:
            delta_all = cache["face_samples"] - nozzle
            candidates = np.flatnonzero(
                np.einsum("ij,ij->i", delta_all, delta_all) <= search_radius * search_radius
            )
        if len(candidates) == 0:
            cache["last_hit_path_index"] = int(path_index)
            cache["last_hit_sample_indices"] = np.empty(0, dtype=np.int64)
            return np.empty(0, dtype=np.int64)

        delta = cache["face_samples"][candidates] - nozzle
        axial = delta @ direction
        frame = cache["nozzle_frames"][path_index]
        local_x = delta @ frame[:, 0]
        local_y = delta @ frame[:, 1]
        radial = np.sqrt(local_x * local_x + local_y * local_y)
        inside_pattern = radial <= axial * cache["cone_tangent"]

        facing = np.einsum(
            "ij,j->i", cache["sample_normals"][candidates], -direction
        ) > 0.02
        inside = (
            (axial > 0.0) & (axial <= max_axial)
            & inside_pattern & facing
        )
        footprint_candidates = candidates[inside]
        if len(footprint_candidates) == 0:
            visible = np.empty(0, dtype=np.int64)
        else:
            locator = cache["occlusion_locator"]
            tolerance = cache["ray_tolerance"]
            visible_indices = []
            # These values are invariant for every candidate ray in this
            # playback frame. Resolve them once rather than once per sample.
            ray_origin = tuple(float(v) for v in nozzle)
            intersection_points = cache["ray_intersection_points"]
            intersection_cell_ids = cache["ray_intersection_cell_ids"]
            for sample_index in footprint_candidates:
                target = cache["face_samples"][sample_index]
                intersection_points.SetNumberOfPoints(0)
                intersection_cell_ids.SetNumberOfIds(0)
                locator.IntersectWithLine(
                    ray_origin,
                    tuple(float(v) for v in target),
                    intersection_points,
                    intersection_cell_ids,
                )
                intersection_count = intersection_points.GetNumberOfPoints()
                if intersection_count:
                    # Read the reusable VTK buffer as a NumPy view. This avoids
                    # allocating one Python tuple per intersection on every ray.
                    intersections = vtk_to_numpy(
                        intersection_points.GetData()
                    )[:intersection_count]
                else:
                    intersections = np.empty((0, 3), dtype=np.float64)
                if not is_line_of_sight_occluded(
                    intersections, nozzle, target, tolerance
                ):
                    visible_indices.append(int(sample_index))
            visible = np.asarray(visible_indices, dtype=np.int64)

        cache["last_hit_path_index"] = int(path_index)
        cache["last_hit_sample_indices"] = visible.copy()
        return visible


    def _mark_surface_samples_from_spray(self, path_index):
        cache = self._prepare_trajectory_playback_cache()
        if cache is None:
            return
        hit_samples = self._get_surface_sample_hits_from_spray(path_index)
        if len(hit_samples):
            mark_sample_hits(
                cache["coverage_sample_hits"], hit_samples, spray_on=True
            )

    def _update_current_spray_footprint(self, path_index):
        """Illuminate only target mesh triangles reached by the current spray sample."""
        cache = self._prepare_trajectory_playback_cache()
        actor = self._playback_spray_actor
        polydata = self._playback_spray_polydata
        if cache is None or actor is None or polydata is None:
            return
        # Keep one RGBA array attached to the VTK polydata. Only clear the
        # cells highlighted by the previous frame and write the new footprint;
        # avoid allocating and copying an entire target-face array per frame.
        rgba = cache["current_footprint_rgba"]
        previous_faces = cache["current_footprint_face_positions"]
        if len(previous_faces):
            rgba[previous_faces] = 0

        current_faces = np.empty(0, dtype=np.int64)
        known = True
        if cache["spray_on_mask"][path_index] and known:
            hits = self._get_surface_sample_hits_from_spray(path_index)
            if len(hits):
                current_faces = np.unique(
                    hits // cache["sample_count_per_face"]
                )
                rgba[current_faces] = [
                    *np.clip(np.rint(cache["paint_rgb"]), 0, 255).astype(np.uint8),
                    190,
                ]

        cache["current_footprint_face_positions"] = current_faces
        vtk_array = polydata.GetCellData().GetArray("Spray Footprint RGBA")
        if vtk_array is not None:
            vtk_array.Modified()
        polydata.Modified()
        actor.SetVisibility(bool(len(current_faces)))



    def _accumulate_film_thickness_for_path(self, path_index):
        """Do not infer film thickness without an explicitly configured motion timeline."""
        cache = self._prepare_trajectory_playback_cache()
        if cache is not None:
            cache["deposition_processed_index"] = int(path_index)
            if hasattr(self, "trajectory_thickness_label"):
                self.trajectory_thickness_label.setText(
                    "PREDICTED DRY FILM THICKNESS | N/A — ROBOT SPEED/TIMING NOT CONFIGURED"
                )
        return

    def _update_film_thickness_legend(self, target_um):
        """Show the exact reference colors and target-relative values used by the renderer."""
        target = float(target_um)
        if not math.isfinite(target) or target <= 0.0:
            self.trajectory_thickness_legend_label.setText(
                "FILM COLOR SCALE | unavailable — set a positive target thickness"
            )
            return

        labels = (
            ("<25%", 0.0, "UNPAINTED / TRACE"),
            ("25%", 0.25, "25–<50%"),
            ("50%", 0.50, "50–<75%"),
            ("75%", 0.75, "75–<100%"),
            ("100%", 1.00, "TARGET REACHED"),
            (">100%", 1.01, "OVER TARGET"),
        )
        entries = []
        for color_hex, (threshold_label, ratio, meaning) in zip(
            FILM_THICKNESS_COLOR_HEX,
            labels,
        ):
            thickness = target * ratio
            value = f"{thickness:.1f} µm" if ratio > 0.0 else "0 µm"
            swatch = (
                f'<span style="color:{color_hex}; font-size:15px;">■</span> '
                f'<span style="color:#E5EDF2;">{threshold_label} ({value})</span> '
                f'<span style="color:#AFC2CB;">{meaning}</span>'
            )
            entries.append(swatch)

        self.trajectory_thickness_legend_label.setText(
            "FILM THICKNESS COLOR KEY  |  " + " &nbsp; ".join(entries)
        )

    def _update_film_thickness_visualization(self):
        """Render target-face predicted dry-film thickness relative to target."""
        cache = self._prepare_trajectory_playback_cache()
        if cache is None or cache["deposition_readiness"] != "INPUTS_READY_UNCALIBRATED":
            if hasattr(self, "trajectory_thickness_label"):
                self.trajectory_thickness_label.setText(
                    "PREDICTED DRY FILM THICKNESS | N/A — INPUTS INCOMPLETE"
                )
            return
        if (
            self._trajectory_surface_polydata is None
            or self._trajectory_surface_base_rgb is None
        ):
            return
        target = float(cache["coating_process"]["target_dry_film_thickness_um"])
        if target <= 0.0:
            return
        base = self._trajectory_surface_base_rgb.copy()
        display_ids = cache["target_display_indices"]
        valid = display_ids >= 0
        face_values = cache["deposition_face_thickness_um"][cache["target_face_ids"]]
        self._update_film_thickness_legend(target)
        mapped = map_film_thickness_to_rgb(face_values, target)
        base[display_ids[valid]] = mapped[valid]
        self._set_trajectory_surface_rgb(base)
        # Include every Stage 2 target face, including zero-film faces.
        # Statistics are weighted by physical triangle area, not face count.
        target_areas = cache["face_areas"]
        if len(face_values) == 0:
            self.trajectory_thickness_label.setText(
                "PREDICTED DRY FILM [µm] | N/A — NO STAGE 2 TARGET FACES"
            )
            return
        stats = calculate_film_thickness_statistics(face_values, target_areas, target)
        self.trajectory_thickness_label.setText(
            "PREDICTED DRY FILM [µm] | "
            f"MIN {stats['min_um']:.1f}  MEAN {stats['mean_um']:.1f}  "
            f"P95 {stats['p95_um']:.1f}  MAX {stats['max_um']:.1f}  "
            f"IN BAND 90–110% {stats['area_in_target_band_percent']:.1f}%  "
            f"UNDER {stats['area_below_target_band_percent']:.1f}%  "
            f"OVER {stats['area_above_target_band_percent']:.1f}% "
            f"(TARGET {target:.1f} µm) | UNCALIBRATED"
        )
        self.trajectory_thickness_label.setToolTip(
            "Predicted dry-film thickness from the time-integrated Double-Beta "
            "spray model. The 90–110% in-band percentage is area-weighted; "
            "under-band and over-band areas are reported separately. Color scale: "
            "grey=<25%, blue=25%, cyan=50%, green=75%, yellow=100%, "
            "red=>100%. Colors are discrete target-relative bands. Only visible samples "
            "inside the spray footprint receive deposition. The model is not "
            "calibrated; valve ON/OFF delays, evaporation, atomization and "
            "measured coupon validation are not included. "
            f"Dry paint supplied: {cache['deposition_input_volume_mm3']:.3f} mm³; "
            f"deposited on modeled surface: {cache['deposition_total_volume_mm3']:.3f} mm³."
        )

    def _update_coverage_correction_plan(self):
        """Analyze the complete Stage-2 target after playback and propose deficits."""
        cache = self._prepare_trajectory_playback_cache()
        if cache is None:
            return None
        if cache["target_mask_status"] != "READY" or len(cache["face_samples"]) == 0:
            self.trajectory_correction_label.setText(
                "COVERAGE CORRECTION | N/A — COMPLETE STAGE 2 TARGET MASK REQUIRED"
            )
            cache["coverage_correction_plan"] = None
            return None
        if cache["deposition_readiness"] != "INPUTS_READY_UNCALIBRATED":
            missing = ", ".join(cache["deposition_missing_inputs"])
            self.trajectory_correction_label.setText(
                "COVERAGE CORRECTION | N/A — PROCESS INPUTS REQUIRED: "
                + missing.upper()
            )
            cache["coverage_correction_plan"] = None
            return None
        process = cache["coating_process"]
        parameters = self.trajectory_data.get("parameters", {})
        sampling_spacing_mm = float(parameters.get("sampling_spacing_mm", 10.0))
        footprint_width_mm = float(cache["footprint_width_mm"])
        # A conservative spatial grouping limit: correction waypoints remain
        # separated at patch boundaries and abrupt normal changes.
        max_gap_mm = max(
            3.0 * sampling_spacing_mm,
            0.75 * footprint_width_mm,
            1.0,
        )
        # Deficit samples are cached in Viewer WORLD coordinates. Convert
        # them back to PRODUCT_CAD before generating correction waypoints.
        from core.spray_coverage_optimizer import world_samples_to_product_cad_mm
        points_cad_mm, normals_cad = world_samples_to_product_cad_mm(
            points_world=cache["face_samples"],
            normals_world=cache["sample_normals"],
            product_transform_4x4=self.trajectory_product_transform,
            mm_per_unit=cache["mm_per_unit"],
        )
        target_mask = np.ones(len(points_cad_mm), dtype=bool)
        plan = build_coverage_correction_plan(
            sample_points_mm=points_cad_mm,
            sample_normals=normals_cad,
            sample_area_mm2=cache["sample_area_mm2"],
            target_mask=target_mask,
            reached_mask=cache["coverage_sample_hits"],
            thickness_um=cache["deposition_sample_thickness_um"],
            target_thickness_um=float(
                process["target_dry_film_thickness_um"]
            ),
            patch_ids=cache["sample_patch_ids"],
            standoff_mm=float(cache["standoff_units"] * cache["mm_per_unit"]),
            max_gap_mm=max_gap_mm,
            lower_ratio=0.90,
            upper_ratio=1.10,
            max_normal_change_deg=30.0,
        )
        cache["coverage_correction_plan"] = plan
        analysis = plan["analysis"]
        strokes = plan["correction_strokes"]
        stroke_count = len(strokes["stroke_ranges"])
        waypoint_count = len(strokes["source_sample_indices"])
        if hasattr(self, "trajectory_correction_apply_button"):
            self.trajectory_correction_apply_button.setEnabled(
                stroke_count > 0
                and not self._correction_preview_appended
                and self._playback_finished
            )
        unreached_area = float(np.sum(
            cache["sample_area_mm2"][analysis["unreached_sample_indices"]]
        ))
        unreached_percent = (
            100.0 * unreached_area / analysis["target_area_mm2"]
            if analysis["target_area_mm2"] > 0.0 else 0.0
        )
        self.trajectory_correction_label.setText(
            "SURFACE DEFICIT | "
            f"REACH {analysis['geometric_reach_percent']:.1f}%  "
            f"BELOW 90% DFT {analysis['below_target_area_percent']:.1f}%  "
            f"IN BAND {analysis['in_target_band_area_percent']:.1f}%  "
            f"OVER {analysis['over_target_area_percent']:.1f}%  "
            f"UNREACHED {unreached_percent:.1f}%\n"
            "CORRECTION CANDIDATES | "
            f"{stroke_count} STROKES / {waypoint_count} WAYPOINTS "
            "| PRODUCT_CAD mm | PREDICTED ONLY — NOT EXECUTION READY"
        )
        self.trajectory_correction_label.setToolTip(
            "Area-weighted analysis of every Stage-2 paintable sample; excluded "
            "faces are omitted. Deficits are samples below 90% of target dry-film "
            "thickness. Candidate paths are grouped by Stage-1 patch, spatial gap "
            "and surface-normal change. They are not collision checked, visibility "
            "rechecked from their new nozzle poses, robot-IK checked, or validated "
            "against calibrated spray deposition. Current process inputs are "
            "uncalibrated. Do not execute these candidates on a robot."
        )
        return plan

    def _apply_coverage_correction_preview(self):
        """Keep correction strokes disabled until robot motion planning is added."""
        self.trajectory_correction_label.setText(
            "CORRECTION APPEND | DISABLED — ROBOT MOTION PLANNING NOT CONFIGURED"
        )
        self.trajectory_correction_apply_button.setEnabled(False)
        return

    def _on_film_thickness_visibility_changed(self, checked):
        cache = self._prepare_trajectory_playback_cache()
        if checked and (
            cache is None
            or cache["deposition_readiness"] != "INPUTS_READY_UNCALIBRATED"
        ):
            self.trajectory_thickness_checkbox.blockSignals(True)
            self.trajectory_thickness_checkbox.setChecked(False)
            self.trajectory_thickness_checkbox.blockSignals(False)
            if hasattr(self, "trajectory_thickness_label"):
                self.trajectory_thickness_label.setText(
                    "PREDICTED DRY FILM THICKNESS | N/A — PROCESS INPUTS INCOMPLETE"
                )
            return
        if checked:
            self._update_film_thickness_visualization()
        else:
            self._restore_trajectory_surface_base_colors()
            if cache is not None and cache["coverage_processed_index"] >= 0:
                old_stride = cache["coverage_render_stride"]
                cache["coverage_render_stride"] = 1
                try:
                    self._update_paint_coverage_visualization(
                        cache["coverage_processed_index"]
                    )
                finally:
                    cache["coverage_render_stride"] = old_stride
        self.plotter.render()

    def _update_paint_coverage_visualization(self, index):
        cache = self._prepare_trajectory_playback_cache()
        if cache is None:
            return

        rewound = index < cache["coverage_processed_index"]
        if rewound:
            cache["coverage_correction_plan"] = None
            if hasattr(self, "trajectory_correction_apply_button"):
                self.trajectory_correction_apply_button.setEnabled(False)
            if hasattr(self, "trajectory_correction_label"):
                self.trajectory_correction_label.setText(
                    "COVERAGE CORRECTION | PLAYBACK REWOUND — ANALYSIS INVALIDATED"
                )
            cache["coverage_sample_hits"].fill(False)
            cache["coverage_processed_index"] = -1
            cache["coverage_last_render_index"] = -1
            cache["deposition_sample_thickness_um"].fill(0.0)
            cache["deposition_sample_volume_mm3"].fill(0.0)
            cache["deposition_face_thickness_um"].fill(0.0)
            cache["deposition_processed_index"] = -1
            cache["deposition_input_volume_mm3"] = 0.0
            cache["deposition_total_volume_mm3"] = 0.0

        final_index = len(cache["nozzle_points"]) - 1
        stride = cache["coverage_render_stride"]
        if (
            not rewound
            and index > 0
            and index < final_index
            and index % stride != 0
        ):
            return

        for path_index in range(
            cache["coverage_processed_index"] + 1, index + 1
        ):
            if cache["spray_on_mask"][path_index]:
                self._mark_surface_samples_from_spray(path_index)
            self._accumulate_film_thickness_for_path(path_index)
        cache["coverage_processed_index"] = max(
            cache["coverage_processed_index"], index
        )

        # Samples are stored face-contiguously; reshape avoids a full-mesh
        # bincount allocation on every coverage refresh.
        (
            face_coverage,
            painted_area,
            target_area,
            coverage_percent,
        ) = calculate_area_weighted_coverage(
            cache["face_areas"],
            cache["coverage_sample_hits"],
            cache["sample_count_per_face"],
        )
        # Keep the denominator immutable and explicit: every Stage 2 paintable
        # candidate face, including faces never referenced by the trajectory.
        if not np.isclose(target_area, cache["target_area"], rtol=1e-10, atol=1e-12):
            raise RuntimeError("Spray coverage target area changed during playback.")

        if (
            self._trajectory_surface_polydata is not None
            and self._trajectory_surface_face_mask is not None
            and self._trajectory_surface_base_rgb is not None
            and not (
                hasattr(self, "trajectory_thickness_checkbox")
                and self.trajectory_thickness_checkbox.isChecked()
            )
        ):
            base = self._trajectory_surface_base_rgb.astype(float)
            face_fraction = np.zeros(len(base), dtype=float)
            valid_display = cache["target_display_indices"] >= 0
            face_fraction[cache["target_display_indices"][valid_display]] = (
                face_coverage[valid_display]
            )
            painted = (
                base * (1.0 - face_fraction[:, None])
                + cache["paint_rgb"][None, :] * face_fraction[:, None]
            )
            colors = np.clip(np.rint(painted), 0, 255).astype(np.uint8)
            self._set_trajectory_surface_rgb(colors)
            # The mapper already references this polydata. Marking the changed
            # VTK array is sufficient; avoid rebuilding/updating the mapper.
            cache["coverage_last_render_index"] = index

        if hasattr(self, "trajectory_coverage_bar"):
            self.trajectory_coverage_bar.setValue(
                int(round(np.clip(coverage_percent, 0.0, 100.0) * 10.0))
            )
        if hasattr(self, "trajectory_thickness_checkbox"):
            ready = cache["deposition_readiness"] == "INPUTS_READY_UNCALIBRATED"
            self.trajectory_thickness_checkbox.setEnabled(ready)
            if not ready and self.trajectory_thickness_checkbox.isChecked():
                self.trajectory_thickness_checkbox.blockSignals(True)
                self.trajectory_thickness_checkbox.setChecked(False)
                self.trajectory_thickness_checkbox.blockSignals(False)
            if hasattr(self, "trajectory_thickness_label"):
                if ready:
                    target = float(cache["coating_process"]["target_dry_film_thickness_um"])
                    face_values = cache["deposition_face_thickness_um"][cache["target_face_ids"]]
                    target_areas = cache["face_areas"]
                    if len(face_values) == 0:
                        self.trajectory_thickness_label.setText(
                            "PREDICTED DRY FILM [µm] | N/A — NO STAGE 2 TARGET FACES"
                        )
                    else:
                        stats = calculate_film_thickness_statistics(face_values, target_areas, target)
                        self.trajectory_thickness_label.setText(
                            "PREDICTED DRY FILM [µm] | "
                            f"MIN {stats['min_um']:.1f}  MEAN {stats['mean_um']:.1f}  "
                            f"P95 {stats['p95_um']:.1f}  MAX {stats['max_um']:.1f}  "
                            f"IN BAND 90–110% {stats['area_in_target_band_percent']:.1f}%  "
            f"UNDER {stats['area_below_target_band_percent']:.1f}%  "
            f"OVER {stats['area_above_target_band_percent']:.1f}% "
            f"(TARGET {target:.1f} µm) | UNCALIBRATED"
                        )
                else:
                    missing = ", ".join(cache["deposition_missing_inputs"])
                    self.trajectory_thickness_label.setText(
                        "PREDICTED DRY FILM THICKNESS | N/A — " + missing.upper()
                    )
        if (
            hasattr(self, "trajectory_thickness_checkbox")
            and self.trajectory_thickness_checkbox.isChecked()
        ):
            self._update_film_thickness_visualization()
        if hasattr(self, "trajectory_coverage_label"):
            model_label = "ROUND CONE"
            if cache["target_mask_status"] != "READY":
                self.trajectory_coverage_label.setText(
                    "N/A — STAGE 2 TARGET MASK REQUIRED"
                )
            elif cache["target_area"] <= 1e-12:
                self.trajectory_coverage_label.setText(
                    "N/A — NO STAGE 2 PAINTABLE TARGET AREA"
                )
            else:
                self.trajectory_coverage_label.setText(
                    f"{coverage_percent:.1f}% SAMPLED REACH | "
                    f"{painted_area * cache['area_display_scale']:.3f} / "
                    f"{cache['target_area'] * cache['area_display_scale']:.3f} "
                    f"{cache['area_display_unit']}"
                )
            if True:
                self.trajectory_coverage_label.setToolTip(
                    "Sampled geometric reach estimate over the complete Stage 2 paintable "
                    "candidate area. Four test points per mesh triangle are used, and "
                    "nozzle-to-sample rays are checked against the complete product mesh. "
                    "This is not deposited paint mass or coating thickness."
                )

    def _create_spray_gun_actors(self):
        cache = self._prepare_trajectory_playback_cache()
        if cache is None:
            return None

        profile = cache["profile"]
        mm_per_unit = cache["mm_per_unit"]
        nozzle_length = max(
            float(profile.get("nozzle_length_mm", 25.0) or 25.0) / mm_per_unit,
            2.0,
        )
        body_length_value = float(profile.get("gun_body_length_mm", 0.0) or 0.0)
        body_diameter_value = float(profile.get("gun_body_diameter_mm", 0.0) or 0.0)
        standoff = cache["standoff_units"]
        body_length = (
            body_length_value / mm_per_unit
            if body_length_value > 0.0 else max(nozzle_length * 2.8, standoff * 0.34)
        )
        body_diameter = (
            body_diameter_value / mm_per_unit
            if body_diameter_value > 0.0 else max(nozzle_length * 0.55, standoff * 0.065)
        )
        orifice = max(
            float(profile.get("nozzle_orifice_mm", 1.4) or 1.4) / mm_per_unit,
            body_diameter * 0.06,
        )

        nozzle_mesh = pv.Cylinder(
            center=(0.0, 0.0, -nozzle_length * 0.5),
            direction=(0.0, 0.0, 1.0),
            radius=max(orifice * 0.9, 0.5),
            height=nozzle_length,
            resolution=20,
        )
        body_mesh = pv.Cylinder(
            center=(0.0, 0.0, -(nozzle_length + body_length * 0.5)),
            direction=(0.0, 0.0, 1.0),
            radius=body_diameter * 0.5,
            height=body_length,
            resolution=24,
        )
        grip_mesh = pv.Cylinder(
            center=(0.0, -body_diameter * 0.62, -(nozzle_length + body_length * 0.48)),
            direction=(0.0, 1.0, 0.0),
            radius=body_diameter * 0.25,
            height=body_diameter * 1.35,
            resolution=20,
        )
        trigger_mesh = pv.Cylinder(
            center=(0.0, body_diameter * 0.48, -(nozzle_length + body_length * 0.28)),
            direction=(0.0, 1.0, 0.0),
            radius=max(body_diameter * 0.045, 0.5),
            height=body_diameter * 0.72,
            resolution=12,
        )

        self._playback_gun_actors = [
            self.plotter.add_mesh(
                body_mesh, color="#C9D4DC", smooth_shading=True,
                specular=0.55, specular_power=30, pickable=False,
            ),
            self.plotter.add_mesh(
                grip_mesh, color="#25333D", smooth_shading=True,
                pickable=False,
            ),
            self.plotter.add_mesh(
                nozzle_mesh, color="#E8EEF2", smooth_shading=True,
                pickable=False,
            ),
            self.plotter.add_mesh(
                trigger_mesh, color="#D9A441", smooth_shading=True,
                pickable=False,
            ),
        ]
        # Surface-conforming overlay: each target triangle can be illuminated
        # only when its samples fall inside the configured spray footprint.
        target_triangles = cache["target_triangles_world"]
        target_points = target_triangles.reshape(-1, 3)
        face_count = len(target_triangles)
        target_faces = np.column_stack([
            np.full(face_count, 3, dtype=np.int64),
            np.arange(face_count, dtype=np.int64) * 3,
            np.arange(face_count, dtype=np.int64) * 3 + 1,
            np.arange(face_count, dtype=np.int64) * 3 + 2,
        ]).reshape(-1)
        self._playback_spray_polydata = pv.PolyData(target_points, target_faces)
        footprint_rgba = np.zeros((face_count, 4), dtype=np.uint8)
        cache["current_footprint_rgba"] = footprint_rgba
        cache["current_footprint_face_positions"] = np.empty(0, dtype=np.int64)
        self._playback_spray_polydata.cell_data["Spray Footprint RGBA"] = (
            footprint_rgba
        )
        self._playback_spray_actor = self.plotter.add_mesh(
            self._playback_spray_polydata,
            scalars="Spray Footprint RGBA",
            rgba=True,
            preference="cell",
            lighting=False,
            pickable=False,
            show_edges=False,
        )
        self._playback_spray_actor.SetVisibility(0)

        # Visible circular-cone spray-state cue.
        sheet_length = cache["standoff_units"]
        sheet_half_width = max(cache["footprint_width_units"] * 0.5, 1e-6)
        # Circular spray volume: nozzle apex to the circular target footprint.
        cone_resolution = 64
        cone_angles = np.linspace(
            0.0, 2.0 * math.pi, cone_resolution, endpoint=False
        )
        cone_ring = np.column_stack((
            sheet_half_width * np.cos(cone_angles),
            sheet_half_width * np.sin(cone_angles),
            np.full(cone_resolution, sheet_length, dtype=float),
        ))
        cone_points = np.vstack((np.zeros((1, 3), dtype=float), cone_ring))
        cone_faces = []
        for cone_index in range(cone_resolution):
            next_index = (cone_index + 1) % cone_resolution
            cone_faces.extend((3, 0, 1 + cone_index, 1 + next_index))
        sheet_mesh = pv.PolyData(
            cone_points, np.asarray(cone_faces, dtype=np.int64)
        )
        self._playback_spray_sheet_actor = self.plotter.add_mesh(
            sheet_mesh,
            color=self._trajectory_paint_color_hex or "#D95B43",
            opacity=0.34,
            show_edges=False,
            edge_color="#FFD08A",
            line_width=2,
            lighting=False,
            pickable=False,
            render=False,
        )
        self._playback_spray_sheet_actor.SetVisibility(0)
        self._playback_nozzle_actor = self._playback_gun_actors[0]
        return self._playback_gun_actors

    def _position_spray_gun(self, index):
        cache = self._prepare_trajectory_playback_cache()
        if cache is None:
            return
        if not self._playback_gun_actors:
            self._create_spray_gun_actors()
        frame = cache["nozzle_frames"][index]
        position = cache["nozzle_points"][index]
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = frame
        matrix[:3, 3] = position
        vtk_matrix = self._vtk_matrix_from_numpy(matrix)
        for actor in self._playback_gun_actors:
            actor.SetOrigin(0.0, 0.0, 0.0)
            actor.SetUserMatrix(vtk_matrix)
        if self._playback_spray_sheet_actor is not None:
            self._playback_spray_sheet_actor.SetOrigin(0.0, 0.0, 0.0)
            self._playback_spray_sheet_actor.SetUserMatrix(vtk_matrix)
            self._playback_spray_sheet_actor.SetVisibility(
                1 if cache["spray_on_mask"][index] else 0
            )

    def _make_playback_path_polydata(self, last_index):
        cache = self._prepare_trajectory_playback_cache()
        points = cache["nozzle_points"][:last_index + 1]
        segments = cache["segment_ids"][:last_index + 1]
        if len(points) == 0:
            return pv.PolyData()

        line_cells = []
        point_offset = 0
        for segment_id in np.unique(segments):
            local = np.flatnonzero(segments == segment_id)
            if len(local) < 2:
                continue
            line_cells.extend([len(local), *local.tolist()])
        polydata = pv.PolyData(points)
        if line_cells:
            polydata.lines = np.asarray(line_cells, dtype=np.int64)
        return polydata

    def _update_trajectory_playback_visuals(self, index, update_coverage=True):
        if self.trajectory_data is None or self.display_mode != "trajectory":
            return
        cache = self._prepare_trajectory_playback_cache()
        nozzle_points = cache["nozzle_points"]
        if len(nozzle_points) == 0:
            return

        index = int(np.clip(index, 0, len(nozzle_points) - 1))
        self._playback_index = index

        path_stride = cache["path_render_stride"]
        path_rewound = index < self._playback_last_path_render_index
        path_due = (
            self._playback_path_actor is None
            or path_rewound
            or index == 0
            or index == len(nozzle_points) - 1
            or index - self._playback_last_path_render_index >= path_stride
        )
        if path_due:
            path = self._make_playback_path_polydata(index)
            if self._playback_path_actor is None:
                if path.n_lines > 0:
                    self._playback_path_actor = self.plotter.add_mesh(
                        path,
                        color="#A9BAC5",
                        line_width=3,
                        opacity=0.28,
                        render_lines_as_tubes=True,
                        lighting=False,
                        pickable=False,
                    )
            else:
                mapper = self._playback_path_actor.GetMapper()
                mapper.SetInputData(path)
                mapper.Update()
            self._playback_last_path_render_index = index

        self._position_spray_gun(index)
        spray_on = bool(cache["spray_on_mask"][index])
        if self._playback_spray_sheet_actor is not None:
            self._playback_spray_sheet_actor.SetVisibility(1 if spray_on else 0)
        self._update_current_spray_footprint(index)
        if hasattr(self, "trajectory_spray_state_label"):
            if spray_on:
                diameter_mm = cache["footprint_width_mm"]
                self.trajectory_spray_state_label.setText(
                    f"SPRAY: ON · ROUND CONE Ø {diameter_mm:.1f} mm"
                )
                self.trajectory_spray_state_label.setStyleSheet(
                    "color: #68D391; font-size: 9px; font-weight: 800; padding: 2px 5px;"
                )
            else:
                self.trajectory_spray_state_label.setText("● SPRAY OFF · MOVE")
                self.trajectory_spray_state_label.setStyleSheet(
                    "color: #91A4AE; font-size: 9px; font-weight: 800; padding: 2px 5px;"
                )
        if update_coverage:
            self._update_paint_coverage_visualization(index)

        total = len(nozzle_points)
        spray_text = "SPRAY ON" if spray_on else "SPRAY OFF"
        self.trajectory_playback_label.setText(
            f"{index + 1:,} / {total:,}  ·  {spray_text}"
        )
        self._update_trajectory_time_label(index)
        self.trajectory_playback_slider.blockSignals(True)
        self.trajectory_playback_slider.setValue(index)
        self.trajectory_playback_slider.blockSignals(False)
        self.plotter.render()

    def _choose_trajectory_paint_color(self):
        """Choose the paint color used by the spray and coverage previews."""
        current = self._trajectory_paint_color_hex or "#D95B43"
        selected = QColorDialog.getColor(
            QColor(current), self, "Select spray paint color"
        )
        if not selected.isValid():
            return

        self._trajectory_paint_color_hex = selected.name().upper()
        if hasattr(self, "trajectory_paint_color_button"):
            self.trajectory_paint_color_button.setStyleSheet(
                "QPushButton { color: %s; font-size: 17px; padding: 0px; "
                "background: #1C303B; border: 1px solid #607986; border-radius: 4px; }"
                % self._trajectory_paint_color_hex
            )
            self.trajectory_paint_color_button.setToolTip(
                "Selected spray paint: %s" % self._trajectory_paint_color_hex
            )

        cache = self._playback_cache
        if cache is not None:
            hex_value = self._trajectory_paint_color_hex.lstrip("#")
            cache["paint_rgb"] = np.asarray(
                [int(hex_value[i:i + 2], 16) for i in (0, 2, 4)],
                dtype=float,
            )
            cache["coverage_sample_hits"].fill(False)
            cache["coverage_processed_index"] = -1
            cache["coverage_last_render_index"] = -1
            if self._playback_spray_sheet_actor is not None:
                self._playback_spray_sheet_actor.GetProperty().SetColor(
                    selected.redF(), selected.greenF(), selected.blueF()
                )
            if self.trajectory_data is not None and self.display_mode == "trajectory":
                self._update_trajectory_playback_visuals(self._playback_index)
                self.plotter.render()

        if getattr(self, "_interactive_spray_display_mode", "thickness") == "paint_color":
            self._interactive_refresh_paint_visual(render=True)

    def _on_trajectory_playback_slider_changed(self, value):
        if self.display_mode != "trajectory" or self.trajectory_data is None:
            return
        self._playback_timer.stop()
        self._pause_playback_wall_clock()
        self._playback_finished = False
        self.trajectory_play_button.setText("▶")
        self._notify_playback_state_changed()
        self._update_trajectory_playback_visuals(value)
        total = len(self.trajectory_data["nozzle_points"])
        if value >= total - 1:
            self._update_coverage_correction_plan()

    def _toggle_trajectory_playback(self):
        if self.trajectory_data is None or self.display_mode != "trajectory":
            return
        total = len(self.trajectory_data["nozzle_points"])
        if total == 0:
            return
        if self._playback_timer.isActive():
            self._playback_timer.stop()
            self._pause_playback_wall_clock()
            self.trajectory_play_button.setText("▶")
            self._notify_playback_state_changed()
            self._update_trajectory_time_label()
            return
        if self._playback_finished or self._playback_index >= total - 1:
            self._playback_finished = False
            self._playback_elapsed_s = 0.0
            self._playback_wall_elapsed_s = 0.0
            self._playback_wall_started_at = None
            self._update_trajectory_playback_visuals(0)
        else:
            if self._playback_path_actor is None and self._playback_nozzle_actor is None:
                self._update_trajectory_playback_visuals(self._playback_index)
        self._playback_wall_started_at = time.monotonic()
        self.trajectory_play_button.setText("Ⅱ")
        self._playback_timer.start()
        self._notify_playback_state_changed()
        self._update_trajectory_time_label()

    def _complete_trajectory_playback(self):
        """Freeze the gun at the final pose and force all spray visuals OFF."""
        self._playback_timer.stop()
        self._pause_playback_wall_clock()
        self._playback_finished = True
        self.trajectory_play_button.setText("▶")
        self._notify_playback_state_changed()

        cache = self._prepare_trajectory_playback_cache()
        if cache is not None and len(cache["nozzle_points"]):
            final_index = len(cache["nozzle_points"]) - 1
            self._playback_index = final_index
            if self._playback_spray_sheet_actor is not None:
                self._playback_spray_sheet_actor.SetVisibility(0)
            if self._playback_spray_actor is not None:
                self._playback_spray_actor.SetVisibility(0)
            if self._playback_spray_polydata is not None:
                rgba = cache["current_footprint_rgba"]
                rgba.fill(0)
                cache["current_footprint_face_positions"] = np.empty(
                    0, dtype=np.int64
                )
                vtk_array = self._playback_spray_polydata.GetCellData().GetArray(
                    "Spray Footprint RGBA"
                )
                if vtk_array is not None:
                    vtk_array.Modified()
                self._playback_spray_polydata.Modified()
            self.trajectory_spray_state_label.setText(
                "SPRAY: OFF · PLAYBACK COMPLETE"
            )
            self.trajectory_spray_state_label.setStyleSheet(
                "color: #8FA3B2; font-weight: 700; padding: 2px 6px;"
            )
            self.trajectory_playback_label.setText(
                f"{final_index + 1} / {len(cache['nozzle_points'])} "
                "| COMPLETE · SPRAY OFF"
            )
            self._update_trajectory_time_label(final_index)
        self.plotter.render()
        if cache is not None and len(cache["nozzle_points"]):
            if cache["coverage_processed_index"] < len(cache["nozzle_points"]) - 1:
                self._update_paint_coverage_visualization(
                    len(cache["nozzle_points"]) - 1
                )
            self._update_coverage_correction_plan()

    def _advance_trajectory_playback(self):
        if self._playback_finished:
            self._playback_timer.stop()
            return
        if self.trajectory_data is None:
            self._stop_trajectory_playback()
            return
        total = len(self.trajectory_data["nozzle_points"])
        if self._playback_index >= total - 1:
            self._complete_trajectory_playback()
            return
        cache = self._prepare_trajectory_playback_cache()
        base_step = max(1, int(np.ceil(total / 600.0)))
        step = max(1, int(round(base_step * self.trajectory_playback_speed.value())))
        next_index = min(total - 1, self._playback_index + step)

        # Never skip a spray-OFF boundary, even at high visual playback speed.
        off_boundaries = cache["spray_off_boundary_indices"]
        boundary_position = int(np.searchsorted(
            off_boundaries, self._playback_index, side="right"
        ))
        if boundary_position < len(off_boundaries):
            next_off_boundary = int(off_boundaries[boundary_position])
            if next_index >= next_off_boundary:
                next_index = next_off_boundary

        self._update_trajectory_playback_visuals(next_index)
        if next_index >= total - 1:
            self._complete_trajectory_playback()

    def _step_trajectory_playback(self):
        if self.trajectory_data is None or self.display_mode != "trajectory":
            return
        self._playback_timer.stop()
        self._pause_playback_wall_clock()
        self._playback_finished = False
        self.trajectory_play_button.setText("▶")
        self._notify_playback_state_changed()
        total = len(self.trajectory_data["nozzle_points"])
        next_index = min(self._playback_index + 1, total - 1)
        self._update_trajectory_playback_visuals(next_index)
        if next_index >= total - 1:
            self._complete_trajectory_playback()

    def _reset_trajectory_playback(self):
        if self.trajectory_data is None or self.display_mode != "trajectory":
            return
        self._stop_trajectory_playback(reset=True)
        self._playback_finished = False
        # Show the initial nozzle pose and footprint without counting the first
        # spray sample as already-painted surface coverage. Coverage advances
        # only when playback actually moves forward from this reset state.
        self._update_trajectory_playback_visuals(0, update_coverage=False)

    def _trajectory_arrow_length(self):
        if self.trajectory_data is None:
            return 20.0

        nozzle = self.trajectory_data["nozzle_points"]
        surface = self.trajectory_data["surface_points"]

        distances = np.linalg.norm(
            nozzle - surface, axis=1
        )

        valid = distances[
            np.isfinite(distances) & (distances > 0)
        ]

        if len(valid) == 0:
            return 20.0

        return float(np.median(valid) * 0.25)

    # =========================================================
    # FIXED WORLD XYZ WORKSPACE AND MOVING PRODUCT CAD FRAME
    # =========================================================

    def _add_world_workspace(self):
        """Render a polished control-room style World frame and work-cell volume."""
        lx, ly, lz = np.asarray(self.workspace_dimensions, dtype=float)
        hx, hy, hz = lx * 0.5, ly * 0.5, lz * 0.5

        # Box corners: World origin is the geometric center.
        # +X = right, +Y = into the work cell, +Z = downward.
        corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz],
            [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz],
            [hx, hy, hz], [-hx, hy, hz],
        ], dtype=float)

        def add_segments(segments, color, width, opacity=1.0):
            if not segments:
                return
            points = []
            cells = []
            for p1, p2 in segments:
                index = len(points)
                points.extend([p1, p2])
                cells.extend([2, index, index + 1])
            lines = pv.PolyData(np.asarray(points, dtype=float))
            lines.lines = np.asarray(cells, dtype=np.int64)
            self.plotter.add_mesh(
                lines, color=color, line_width=width,
                opacity=opacity, lighting=False, render_lines_as_tubes=True,
                pickable=False,
            )

        # World reference volume: thin, pale geometry stays in the background.
        x_edges = [(corners[a], corners[b]) for a, b in
                   [(0, 1), (3, 2), (4, 5), (7, 6)]]
        y_edges = [(corners[a], corners[b]) for a, b in
                   [(0, 3), (1, 2), (4, 7), (5, 6)]]
        z_edges = [(corners[a], corners[b]) for a, b in
                   [(0, 4), (1, 5), (2, 6), (3, 7)]]
        add_segments(x_edges, "#426273", 1.1, 0.34)
        add_segments(y_edges, "#385D68", 1.0, 0.30)
        add_segments(z_edges, "#40566D", 1.0, 0.30)

        # Fine World reference grid on the central XY plane.
        grid_segments = []
        grid_count = 10
        for i in range(-grid_count, grid_count + 1):
            x = hx * i / grid_count
            y = hy * i / grid_count
            grid_segments.append((np.array([x, -hy, 0.0]), np.array([x, hy, 0.0])))
            grid_segments.append((np.array([-hx, y, 0.0]), np.array([hx, y, 0.0])))
        add_segments(grid_segments, "#29404D", 0.7, 0.20)

        # World XYZ axes are intentionally much longer than the Product
        # axes so the fixed reference direction remains easy to compare while
        # the product rotates around the CAD origin.
        product_bounds = np.asarray(self._pose_actor.GetBounds(), dtype=float).reshape(3, 2)
        product_size = max(float(np.max(product_bounds[:, 1] - product_bounds[:, 0])), 1.0)
        world_axis_length = max(product_size * 2.0, 20.0)
        world_axes = [
            (np.array([1.0, 0.0, 0.0]), "#E2A0AC", "X"),
            (np.array([0.0, 1.0, 0.0]), "#A1D7B7", "Y"),
            (np.array([0.0, 0.0, 1.0]), "#A9C9EA", "Z"),
        ]
        for direction, color, label in world_axes:
            dash_segments = []
            dash_count = 12
            for dash in range(dash_count):
                start_t = (dash / dash_count) * world_axis_length
                end_t = ((dash + 0.72) / dash_count) * world_axis_length
                dash_segments.append((
                    direction * start_t,
                    direction * end_t,
                ))
            add_segments(dash_segments, color, 2.2, 0.84)
            self.plotter.add_point_labels(
                np.asarray([direction * world_axis_length * 1.04]),
                [label],
                font_size=11,
                text_color=color,
                shape=None,
                margin=1,
                point_size=0,
                always_visible=True,
                bold=True,
            )

        # Small, understated World origin marker.
        self.plotter.add_mesh(
            pv.Sphere(
                radius=max(min(lx, ly, lz) * 0.004, 1.5),
                center=(0.0, 0.0, 0.0),
                theta_resolution=12,
                phi_resolution=12,
            ),
            color="#AABBC5",
            opacity=0.45,
            lighting=False,
            pickable=False,
        )

        # Compact telemetry-style overlays; avoid long labels around the axes.

    def _add_product_coordinate_frame(self):
        if self._pose_actor is None:
            return
        from vtkmodules.vtkRenderingAnnotation import vtkAxesActor

        axes = vtkAxesActor()
        product_bounds = np.asarray(self._pose_actor.GetBounds(), dtype=float).reshape(3, 2)
        product_size = max(float(np.max(product_bounds[:, 1] - product_bounds[:, 0])), 1.0)
        # The Product frame is intentionally larger and brighter than
        # the short, dashed World reference axes.
        axis_length = product_size * 0.68
        axes.SetTotalLength(axis_length, axis_length, axis_length)

        axes.SetShaftTypeToLine()
        axes.SetConeRadius(0.060)
        axes.SetSphereRadius(0.035)
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        axes.SetAxisLabels(True)

        axis_styles = (
            (axes.GetXAxisShaftProperty(), axes.GetXAxisTipProperty(), (1.00, 0.43, 0.52)),
            (axes.GetYAxisShaftProperty(), axes.GetYAxisTipProperty(), (0.28, 0.92, 0.62)),
            (axes.GetZAxisShaftProperty(), axes.GetZAxisTipProperty(), (0.30, 0.70, 1.00)),
        )
        for shaft, tip, color in axis_styles:
            shaft.SetColor(*color)
            tip.SetColor(*color)
            shaft.SetOpacity(0.92)
            tip.SetOpacity(1.0)
            shaft.SetLineWidth(3.2)

        # Small, crisp XYZ glyphs, color-matched to their respective axes.
        caption_styles = (
            (axes.GetXAxisCaptionActor2D(), (1.00, 0.64, 0.69)),
            (axes.GetYAxisCaptionActor2D(), (0.55, 1.00, 0.76)),
            (axes.GetZAxisCaptionActor2D(), (0.58, 0.82, 1.00)),
        )
        for caption, color in caption_styles:
            text_property = caption.GetCaptionTextProperty()
            text_property.SetFontFamilyToArial()
            text_property.SetFontSize(11)
            text_property.SetBold(True)
            text_property.SetItalic(False)
            text_property.SetShadow(False)
            text_property.SetColor(*color)
            text_property.SetOpacity(1.0)

        axes.SetUserMatrix(self._vtk_matrix_from_numpy(self._actor_matrix_array()))
        self._cad_axes_actor = axes
        self.plotter.add_actor(axes, reset_camera=False)

    def _add_world_orientation_indicator(self):
        """Show the fixed World XYZ orientation triad in the lower-left viewport."""
        from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
        from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget

        axes = vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.035)
        axes.SetConeRadius(0.12)
        axes.SetSphereRadius(0.08)
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        axes.SetAxisLabels(True)
        axes.GetXAxisShaftProperty().SetColor(1.0, 0.40, 0.48)
        axes.GetXAxisTipProperty().SetColor(1.0, 0.40, 0.48)
        axes.GetYAxisShaftProperty().SetColor(0.30, 0.86, 0.65)
        axes.GetYAxisTipProperty().SetColor(0.30, 0.86, 0.65)
        axes.GetZAxisShaftProperty().SetColor(0.28, 0.75, 1.0)
        axes.GetZAxisTipProperty().SetColor(0.28, 0.75, 1.0)

        widget = vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.plotter.iren.interactor)
        widget.SetViewport(0.015, 0.015, 0.17, 0.17)
        widget.SetEnabled(1)
        widget.InteractiveOff()
        self._world_orientation_widget = widget
        self._world_orientation_axes = axes

    def _add_pose_coordinate_frames(self):
        # Keep the fixed World reference and CAD-origin Product frame
        # visible during both the initial product view and Stage 1/2
        # surface-patch inspection. Patch mode uses the same posed CAD
        # actor, so these references remain spatially registered.
        if self.display_mode not in ("product", "patches", "trajectory", "spray_point_preview"):
            return

        if self._pose_actor is None:
            return

        # The World frame remains fixed in the work cell.
        self._add_world_workspace()

        # The Product/CAD frame is attached to the product actor and
        # follows its complete 4x4 pose transform during RPY changes.
        self._add_product_coordinate_frame()

        # Viewport orientation widget represents the camera-relative
        # appearance of the fixed World frame, not the product frame.
        self._add_world_orientation_indicator()

    # =========================================================
    # MAIN RENDERING
    # =========================================================

    
    def _render_product(self, reset_camera=False):
        self._stop_trajectory_playback(reset=True)
        if self.current_mesh is None:
            self._draw_empty_scene()
            return

        previous_camera = None
        previous_product_pose = None
        if not reset_camera and self._pose_actor is not None:
            previous_product_pose = self._actor_matrix_array().copy()

        if not reset_camera:
            try:
                previous_camera = self.plotter.camera_position
            except Exception:
                previous_camera = None
        self._attitude_text_actor = None
        if self._world_orientation_widget is not None:
            try:
                self._world_orientation_widget.SetEnabled(0)
            except Exception:
                pass
        self._world_orientation_widget = None
        self._world_orientation_axes = None
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        self.plotter.clear()
        self._trajectory_safety_actor = None
        # Actors from the previous scene are no longer in the renderer.
        self._hole_overlay_actors = []
        self.plotter.set_background("#07111D", top="#142B40")

        # -------------------------------------------------
        # CLASSIFICATION MODE
        # -------------------------------------------------
        if (
            self.display_mode == "classification"
            and self.face_status_labels is not None
        ):
            polydata = self._to_polydata(self.current_mesh)

            if len(self.face_status_labels) != len(
                self.current_mesh.faces
            ):
                raise ValueError(
                    "Classification labels do not match mesh faces."
                )

            polydata.cell_data["Surface Status"] = (
                self.face_status_labels.astype(np.int32)
            )

            self._pose_actor = self.plotter.add_mesh(
                polydata,
                scalars="Surface Status",
                cmap=CLASSIFICATION_COLORS,
                clim=(-0.5, 3.5),
                categories=True,
                show_edges=True,
                edge_color="#182C38",
                line_width=1,
                smooth_shading=False,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Surface Status",
                    "color": "#D8E5ED",
                    "title_font_size": 11,
                    "label_font_size": 9,
                    "n_labels": 4,
                    "fmt": "%.0f",
                },
            )
            self._pose_actor.SetOrigin(0.0, 0.0, 0.0)
            self._pose_actor.SetUserMatrix(
                self._vtk_matrix_from_numpy(self._last_product_transform)
            )

            self._add_classification_legend()

        # -------------------------------------------------
        # PATCH MODE
        # -------------------------------------------------
        elif (
            self.display_mode == "patches"
            and self.patch_labels is not None
            and len(self.patch_labels)
            == self.current_mesh.faces.shape[0]
        ):
            polydata = self._to_polydata(self.current_mesh)

            polydata.cell_data["Surface Patch"] = (
                self.patch_labels.astype(float)
            )

            # Keep every original triangle in this actor so direct picking
            # still maps its cell ID to the original patch_labels array.
            # The selected patch is made transparent here and drawn once,
            # at its exact original surface location, by the highlight actor.
            # Before a Stage 2 patch is selected, keep the complete
            # product fully opaque. Fade the surrounding surfaces only
            # while an individual paintable candidate is being inspected.
            surface_opacity = np.ones(
                len(self.patch_labels), dtype=float
            )
            if self.selected_patch_id is not None:
                surface_opacity.fill(0.16)
                surface_opacity[
                    self.patch_labels == self.selected_patch_id
                ] = 0.0
            polydata.cell_data["Surface Opacity"] = surface_opacity

            self._pose_actor = self.plotter.add_mesh(
                polydata,
                scalars="Surface Patch",
                cmap=PATCH_COLORS,
                categories=True,
                opacity="Surface Opacity",
                show_edges=True,
                edge_color="#182C38",
                line_width=1,
                smooth_shading=False,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Surface Patches",
                    "color": "#D8E5ED",
                    "title_font_size": 11,
                    "label_font_size": 9,
                },
            )

            # Reapply the exact STEP CAD actor pose captured before
            # clearing the renderer. This keeps the displayed patch map
            # aligned with the user's confirmed Roll/Pitch/Yaw pose.
            if previous_product_pose is not None:
                from vtkmodules.vtkCommonTransforms import vtkTransform
                patch_transform = vtkTransform()
                patch_transform.SetMatrix(
                    self._vtk_matrix_from_numpy(previous_product_pose)
                )
                self._pose_actor.SetOrigin(0.0, 0.0, 0.0)
                self._pose_actor.SetOrientation(
                    *patch_transform.GetOrientation()
                )
                self._pose_actor.SetPosition(
                    *previous_product_pose[:3, 3].tolist()
                )

            # -------------------------------------------------
            # SELECTED PATCH - HIGH-VISIBILITY GLOW HIGHLIGHT
            # -------------------------------------------------
            if self.selected_patch_id is not None:
                selected_mask = (
                    self.patch_labels == self.selected_patch_id
                )

                selected_polydata = self._to_polydata(
                    self.current_mesh,
                    face_mask=selected_mask,
                )

                if selected_polydata.n_cells > 0:
                    # Read the exact color assigned to this Patch ID by
                    # the same Turbo lookup table used for the product.
                    patch_rgb = [0.0, 0.0, 0.0]
                    lookup_table = (
                        self._pose_actor.GetMapper().GetLookupTable()
                    )
                    lookup_table.GetColor(
                        float(self.selected_patch_id),
                        patch_rgb,
                    )
                    patch_color = "#{:02X}{:02X}{:02X}".format(
                        *[
                            int(round(max(0.0, min(1.0, channel)) * 255))
                            for channel in patch_rgb
                        ]
                    )

                    # Draw the selected triangles directly on the original
                    # CAD surface. No offset membrane, cylinder, beam, or
                    # secondary inner ring is generated.
                    selected_actor = self.plotter.add_mesh(
                        selected_polydata,
                        color=patch_color,
                        opacity=1.0,
                        show_edges=False,
                        smooth_shading=False,
                        lighting=False,
                        ambient=1.0,
                        diffuse=0.0,
                        specular=0.0,
                        pickable=False,
                    )

                    # The outer contour uses the patch's own color, not a
                    # universal cyan/white highlight.
                    edge_actor = self.plotter.add_mesh(
                        selected_polydata,
                        style="wireframe",
                        color=patch_color,
                        opacity=1.0,
                        line_width=4.5,
                        render_lines_as_tubes=True,
                        lighting=False,
                        pickable=False,
                    )

                    if previous_product_pose is not None:
                        for selected_actor in (
                            selected_actor,
                            edge_actor,
                        ):
                            selected_actor.SetOrigin(0.0, 0.0, 0.0)
                            selected_actor.SetOrientation(
                                *patch_transform.GetOrientation()
                            )
                            selected_actor.SetPosition(
                                *previous_product_pose[:3, 3].tolist()
                            )

        # -------------------------------------------------
        # TRAJECTORY MODE
        # -------------------------------------------------
        elif (
            self.display_mode == "trajectory"
            and self.trajectory_data is not None
        ):
            # Keep every CAD face in ONE actor. This prevents gaps where
            # face classification is unknown or where separate overlay meshes
            # would otherwise fail to cover the original product.
            polydata = self._to_polydata(self.current_mesh)
            face_count = len(self.current_mesh.faces)
            base_rgb = np.tile(
                np.asarray([178, 188, 195], dtype=np.uint8),
                (face_count, 1),
            )

            face_opacity = np.ones(face_count, dtype=np.float32)
            if self.face_surface_types is not None:
                surface_types = np.char.lower(
                    np.asarray(self.face_surface_types).astype(str)
                )
                planar_mask = surface_types == "planar"
                curved_mask = surface_types == "curved"
                classified_mask = planar_mask | curved_mask
                planar_selected = bool(self.trajectory_visibility["planar"])
                curved_selected = bool(self.trajectory_visibility["curved"])

                # Color only the selected surface class. Unselected
                # classified faces remain in the mesh but become translucent.
                # When both classes are selected, show the original neutral CAD
                # appearance without any classification colors.
                if planar_selected and curved_selected:
                    pass
                elif planar_selected:
                    base_rgb[planar_mask] = [75, 155, 220]  # soft blue
                    face_opacity[classified_mask & ~planar_mask] = 0.20
                elif curved_selected:
                    base_rgb[curved_mask] = [230, 150, 75]  # soft orange
                    face_opacity[classified_mask & ~curved_mask] = 0.20

            # VTK/PyVista handles per-cell alpha reliably when it is part
            # of the direct RGBA color array. Do not also pass an opacity
            # scalar to add_mesh: that multiplies the RGBA alpha a second
            # time and causes a shape-broadcasting error in some versions.
            face_alpha = np.clip(
                np.rint(face_opacity * 255.0), 0, 255
            ).astype(np.uint8)

            if self.face_status_labels is not None:
                status = np.asarray(self.face_status_labels, dtype=np.uint8)
                base_rgb[status == FACE_EXCLUDED] = [89, 99, 107]
                # A paintable candidate remains unpainted until spray samples hit it.
                base_rgb[status == FACE_PAINTABLE_CANDIDATE] = [178, 188, 195]
                base_rgb[status == FACE_REQUIRES_INSPECTION] = [255, 215, 64]
                base_rgb[status == FACE_INVALID] = [255, 82, 82]

            self._trajectory_surface_face_mask = np.ones(face_count, dtype=bool)
            self._trajectory_surface_base_rgb = base_rgb.copy()
            self._trajectory_surface_base_alpha = face_alpha.copy()
            self._trajectory_surface_polydata = polydata
            polydata.cell_data["Paint Preview RGBA"] = np.column_stack(
                (base_rgb, face_alpha)
            ).astype(np.uint8)

            self._pose_actor = self.plotter.add_mesh(
                polydata,
                scalars="Paint Preview RGBA",
                rgb=True,
                preference="cell",
                show_edges=False,
                edge_color="#35434B",
                line_width=0.7,
                smooth_shading=True,
                pickable=False,
            )
            self._pose_actor.SetOrigin(0.0, 0.0, 0.0)
            self._pose_actor.SetUserMatrix(
                self._vtk_matrix_from_numpy(self.trajectory_product_transform)
            )

            self._add_trajectory_visualization()
            if self.face_status_labels is not None:
                self._add_spray_coverage_legend()

        # DEFAULT PRODUCT MODE
        # -------------------------------------------------
        else:
            # Product mode displays the CAD mesh as the single movable
            # actor. The World box, axes, and reference grid are static.
            # The product is the only movable 3D actor. World reference
            # geometry is rendered separately and remains fixed.
            if (
                self.display_mode == "spray_point_preview"
                and self._interactive_spray_face_thickness_um is not None
            ):
                polydata = self._to_polydata(self.current_mesh)
                paint_rgb = self._interactive_paint_rgb()
                if paint_rgb is not None and len(paint_rgb) == polydata.n_cells:
                    polydata.cell_data["Interactive Paint RGB"] = paint_rgb
                    self._pose_actor = self.plotter.add_mesh(
                        polydata,
                        scalars="Interactive Paint RGB",
                        rgb=True,
                        opacity=1.0,
                        show_edges=False,
                        edge_color="#555555",
                        line_width=1.0,
                        smooth_shading=False,
                        specular=0.35,
                        specular_power=25,
                        pickable=True,
                    )
                else:
                    self._pose_actor = self.plotter.add_mesh(
                        self.current_mesh,
                        color="#A8A8A8",
                        opacity=1.0,
                        show_edges=False,
                        edge_color="#555555",
                        line_width=1.0,
                        smooth_shading=False,
                        specular=0.35,
                        specular_power=25,
                        pickable=True,
                    )
            else:
                self._pose_actor = self.plotter.add_mesh(
                    self.current_mesh,
                    color="#A8A8A8",
                    opacity=1.0,
                    show_edges=False,
                    edge_color="#555555",
                    line_width=1.0,
                    smooth_shading=False,
                    specular=0.35,
                    specular_power=25,
                    pickable=True,
                )
            # CAD (0,0,0) is the rotation pivot and is aligned exactly
            # with World (0,0,0) at initial load.
            self.rotation_pivot = np.zeros(3, dtype=float)
            self._pose_actor.SetOrigin(0.0, 0.0, 0.0)
            self._pose_actor.SetPosition(0.0, 0.0, 0.0)
            self._pose_actor.SetOrientation(0.0, 0.0, 0.0)
            if previous_product_pose is not None:
                from vtkmodules.vtkCommonTransforms import vtkTransform
                pose_transform = vtkTransform()
                pose_transform.SetMatrix(self._vtk_matrix_from_numpy(previous_product_pose))
                self._pose_actor.SetOrientation(*pose_transform.GetOrientation())
                self._pose_actor.SetPosition(*previous_product_pose[:3, 3].tolist())

        # -------------------------------------------------
        # COMMON VIEWER ELEMENTS
        # -------------------------------------------------
        # Fixed World XYZ box and moving Product CAD frame.
        self._add_pose_coordinate_frames()

        # Overall dimensions are a permanent, subdued reference in the
        # product viewer. Keep this call in the common render path so the
        # dimensions appear in product, patch, and trajectory views.
        self._add_dimension_box()

        self._add_hole_candidate_overlay()

        # -------------------------------------------------
        # CAMERA RESTORATION
        # -------------------------------------------------
        if previous_camera is not None:
            try:
                self.plotter.camera_position = previous_camera
                self.plotter.reset_camera_clipping_range()
            except Exception:
                self._set_standard_camera()
        elif (
            self.display_mode in ("product", "classification", "trajectory", "spray_point_preview")
            and self._pose_actor is not None
        ):
            self._set_product_camera()
        else:
            self._set_standard_camera()

        self.plotter.render()

        # Direct VTK cell picking is installed once on the interactor.
        # It is active only while Stage 1 patch mode is displayed.

        if self._pose_actor is not None:
            self._update_pose_readout()

    def _add_spray_coverage_legend(self):
        """Explain target eligibility and sampled spray hits separately."""
        legend = (
            "SPRAY PREVIEW STATUS\n"
            "LIGHT GREY - APPROVED TARGET, NOT REACHED\n"
            "PAINT COLOR - SPRAY-ENVELOPE HIT\n"
            "DARK GREY - EXCLUDED FROM TARGET AREA\n"
            "YELLOW - REQUIRES INSPECTION\n"
            "RED - INVALID SURFACE\n\n"
            "WHEN FILM THICKNESS IS OFF: COVERAGE COLORS SHOW GEOMETRIC REACH"
        )
        self.plotter.add_text(
            legend,
            position="upper_right",
            font_size=9,
            color="#E1EAF0",
            shadow=True,
        )

    def _add_classification_legend(self):
        legend = (
            "SURFACE CLASSIFICATION\n"
            "GREEN   - PAINTABLE CANDIDATE\n"
            "GREY    - EXCLUDED\n"
            "YELLOW  - REQUIRES INSPECTION\n"
            "RED     - INVALID\n\n"
            "ROBOT ACCESSIBILITY: NOT CHECKED"
        )

        self.plotter.add_text(
            legend,
            position="upper_right",
            font_size=9,
            color="#E1EAF0",
            shadow=True,
        )

    # =========================================================
    # GRID SPACING
    # =========================================================

    @staticmethod
    def _nice_grid_spacing(value):
        if not np.isfinite(value) or value <= 0:
            return 1.0

        raw = value / 8.0
        magnitude = 10 ** np.floor(np.log10(raw))
        normalized = raw / magnitude

        if normalized <= 1:
            step = 1
        elif normalized <= 2:
            step = 2
        elif normalized <= 5:
            step = 5
        else:
            step = 10

        return float(step * magnitude)

    @staticmethod
    def _grid_values(start, stop, spacing):
        first = np.ceil(start / spacing) * spacing

        return np.arange(
            first,
            stop + spacing * 0.5,
            spacing,
        )

    # =========================================================
    # LINE GEOMETRY
    # =========================================================

    @staticmethod
    def _build_line_polydata(segments):
        if not segments:
            return None

        points = []
        lines = []

        for p1, p2 in segments:
            index = len(points)

            points.append(p1)
            points.append(p2)

            lines.extend([2, index, index + 1])

        polydata = pv.PolyData(
            np.asarray(points, dtype=float)
        )

        polydata.lines = np.asarray(
            lines, dtype=np.int64
        )

        return polydata

    # =========================================================
    # GRID LABEL FORMATTING
    # =========================================================

    def _format_coordinate(self, value_source):
        value = self._convert_source_length_to_display(
            value_source
        )

        if self.source_unit == "unknown":
            precision = 3
        elif self.display_unit in ("mm", "cm"):
            precision = 2
        else:
            precision = 4

        if abs(value) < 10 ** (-precision) / 2:
            value = 0.0

        return f"{value:.{precision}f}"

    def _make_grid_labels(self, values, axis_name):
        unit_label = self._display_unit_label()

        return [
            f"{axis_name} {self._format_coordinate(value)} {unit_label}"
            for value in values
        ]

    def _add_grid_coordinate_labels(
        self,
        x_values,
        y_values,
        z_values,
        xmin,
        ymin,
        zmin,
    ):
        x_points = np.array([
            [x, ymin, zmin] for x in x_values
        ], dtype=float)

        if len(x_points):
            self.plotter.add_point_labels(
                x_points,
                self._make_grid_labels(x_values, "X"),
                font_size=9,
                text_color="#00E5FF",
                shape="rounded_rect",
                shape_color="#071018",
                shape_opacity=0.8,
                always_visible=True,
                show_points=False,
                margin=2,
            )

        y_points = np.array([
            [xmin, y, zmin] for y in y_values
        ], dtype=float)

        if len(y_points):
            self.plotter.add_point_labels(
                y_points,
                self._make_grid_labels(y_values, "Y"),
                font_size=9,
                text_color="#FFD740",
                shape="rounded_rect",
                shape_color="#071018",
                shape_opacity=0.8,
                always_visible=True,
                show_points=False,
                margin=2,
            )

        z_points = np.array([
            [xmin, ymin, z] for z in z_values
        ], dtype=float)

        if len(z_points):
            self.plotter.add_point_labels(
                z_points,
                self._make_grid_labels(z_values, "Z"),
                font_size=9,
                text_color="#00E676",
                shape="rounded_rect",
                shape_color="#071018",
                shape_opacity=0.8,
                always_visible=True,
                show_points=False,
                margin=2,
            )

    # =========================================================
    # SIX-FACE 3D ENGINEERING GRID
    # =========================================================

    def _add_3d_box_grid(self):
        bounds = np.asarray(
            self.current_mesh.bounds,
            dtype=float,
        )

        minimum = bounds[0]
        maximum = bounds[1]

        dimensions = maximum - minimum

        size = max(float(np.max(dimensions)), 1e-9)
        margin = size * 0.18

        grid_min = minimum - margin
        grid_max = maximum + margin

        xmin, ymin, zmin = grid_min
        xmax, ymax, zmax = grid_max

        spacing = self._nice_grid_spacing(size)

        while True:
            x_values = self._grid_values(
                xmin, xmax, spacing
            )
            y_values = self._grid_values(
                ymin, ymax, spacing
            )
            z_values = self._grid_values(
                zmin, zmax, spacing
            )

            if max(
                len(x_values),
                len(y_values),
                len(z_values),
            ) <= 30:
                break

            spacing *= 2.0

        self.grid_spacing_coordinate_units = spacing
        self.grid_spacing_mm = spacing

        minor_segments = []

        for x in (xmin, xmax):
            for y in y_values:
                minor_segments.append(((x, y, zmin), (x, y, zmax)))
            for z in z_values:
                minor_segments.append(((x, ymin, z), (x, ymax, z)))

        for y in (ymin, ymax):
            for x in x_values:
                minor_segments.append(((x, y, zmin), (x, y, zmax)))
            for z in z_values:
                minor_segments.append(((xmin, y, z), (xmax, y, z)))

        for z in (zmin, zmax):
            for x in x_values:
                minor_segments.append(((x, ymin, z), (x, ymax, z)))
            for y in y_values:
                minor_segments.append(((xmin, y, z), (xmax, y, z)))

        minor_grid = self._build_line_polydata(minor_segments)

        if minor_grid is not None:
            self.plotter.add_mesh(
                minor_grid,
                color="#29404D",
                line_width=1,
                opacity=0.22,
                lighting=False,
            )

        major_spacing = spacing * 5.0

        major_x = self._grid_values(xmin, xmax, major_spacing)
        major_y = self._grid_values(ymin, ymax, major_spacing)
        major_z = self._grid_values(zmin, zmax, major_spacing)

        major_segments = []

        for x in (xmin, xmax):
            for y in major_y:
                major_segments.append(((x, y, zmin), (x, y, zmax)))
            for z in major_z:
                major_segments.append(((x, ymin, z), (x, ymax, z)))

        for y in (ymin, ymax):
            for x in major_x:
                major_segments.append(((x, y, zmin), (x, y, zmax)))
            for z in major_z:
                major_segments.append(((xmin, y, z), (xmax, y, z)))

        for z in (zmin, zmax):
            for x in major_x:
                major_segments.append(((x, ymin, z), (x, ymax, z)))
            for y in major_y:
                major_segments.append(((xmin, y, z), (xmax, y, z)))

        major_grid = self._build_line_polydata(major_segments)

        if major_grid is not None:
            self.plotter.add_mesh(
                major_grid,
                color="#456273",
                line_width=1.5,
                opacity=0.38,
                lighting=False,
            )

        grid_box = pv.Box(
            bounds=(xmin, xmax, ymin, ymax, zmin, zmax)
        )

        self.plotter.add_mesh(
            grid_box,
            style="wireframe",
            color="#557887",
            line_width=1.5,
            opacity=0.55,
            lighting=False,
        )

        label_x = self._grid_values(xmin, xmax, major_spacing)
        label_y = self._grid_values(ymin, ymax, major_spacing)
        label_z = self._grid_values(zmin, zmax, major_spacing)

        if len(label_x) > 12:
            label_x = label_x[::2]
        if len(label_y) > 12:
            label_y = label_y[::2]
        if len(label_z) > 12:
            label_z = label_z[::2]

        self._add_grid_coordinate_labels(
            label_x,
            label_y,
            label_z,
            xmin,
            ymin,
            zmin,
        )

    # =========================================================
    # PRODUCT DIMENSION BOX
    # =========================================================

    def _add_dimension_box(self):
        bounds = np.asarray(
            self.current_mesh.bounds,
            dtype=float,
        )

        minimum = bounds[0]
        maximum = bounds[1]

        xmin, ymin, zmin = minimum
        xmax, ymax, zmax = maximum

        width_source = xmax - xmin
        depth_source = ymax - ymin
        height_source = zmax - zmin

        width = self._convert_source_length_to_display(width_source)
        depth = self._convert_source_length_to_display(depth_source)
        height = self._convert_source_length_to_display(height_source)

        unit_label = self._display_unit_label()

        box = pv.Box(
            bounds=(xmin, xmax, ymin, ymax, zmin, zmax)
        )

        # Keep overall product dimensions visible as a subtle reference,
        # so the hook-compatible hole annotations remain visually dominant.
        self.plotter.add_mesh(
            box,
            style="wireframe",
            color="#71818C",
            line_width=1,
            opacity=0.28,
            lighting=False,
        )

        width_points = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
        ])

        depth_points = np.array([
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
        ])

        height_points = np.array([
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
        ])

        self.plotter.add_lines(width_points, color="#71818C", width=1)
        self.plotter.add_lines(depth_points, color="#71818C", width=1)
        self.plotter.add_lines(height_points, color="#71818C", width=1)

        label_points = np.array([
            [(xmin + xmax) / 2, ymin, zmin],
            [xmin, (ymin + ymax) / 2, zmin],
            [xmin, ymin, (zmin + zmax) / 2],
        ])

        labels = [
            f"LENGTH (X): {width:.2f} {unit_label}",
            f"WIDTH (Y): {depth:.2f} {unit_label}",
            f"HEIGHT (Z): {height:.2f} {unit_label}",
        ]

        self.plotter.add_point_labels(
            label_points,
            labels,
            font_size=12,
            text_color="#B8C4CC",
            shape="rounded_rect",
            shape_color="#101B24",
            shape_opacity=0.58,
            always_visible=True,
            show_points=False,
            margin=4,
        )

        self.plotter.add_text(
            (
                f"PRODUCT DIMENSIONS  |  "
                f"LENGTH {width:.2f}  "
                f"WIDTH {depth:.2f}  "
                f"HEIGHT {height:.2f} "
                f"{unit_label}"
            ),
            position="upper_left",
            font_size=11,
            color="#AAB5BD",
        )

    # =========================================================
    # GLOBAL COORDINATE AXES
    # =========================================================

    def _add_coordinate_axes(self):
        bounds = np.asarray(
            self.current_mesh.bounds,
            dtype=float,
        )

        minimum = bounds[0]
        maximum = bounds[1]

        dimensions = maximum - minimum
        size = max(float(np.max(dimensions)), 1e-9)
        axis_length = size * 0.35

        x_positive = max(
            axis_length,
            maximum[0] * 1.05 if maximum[0] > 0 else axis_length,
        )
        x_negative = max(
            axis_length,
            abs(minimum[0]) * 1.05 if minimum[0] < 0 else axis_length,
        )
        y_positive = max(
            axis_length,
            maximum[1] * 1.05 if maximum[1] > 0 else axis_length,
        )
        y_negative = max(
            axis_length,
            abs(minimum[1]) * 1.05 if minimum[1] < 0 else axis_length,
        )
        z_positive = max(
            axis_length,
            maximum[2] * 1.05 if maximum[2] > 0 else axis_length,
        )
        z_negative = max(
            axis_length,
            abs(minimum[2]) * 1.05 if minimum[2] < 0 else axis_length,
        )

        self.plotter.add_lines(
            np.array([
                [-x_negative, 0.0, 0.0],
                [x_positive, 0.0, 0.0],
            ]),
            color="#00E5FF",
            width=4,
        )

        self.plotter.add_lines(
            np.array([
                [0.0, -y_negative, 0.0],
                [0.0, y_positive, 0.0],
            ]),
            color="#FFD740",
            width=4,
        )

        self.plotter.add_lines(
            np.array([
                [0.0, 0.0, -z_negative],
                [0.0, 0.0, z_positive],
            ]),
            color="#00E676",
            width=4,
        )

        label_points = np.array([
            [x_positive, 0.0, 0.0],
            [0.0, y_positive, 0.0],
            [0.0, 0.0, z_positive],
        ])

        self.plotter.add_point_labels(
            label_points,
            ["+X", "+Y", "+Z"],
            font_size=15,
            text_color="#FFFFFF",
            shape="rounded_rect",
            shape_color="#071018",
            shape_opacity=0.9,
            always_visible=True,
            show_points=False,
            margin=4,
        )

    # =========================================================
    # GLOBAL ORIGIN MARKER
    # =========================================================

    def _add_origin_marker(self):
        origin = np.zeros(3, dtype=float)

        bounds = np.asarray(
            self.current_mesh.bounds,
            dtype=float,
        )

        dimensions = bounds[1] - bounds[0]
        size = max(float(np.max(dimensions)), 1e-9)

        marker = pv.Sphere(
            radius=size * 0.012,
            center=origin,
            theta_resolution=16,
            phi_resolution=16,
        )

        self.plotter.add_mesh(
            marker,
            color="#FFFFFF",
            smooth_shading=True,
        )

        self.plotter.add_point_labels(
            np.array([origin]),
            ["CAD ORIGIN (0, 0, 0)"],
            font_size=12,
            text_color="#FFFFFF",
            shape="rounded_rect",
            shape_color="#071018",
            shape_opacity=0.95,
            always_visible=True,
            show_points=False,
            margin=4,
        )

        self.plotter.add_text(
            (
                "COORDINATE FRAME: ORIGINAL CAD/STL  |  "
                f"STL UNIT: {self._source_unit_label()}  |  "
                f"DISPLAY UNIT: {self._display_unit_label()}"
            ),
            position="lower_left",
            font_size=9,
            color="#A7BAC5",
        )

    # =========================================================
    # STANDARD CAMERA
    # =========================================================

    def _set_product_camera(self):
        """Fit the loaded product tightly while keeping the oblique CAD view."""
        if self._pose_actor is None:
            self._set_standard_camera()
            return
        bounds = np.asarray(self._pose_actor.GetBounds(), dtype=float).reshape(3, 2)
        minimum, maximum = bounds[:, 0], bounds[:, 1]
        center = (minimum + maximum) * 0.5
        half_diagonal = max(float(np.linalg.norm(maximum - minimum) * 0.5), 1.0)
        camera = self.plotter.camera
        view_angle = max(float(camera.GetViewAngle()), 10.0)
        distance = half_diagonal / max(math.sin(math.radians(view_angle * 0.5)), 0.1) * 1.08
        direction = np.array([1.25, 2.0, -1.15], dtype=float)
        direction /= np.linalg.norm(direction)
        position = center + direction * distance
        camera.SetPosition(*position.tolist())
        camera.SetFocalPoint(*center.tolist())
        camera.SetViewUp(0.0, 0.0, -1.0)
        camera.OrthogonalizeViewUp()
        self.plotter.reset_camera_clipping_range()

    def _set_standard_camera(self):
        # Operator is in front of the box at +Y and looks toward -Y.
        # Screen-right is +X; the downward World axis is +Z.
        lx, ly, lz = np.asarray(self.workspace_dimensions, dtype=float)
        # The box and World origin are centered in X, Y, and Z.
        center = np.zeros(3, dtype=float)
        distance = max(float(np.linalg.norm(self.workspace_dimensions)), 100.0)

        # Oblique operator-side view: all three positive axes remain
        # visually separated instead of Y disappearing into depth.
        camera_position = center + np.array(
            [1.65 * distance, 2.55 * distance, -1.75 * distance],
            dtype=float,
        )

        self.plotter.camera_position = (
            tuple(camera_position.tolist()),
            tuple(center.tolist()),
            (0.0, 0.0, -1.0),
        )
        self.plotter.camera.zoom(1.12)
        self.plotter.reset_camera_clipping_range()

    def reset_camera(self):
        if self.current_mesh is None:
            return

        # RESET CAMERA always frames the product actor, including
        # Stage 1 patch and Stage 2 classification display modes.
        # The product pose and world position are not modified.
        if self._pose_actor is not None:
            self._set_product_camera()
        else:
            self._set_standard_camera()
        self.plotter.render()

    # =========================================================
    # SCREENSHOT
    # =========================================================

    def save_screenshot(self, path):
        self.plotter.screenshot(str(path))

    # =========================================================
    # SHUTDOWN
    # =========================================================

    def closeEvent(self, event):
        try:
            self.plotter.close()
        except (RuntimeError, AttributeError):
            pass

        super().closeEvent(event)
