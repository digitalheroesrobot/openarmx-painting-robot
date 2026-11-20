
from __future__ import annotations

import math
import numpy as np
import pyvista as pv

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
    QPushButton,
)
from pyvistaqt import QtInteractor


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

MAX_DIRECTION_ARROWS = 300


class ProductViewer(QWidget):

    pose_confirmed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_mesh = None
        self.patch_labels = None
        self.face_status_labels = None
        self.trajectory_data = None
        self.face_surface_types = None

        # Currently selected Stage 2 surface patch.
        self.selected_patch_id = None

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
        self.pose_manipulation_enabled = False
        self._pose_actor = None
        self._actor_style = None
        self._pose_interaction_callback = None
        self._cad_axes_actor = None
        self._rotation_widget = None
        self._rotation_representation = None
        self._rotation_last_matrix = None
        self._selection_observer_tag = None
        self._product_selected = False
        self._rotation_ring_actors = []
        self._rotation_observer_tags = []
        self._rotation_drag_axis = None
        self._rotation_drag_start_vector = None
        self._rotation_drag_start_matrix = None
        self._camera_style = None
        self._rotation_drag_style = None
        self._attitude_text_actor = None
        # Display-only bounds; operator must set actual work-cell extents.
        self.workspace_dimensions = np.array([1000.0, 1000.0, 1000.0], dtype=float)

        self.source_to_view_transform = np.eye(4, dtype=float)
        self.view_to_source_transform = np.eye(4, dtype=float)

        # Compatibility field: source-coordinate spacing, not always mm.
        self.grid_spacing_mm = 10.0
        self.grid_spacing_coordinate_units = 10.0

        self.trajectory_visibility = {
            "planar": True,
            "curved": True,
            "nozzle": True,
            "arrows": True,
            "paths": True,
            "samples": True,
        }

        self.plotter = QtInteractor(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.display_control_panel = self._build_trajectory_controls()
        layout.addWidget(self.display_control_panel)
        self.pose_control_panel = self._build_pose_controls()
        layout.addWidget(self.pose_control_panel)
        layout.addWidget(self.plotter.interactor, 1)

        self.plotter.set_background("#07111D", top="#142B40")

        self.display_control_panel.hide()
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

    def _build_trajectory_controls(self):

        panel = QFrame()
        panel.setObjectName("TrajectoryDisplayControls")

        panel.setStyleSheet("""
            QFrame#TrajectoryDisplayControls {
                background-color: #0D1822;
                border: 1px solid #203746;
                border-radius: 5px;
            }

            QLabel {
                color: #00E5FF;
                font-weight: bold;
                padding-right: 5px;
            }

            QCheckBox {
                color: #D8E5ED;
                spacing: 5px;
                padding: 3px;
            }

            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }

            QCheckBox::indicator:unchecked {
                background-color: #14232E;
                border: 1px solid #557887;
                border-radius: 2px;
            }

            QCheckBox::indicator:checked {
                background-color: #00E676;
                border: 1px solid #00E676;
                border-radius: 2px;
            }
        """)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(8)

        layout.addWidget(QLabel("3D DISPLAY:"))

        self.trajectory_checkboxes = {}

        checkbox_items = [
            ("planar", "Planar surfaces"),
            ("curved", "Curved surfaces"),
            ("nozzle", "Nozzle positions"),
            ("arrows", "Direction arrows"),
            ("paths", "Trajectory paths"),
            ("samples", "Surface samples"),
        ]

        for key, label_text in checkbox_items:
            checkbox = QCheckBox(label_text)
            checkbox.setChecked(True)

            checkbox.toggled.connect(
                lambda checked, item=key:
                self._on_trajectory_visibility_changed(
                    item, checked
                )
            )

            layout.addWidget(checkbox)
            self.trajectory_checkboxes[key] = checkbox

        layout.addStretch()
        return panel

    def _on_trajectory_visibility_changed(self, key, checked):
        self.trajectory_visibility[key] = bool(checked)

        if (
            self.display_mode == "trajectory"
            and self.current_mesh is not None
        ):
            self._render_product(reset_camera=False)

    def _set_trajectory_controls_visible(self, visible):
        self.display_control_panel.setVisible(bool(visible))

    # =========================================================
    # MANUAL REFERENCE POSE CONTROLS
    # =========================================================

    def _build_pose_controls(self):
        panel = QFrame()
        panel.setObjectName("PoseReferenceControls")
        panel.setStyleSheet("""
            QFrame#PoseReferenceControls { background-color: #0D1822; border: 1px solid #294858; border-radius: 5px; }
            QLabel { color: #D8E5ED; padding: 2px; }
            QPushButton { color: #EAF4F8; background-color: #173747; border: 1px solid #3D7185; border-radius: 4px; padding: 5px 8px; font-weight: bold; }
            QPushButton:disabled { color: #71838D; background-color: #17232B; }
        """)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(5)

        layout.addWidget(QLabel("CAD ORIGIN / WORLD XYZ:"))
        self.world_position_inputs = []
        for axis in "XYZ":
            field = QDoubleSpinBox()
            field.setRange(-10000000.0, 10000000.0)
            field.setDecimals(3)
            field.setSingleStep(10.0)
            field.setSuffix(f" {axis}")
            field.setFixedWidth(82)
            field.setToolTip(f"World {axis} coordinate of STEP CAD origin (0,0,0), in source CAD units.")
            layout.addWidget(field)
            self.world_position_inputs.append(field)

        self.apply_position_button = QPushButton("SET POSITION")
        self.apply_position_button.clicked.connect(self._apply_world_position)
        layout.addWidget(self.apply_position_button)

        layout.addWidget(QLabel("BOX X/Y/Z:"))
        self.workspace_size_inputs = []
        for axis in "XYZ":
            field = QDoubleSpinBox()
            field.setRange(1.0, 10000000.0)
            field.setDecimals(1)
            field.setSingleStep(100.0)
            field.setValue(1000.0)
            field.setSuffix(f" {axis}")
            field.setFixedWidth(72)
            field.setToolTip("Configurable display-box extent in source CAD units; not a measured work-cell dimension.")
            layout.addWidget(field)
            self.workspace_size_inputs.append(field)

        self.apply_workspace_button = QPushButton("APPLY BOX")
        self.apply_workspace_button.clicked.connect(self._apply_workspace_bounds)
        layout.addWidget(self.apply_workspace_button)

        layout.addWidget(QLabel("HOOK LENGTH:"))
        self.hook_length_input = QDoubleSpinBox()
        self.hook_length_input.setRange(0.0, 10000000.0)
        self.hook_length_input.setDecimals(2)
        self.hook_length_input.setSingleStep(10.0)
        self.hook_length_input.setValue(self.hook_length_source_units)
        self.hook_length_input.setSuffix(" CAD units")
        self.hook_length_input.setFixedWidth(130)
        self.hook_length_input.setToolTip("Vertical distance from the conveyor hook suspension point to the hook engagement point on the product, measured downward along +Z.")
        self.hook_length_input.valueChanged.connect(self._on_hook_length_changed)
        layout.addWidget(self.hook_length_input)

        self.pick_hook_point_button = QPushButton("PICK PRODUCT HOOK POINT")
        self.pick_hook_point_button.clicked.connect(self._start_hook_point_picking)
        layout.addWidget(self.pick_hook_point_button)

        self.pose_status_label = QLabel("World: +X conveyor travel, -Y viewing direction, +Z downward. Select the product hook point to place it.")
        self.pose_status_label.setMinimumWidth(230)
        layout.addWidget(self.pose_status_label, 1)

        self.confirm_pose_button = QPushButton("CONFIRM & SAVE POSE")
        self.confirm_pose_button.setEnabled(False)
        self.confirm_pose_button.clicked.connect(self._confirm_reference_pose)
        layout.addWidget(self.confirm_pose_button)

        layout.addWidget(QLabel("ATTITUDE RPY (°):"))
        self.rpy_inputs = []
        for label in ("ROLL", "PITCH", "YAW"):
            field = QDoubleSpinBox()
            field.setRange(-3600.0, 3600.0)
            field.setDecimals(2)
            field.setSingleStep(1.0)
            field.setSuffix(f"° {label}")
            field.setFixedWidth(92)
            field.setToolTip(f"Absolute {label.lower()} angle in the World XYZ frame.")
            layout.addWidget(field)
            self.rpy_inputs.append(field)

        self.apply_rpy_button = QPushButton("APPLY RPY")
        self.apply_rpy_button.clicked.connect(self._apply_rpy_inputs)
        layout.addWidget(self.apply_rpy_button)

        self.reset_pose_button = QPushButton("RESET")
        self.reset_pose_button.clicked.connect(self._reset_reference_pose)
        layout.addWidget(self.reset_pose_button)
        return panel

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
        matrix = self._actor_matrix_array()
        matrix[:3, :3] = rz @ ry @ rx
        from vtkmodules.vtkCommonTransforms import vtkTransform
        transform = vtkTransform()
        transform.SetMatrix(self._vtk_matrix_from_numpy(matrix))
        self._pose_actor.SetOrientation(*transform.GetOrientation())
        self._update_rotation_gizmo_transform()
        self._update_pose_readout()
        self.pose_status_label.setText("ATTITUDE UPDATED  |  Product set to the entered absolute RPY.")
        self.plotter.render()

    def _apply_world_position(self):
        if self._pose_actor is None:
            return
        position = np.asarray([field.value() for field in self.world_position_inputs], dtype=float)
        if not np.all(np.isfinite(position)):
            self.pose_status_label.setText("Invalid World XYZ position.")
            return
        self._set_cad_origin_world_position(position)
        self._update_rotation_gizmo_transform()
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

    def _install_actor_interaction(self):
        """Install a selectable local-CAD X/Y/Z rotation-ring manipulator."""
        if self._pose_actor is None:
            return
        try:
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleUser
            from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkPropPicker

            interactor = self.plotter.interactor
            self._camera_style = vtkInteractorStyleTrackballCamera()
            self._camera_style.SetDefaultRenderer(self.plotter.renderer)
            self._rotation_drag_style = vtkInteractorStyleUser()
            self._rotation_drag_style.SetDefaultRenderer(self.plotter.renderer)
            interactor.SetInteractorStyle(self._camera_style)

            self._create_rotation_gizmo()
            ring_picker = vtkCellPicker()
            ring_picker.SetTolerance(0.012)
            ring_picker.PickFromListOn()
            for ring_actor in self._rotation_ring_actors:
                ring_picker.AddPickList(ring_actor)
            product_picker = vtkPropPicker()

            def on_press(caller, event):
                x, y = interactor.GetEventPosition()
                hit_ring = ring_picker.Pick(x, y, 0, self.plotter.renderer)
                ring_actor = ring_picker.GetActor() if hit_ring else None
                if ring_actor in self._rotation_ring_actors and self._product_selected:
                    axis_index = self._rotation_ring_actors.index(ring_actor)
                    start_vector = self._screen_vector_from_cad_origin(x, y, axis_index)
                    if start_vector is not None:
                        self._rotation_drag_axis = axis_index
                        self._rotation_drag_start_vector = start_vector
                        self._rotation_drag_start_matrix = self._actor_matrix_array().copy()
                        interactor.SetInteractorStyle(self._rotation_drag_style)
                        return
                hit_product = product_picker.Pick(x, y, 0, self.plotter.renderer)
                picked = product_picker.GetActor() if hit_product else None
                self._set_product_selected(picked == self._pose_actor)

            def on_move(caller, event):
                if self._rotation_drag_axis is None:
                    return
                x, y = interactor.GetEventPosition()
                current = self._screen_vector_from_cad_origin(x, y, self._rotation_drag_axis)
                if current is None:
                    return
                start = self._rotation_drag_start_vector
                index = self._rotation_drag_axis
                local_axis = np.eye(3, dtype=float)[index]
                base = self._rotation_drag_start_matrix
                base_rotation = base[:3, :3]
                world_axis = base_rotation @ local_axis
                angle = math.atan2(
                    float(np.dot(world_axis, np.cross(start, current))),
                    float(np.clip(np.dot(start, current), -1.0, 1.0)),
                )
                ux, uy, uz = local_axis
                skew = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
                delta = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
                updated = base.copy()
                updated[:3, :3] = base_rotation @ delta
                from vtkmodules.vtkCommonTransforms import vtkTransform
                transform = vtkTransform()
                transform.SetMatrix(self._vtk_matrix_from_numpy(updated))
                self._pose_actor.SetOrientation(*transform.GetOrientation())
                self._update_rotation_gizmo_transform()
                self._update_pose_readout()
                self.plotter.render()

            def on_release(caller, event):
                if self._rotation_drag_axis is None:
                    return
                self._rotation_drag_axis = None
                self._rotation_drag_start_vector = None
                self._rotation_drag_start_matrix = None
                interactor.SetInteractorStyle(self._camera_style)
                self._update_pose_readout()
                self.pose_status_label.setText("ROTATION COMPLETE  |  Camera orbit and zoom are available.")

            self._rotation_observer_tags = [
                interactor.AddObserver("LeftButtonPressEvent", on_press, 1.0),
                interactor.AddObserver("MouseMoveEvent", on_move, 1.0),
                interactor.AddObserver("LeftButtonReleaseEvent", on_release, 1.0),
            ]
            self._update_rotation_gizmo_transform()
            self._update_pose_readout()
            self.plotter.render()
        except Exception as exc:
            self.pose_status_label.setText(f"Rotation manipulator unavailable: {exc}")
            self.confirm_pose_button.setEnabled(False)

    def _create_rotation_gizmo(self):
        """Build three colored rings in CAD-local planes, centered on CAD origin."""
        if self._pose_actor is None:
            return
        bounds = np.asarray(self.current_mesh.bounds, dtype=float)
        diagonal = float(np.linalg.norm(bounds[[1, 3, 5]] - bounds[[0, 2, 4]]))
        radius = max(diagonal * 0.62, 10.0)
        tube_radius = max(radius * 0.012, diagonal * 0.002)
        planes = (
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        )
        self._rotation_ring_actors = []
        for index, (u, v) in enumerate(planes):
            theta = np.linspace(0.0, 2.0 * math.pi, 129)
            points = radius * np.cos(theta)[:, None] * u + radius * np.sin(theta)[:, None] * v
            line = pv.PolyData(points)
            line.lines = np.hstack((np.array([len(points)], dtype=np.int64), np.arange(len(points), dtype=np.int64)))
            tube = line.tube(radius=tube_radius, n_sides=10)
            actor = self.plotter.add_mesh(
                tube, color=("#FF5252", "#00E676", "#40C4FF")[index],
                lighting=False, pickable=True, render=False,
            )
            actor.SetVisibility(False)
            self._rotation_ring_actors.append(actor)

    def _update_rotation_gizmo_transform(self):
        if self._pose_actor is None:
            return
        matrix = self._vtk_matrix_from_numpy(self._actor_matrix_array())
        for ring_actor in self._rotation_ring_actors:
            ring_actor.SetUserMatrix(matrix)
            ring_actor.SetVisibility(self._product_selected)

    def _screen_vector_from_cad_origin(self, display_x, display_y, axis_index):
        """Project a display point onto the plane normal to a local CAD axis."""
        renderer = self.plotter.renderer
        matrix = self._actor_matrix_array()
        origin = matrix[:3, 3]
        renderer.SetWorldPoint(float(origin[0]), float(origin[1]), float(origin[2]), 1.0)
        renderer.WorldToDisplay()
        depth = renderer.GetDisplayPoint()[2]
        renderer.SetDisplayPoint(float(display_x), float(display_y), float(depth))
        renderer.DisplayToWorld()
        world = np.asarray(renderer.GetWorldPoint(), dtype=float)
        if abs(world[3]) < 1e-12:
            return None
        point = world[:3] / world[3]
        vector = point - origin
        axis = matrix[:3, :3] @ np.eye(3, dtype=float)[axis_index]
        vector -= axis * float(np.dot(vector, axis))
        norm = float(np.linalg.norm(vector))
        return None if norm < 1e-9 else vector / norm

    @staticmethod
    def _vtk_transform_matrix(transform):
        matrix = transform.GetMatrix()
        return np.array(
            [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)],
            dtype=float,
        )

    def _set_product_selected(self, selected):
        self._product_selected = bool(selected)
        self._update_rotation_gizmo_transform()
        self.pose_status_label.setText(
            "PRODUCT SELECTED  |  Drag red X, green Y, or blue Z ring to rotate."
            if self._product_selected else
            "PRODUCT DESELECTED  |  Click the product to show its rotation rings."
        )
        self.plotter.render()

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
            return np.eye(4, dtype=float)
        matrix = self._pose_actor.GetMatrix()
        return np.array([[matrix.GetElement(row, col) for col in range(4)] for row in range(4)], dtype=float)

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
        text = f"ROLL  {rpy[0]:.2f}°    PITCH  {rpy[1]:.2f}°    YAW  {rpy[2]:.2f}°"
        if self._attitude_text_actor is None:
            self._attitude_text_actor = self.plotter.add_text(
                text, position="upper_left", font_size=12, color="#FFFFFF", shadow=True
            )
        else:
            self._attitude_text_actor.SetInput(text)
        if self._cad_axes_actor is not None:
            self._cad_axes_actor.SetUserMatrix(self._vtk_matrix_from_numpy(self._actor_matrix_array()))

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
            self._update_rotation_gizmo_transform()
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
        mesh_copy = mesh.copy()

        self.source_origin = self._calculate_cad_origin(mesh_copy)
        bounds = np.asarray(mesh_copy.bounds, dtype=float)
        self.product_center_pivot = (bounds[0] + bounds[1]) * 0.5
        # Rotation pivot is the STEP CAD coordinate origin, not the mesh center.
        self.rotation_pivot = np.zeros(3, dtype=float)

        self.source_to_view_transform = np.eye(4, dtype=float)
        self.view_to_source_transform = np.eye(4, dtype=float)

        return mesh_copy

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
        faces = np.asarray(mesh.faces, dtype=np.int64)

        if face_mask is not None:
            face_mask = np.asarray(face_mask, dtype=bool)

            if len(face_mask) != len(faces):
                raise ValueError(
                    "Face visibility mask does not match mesh faces."
                )

            faces = faces[face_mask]

        if len(faces) == 0:
            return pv.PolyData()

        packed_faces = np.hstack([
            np.full((len(faces), 1), 3, dtype=np.int64),
            faces,
        ])

        return pv.PolyData(
            np.asarray(mesh.vertices),
            packed_faces.ravel(),
        )

    # =========================================================
    # EMPTY SCENE
    # =========================================================

    def _draw_empty_scene(self):
        self.plotter.clear()
        self._pose_actor = None
        self._actor_style = None
        self._cad_axes_actor = None
        self.plotter.set_background("#07111D", top="#142B40")

        self.plotter.render()

    # =========================================================
    # PRODUCT DISPLAY MODES
    # =========================================================

    def show_product(self, mesh):
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
        self.pose_manipulation_enabled = True
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
        self.pose_control_panel.hide()

        self._render_product(reset_camera=True)

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
        self.trajectory_data = None
        self.face_surface_types = None
        self.selected_patch_id = None

        self.display_mode = "classification"
        self._set_trajectory_controls_visible(False)
        self.pose_control_panel.hide()

        self._render_product(reset_camera=True)

    def show_spray_trajectory(
        self,
        mesh,
        trajectory_data,
        face_surface_types=None,
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
        self.face_status_labels = None
        self.selected_patch_id = None
        self.trajectory_data = data

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
        self._set_trajectory_controls_visible(True)
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

    # =========================================================
    # STAGE 3 TRAJECTORY RENDERING
    # =========================================================

    def _add_trajectory_visualization(self):
        data = self.trajectory_data

        if data is None:
            return

        surface_points = data["surface_points"]
        nozzle_points = data["nozzle_points"]
        spray_directions = data["spray_directions"]
        segment_ids = data["segment_ids"]
        sample_indices = data["sample_indices"]

        count = len(surface_points)

        if self.trajectory_visibility["samples"]:
            self.plotter.add_points(
                surface_points,
                color="#00E676",
                point_size=6,
                render_points_as_spheres=True,
            )

        if self.trajectory_visibility["nozzle"]:
            self.plotter.add_points(
                nozzle_points,
                color="#00E5FF",
                point_size=6,
                render_points_as_spheres=True,
            )

        if self.trajectory_visibility["arrows"]:
            stride = max(
                1,
                int(np.ceil(count / MAX_DIRECTION_ARROWS)),
            )

            arrow_indices = np.arange(
                0, count, stride, dtype=np.int64
            )

            arrow_cloud = pv.PolyData(
                nozzle_points[arrow_indices]
            )
            arrow_cloud["spray_vectors"] = (
                spray_directions[arrow_indices]
            )

            arrow_source = pv.Arrow(
                tip_length=0.25,
                tip_radius=0.08,
                shaft_radius=0.025,
            )

            arrow_glyphs = arrow_cloud.glyph(
                orient="spray_vectors",
                scale=False,
                factor=self._trajectory_arrow_length(),
                geom=arrow_source,
            )

            self.plotter.add_mesh(
                arrow_glyphs,
                color="#FFD740",
                lighting=False,
            )

        if self.trajectory_visibility["paths"]:
            line_points = []
            line_cells = []

            for segment_id in np.unique(segment_ids):
                indices = np.where(
                    segment_ids == segment_id
                )[0]

                if len(indices) < 2:
                    continue

                order = np.argsort(
                    sample_indices[indices],
                    kind="stable",
                )
                indices = indices[order]

                start = len(line_points)
                line_points.extend(
                    nozzle_points[indices].tolist()
                )

                local_indices = np.arange(
                    start,
                    start + len(indices),
                    dtype=np.int64,
                )

                line_cells.extend([
                    len(local_indices),
                    *local_indices.tolist(),
                ])

            if line_points:
                path_polydata = pv.PolyData(
                    np.asarray(line_points, dtype=float)
                )
                path_polydata.lines = np.asarray(
                    line_cells, dtype=np.int64
                )

                self.plotter.add_mesh(
                    path_polydata,
                    color="#00E676",
                    line_width=3,
                    lighting=False,
                )

        visible_layers = []

        for key, label in [
            ("planar", "PLANAR SURFACES"),
            ("curved", "CURVED SURFACES"),
            ("nozzle", "NOZZLE POSITIONS"),
            ("arrows", "DIRECTION ARROWS"),
            ("paths", "TRAJECTORY PATHS"),
            ("samples", "SURFACE SAMPLES"),
        ]:
            state = "ON" if self.trajectory_visibility[key] else "OFF"
            visible_layers.append(f"{label}: {state}")

        legend = (
            "STAGE 3 - SPRAY TRAJECTORY\n"
            "GREEN   - SURFACE SAMPLES / PATH\n"
            "CYAN    - NOZZLE REFERENCE POINT\n"
            "YELLOW  - SPRAY DIRECTION\n\n"
            + "\n".join(visible_layers)
            + "\n\n"
            + f"TRAJECTORY SEGMENTS: {len(np.unique(segment_ids))}\n"
            + f"SAMPLE POINTS: {count}\n"
            + f"STL UNIT: {self._source_unit_label()}\n"
            + f"DISPLAY UNIT: {self._display_unit_label()}\n\n"
            + "ROBOT REACHABILITY: NOT CHECKED\n"
            + "COLLISION: NOT CHECKED\n"
            + "SPRAY COVERAGE: NOT VERIFIED\n"
            + "TCP CALIBRATION: NOT VERIFIED"
        )

        self.plotter.add_text(
            legend,
            position="upper_right",
            font_size=9,
            color="#E1EAF0",
            shadow=True,
        )

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

        # Three differently tinted edge families give the volume depth cues.
        x_edges = [(corners[a], corners[b]) for a, b in
                   [(0, 1), (3, 2), (4, 5), (7, 6)]]
        y_edges = [(corners[a], corners[b]) for a, b in
                   [(0, 3), (1, 2), (4, 7), (5, 6)]]
        z_edges = [(corners[a], corners[b]) for a, b in
                   [(0, 4), (1, 5), (2, 6), (3, 7)]]
        add_segments(x_edges, "#27C7F7", 3.0, 1.0)
        add_segments(y_edges, "#168F9F", 2.6, 0.98)
        add_segments(z_edges, "#426B91", 2.6, 0.98)

        # Fine reference grid on the central XY plane, passing through origin.
        grid_segments = []
        grid_count = 10
        for i in range(-grid_count, grid_count + 1):
            x = hx * i / grid_count
            y = hy * i / grid_count
            grid_segments.append((np.array([x, -hy, 0.0]), np.array([x, hy, 0.0])))
            grid_segments.append((np.array([-hx, y, 0.0]), np.array([hx, y, 0.0])))
        add_segments(grid_segments, "#24465B", 1.0, 0.42)

        # Center crosshair: a subtle reference marker around World (0,0,0).
        cross = min(lx, ly, lz) * 0.035
        add_segments([
            (np.array([-cross, 0.0, 0.0]), np.array([cross, 0.0, 0.0])),
            (np.array([0.0, -cross, 0.0]), np.array([0.0, cross, 0.0])),
            (np.array([0.0, 0.0, -cross]), np.array([0.0, 0.0, cross])),
        ], "#D7F6FF", 2.0, 0.9)

        # Positive coordinate arrows; keep their labels deliberately simple.
        axis_length = max(min(lx, ly, lz) * 0.24, 30.0)
        axes = [
            (np.array([1.0, 0.0, 0.0]), "#FF657A", "X"),
            (np.array([0.0, 1.0, 0.0]), "#4DDBA5", "Y"),
            (np.array([0.0, 0.0, 1.0]), "#48BFFF", "Z"),
        ]
        for direction, color, label in axes:
            self.plotter.add_mesh(
                pv.Arrow(
                    start=(0.0, 0.0, 0.0),
                    direction=direction,
                    scale=axis_length,
                    tip_length=0.28,
                    tip_radius=0.105,
                    shaft_radius=0.027,
                ),
                color=color,
                lighting=False,
                pickable=False,
            )
            self.plotter.add_point_labels(
                np.asarray([direction * axis_length * 1.22]),
                [label],
                font_size=20,
                text_color=color,
                shape="rounded_rect",
                shape_color="#07111D",
                shape_opacity=0.92,
                margin=4,
                point_size=0,
                always_visible=True,
                bold=True,
            )

        # Bright origin marker, centered exactly at the box midpoint.
        self.plotter.add_mesh(
            pv.Sphere(
                radius=max(min(lx, ly, lz) * 0.012, 3.0),
                center=(0.0, 0.0, 0.0),
                theta_resolution=24,
                phi_resolution=24,
            ),
            color="#F4FBFF",
            smooth_shading=True,
            specular=0.8,
            specular_power=35,
            pickable=False,
        )

        # Compact telemetry-style overlays; avoid long labels around the axes.

    def _add_product_coordinate_frame(self):
        if self._pose_actor is None:
            return
        from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
        axes = vtkAxesActor()
        axes.SetTotalLength(*[max(float(value) * 0.12, 10.0) for value in self.workspace_dimensions])
        axes.SetUserMatrix(self._vtk_matrix_from_numpy(self._actor_matrix_array()))
        self._cad_axes_actor = axes
        self.plotter.add_actor(axes, reset_camera=False)

    def _add_pose_coordinate_frames(self):
        if self.display_mode != "product":
            return

        # Initial setup stage: only the fixed World frame and work-cell
        # box are displayed. Product geometry is added in a later step.
        self._add_world_workspace()

    # =========================================================
    # MAIN RENDERING
    # =========================================================

    
    def _render_product(self, reset_camera=False):
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

        for tag in self._rotation_observer_tags:
            try:
                self.plotter.interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._rotation_observer_tags = []
        if self._selection_observer_tag is not None:
            try:
                self.plotter.interactor.RemoveObserver(self._selection_observer_tag)
            except Exception:
                pass
            self._selection_observer_tag = None
        self._rotation_drag_axis = None
        self._rotation_ring_actors = []
        self._attitude_text_actor = None
        if self._rotation_widget is not None:
            try:
                self._rotation_widget.SetEnabled(0)
            except Exception:
                pass
        self._rotation_widget = None
        self._rotation_representation = None
        self._product_selected = False
        self.plotter.clear()
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

            self.plotter.add_mesh(
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

            self.plotter.add_mesh(
                polydata,
                scalars="Surface Patch",
                cmap="turbo",
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

            # -------------------------------------------------
            # SELECTED PATCH HIGHLIGHT
            # No text overlay.
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
                    self.plotter.add_mesh(
                        selected_polydata,
                        color="#FFD740",
                        show_edges=True,
                        edge_color="#FFFFFF",
                        line_width=2.5,
                        smooth_shading=False,
                        lighting=False,
                    )

        # -------------------------------------------------
        # TRAJECTORY MODE
        # -------------------------------------------------
        elif (
            self.display_mode == "trajectory"
            and self.trajectory_data is not None
        ):
            face_mask = self._get_trajectory_surface_mask()

            polydata = self._to_polydata(
                self.current_mesh,
                face_mask=face_mask,
            )

            if polydata.n_cells > 0:
                self.plotter.add_mesh(
                    polydata,
                    color="#4C8DAD",
                    opacity=0.38,
                    show_edges=True,
                    edge_color="#29404D",
                    line_width=1,
                    smooth_shading=False,
                )

            self._add_trajectory_visualization()

        # -------------------------------------------------
        # DEFAULT PRODUCT MODE
        # -------------------------------------------------
        else:
            # Product mode displays the CAD mesh as the single movable
            # actor. The World box, axes, and reference grid are static.
            # The product is the only movable 3D actor. World reference
            # geometry is rendered separately and remains fixed.
            self._pose_actor = self.plotter.add_mesh(
                self.current_mesh,
                color="#9FB7C7",
                show_edges=True,
                edge_color="#263E4D",
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

        # -------------------------------------------------
        # CAMERA RESTORATION
        # -------------------------------------------------
        if previous_camera is not None:
            try:
                self.plotter.camera_position = previous_camera
                self.plotter.reset_camera_clipping_range()
            except Exception:
                self._set_standard_camera()
        elif self.display_mode == "product" and self._pose_actor is not None:
            self._set_product_camera()
        else:
            self._set_standard_camera()

        self.plotter.render()

        # Reinstall actor manipulation after a redraw (for example, a
        # display-unit change) without enabling it in analysis stages.
        if self.display_mode == "product" and self.pose_manipulation_enabled:
            self._install_actor_interaction()

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

        self.plotter.add_mesh(
            box,
            style="wireframe",
            color="#00E5FF",
            line_width=2,
            opacity=0.9,
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

        self.plotter.add_lines(width_points, color="#00E5FF", width=4)
        self.plotter.add_lines(depth_points, color="#FFD740", width=4)
        self.plotter.add_lines(height_points, color="#00E676", width=4)

        label_points = np.array([
            [(xmin + xmax) / 2, ymin, zmin],
            [xmin, (ymin + ymax) / 2, zmin],
            [xmin, ymin, (zmin + zmax) / 2],
        ])

        labels = [
            f"W: {width:.3f} {unit_label}",
            f"D: {depth:.3f} {unit_label}",
            f"H: {height:.3f} {unit_label}",
        ]

        self.plotter.add_point_labels(
            label_points,
            labels,
            font_size=14,
            text_color="#FFFFFF",
            shape="rounded_rect",
            shape_color="#071018",
            shape_opacity=0.85,
            always_visible=True,
            show_points=False,
            margin=4,
        )

        self.plotter.add_text(
            (
                f"PRODUCT SIZE  |  "
                f"W {width:.3f}  "
                f"D {depth:.3f}  "
                f"H {height:.3f} "
                f"{unit_label}"
            ),
            position="upper_left",
            font_size=10,
            color="#00E5FF",
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

        if self.display_mode == "product" and self._pose_actor is not None:
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
