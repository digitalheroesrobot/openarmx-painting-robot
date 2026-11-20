
from __future__ import annotations

import math
import numpy as np
import pyvista as pv

from PySide6.QtCore import QEvent, Qt, Signal
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
        self.pose_control_panel.setMinimumHeight(42)
        self.pose_control_panel.setMaximumHeight(48)
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
        # RPY is shown only in the three numeric selectors above the viewport.
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
        if self.display_mode != "product":
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

            self._pose_actor = self.plotter.add_mesh(
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
                    # Translucent cyan surface wash.
                    glow_actor = self.plotter.add_mesh(
                        selected_polydata,
                        color="#00E5FF",
                        opacity=0.38,
                        show_edges=False,
                        smooth_shading=False,
                        lighting=False,
                        ambient=1.0,
                        diffuse=0.0,
                    )

                    # Broad luminous edge pass.
                    outer_glow_actor = self.plotter.add_mesh(
                        selected_polydata,
                        style="wireframe",
                        color="#00E5FF",
                        opacity=0.78,
                        line_width=7.0,
                        render_lines_as_tubes=True,
                        lighting=False,
                    )

                    # Bright inner edge makes the selected region
                    # unmistakable against adjacent patch colors.
                    edge_actor = self.plotter.add_mesh(
                        selected_polydata,
                        style="wireframe",
                        color="#E9FFFF",
                        opacity=1.0,
                        line_width=2.5,
                        render_lines_as_tubes=True,
                        lighting=False,
                    )

                    if previous_product_pose is not None:
                        for selected_actor in (
                            glow_actor,
                            outer_glow_actor,
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
                opacity=0.42,
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
        if self._pose_actor is not None:
            self._update_pose_readout()

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
