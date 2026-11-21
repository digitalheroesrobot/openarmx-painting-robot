
from __future__ import annotations

import math
import numpy as np
import pyvista as pv

from core.spray_coverage import (
    build_paintable_face_mask,
    calculate_area_weighted_coverage,
    is_line_of_sight_occluded,
    mark_sample_hits,
)

from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkFiltersGeneral import vtkOBBTree

from PySide6.QtCore import QEvent, Qt, Signal, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
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


class ProductViewer(QWidget):

    pose_confirmed = Signal(dict)
    patch_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_mesh = None
        self.patch_labels = None
        self.face_status_labels = None
        self.trajectory_data = None
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
            "paths": False,
            "samples": False,
            "frames": False,
        }
        self.trajectory_product_transform = np.eye(4, dtype=float)
        self._last_product_transform = np.eye(4, dtype=float)

        # Stage 3 sequential trajectory playback state.
        self._playback_index = 0
        self._playback_elapsed_s = 0.0
        self._playback_finished = False
        self._playback_last_path_render_index = -1
        self._playback_path_actor = None
        self._playback_nozzle_actor = None
        self._playback_gun_actors = []
        self._playback_cache = None
        self._trajectory_paint_color_hex = None
        self._trajectory_surface_polydata = None
        self._trajectory_surface_face_mask = None
        self._trajectory_surface_base_rgb = None
        self._trajectory_surface_base_alpha = None
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

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
        playback_layout.addLayout(coverage_row)

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
        return panel

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

    def _on_patch_mouse_press(self, interactor, event):
        """Remember the mouse-down point; camera drags are not selections."""
        self._patch_mouse_press_position = tuple(
            interactor.GetEventPosition()
        )

    def _on_patch_mouse_release(self, interactor, event):
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
        face_status_labels=None,
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

    def _add_trajectory_visualization(self):
        data = self.trajectory_data

        if data is None:
            return

        transform = self.trajectory_product_transform
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        surface_points = data["surface_points"] @ rotation.T + translation
        nozzle_points = data["nozzle_points"] @ rotation.T + translation
        spray_directions = data["spray_directions"] @ rotation.T
        nozzle_frames = (
            np.einsum("ij,njk->nik", rotation, data["nozzle_frames"])
            if "nozzle_frames" in data else None
        )
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

        if (
            self.trajectory_visibility.get("frames", False)
            and nozzle_frames is not None
            and count > 0
        ):
            frame_stride = max(1, int(np.ceil(count / 40)))
            frame_indices = np.arange(0, count, frame_stride, dtype=np.int64)
            frame_length = max(
                float(np.median(np.linalg.norm(
                    nozzle_points[frame_indices] - surface_points[frame_indices],
                    axis=1,
                ))) * 0.12,
                1.0,
            )
            for axis_index, axis_color in (
                (0, "#FF6464"),
                (1, "#55E69A"),
                (2, "#55A8FF"),
            ):
                starts = nozzle_points[frame_indices]
                ends = starts + nozzle_frames[frame_indices, :, axis_index] * frame_length
                line_vertices = np.stack((starts, ends), axis=1).reshape(-1, 3)
                line_cells = np.column_stack((
                    np.full(len(frame_indices), 2, dtype=np.int64),
                    np.arange(0, len(frame_indices) * 2, 2, dtype=np.int64),
                    np.arange(1, len(frame_indices) * 2, 2, dtype=np.int64),
                )).reshape(-1)
                frame_polydata = pv.PolyData(line_vertices)
                frame_polydata.lines = line_cells
                self.plotter.add_mesh(
                    frame_polydata,
                    color=axis_color,
                    line_width=2.5,
                    render_lines_as_tubes=True,
                    lighting=False,
                    pickable=False,
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

            # Direct links between consecutive strokes are shown separately
            # in amber. They are spray-OFF visualization estimates only.
            travel_plan = data.get("inter_segment_travel_plan", {})
            transitions = travel_plan.get("transitions", [])
            if transitions:
                travel_points = []
                travel_cells = []
                for transition in transitions:
                    start = np.asarray(
                        transition.get("start_point_coordinate_units", []),
                        dtype=float,
                    )
                    end = np.asarray(
                        transition.get("end_point_coordinate_units", []),
                        dtype=float,
                    )
                    if (
                        start.shape != (3,) or end.shape != (3,)
                        or not np.all(np.isfinite(start))
                        or not np.all(np.isfinite(end))
                    ):
                        continue
                    start_world = start @ rotation.T + translation
                    end_world = end @ rotation.T + translation
                    point_index = len(travel_points)
                    travel_points.extend((start_world.tolist(), end_world.tolist()))
                    travel_cells.extend((2, point_index, point_index + 1))
                if travel_points:
                    travel_polydata = pv.PolyData(
                        np.asarray(travel_points, dtype=float)
                    )
                    travel_polydata.lines = np.asarray(
                        travel_cells, dtype=np.int64
                    )
                    self.plotter.add_mesh(
                        travel_polydata,
                        color="#FFB454",
                        line_width=2,
                        opacity=0.9,
                        lighting=False,
                        pickable=False,
                    )

        # Keep the 3D viewport free of diagnostic text. Stage 3 metrics
        # belong in the Summary panel, not over the product geometry.

    def _stop_trajectory_playback(self, reset=False):
        if hasattr(self, "_playback_timer"):
            self._playback_timer.stop()
        if hasattr(self, "trajectory_play_button"):
            self.trajectory_play_button.setText("▶")
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
            self._playback_finished = False
            self._playback_last_path_render_index = -1
            self._playback_cache = None
            if hasattr(self, "trajectory_coverage_bar"):
                self.trajectory_coverage_bar.setValue(0)
            if hasattr(self, "trajectory_coverage_label"):
                self.trajectory_coverage_label.setText("0.0% | 0.000 / 0.000 area")
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
        pattern_type = str(profile.get("pattern_type", "flat_fan")).strip().lower()
        standoff_mm = standoff_units * mm_per_unit
        footprint_width_units = 2.0 * standoff_units * cone_tangent
        fan_reference_distance_units = max(
            float(profile.get("fan_reference_distance_mm", 200.0) or 200.0) / mm_per_unit,
            1e-9,
        )
        fan_thickness_units = max(
            0.0,
            float(profile.get("fan_thickness_at_reference_mm", 0.0) or 0.0) / mm_per_unit,
        )
        footprint_thickness_units = (
            footprint_width_units if pattern_type == "round_cone"
            else fan_thickness_units * standoff_units / fan_reference_distance_units
        )

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

        self._playback_cache = {
            "nozzle_points": nozzle_points,
            "spray_directions": spray_directions,
            "nozzle_frames": nozzle_frames,
            "segment_ids": segment_ids,
            "spray_on_mask": spray_on_mask,
            "travel_speeds_mm_s": np.asarray(
                data.get("travel_speeds_mm_s", np.full(len(nozzle_points), 150.0)),
                dtype=float,
            ),
            "trajectory_time_s": np.asarray(
                data.get("trajectory_time_s", np.arange(len(nozzle_points), dtype=float) * 0.1),
                dtype=float,
            ),
            "face_samples": face_samples,
            "sample_normals": sample_normals,
            "face_areas": face_areas,
            "target_face_ids": target_face_ids,
            "target_display_indices": target_display_indices,
            "target_area": target_area,
            "target_mask_status": target_mask_status,
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
            "fan_thickness_units": fan_thickness_units,
            "fan_reference_distance_units": fan_reference_distance_units,
            "footprint_width_units": footprint_width_units,
            "footprint_thickness_units": footprint_thickness_units,
            "footprint_width_mm": footprint_width_units * mm_per_unit,
            "footprint_thickness_mm": footprint_thickness_units * mm_per_unit,
            "paint_rgb": paint_rgb,
            "profile": dict(profile),
            "mm_per_unit": mm_per_unit,
        }
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

        The footprint test is only the geometric envelope. A VTK OBBTree
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
        # Never invent a circular footprint for a flat fan with unknown thickness.
        if cache["pattern_type"] == "flat_fan" and cache["fan_thickness_units"] <= 0.0:
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
        if cache["pattern_type"] == "flat_fan":
            half_width = axial * cache["cone_tangent"]
            half_thickness = (
                0.5 * cache["fan_thickness_units"]
                * axial / cache["fan_reference_distance_units"]
            )
            inside_pattern = (
                (np.abs(local_x) <= half_width)
                & (np.abs(local_y) <= half_thickness)
            )
        else:
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
            for sample_index in footprint_candidates:
                target = cache["face_samples"][sample_index]
                intersection_points = vtkPoints()
                intersection_cell_ids = vtkIdList()
                locator.IntersectWithLine(
                    tuple(float(v) for v in nozzle),
                    tuple(float(v) for v in target),
                    intersection_points,
                    intersection_cell_ids,
                )
                intersections = np.asarray([
                    intersection_points.GetPoint(point_index)
                    for point_index in range(intersection_points.GetNumberOfPoints())
                ], dtype=np.float64).reshape(-1, 3)
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
        rgba = np.zeros((len(cache["target_face_ids"]), 4), dtype=np.uint8)
        known = cache["pattern_type"] == "round_cone" or cache["fan_thickness_units"] > 0.0
        if cache["spray_on_mask"][path_index] and known:
            hits = self._get_surface_sample_hits_from_spray(path_index)
            if len(hits):
                face_positions = np.unique(hits // cache["sample_count_per_face"])
                rgba[face_positions] = [*np.clip(np.rint(cache["paint_rgb"]), 0, 255).astype(np.uint8), 190]
        polydata.cell_data["Spray Footprint RGBA"] = rgba
        vtk_array = polydata.GetCellData().GetArray("Spray Footprint RGBA")
        if vtk_array is not None:
            vtk_array.Modified()
        polydata.Modified()
        actor.SetVisibility(bool(np.any(rgba[:, 3] > 0)))


    def _update_paint_coverage_visualization(self, index):
        cache = self._prepare_trajectory_playback_cache()
        if cache is None:
            return

        rewound = index < cache["coverage_processed_index"]
        if rewound:
            cache["coverage_sample_hits"].fill(False)
            cache["coverage_processed_index"] = -1
            cache["coverage_last_render_index"] = -1

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
        if hasattr(self, "trajectory_coverage_label"):
            if cache["pattern_type"] == "flat_fan":
                model_label = (
                    "FLAT FAN" if cache["fan_thickness_units"] > 0.0
                    else "FAN THICKNESS REQUIRED"
                )
            else:
                model_label = "ROUND CONE"
            if cache["target_mask_status"] != "READY":
                self.trajectory_coverage_label.setText(
                    "N/A — STAGE 2 TARGET MASK REQUIRED"
                )
            elif cache["target_area"] <= 1e-12:
                self.trajectory_coverage_label.setText(
                    "N/A — NO STAGE 2 PAINTABLE TARGET AREA"
                )
            elif (
                cache["pattern_type"] == "flat_fan"
                and cache["fan_thickness_units"] <= 0.0
            ):
                self.trajectory_coverage_label.setText(
                    "N/A — FAN THICKNESS REQUIRED | "
                    f"TARGET {cache['target_area'] * cache['area_display_scale']:.3f} "
                    f"{cache['area_display_unit']}"
                )
            else:
                self.trajectory_coverage_label.setText(
                    f"{coverage_percent:.1f}% SAMPLED REACH | "
                    f"{painted_area * cache['area_display_scale']:.3f} / "
                    f"{cache['target_area'] * cache['area_display_scale']:.3f} "
                    f"{cache['area_display_unit']}"
                )
            if (
                cache["pattern_type"] == "flat_fan"
                and cache["fan_thickness_units"] <= 0.0
            ):
                self.trajectory_coverage_label.setToolTip(
                    "Flat-fan footprint is not displayed or counted because its "
                    "measured thickness is zero/unknown. Enter the measured fan "
                    "thickness at the reference distance. No conical substitute is used."
                )
            else:
                self.trajectory_coverage_label.setToolTip(
                    "Sampled geometric reach estimate over the complete Stage 2 paintable "
                    "candidate area. Four test points per mesh triangle are used. This "
                    "does not yet model occlusion, deposited paint mass, or coating thickness."
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
        self._playback_spray_polydata.cell_data["Spray Footprint RGBA"] = (
            np.zeros((face_count, 4), dtype=np.uint8)
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

        # Visible spray-state cue. Round-cone geometry is explicitly built
        # from nozzle apex to the configured target-plane radius. Flat fan is
        # shown as a thin sheet; its thickness is not invented if unmeasured.
        sheet_length = cache["standoff_units"]
        sheet_half_width = max(cache["footprint_width_units"] * 0.5, 1e-6)
        if cache["pattern_type"] == "round_cone":
            # Build the spray volume explicitly to avoid cone-orientation
            # ambiguity: apex is exactly at the nozzle (z=0), and the circular
            # footprint is exactly at the target plane (z=standoff).
            # For a full included spray angle theta:
            # target radius = standoff * tan(theta / 2).
            cone_resolution = 64
            cone_angles = np.linspace(
                0.0, 2.0 * math.pi, cone_resolution, endpoint=False
            )
            cone_ring = np.column_stack((
                sheet_half_width * np.cos(cone_angles),
                sheet_half_width * np.sin(cone_angles),
                np.full(cone_resolution, sheet_length, dtype=float),
            ))
            cone_points = np.vstack((
                np.zeros((1, 3), dtype=float),
                cone_ring,
            ))
            cone_faces = []
            for cone_index in range(cone_resolution):
                next_index = (cone_index + 1) % cone_resolution
                cone_faces.extend((
                    3, 0, 1 + cone_index, 1 + next_index
                ))
            sheet_mesh = pv.PolyData(
                cone_points, np.asarray(cone_faces, dtype=np.int64)
            )
        else:
            sheet_points = np.asarray([
                [0.0, 0.0, 0.0],
                [-sheet_half_width, 0.0, sheet_length],
                [sheet_half_width, 0.0, sheet_length],
            ], dtype=float)
            sheet_faces = np.asarray([3, 0, 1, 2], dtype=np.int64)
            sheet_mesh = pv.PolyData(sheet_points, sheet_faces)
        self._playback_spray_sheet_actor = self.plotter.add_mesh(
            sheet_mesh,
            color=self._trajectory_paint_color_hex or "#D95B43",
            opacity=0.34,
            show_edges=(cache["pattern_type"] == "flat_fan"),
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
                if cache["pattern_type"] == "flat_fan" and cache["fan_thickness_units"] <= 0.0:
                    self.trajectory_spray_state_label.setText(
                        f"● SPRAY ON · FAN SHEET "
                        f"({cache['footprint_width_mm']:.1f} mm width; thickness unknown)"
                    )
                else:
                    self.trajectory_spray_state_label.setText(
                        f"SPRAY: ON · {cache['footprint_width_mm']:.1f} × "
                        f"{cache['footprint_thickness_mm']:.1f} mm"
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
        current_speed = float(cache["travel_speeds_mm_s"][index])
        spray_text = "SPRAY ON" if spray_on else "SPRAY OFF"
        self.trajectory_playback_label.setText(
            f"{index + 1:,} / {total:,}  ·  {current_speed:.0f} mm/s  ·  {spray_text}"
        )
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

    def _on_trajectory_playback_slider_changed(self, value):
        if self.display_mode != "trajectory" or self.trajectory_data is None:
            return
        self._playback_timer.stop()
        self._playback_finished = False
        self.trajectory_play_button.setText("▶")
        self._update_trajectory_playback_visuals(value)

    def _toggle_trajectory_playback(self):
        if self.trajectory_data is None or self.display_mode != "trajectory":
            return
        total = len(self.trajectory_data["nozzle_points"])
        if total == 0:
            return
        if self._playback_timer.isActive():
            self._playback_timer.stop()
            self.trajectory_play_button.setText("▶")
            return
        if self._playback_finished or self._playback_index >= total - 1:
            self._playback_finished = False
            self._playback_elapsed_s = 0.0
            self._update_trajectory_playback_visuals(0)
        else:
            cache = self._prepare_trajectory_playback_cache()
            times = cache["trajectory_time_s"]
            if len(times) == total and np.all(np.isfinite(times)):
                self._playback_elapsed_s = float(times[self._playback_index])
            if self._playback_path_actor is None and self._playback_nozzle_actor is None:
                self._update_trajectory_playback_visuals(self._playback_index)
        self.trajectory_play_button.setText("Ⅱ")
        self._playback_timer.start()

    def _complete_trajectory_playback(self):
        """Freeze the gun at the final pose and force all spray visuals OFF."""
        self._playback_timer.stop()
        self._playback_finished = True
        self.trajectory_play_button.setText("▶")

        cache = self._prepare_trajectory_playback_cache()
        if cache is not None and len(cache["nozzle_points"]):
            final_index = len(cache["nozzle_points"]) - 1
            self._playback_index = final_index
            if self._playback_spray_sheet_actor is not None:
                self._playback_spray_sheet_actor.SetVisibility(0)
            if self._playback_spray_actor is not None:
                self._playback_spray_actor.SetVisibility(0)
            if self._playback_spray_polydata is not None:
                rgba = np.zeros(
                    (len(cache["target_face_ids"]), 4), dtype=np.uint8
                )
                self._playback_spray_polydata.cell_data[
                    "Spray Footprint RGBA"
                ] = rgba
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
        self.plotter.render()

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
        sample_times = cache["trajectory_time_s"]
        if len(sample_times) != total or not np.all(np.isfinite(sample_times)):
            base_step = max(1, int(np.ceil(total / 600.0)))
            step = max(1, int(round(base_step * self.trajectory_playback_speed.value())))
            next_index = min(total - 1, self._playback_index + step)
        else:
            # The visual playback follows the calculated mm/s speed profile.
            # The UI multiplier changes playback time only, not robot speed.
            self._playback_elapsed_s += (
                self._playback_timer.interval() / 1000.0
                * self.trajectory_playback_speed.value()
            )
            next_index = int(np.searchsorted(
                sample_times, self._playback_elapsed_s, side="right"
            ) - 1)
            next_index = int(np.clip(next_index, self._playback_index, total - 1))
            if next_index == self._playback_index and next_index < total - 1:
                return

        # Never skip a spray-OFF boundary, even at high visual playback speed.
        off_boundaries = np.flatnonzero(~cache["spray_on_mask"])
        future_boundaries = off_boundaries[
            off_boundaries > self._playback_index
        ]
        if len(future_boundaries):
            next_off_boundary = int(future_boundaries[0])
            if next_index >= next_off_boundary:
                next_index = next_off_boundary

        self._update_trajectory_playback_visuals(next_index)
        if next_index >= total - 1:
            self._complete_trajectory_playback()

    def _step_trajectory_playback(self):
        if self.trajectory_data is None or self.display_mode != "trajectory":
            return
        self._playback_timer.stop()
        self._playback_finished = False
        self.trajectory_play_button.setText("▶")
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
        if self.display_mode not in ("product", "patches", "trajectory"):
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
                show_edges=True,
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
            self.display_mode in ("product", "classification", "trajectory")
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
            "GEOMETRIC REACH ONLY - NO FILM THICKNESS"
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
