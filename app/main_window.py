
from pathlib import Path
import json
import traceback

import numpy as np

from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtCore import QThread, Signal, Qt, QSize
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QDialog,
    QWidget,
    QFrame,
    QLabel,
    QPushButton,
    QToolButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStatusBar,
    QProgressBar,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QColorDialog,
    QAbstractItemView,
    QListWidget,
    QListWidgetItem
)

from core.step_loader import load_step, convert_step_report_units
from core.result_manager import ResultManager
from core.geometry_analyzer import detect_hole_candidates
from core.coating_process import load_coating_process

from stages.stage1_geometry import execute_stage1
from stages.stage2_surface_identification import execute_stage2
from stages.stage3_spray_trajectory import (
    execute_stage3,
    _calculate_spray_footprint,
)

from ui.viewer_3d import ProductViewer
from ui.coating_process_settings import CoatingProcessSettingsDialog
from ui.robot_settings import RobotSettingsDialog, load_robot_settings, save_robot_settings
from ui.sprayer_settings import (
    SprayerSettingsDialog,
    load_active_sprayer_profile_name,
    load_sprayer_profiles,
    save_sprayer_profiles,
)


class ReorderableSprayGroupList(QListWidget):
    """Internal drag-and-drop list that reports completed row reordering."""

    orderChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(False)
        self.setSpacing(4)
        self.setMinimumHeight(90)
        self.setMaximumHeight(280)

    def dropEvent(self, event):
        super().dropEvent(event)
        if event.isAccepted():
            self.orderChanged.emit()


# ============================================================
# UNIT DEFINITIONS
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
    "unknown": "unknown",
}

MM_PER_UNIT = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "ft": 304.8,
}

STAGE_PRODUCT_SETUP = 0
STAGE_GEOMETRY_INSPECTION = 1
STAGE_SURFACE_IDENTIFICATION = 2
STAGE_SPRAY_TRAJECTORY = 3
STAGE_SPRAY_SIMULATION = 4

FACE_EXCLUDED = 0
FACE_PAINTABLE_CANDIDATE = 1
FACE_REQUIRES_INSPECTION = 2
FACE_INVALID = 3

DECISION_REQUIRES_INSPECTION = "requires_inspection"
DECISION_PAINT = "paint"
DECISION_NO_PAINT = "no_paint"

DECISION_TO_FACE_STATUS = {
    DECISION_REQUIRES_INSPECTION: FACE_REQUIRES_INSPECTION,
    DECISION_PAINT: FACE_PAINTABLE_CANDIDATE,
    DECISION_NO_PAINT: FACE_EXCLUDED,
}

WORKFLOW_STAGES = [
    {
        "name": "PRODUCT SETUP",
        "description": (
            "Load the STEP AP214 product model and establish "
            "the active product coordinate system."
        ),
        "action": "LOAD STEP PRODUCT",
    },
    {
        "name": "GEOMETRY INSPECTION",
        "description": (
            "Inspect CAD tessellation integrity, dimensions, connected "
            "components, and geometric surface patches."
        ),
        "action": "ANALYZE GEOMETRY",
    },
    {
        "name": "SURFACE IDENTIFICATION",
        "description": (
            "Inspect each geometric patch and assign its "
            "painting decision."
        ),
        "action": "IDENTIFY PAINTABLE SURFACES",
    },
    {
        "name": "ROUTE GENERATION",
        "description": (
            "Record spray strokes in the exact desired order. MoveIt2 creates "
            "the collision-free transfer motion between separate spray groups."
        ),
        "action": "RECORD SPRAY ROUTE",
    },
    {
        "name": "SPRAY SIMULATION",
        "description": (
            "Replay the saved route from a bare product and visually "
            "verify progressive coating coverage."
        ),
        "action": "SIMULATE SPRAY ROUTE",
    },
]

PIPELINE_ITEMS = [
    "Product loading",
    "Geometry inspection",
    "Paintable surface identification",
    "Spray trajectory generation",
]


# ============================================================
# UNIT UTILITIES
# ============================================================

def normalize_unit(unit):
    if not isinstance(unit, str):
        return "unknown"

    return SUPPORTED_UNITS.get(unit.strip().lower(), "unknown")


def get_mesh_source_unit(mesh):
    metadata = getattr(mesh, "metadata", {})

    if not isinstance(metadata, dict):
        return "unknown"

    return normalize_unit(metadata.get("source_unit"))


def convert_coordinate_value(value, source_unit, display_unit):
    source_unit = normalize_unit(source_unit)
    display_unit = normalize_unit(display_unit)

    if source_unit == "unknown" or display_unit == "unknown":
        return None

    mm_value = float(value) * MM_PER_UNIT[source_unit]
    return mm_value / MM_PER_UNIT[display_unit]


def get_display_area_factor(source_unit, display_unit):
    source_unit = normalize_unit(source_unit)
    display_unit = normalize_unit(display_unit)

    if source_unit == "unknown" or display_unit == "unknown":
        return None

    factor = MM_PER_UNIT[source_unit] / MM_PER_UNIT[display_unit]
    return factor ** 2


def get_display_volume_factor(source_unit, display_unit):
    source_unit = normalize_unit(source_unit)
    display_unit = normalize_unit(display_unit)

    if source_unit == "unknown" or display_unit == "unknown":
        return None

    factor = MM_PER_UNIT[source_unit] / MM_PER_UNIT[display_unit]
    return factor ** 3


# ============================================================
# OPERATION WORKER
# ============================================================

class OperationWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.operation(*self.args, **self.kwargs)
            self.succeeded.emit(result)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))


class PatchDistributionChart(QWidget):
    """Dependency-free Qt pie chart for Stage 1 patch classification."""

    COLORS = {
        "planar": "#00E676",
        "curved": "#00B8D9",
        "uncertain": "#FFD740",
        "invalid": "#FF5252",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = {
            "planar": 0,
            "curved": 0,
            "uncertain": 0,
            "invalid": 0,
        }
        self.setFixedSize(126, 126)
        self.setToolTip(
            "Stage 1 patch classification: planar, curved, uncertain, invalid"
        )

    def set_values(self, values):
        self.values = {
            key: max(0, int(values.get(key, 0)))
            for key in self.COLORS
        }
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        margin = 7
        rect = self.rect().adjusted(
            margin, margin, -margin, -margin
        )
        total = sum(self.values.values())

        if total <= 0:
            painter.setPen(QPen(QColor("#29404D"), 1))
            painter.setBrush(QColor("#132431"))
            painter.drawEllipse(rect)
            painter.end()
            return

        start_angle = 90 * 16
        remaining_angle = 360 * 16
        nonzero = [
            key for key in self.COLORS
            if self.values[key] > 0
        ]

        for index, key in enumerate(nonzero):
            if index == len(nonzero) - 1:
                span_angle = remaining_angle
            else:
                span_angle = int(
                    round(360 * 16 * self.values[key] / total)
                )
                span_angle = min(span_angle, remaining_angle)

            painter.setPen(QPen(QColor("#071018"), 1.5))
            painter.setBrush(QColor(self.COLORS[key]))
            painter.drawPie(rect, start_angle, span_angle)

            start_angle += span_angle
            remaining_angle -= span_angle

        painter.end()


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self, project_root: Path, fonts: dict):
        super().__init__()

        # Increase the inherited application UI font modestly while keeping
        # widget-specific font families and explicit emphasis intact.
        base_font = self.font()
        base_font.setPointSize(max(base_font.pointSize() + 1, 10))
        self.setFont(base_font)

        self.project_root = Path(project_root).resolve()
        self.fonts = fonts

        self.sprayer_config_path = (
            self.project_root / "config" / "sprayer_profiles.json"
        )
        self.sprayer_profiles = load_sprayer_profiles(
            self.sprayer_config_path
        )
        self.active_sprayer_profile_name = (
            load_active_sprayer_profile_name(self.sprayer_config_path)
        )
        self.sprayer_profile = dict(
            next(
                (
                    profile for profile in self.sprayer_profiles
                    if str(profile.get("name", "")).strip().lower()
                    == self.active_sprayer_profile_name.strip().lower()
                ),
                next(
                    (
                        profile for profile in self.sprayer_profiles
                        if str(profile.get("name", "")).strip().lower()
                        == "default sprayer"
                    ),
                    self.sprayer_profiles[0],
                ),
            )
        )

        self.robot_settings_path = self.project_root / "config" / "robot_settings.json"
        self.robot_settings = load_robot_settings(self.robot_settings_path)

        self.coating_process_config_path = (
            self.project_root / "config" / "coating_process.json"
        )
        self.coating_process = load_coating_process(
            self.coating_process_config_path, self.sprayer_profile
        )

        self.result_manager = ResultManager(self.project_root)

        self.current_mesh = None
        self.current_stl_path = None
        self.current_step_report = None
        self.current_step_report_path = None

        self.current_stage1_result = None
        self.current_report = None
        self.current_stage2_result = None
        self.current_stage3_result = None
        self.interactive_spray_path = None

        self.source_unit_from_stl = "unknown"
        self.coordinate_unit = "unknown"
        self.manual_unit_selected = False

        self.current_stage3_parameters = None
        self.pending_stage3_parameters = None

        self._worker = None
        self._operation_running = False
        self._simulation_loading = False

        self.patch_inspection = {}
        self.patch_row_widgets = {}
        self.selected_patch_id = None

        self._stage2_automatic_face_status_labels = None
        self.patch_inspection_path = None

        # Right sidebar visibility state.
        self._right_sidebar_visible = True

        # Individual right-panel section visibility state.
        self._right_section_bodies = {}
        self._right_section_buttons = {}

        self.setWindowTitle("OPENARMX PAINTING CONTROL SYSTEM")
        self.resize(1600, 950)
        self.setMinimumSize(1200, 750)

        self.active_stage = STAGE_PRODUCT_SETUP

        self.stage_completed = {
            STAGE_PRODUCT_SETUP: False,
            STAGE_GEOMETRY_INSPECTION: False,
            STAGE_SURFACE_IDENTIFICATION: False,
            STAGE_SPRAY_TRAJECTORY: False,
            STAGE_SPRAY_SIMULATION: False,
        }

        self._build_ui()

        # The workflow intentionally ends at spray trajectory generation.
        # Robot configuration and execution stages are removed from this build.
        self.viewer.display_control_panel.hide()

        self._refresh_workflow_ui()

        models_dir = self.project_root / "models"
        preferred_step_candidates = [
            models_dir / "product.step",
            models_dir / "product.STEP",
            models_dir / "product.stp",
            models_dir / "product.STP",
        ]
        discovered_step_candidates = sorted(
            (
                path for path in models_dir.iterdir()
                if path.is_file() and path.suffix.lower() in (".step", ".stp")
            ),
            key=lambda path: path.name.lower(),
        ) if models_dir.is_dir() else []

        # Prefer the conventional product filename, then use an available
        # STEP model (for example valvemodel.STEP) before falling back to STL.
        step_candidates = preferred_step_candidates + discovered_step_candidates
        seen_step_paths = set()
        default_step = None
        for candidate in step_candidates:
            resolved = candidate.resolve()
            if resolved in seen_step_paths:
                continue
            seen_step_paths.add(resolved)
            if candidate.is_file():
                default_step = candidate
                break

        if default_step is not None:
            self.load_part(default_step)

    # ========================================================
    # UI HELPERS
    # ========================================================

    def _panel(self):
        frame = QFrame()
        frame.setObjectName("Panel")
        return frame

    def _section(self, text):
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _metric(self, title, value="--"):
        name = QLabel(title)
        name.setObjectName("Muted")

        result = QLabel(value)
        result.setObjectName("MetricValue")

        return name, result

    def _make_collapsible_section(self, title, section_id, content):
        """
        Create an accordion-style right-sidebar section.

        Only one section can be expanded at a time.
        Clicking the expanded section collapses it.
        Existing content widgets are preserved.
        """

        container = QWidget()
        container.setObjectName("CollapsibleSection")

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(3)

        # Check whether another section is already expanded.
        another_section_open = any(
            button.isChecked()
            for button in self._right_section_buttons.values()
        )

        # Keep the original, ordinary QToolButton appearance.
        header = QToolButton()
        header.setText(title)
        header.setCheckable(True)

        # Only the first section starts expanded.
        initially_expanded = not another_section_open
        header.setChecked(initially_expanded)

        header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        header.setArrowType(
            Qt.DownArrow if initially_expanded else Qt.RightArrow
        )
        header.setObjectName("CollapsibleSectionHeader")

        header.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Fixed,
        )

        content.setObjectName("CollapsibleSectionBody")
        content.setVisible(initially_expanded)

        def on_toggled(expanded, body=content, button=header):
            # If this section opens, close all other sections.
            if expanded:
                for other_id, other_button in (
                    self._right_section_buttons.items()
                ):
                    if other_button is not button:
                        other_button.setChecked(False)

            body.setVisible(expanded)

            button.setArrowType(
                Qt.DownArrow if expanded else Qt.RightArrow
            )

        header.toggled.connect(on_toggled)

        # Place each compact accordion button at the left edge.
        container_layout.addWidget(header, 0, Qt.AlignLeft)
        container_layout.addWidget(content)

        # Register this section.
        self._right_section_bodies[section_id] = content
        self._right_section_buttons[section_id] = header

        return container



    def _toggle_right_sidebar(self):
        self._right_sidebar_visible = not self._right_sidebar_visible

        self.right_panel.setVisible(
            self._right_sidebar_visible
        )

        if self._right_sidebar_visible:
            # Sidebar visible → LEFT arrow → click to collapse.
            self.right_sidebar_toggle.setIcon(
                self.style().standardIcon(
                    QStyle.StandardPixmap.SP_ArrowRight
                )
            )

            self.right_sidebar_toggle.setToolTip(
                "Collapse right sidebar"
            )

        else:
            # Sidebar hidden → RIGHT arrow → click to expand.
            self.right_sidebar_toggle.setIcon(
                self.style().standardIcon(
                    QStyle.StandardPixmap.SP_ArrowLeft
                )
            )

            self.right_sidebar_toggle.setToolTip(
                "Expand right sidebar"
            )

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    
    def _collapse_patch_inspection_section(self):
        """
        Collapse the Patch Inspection accordion section.
        Keep its header available in the sidebar.
        """

        button = self._right_section_buttons.get(
            "patch_inspection"
        )

        if button is not None:
            button.setChecked(False)

    def _expand_patch_inspection_section(self):
        button = self._right_section_buttons.get(
            "patch_inspection"
        )

        if button is not None:
            button.setChecked(True)


    def _expand_summary_section(self):
        button = self._right_section_buttons.get("stage_summary")
        if button is not None:
            button.setChecked(True)


    def _set_patch_inspection_readonly(self, readonly=True):
        # Disable the complete row container in one Qt operation. Iterating
        # over every patch widget on each workflow refresh made Stage 3 entry
        # increasingly slow on products with many patches.
        self.patch_rows_container.setEnabled(not readonly)

    def _build_ui(self):
        central = QWidget()

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        root.addLayout(self._build_header())

        # ----------------------------------------------------
        # WORKFLOW NAVIGATION
        # ----------------------------------------------------

        # Show the workflow navigation through the first spray-trajectory stage.
        self.workflow_bar = self._build_workflow_bar()
        root.addWidget(self.workflow_bar)
        self.workflow_bar.show()

        # ----------------------------------------------------
        # MAIN WORKSPACE
        # ----------------------------------------------------

        workspace = QHBoxLayout()
        workspace.setSpacing(6)

        # 3D Viewer
        self.viewer = ProductViewer()
        self.viewer.set_robot_motion_settings(self.robot_settings)
        self.viewer.pose_confirmed.connect(self._save_reference_pose)
        self.viewer.patch_selected.connect(
            self._on_viewer_patch_selected
        )
        self.viewer.route_readiness_changed.connect(
            self._on_manual_route_readiness_changed
        )
        self.viewer.spray_samples_changed.connect(
            self._on_spray_samples_changed
        )

        # Left sidebar
        self.left_panel = self._build_left_panel()
        self.left_panel.setFixedWidth(330)

        # Right sidebar
        self.right_panel = self._build_right_panel()
        self.right_panel.setFixedWidth(400)
        self._move_route_controls_to_sidebar()

        # ----------------------------------------------------
        # RIGHT SIDEBAR TOGGLE BUTTON
        # ----------------------------------------------------

        self._right_sidebar_visible = True

        self.right_sidebar_toggle = QToolButton()
        self.right_sidebar_toggle.setFixedSize(48, 78)

        self.right_sidebar_toggle.setIconSize(QSize(24, 24))
        self.right_sidebar_toggle.setToolButtonStyle(
            Qt.ToolButtonIconOnly
        )
        self.right_sidebar_toggle.setAutoRaise(False)

        # Initial state: sidebar visible, arrow points LEFT.
        self.right_sidebar_toggle.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_ArrowRight
            )
        )

        self.right_sidebar_toggle.setToolTip(
            "Collapse right sidebar"
        )

        self.right_sidebar_toggle.setStyleSheet("""
            QToolButton {
                background-color: #14232E;
                border: 2px solid #203B4B;
                border-radius: 8px;
                padding: 8px;
            }

            QToolButton:hover {
                background-color: #203B4B;
                border: 2px solid #00BCD4;
            }

            QToolButton:pressed {
                background-color: #0D1822;
            }
        """)

        self.right_sidebar_toggle.clicked.connect(
            self._toggle_right_sidebar
        )

        # ----------------------------------------------------
        # WORKSPACE WIDGET ORDER
        # ----------------------------------------------------

        workspace.addWidget(self.left_panel)
        workspace.addWidget(self.viewer, 1)

        workspace.addWidget(
            self.right_sidebar_toggle,
            0,
            Qt.AlignVCenter,
        )

        workspace.addWidget(self.right_panel)

        root.addLayout(workspace, 1)

        # ----------------------------------------------------
        # CENTRAL WIDGET
        # ----------------------------------------------------

        self.setCentralWidget(central)

        # ----------------------------------------------------
        # STATUS BAR
        # ----------------------------------------------------

        self.status_bar = QStatusBar()
        self.status_bar.showMessage(
            "SYSTEM READY  |  WAITING FOR PRODUCT"
        )
        self.setStatusBar(self.status_bar)



    def _move_route_controls_to_sidebar(self):
        """Place spray setup controls and recorded-path controls in separate accordions."""
        viewer_layout = self.viewer.layout()
        interaction_controls = self.viewer.interaction_mode_panel
        viewer_layout.removeWidget(interaction_controls)
        interaction_controls.setParent(self.spray_settings_scroll.widget())
        interaction_controls.setMinimumWidth(0)
        interaction_controls.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Maximum,
        )
        self.spray_settings_layout.addWidget(interaction_controls)
        interaction_controls.show()

        for widget in (
            self.viewer.interactive_recorded_path_panel,
            self.viewer.display_control_panel,
        ):
            viewer_layout.removeWidget(widget)
            widget.setParent(self.route_controls_scroll.widget())
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Maximum,
            )
            self.route_controls_layout.addWidget(widget)

        # Keep simulation Start/Stop controls directly visible at the top
        # of the Recorded Spray Path accordion. They are separated from the
        # recorded-path information panel so their visibility does not depend
        # on that panel's content visibility.
        recorded_panel_layout = self.viewer.interactive_recorded_path_panel.layout()
        simulation_controls = QFrame()
        simulation_controls.setObjectName("SpraySimulationControls")
        simulation_controls_layout = QHBoxLayout(simulation_controls)
        simulation_controls_layout.setContentsMargins(4, 4, 4, 4)
        simulation_controls_layout.setSpacing(8)
        for button in (
            self.viewer.recorded_path_replay_button,
            self.viewer.recorded_path_stop_button,
        ):
            recorded_panel_layout.removeWidget(button)
            button.setParent(simulation_controls)
            button.setMinimumHeight(40)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            simulation_controls_layout.addWidget(button, 1)
        self.route_controls_layout.insertWidget(0, simulation_controls)
        self.simulation_playback_controls = simulation_controls
        simulation_controls.hide()

        # CSV import/export are simulation file actions, not playback
        # controls. Keep them in a fixed footer at the bottom of the
        # entire right sidebar, outside every accordion and scroll area.
        recorded_layout = self.viewer.interactive_recorded_path_panel.layout()
        csv_footer = QFrame()
        csv_footer.setObjectName("SimulationCsvFooter")
        csv_footer.setStyleSheet("""
            QFrame#SimulationCsvFooter {
                background: #10212C;
                border: 1px solid #36566A;
                border-radius: 6px;
            }
            QFrame#SimulationCsvFooter QLabel {
                color: #00E5FF;
                font-weight: 800;
                border: none;
            }
            QFrame#SimulationCsvFooter QPushButton {
                min-height: 40px;
                background: #183244;
                color: #E2EDF4;
                border: 1px solid #3B687E;
                border-radius: 5px;
                padding: 6px 10px;
                font-weight: 800;
            }
            QFrame#SimulationCsvFooter QPushButton:hover {
                background: #21485D;
                border-color: #00DDF5;
            }
            QFrame#SimulationCsvFooter QPushButton:disabled {
                background: #18232C;
                color: #586873;
                border: 1px solid #293943;
            }
        """)
        csv_layout = QVBoxLayout(csv_footer)
        csv_layout.setContentsMargins(10, 8, 10, 8)
        csv_layout.setSpacing(7)
        csv_title = QLabel("CSV FILE")
        csv_layout.addWidget(csv_title)
        csv_buttons = QHBoxLayout()
        csv_buttons.setSpacing(8)
        for button in (
            self.viewer.recorded_path_import_button,
            self.viewer.recorded_path_export_button,
        ):
            recorded_layout.removeWidget(button)
            button.setParent(csv_footer)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            csv_buttons.addWidget(button, 1)
        csv_layout.addLayout(csv_buttons)

        right_panel_layout = self.right_panel.layout()
        # _build_right_panel already places a stretch before the action footer.
        # Do not add another stretch here: it would push the action buttons
        # upward and leave empty space beneath them when the CSV footer is hidden.
        right_panel_layout.addWidget(csv_footer)
        self.simulation_csv_footer = csv_footer
        csv_footer.hide()

        self.route_controls_layout.addStretch(1)

    # ========================================================
    # HEADER
    # ========================================================

    def _build_header(self):
        header = QHBoxLayout()
        title_block = QVBoxLayout()

        title = QLabel("OPENARMX PAINTING CONTROL SYSTEM")
        title.setObjectName("MainTitle")

        subtitle = QLabel(
            "INDUSTRIAL SPRAY PAINTING  "
            "•  PRODUCT → GEOMETRY → SURFACES → SPRAY TRAJECTORY"
        )
        subtitle.setObjectName("Subtitle")

        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        header.addLayout(title_block)
        header.addStretch()

        self.system_status = QLabel("● SYSTEM ONLINE")
        self.system_status.setStyleSheet(
            "color: #00E676; font-weight: bold;"
        )
        header.addWidget(self.system_status)

        return header

    # ========================================================
    # WORKFLOW NAVIGATION
    # ========================================================

    def _build_workflow_bar(self):
        panel = self._panel()

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self.workflow_buttons = []

        for index, stage in enumerate(WORKFLOW_STAGES):
            button = QPushButton(stage["name"])
            button.setCheckable(True)

            button.clicked.connect(
                lambda checked=False, i=index:
                self._select_workflow_stage(i)
            )

            layout.addWidget(button, 1)
            self.workflow_buttons.append(button)

        return panel

    # ========================================================
    # LEFT PANEL
    # ========================================================

    def _build_left_panel(self):
        panel = self._panel()

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        layout.addWidget(self._section("PRODUCT"))

        self.part_name_label = QLabel("NO PRODUCT LOADED")
        self.part_name_label.setWordWrap(True)
        self.part_name_label.setObjectName("MetricValue")
        layout.addWidget(self.part_name_label)

        self.load_button = QPushButton("LOAD STEP PRODUCT")
        self.load_button.clicked.connect(self.load_stl_dialog)
        layout.addWidget(self.load_button)

        layout.addSpacing(5)
        layout.addWidget(self._section("MODEL COORDINATE UNIT"))

        self.coordinate_unit_selector = QComboBox()
        self.coordinate_unit_selector.addItem(
            "Select coordinate unit",
            "unknown",
        )

        for unit in ("mm", "cm", "m", "in", "ft"):
            self.coordinate_unit_selector.addItem(unit, unit)

        self.coordinate_unit_selector.currentIndexChanged.connect(
            self._on_coordinate_unit_changed
        )
        layout.addWidget(self.coordinate_unit_selector)

        layout.addSpacing(5)
        layout.addWidget(self._section("DISPLAY UNIT"))

        self.unit_selector = QComboBox()
        self.unit_selector.addItems(["mm", "cm", "m", "in", "ft"])
        self.unit_selector.setCurrentText("mm")
        self.unit_selector.currentTextChanged.connect(
            self._on_unit_changed
        )
        layout.addWidget(self.unit_selector)

        # Geometry statistics and bounding dimensions are intentionally
        # removed from the left sidebar. Geometry analysis remains available
        # through the workflow and its summary panels.

        layout.addSpacing(8)
        layout.addWidget(self._section("VIEW"))

        self.reset_camera_button = QPushButton("RESET CAMERA")
        self.reset_camera_button.clicked.connect(
            self.viewer.reset_camera
        )
        layout.addWidget(self.reset_camera_button)

        layout.addSpacing(8)
        layout.addWidget(self._section("SPRAYER"))

        self.sprayer_settings_button = QPushButton(
            "SPRAYER SETTINGS"
        )
        self.sprayer_settings_button.setObjectName("PrimaryButton")
        self.sprayer_settings_button.setToolTip(
            "Configure sprayer technical specifications and preview its spray pattern."
        )
        self.sprayer_settings_button.clicked.connect(
            self._open_sprayer_settings
        )
        layout.addWidget(self.sprayer_settings_button)

        self.robot_settings_button = QPushButton("ROBOT ARM SETTINGS")
        self.robot_settings_button.setObjectName("PrimaryButton")
        self.robot_settings_button.setToolTip(
            "Set painting travel speed and the future spray-point control frequency."
        )
        self.robot_settings_button.clicked.connect(self._open_robot_settings)
        layout.addWidget(self.robot_settings_button)

        self.sprayer_profile_label = QLabel(
            f"ACTIVE PROFILE: {self.sprayer_profile.get('name', 'default sprayer')}"
        )
        self.sprayer_profile_label.setObjectName("Muted")
        self.sprayer_profile_label.setWordWrap(True)
        layout.addWidget(self.sprayer_profile_label)

        self.coating_process_button = QPushButton("COATING PROCESS PARAMETERS")
        self.coating_process_button.setObjectName("PrimaryButton")
        self.coating_process_button.setToolTip(
            "Configure target dry-film thickness, paint flow, transfer efficiency, solids and process motion."
        )
        self.coating_process_button.clicked.connect(self._open_coating_process_settings)
        layout.addWidget(self.coating_process_button)

        self.coating_process_label = QLabel(
            f"PROCESS: {self.coating_process.get('name', 'default coating process')}  |  "
            f"TARGET: {self.coating_process.get('target_dry_film_thickness_um', 0.0):.1f} µm"
        )
        self.coating_process_label.setObjectName("Muted")
        self.coating_process_label.setWordWrap(True)
        layout.addWidget(self.coating_process_label)

        layout.addStretch()
        return panel

    # ========================================================
    # RIGHT PANEL - COLLAPSIBLE SECTIONS
    # ========================================================

    
    
    def _build_right_panel(self):
        panel = self._panel()

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ----------------------------------------------------
        # SPRAY SETTING
        # ----------------------------------------------------
        self.spray_settings_content = QWidget()
        spray_settings_outer = QVBoxLayout(self.spray_settings_content)
        spray_settings_outer.setContentsMargins(4, 4, 4, 4)
        spray_settings_outer.setSpacing(0)

        self.spray_settings_scroll = QScrollArea()
        self.spray_settings_scroll.setObjectName("SpraySettingsScroll")
        self.spray_settings_scroll.setWidgetResizable(True)
        self.spray_settings_scroll.setFrameShape(QFrame.NoFrame)
        self.spray_settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.spray_settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.spray_settings_scroll.setMaximumHeight(520)

        spray_settings_body = QWidget()
        self.spray_settings_layout = QVBoxLayout(spray_settings_body)
        self.spray_settings_layout.setContentsMargins(4, 4, 12, 8)
        self.spray_settings_layout.setSpacing(8)
        self.spray_settings_scroll.setWidget(spray_settings_body)
        spray_settings_outer.addWidget(self.spray_settings_scroll)

        self.spray_settings_section = self._make_collapsible_section(
            "SPRAY SETTING",
            "spray_settings",
            self.spray_settings_content,
        )
        self._right_section_buttons["spray_settings"].setMinimumWidth(360)
        self.spray_settings_section.hide()

        # ----------------------------------------------------
        # RECORDED SPRAY PATH
        # ----------------------------------------------------
        self.route_controls_content = QWidget()
        route_outer_layout = QVBoxLayout(self.route_controls_content)
        route_outer_layout.setContentsMargins(4, 4, 4, 4)
        route_outer_layout.setSpacing(0)
        self.route_controls_outer_layout = route_outer_layout

        self.route_controls_scroll = QScrollArea()
        self.route_controls_scroll.setObjectName("RecordedSprayPathScroll")
        self.route_controls_scroll.setWidgetResizable(True)
        self.route_controls_scroll.setFrameShape(QFrame.NoFrame)
        self.route_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.route_controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.route_controls_scroll.setMaximumHeight(800)

        route_controls_body = QWidget()
        self.route_controls_layout = QVBoxLayout(route_controls_body)
        self.route_controls_layout.setContentsMargins(4, 4, 12, 8)
        self.route_controls_layout.setSpacing(8)

        self.recorded_spray_group_list = ReorderableSprayGroupList()
        self.recorded_spray_group_list.setToolTip(
            "Drag a spray group up or down to change the execution order. "
            "The path shape and direction inside each group remain unchanged."
        )
        self.recorded_spray_group_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.recorded_spray_group_list.orderChanged.connect(
            self._on_recorded_spray_groups_reordered
        )
        self.route_controls_layout.addWidget(self.recorded_spray_group_list)

        self.recorded_spray_path_empty_label = QLabel(
            "No finalized spray groups yet. Draw independent strokes, then generate the spray route."
        )
        self.recorded_spray_path_empty_label.setObjectName("Muted")
        self.recorded_spray_path_empty_label.setWordWrap(True)
        self.route_controls_layout.addWidget(self.recorded_spray_path_empty_label)

        self.recorded_spray_path_total_label = QLabel(
            "TOTAL SPRAY LENGTH: 0.0 mm  |  TOTAL EXECUTION TIME: 00:00"
        )
        self.recorded_spray_path_total_label.setObjectName("MetricValue")
        self.recorded_spray_path_total_label.setWordWrap(True)
        self.route_controls_layout.addWidget(self.recorded_spray_path_total_label)

        self.recorded_spray_path_note_label = QLabel(
            "Execution time is spray motion only; MoveIt2 transfer time between groups is not included."
        )
        self.recorded_spray_path_note_label.setObjectName("Muted")
        self.recorded_spray_path_note_label.setWordWrap(True)
        self.route_controls_layout.addWidget(self.recorded_spray_path_note_label)

        self.route_controls_scroll.setWidget(route_controls_body)
        route_outer_layout.addWidget(self.route_controls_scroll)
        self.route_controls_section = self._make_collapsible_section(
            "RECORDED SPRAY PATH",
            "recorded_spray_path",
            self.route_controls_content,
        )
        self._right_section_buttons["recorded_spray_path"].setMinimumWidth(360)
        self.route_controls_section.hide()
        self._recorded_spray_list_updating = False

        # ----------------------------------------------------
        # ACTIVE WORKFLOW SECTION
        # ----------------------------------------------------
        active_content = QWidget()
        active_layout = QVBoxLayout(active_content)
        active_layout.setContentsMargins(5, 2, 5, 4)
        active_layout.setSpacing(5)

        self.current_stage_label = QLabel("PRODUCT SETUP")
        self.current_stage_label.setObjectName("MetricValue")
        self.current_stage_label.setWordWrap(True)
        active_layout.addWidget(self.current_stage_label)

        self.workflow_description = QLabel(
            "Load a product model to begin."
        )
        self.workflow_description.setObjectName("Muted")
        self.workflow_description.setWordWrap(True)
        active_layout.addWidget(self.workflow_description)

        self.workflow_status_label = QLabel(
            "STATUS: WAITING FOR PRODUCT"
        )
        self.workflow_status_label.setStyleSheet(
            "color: #FFD740; font-weight: bold;"
        )
        self.workflow_status_label.setWordWrap(True)
        active_layout.addWidget(self.workflow_status_label)

        active_section = self._make_collapsible_section(
            "ACTIVE WORKFLOW",
            "active_workflow",
            active_content,
        )

        self._right_section_buttons[
            "active_workflow"
        ].setChecked(False)

        layout.addWidget(active_section)

        # ----------------------------------------------------
        # PROCESS OVERVIEW SECTION
        # ----------------------------------------------------
        process_content = QWidget()
        process_layout = QVBoxLayout(process_content)
        process_layout.setContentsMargins(5, 2, 5, 4)
        process_layout.setSpacing(4)

        self.pipeline_labels = []

        for item in PIPELINE_ITEMS:
            label = QLabel("○  " + item)
            label.setObjectName("Muted")
            label.setWordWrap(True)

            process_layout.addWidget(label)
            self.pipeline_labels.append(label)

        process_section = self._make_collapsible_section(
            "PROCESS OVERVIEW",
            "process_overview",
            process_content,
        )

        # The first (Product Setup) page should open Process Overview by
        # default, regardless of the hidden route-controls accordion created
        # earlier in the sidebar build sequence.
        self._right_section_buttons["process_overview"].setChecked(True)

        layout.addWidget(process_section)

        # ----------------------------------------------------
        # STAGE SUMMARY SECTION
        # ----------------------------------------------------
        summary_content = QWidget()
        summary_layout = QVBoxLayout(summary_content)
        summary_layout.setContentsMargins(5, 2, 5, 4)
        summary_layout.setSpacing(4)

        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(8)

        self.summary_chart = PatchDistributionChart()
        self.summary_chart.hide()
        summary_row.addWidget(
            self.summary_chart,
            0,
            Qt.AlignVCenter,
        )

        self.analysis_summary = QLabel("AWAITING PRODUCT")
        self.analysis_summary.setWordWrap(True)
        self.analysis_summary.setTextFormat(Qt.RichText)
        self.analysis_summary.setObjectName("Muted")
        summary_row.addWidget(self.analysis_summary, 1)

        summary_layout.addLayout(summary_row)

        # The collapsible section title is the single SUMMARY heading.
        # Keep the legacy inner label hidden for compatibility with
        # existing stage callbacks that still update its text.
        self.summary_section = QLabel("SUMMARY")
        self.summary_section.setObjectName("Muted")
        self.summary_section.hide()
        summary_layout.addWidget(self.summary_section)

        summary_section = self._make_collapsible_section(
            "SUMMARY",
            "stage_summary",
            summary_content,
        )

        self._right_section_buttons[
            "stage_summary"
        ].setChecked(False)

        layout.addWidget(summary_section)

        # ----------------------------------------------------
        # PAINTING PATCHES SECTION
        # ----------------------------------------------------
        self.patch_inspection_panel = QFrame()
        self.patch_inspection_panel.setObjectName(
            "PatchInspectionPanel"
        )

        patch_outer_layout = QVBoxLayout(
            self.patch_inspection_panel
        )
        patch_outer_layout.setContentsMargins(0, 0, 0, 0)
        patch_outer_layout.setSpacing(0)

        patch_content = QWidget()
        patch_layout = QVBoxLayout(patch_content)
        patch_layout.setContentsMargins(5, 2, 5, 4)
        patch_layout.setSpacing(5)

        # ----------------------------------------------------
        # PAINTING SELECTION SUMMARY
        # ----------------------------------------------------
        self.patch_progress_label = QLabel(
            "PAINTABLE PATCHES: 0 / 0 (0%)"
        )
        self.patch_progress_label.setObjectName("MetricValue")
        patch_layout.addWidget(self.patch_progress_label)

        # ----------------------------------------------------
        # PAINTING SELECTION STATUS
        # Exactly one status label
        # ----------------------------------------------------
        self.patch_inspection_status_label = QLabel(
            "PAINTING SELECTION INCOMPLETE"
        )
        self.patch_inspection_status_label.setObjectName(
            "PatchInspectionStatus"
        )
        self.patch_inspection_status_label.setAlignment(
            Qt.AlignCenter
        )
        self.patch_inspection_status_label.setWordWrap(True)
        self.patch_inspection_status_label.setStyleSheet("""
            QLabel {
                color: #FFD740;
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.patch_inspection_status_label.hide()

        patch_layout.addWidget(
            self.patch_inspection_status_label
        )

        # ----------------------------------------------------
        # PAINTING SELECTION PROGRESS
        # Exactly one patch progress bar
        # ----------------------------------------------------
        self.patch_progress_bar = QProgressBar()
        self.patch_progress_bar.setRange(0, 100)
        self.patch_progress_bar.setValue(100)
        self.patch_progress_bar.setFormat("%p%")
        self.patch_progress_bar.setFixedHeight(10)

        patch_layout.addWidget(self.patch_progress_bar)

        # ----------------------------------------------------
        # FULL PRODUCT VIEW
        # Restore full opacity after the operator finishes
        # inspecting individual patches.
        # ----------------------------------------------------
        self.show_full_product_button = QPushButton(
            "SHOW FULL PRODUCT"
        )
        self.show_full_product_button.setObjectName(
            "PatchViewButton"
        )
        self.show_full_product_button.setToolTip(
            "Restore the complete product view and full surface opacity."
        )
        self.show_full_product_button.setStyleSheet("""
            QPushButton#PatchViewButton {
                background-color: #17465A;
                color: #EAFBFF;
                border: 1px solid #00B8D9;
                border-radius: 4px;
                padding: 7px 10px;
                font-weight: bold;
            }
            QPushButton#PatchViewButton:hover {
                background-color: #21627A;
                border-color: #4DFFE1;
            }
            QPushButton#PatchViewButton:pressed {
                background-color: #103546;
            }
        """)
        self.show_full_product_button.clicked.connect(
            self._show_full_product_view
        )
        patch_layout.addWidget(self.show_full_product_button)

        # ----------------------------------------------------
        # PATCH LIST
        # ----------------------------------------------------
        self.patch_scroll = QScrollArea()
        self.patch_scroll.setWidgetResizable(True)
        self.patch_scroll.setMinimumHeight(320)
        self.patch_scroll.setMaximumHeight(520)
        self.patch_scroll.setFrameShape(QFrame.NoFrame)

        self.patch_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOn
        )
        self.patch_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.patch_rows_container = QWidget()

        self.patch_rows_layout = QVBoxLayout(
            self.patch_rows_container
        )
        self.patch_rows_layout.setContentsMargins(
            2, 2, 2, 2
        )
        self.patch_rows_layout.setSpacing(4)

        self.patch_scroll.setWidget(
            self.patch_rows_container
        )

        patch_layout.addWidget(
            self.patch_scroll
        )

        # ----------------------------------------------------
        # PATCH ACCORDION
        # Keep the internal ID unchanged because other
        # functions use "patch_inspection".
        # ----------------------------------------------------
        patch_section = self._make_collapsible_section(
            "PAINTING PATCHES",
            "patch_inspection",
            patch_content,
        )

        patch_outer_layout.addWidget(
            patch_section
        )

        # Initially hidden until Stage 2.
        self.patch_inspection_panel.hide()

        layout.addWidget(
            self.patch_inspection_panel
        )

        # Stage-specific controls: settings while drawing, recorded paths after generation.
        layout.addWidget(self.spray_settings_section)
        layout.addWidget(self.route_controls_section)

        # ----------------------------------------------------
        # BOTTOM ACTION AREA
        # ----------------------------------------------------
        layout.addStretch()

        self.action_footer = QFrame()
        self.action_footer.setObjectName("RightPanelActionFooter")
        self.action_footer.setStyleSheet("""
            QFrame#RightPanelActionFooter {
                background: #0D1B26;
                border: none;
                border-radius: 7px;
            }
            QFrame#RightPanelActionFooter QLabel {
                color: #8FA8B8;
                font-size: 8pt;
                font-weight: 800;
                border: none;
            }
            /* Preserve the application's original enabled QPushButton appearance.
               Only disabled controls receive a subdued state below. */
            QFrame#RightPanelActionFooter QPushButton:disabled {
                background: #18232C;
                color: #586873;
                border: 1px solid #293943;
            }
        """)
        action_layout = QVBoxLayout(self.action_footer)
        action_layout.setContentsMargins(9, 8, 9, 9)
        action_layout.setSpacing(6)

        self.start_analysis_button = QPushButton(
            "LOAD STL PRODUCT"
        )
        self.start_analysis_button.setObjectName(
            "PrimaryButton"
        )
        # Space normally activates a focused QPushButton. This workflow
        # action must not steal the 3D viewer's Space-bar spray-mode command.
        # Mouse activation remains available.
        self.start_analysis_button.setFocusPolicy(Qt.NoFocus)
        self.start_analysis_button.clicked.connect(
            self._handle_active_stage_action
        )

        action_layout.addWidget(self.start_analysis_button)

        self.import_route_csv_button = QPushButton("IMPORT ROUTE CSV")
        self.import_route_csv_button.setObjectName("SecondaryButton")
        self.import_route_csv_button.setToolTip(
            "Import an existing product-local spray route CSV, preserve its group order, "
            "then continue drawing additional spray groups."
        )
        self.import_route_csv_button.clicked.connect(
            self._import_interactive_spray_route_csv
        )
        self.import_route_csv_button.hide()
        action_layout.addWidget(self.import_route_csv_button)

        self.route_generation_progress = QProgressBar()
        self.route_generation_progress.setObjectName("RouteGenerationProgress")
        self.route_generation_progress.setRange(0, 0)
        self.route_generation_progress.setTextVisible(False)
        self.route_generation_progress.setFixedHeight(7)
        self.route_generation_progress.setStyleSheet(
            "QProgressBar { background: #172B38; border: 1px solid #36566A; "
            "border-radius: 3px; } "
            "QProgressBar::chunk { background: #39B8A5; border-radius: 3px; }"
        )
        self.route_generation_progress.hide()
        action_layout.addWidget(self.route_generation_progress)

        self.finish_spray_path_button = QPushButton("FINISH SPRAY PATH")
        self.finish_spray_path_button.setObjectName("PrimaryButton")
        self.finish_spray_path_button.setToolTip(
            "Stop recording the current spray strokes and generate the finalized spray route."
        )
        self.finish_spray_path_button.clicked.connect(
            self._finish_interactive_spray_path
        )
        self.finish_spray_path_button.hide()
        action_layout.addWidget(self.finish_spray_path_button)

        self.interactive_paint_color_button = QPushButton("PAINT COLOR")
        self.interactive_paint_color_button.setObjectName("SecondaryButton")
        self.interactive_paint_color_button.setToolTip(
            "Choose the paint color used by the interactive coating-thickness visualization."
        )
        self.interactive_paint_color_button.clicked.connect(
            self._choose_interactive_paint_color
        )
        self.interactive_paint_color_button.hide()
        action_layout.addWidget(self.interactive_paint_color_button)

        self.route_next_button = QPushButton("SPRAY SIMULATION")
        self.route_next_button.setObjectName("PrimaryButton")
        self.route_next_button.clicked.connect(self._on_route_next_clicked)
        self.route_next_button.hide()
        action_layout.addWidget(self.route_next_button)

        self.simulation_loading_progress = QProgressBar()
        self.simulation_loading_progress.setObjectName("RouteGenerationProgress")
        self.simulation_loading_progress.setRange(0, 0)
        self.simulation_loading_progress.setTextVisible(False)
        self.simulation_loading_progress.setFixedHeight(7)
        self.simulation_loading_progress.setStyleSheet(
            "QProgressBar { background: #172B38; border: 1px solid #36566A; "
            "border-radius: 3px; } "
            "QProgressBar::chunk { background: #39B8A5; border-radius: 3px; }"
        )
        self.simulation_loading_progress.hide()
        action_layout.addWidget(self.simulation_loading_progress)

        self.route_edit_button = QPushButton("ADD TO EXISTING ROUTE")
        self.route_edit_button.setObjectName("SecondaryButton")
        self.route_edit_button.setToolTip(
            "Continue from the existing route. The current sprayed/coated state remains visible; add only the missing spray strokes."
        )
        self.route_edit_button.clicked.connect(self._continue_interactive_spray_path)
        self.route_edit_button.hide()
        action_layout.addWidget(self.route_edit_button)

        self.simulation_back_button = QPushButton("BACK TO ROUTE GENERATION")
        self.simulation_back_button.setObjectName("SecondaryButton")
        self.simulation_back_button.clicked.connect(
            lambda: self._select_workflow_stage(STAGE_SPRAY_TRAJECTORY)
        )
        self.simulation_back_button.hide()
        action_layout.addWidget(self.simulation_back_button)

        # ----------------------------------------------------
        # OPERATION PROGRESS
        # This is a separate progress indicator for running
        # operations, not the patch-selection progress bar.
        # ----------------------------------------------------
        self.operation_panel = QFrame()
        self.operation_panel.setObjectName(
            "OperationProgressPanel"
        )

        progress_layout = QVBoxLayout(
            self.operation_panel
        )
        progress_layout.setContentsMargins(
            8, 8, 8, 8
        )
        progress_layout.setSpacing(6)

        self.operation_status_label = QLabel(
            "PREPARING OPERATION..."
        )
        self.operation_status_label.setStyleSheet(
            "color: #00E5FF; font-weight: bold;"
        )

        progress_layout.addWidget(
            self.operation_status_label
        )

        self.operation_progress = QProgressBar()
        self.operation_progress.setRange(0, 0)
        self.operation_progress.setTextVisible(False)
        self.operation_progress.setFixedHeight(9)

        progress_layout.addWidget(
            self.operation_progress
        )

        self.operation_panel.setStyleSheet("""
            QFrame#OperationProgressPanel {
                background-color: #0D1822;
                border: 1px solid #203746;
                border-radius: 5px;
            }
        """)

        self.operation_panel.hide()

        layout.addWidget(
            self.operation_panel
        )

        layout.addWidget(self.action_footer)

        return panel


    # ========================================================
    # STAGE 2 PATCH LIST
    # ========================================================

    def _clear_patch_rows(self):
        self.patch_row_widgets.clear()

        while self.patch_rows_layout.count():
            item = self.patch_rows_layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def _get_stage1_patch_ids(self):
        if self.current_stage1_result is None:
            return []

        labels = np.asarray(
            self.current_stage1_result.get(
                "patch_labels",
                [],
            )
        )

        if labels.ndim != 1 or labels.size == 0:
            return []

        return sorted(
            int(value)
            for value in np.unique(labels)
            if int(value) >= 0
        )

    
    
    def _initialize_patch_inspection(self):
        """
        Assign every Stage 1 surface patch to exactly one initial
        Stage 2 decision.

        PLANAR / CURVED -> Paint (included, inspection complete)
        UNCERTAIN / INVALID / unknown -> Requires Inspection

        No patch is silently omitted. Stage 1 patch IDs must cover
        every mesh triangle and match the Stage 1 patch report.
        """

        if self.current_stage1_result is None:
            raise RuntimeError(
                "Stage 1 results are required to initialize patch decisions."
            )

        labels = np.asarray(
            self.current_stage1_result.get("patch_labels", [])
        )

        if (
            labels.ndim != 1
            or labels.size == 0
            or len(labels) != len(self.current_mesh.faces)
        ):
            raise ValueError(
                "Stage 1 patch labels do not cover the complete mesh."
            )

        if np.any(labels < 0):
            raise ValueError(
                "Some mesh triangles have no Surface Patch ID. "
                "Complete surface coverage cannot be guaranteed."
            )

        report_patches = (
            self.current_stage1_result.get("report", {})
            .get("surface_segmentation", {})
            .get("patches", [])
        )

        geometry_by_id = {
            int(patch["patch_id"]): str(
                patch.get("classification", "unknown")
            ).strip().lower()
            for patch in report_patches
        }

        mesh_patch_ids = {
            int(value) for value in np.unique(labels)
        }

        if not geometry_by_id or mesh_patch_ids != set(geometry_by_id):
            raise ValueError(
                "Stage 1 patch report and mesh patch labels do not match. "
                "Every product surface must belong to a reported patch."
            )

        self.patch_inspection = {}

        for patch_id in sorted(mesh_patch_ids):
            geometry_class = geometry_by_id[patch_id]

            if geometry_class in ("planar", "curved"):
                self.patch_inspection[patch_id] = {
                    "inspected": True,
                    "decision": DECISION_PAINT,
                }
            else:
                self.patch_inspection[patch_id] = {
                    "inspected": False,
                    "decision": DECISION_REQUIRES_INSPECTION,
                }

        self.selected_patch_id = None

        self._build_patch_rows()
        self._refresh_patch_progress()



    
    def _build_patch_rows(self):
        """
        Build the Stage 2 patch inspection list.

        Widget values are initialized from patch_inspection.
        """

        self._clear_patch_rows()

        if not self.patch_inspection:
            self.patch_inspection_panel.hide()
            return

        self.patch_inspection_panel.show()

        for patch_id in sorted(self.patch_inspection):

            state = self.patch_inspection[patch_id]

            row = QFrame()
            row.setObjectName("PatchRow")

            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(3, 2, 3, 2)
            row_layout.setSpacing(4)

            # Patch selection button
            select_button = QPushButton(f"PATCH {patch_id}")
            select_button.setFixedWidth(82)
            select_button.setToolTip(
                f"Select Patch {patch_id} in the 3D viewer"
            )

            select_button.clicked.connect(
                lambda checked=False, pid=patch_id:
                self._select_patch_in_viewer(pid)
            )

            # Inspected checkbox
            inspected_checkbox = QCheckBox("Inspected")
            inspected_checkbox.setToolTip(
                "Mark this Patch as visually inspected."
            )
            inspected_checkbox.setChecked(
                bool(state.get("inspected", True))
            )

            # Decision dropdown
            decision_selector = QComboBox()

            decision_selector.addItem("Paint", DECISION_PAINT)
            decision_selector.addItem("No Paint", DECISION_NO_PAINT)
            decision_selector.addItem(
                "Requires Inspection",
                DECISION_REQUIRES_INSPECTION
            )

            decision_selector.setMinimumWidth(125)
            decision_selector.setToolTip(
                "Manual painting decision for this Patch."
            )

            current_decision = state.get(
                "decision",
                DECISION_PAINT
            )

            decision_index = decision_selector.findData(
                current_decision
            )

            if decision_index < 0:
                decision_index = decision_selector.findData(
                    DECISION_PAINT
                )

            decision_selector.setCurrentIndex(decision_index)

            # Connect signals after initializing widget values.
            inspected_checkbox.toggled.connect(
                lambda checked, pid=patch_id:
                self._on_patch_inspected_changed(pid, checked)
            )

            decision_selector.currentIndexChanged.connect(
                lambda index, pid=patch_id, combo=decision_selector:
                self._on_patch_decision_changed(
                    pid,
                    combo.currentData()
                )
            )

            row_layout.addWidget(select_button)
            row_layout.addWidget(inspected_checkbox)
            row_layout.addWidget(decision_selector, 1)

            self.patch_rows_layout.addWidget(row)

            self.patch_row_widgets[patch_id] = {
                "row": row,
                "select_button": select_button,
                "checkbox": inspected_checkbox,
                "decision": decision_selector,
            }

        self.patch_rows_layout.addStretch()


    def _highlight_patch_row(self, patch_id, scroll=True):
        """Highlight and reveal the row matching a selected 3D Patch ID."""
        patch_id = int(patch_id)
        widgets = self.patch_row_widgets.get(patch_id)
        if widgets is None:
            return

        for current_id, row_widgets in self.patch_row_widgets.items():
            selected = int(current_id) == patch_id
            row = row_widgets["row"]
            row.setStyleSheet(
                "QFrame#PatchRow {"
                " background-color: #18394A;"
                " border: 2px solid #00E5FF;"
                " border-radius: 5px;"
                "}"
                if selected
                else ""
            )
            row_widgets["select_button"].setStyleSheet(
                "background-color: #D6A500; color: #101820; "
                "font-weight: bold; border: 2px solid #FFF0A6;"
                if selected
                else ""
            )

        if scroll:
            self._expand_patch_inspection_section()
            self.patch_inspection_panel.show()
            self.patch_scroll.ensureWidgetVisible(
                widgets["row"],
                0,
                24,
            )

    def _show_full_product_view(self):
        """
        Clear the active patch highlight and restore full product opacity.

        Patch inspection decisions and Inspected checkboxes are preserved.
        The operator can select another Patch at any time to resume
        individual-surface inspection.
        """
        if (
            self.viewer.display_mode != "patches"
            or self.viewer.patch_labels is None
        ):
            return

        try:
            self.viewer.set_selected_patch(None)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Full Product View",
                "The complete product view could not be restored.\n\n"
                f"{exc}",
            )
            return

        self.selected_patch_id = None

        # Remove the active row highlight without changing inspection data.
        for widgets in self.patch_row_widgets.values():
            row = widgets.get("row")
            select_button = widgets.get("select_button")
            if row is not None:
                row.setStyleSheet("")
            if select_button is not None:
                select_button.setStyleSheet("")

        self.status_bar.showMessage(
            "FULL PRODUCT VIEW  |  ALL PATCHES AT ORIGINAL OPACITY"
        )

    def _on_viewer_patch_selected(self, patch_id):
        """Synchronize a 3D surface click with the right-side Patch list."""
        if int(patch_id) not in self.patch_inspection:
            return

        self.selected_patch_id = int(patch_id)
        self._highlight_patch_row(self.selected_patch_id, scroll=True)
        self.status_bar.showMessage(
            f"SELECTED PATCH {self.selected_patch_id}  |  "
            "3D SURFACE AND PATCH LIST SYNCHRONIZED"
        )

    def _select_patch_in_viewer(self, patch_id):
        if patch_id not in self.patch_inspection:
            return

        self.selected_patch_id = int(patch_id)

        try:
            self.viewer.select_surface_patch(
                self.selected_patch_id
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Patch Selection",
                "The 3D viewer could not select this Patch.\n\n"
                f"{exc}",
            )
            return

        self._highlight_patch_row(self.selected_patch_id, scroll=True)

        self.status_bar.showMessage(
            f"SELECTED PATCH {patch_id}  |  "
            "INSPECT THE HIGHLIGHTED SURFACE"
        )

    def _on_patch_inspected_changed(self, patch_id, checked):
        if patch_id not in self.patch_inspection:
            return

        self.patch_inspection[patch_id]["inspected"] = bool(
            checked
        )

        self._on_patch_inspection_updated()

    def _on_patch_decision_changed(self, patch_id, decision):
        if patch_id not in self.patch_inspection:
            return

        if decision not in DECISION_TO_FACE_STATUS:
            return

        state = self.patch_inspection[patch_id]

        state["decision"] = decision

        # Paint or No Paint automatically completes inspection.
        if decision in (
            DECISION_PAINT,
            DECISION_NO_PAINT,
        ):
            state["inspected"] = True
        else:
            state["inspected"] = False

        # Update the checkbox without triggering another callback.
        widgets = self.patch_row_widgets.get(patch_id)

        if widgets is not None:
            checkbox = widgets["checkbox"]

            checkbox.blockSignals(True)
            checkbox.setChecked(state["inspected"])
            checkbox.blockSignals(False)

        self._on_patch_inspection_updated()

    def _on_patch_inspection_updated(self):
        """
        Handle changes to patch inspection checkboxes
        and painting decisions.
        """

        try:
            self._apply_manual_patch_decisions()

        except Exception as exc:
            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Patch Decision Error",
                str(exc),
            )

            return

        self._refresh_patch_progress()

        self._persist_patch_inspection()

        all_inspected = self._all_patches_inspected()

        self.stage_completed[
            STAGE_SURFACE_IDENTIFICATION
        ] = all_inspected

        self._refresh_workflow_ui()

        if all_inspected:
            self.status_bar.showMessage(
                "PAINTING SELECTION COMPLETE  |  "
                "100% OF PATCHES INCLUDED  |  "
                "SPRAY TRAJECTORY AVAILABLE"
            )
        else:
            self.status_bar.showMessage(
                "PAINTING SELECTION INCOMPLETE"
            )

        if self.active_stage == STAGE_SURFACE_IDENTIFICATION:
            self._show_surface_identification_summary()

    def _all_patches_inspected(self):
        if not self.patch_inspection:
            return False

        return all(
            item["inspected"]
            for item in self.patch_inspection.values()
        )

    def _refresh_patch_progress(self):
        """Update the percentage of patches selected for painting."""

        total = len(self.patch_inspection)

        selected = sum(
            1
            for state in self.patch_inspection.values()
            if state.get("inspected", True)
        )

        percentage = (
            round(selected * 100 / total)
            if total > 0
            else 0
        )

        self.patch_progress_label.setText(
            f"PAINTABLE PATCHES: {selected} / {total} "
            f"({percentage}%)"
        )

        self.patch_progress_bar.setValue(percentage)

        all_selected = total > 0 and selected == total

        self.stage_completed[
            STAGE_SURFACE_IDENTIFICATION
        ] = all_selected

        self._refresh_stage2_action_visibility()

    
    
    # ========================================================
    # STAGE 2 MANUAL DECISION APPLICATION
    # ========================================================

    def _apply_manual_patch_decisions(self):
        """
        Rebuild the final Stage 2 face labels from the immutable
        automatic Stage 2 baseline plus currently inspected
        manual patch decisions.

        IMPORTANT:
        Never modify the automatic baseline in-place.

        This gives us reversible behavior:

            automatic
                ↓
            inspected + Paint
                ↓
            uncheck Inspected
                ↓
            automatic again
        """

        if self.current_stage2_result is None:
            return

        if self.current_stage1_result is None:
            return

        automatic_labels = (
            self._stage2_automatic_face_status_labels
        )

        if automatic_labels is None:
            # Defensive fallback for older/reloaded state.
            automatic_labels = np.asarray(
                self.current_stage2_result.get(
                    "face_status_labels",
                    [],
                )
            ).copy()

            self._stage2_automatic_face_status_labels = (
                automatic_labels.copy()
            )

        source_labels = np.asarray(
            self.current_stage1_result["patch_labels"]
        )

        automatic_labels = np.asarray(
            automatic_labels
        )

        if (
            source_labels.ndim != 1
            or automatic_labels.ndim != 1
            or len(source_labels) != len(automatic_labels)
        ):
            raise ValueError(
                "Stage 1 Patch Labels and automatic Stage 2 "
                "Face Labels have incompatible dimensions."
            )

        # Start every time from the untouched automatic result.
        face_labels = automatic_labels.copy()

        # Validate only triangles already marked INVALID. The previous
        # implementation materialized full-mesh area and normal arrays
        # inside every patch iteration, repeating large allocations and
        # caching data that this decision step does not otherwise need.
        invalid_indices = np.flatnonzero(
            automatic_labels == FACE_INVALID
        )
        genuinely_invalid = np.zeros(
            len(automatic_labels), dtype=bool
        )
        if invalid_indices.size:
            invalid_faces = np.asarray(
                self.current_mesh.faces[invalid_indices]
            )
            vertices = np.asarray(self.current_mesh.vertices)
            triangles = vertices[invalid_faces]
            finite_triangles = np.all(
                np.isfinite(triangles), axis=(1, 2)
            )
            cross_products = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            twice_areas = np.linalg.norm(cross_products, axis=1)
            valid_invalid_triangles = (
                finite_triangles
                & np.isfinite(twice_areas)
                & (twice_areas > 0.0)
            )
            genuinely_invalid[invalid_indices[~valid_invalid_triangles]] = True

        for patch_id, state in self.patch_inspection.items():

            decision = state.get("decision")

            if decision not in DECISION_TO_FACE_STATUS:
                continue

            mask = source_labels == int(patch_id)

            if not np.any(mask):
                continue

            # Keep every uncertain / invalid patch explicitly in
            # the REVIEW REQUIRED surface category until an operator
            # resolves it. This also maps Stage 2's provisional
            # INVALID patch result into the review bucket; it does
            # not make that surface paintable.
            if decision == DECISION_REQUIRES_INSPECTION:
                face_labels[mask] = FACE_REQUIRES_INSPECTION
                continue

            # An unchecked patch has not been manually reviewed.
            if not bool(state.get("inspected", False)):
                continue

            manual_status = DECISION_TO_FACE_STATUS[decision]

            # An operator may approve a valid triangle whose automatic
            # surface fit failed. Truly degenerate/non-finite triangles
            # remain INVALID and cannot become spray targets.
            apply_mask = mask & ~genuinely_invalid
            face_labels[apply_mask] = manual_status

        self.current_stage2_result[
            "face_status_labels"
        ] = face_labels

        report = self.current_stage2_result.setdefault(
            "report",
            {},
        )

        report["manual_patch_inspection"] = {
            str(patch_id): {
                "inspected": bool(state["inspected"]),
                "decision": state["decision"],
            }
            for patch_id, state in self.patch_inspection.items()
        }

        report["manual_inspection_complete"] = (
            self._all_patches_inspected()
        )

        self._update_stage2_manual_summary()

    def _update_stage2_manual_summary(self):
        if self.current_stage2_result is None:
            return

        labels = np.asarray(
            self.current_stage2_result["face_status_labels"]
        )

        report = self.current_stage2_result.setdefault(
            "report",
            {},
        )

        summary = report.setdefault("summary", {})

        paint_faces = int(
            np.count_nonzero(
                labels == FACE_PAINTABLE_CANDIDATE
            )
        )

        excluded_faces = int(
            np.count_nonzero(
                labels == FACE_EXCLUDED
            )
        )

        inspection_faces = int(
            np.count_nonzero(
                labels == FACE_REQUIRES_INSPECTION
            )
        )

        invalid_faces = int(
            np.count_nonzero(
                labels == FACE_INVALID
            )
        )

        summary["manual_paint_faces"] = paint_faces
        summary["manual_no_paint_faces"] = excluded_faces
        summary["manual_requires_inspection_faces"] = (
            inspection_faces
        )
        summary["invalid_faces"] = invalid_faces

        # ----------------------------------------------------
        # IMPORTANT:
        # Patch decision counts represent completed human
        # decisions only.
        #
        # Uninspected patches retain the default decision value
        # "requires_inspection", but they are NOT counted as a
        # completed manual decision.
        # ----------------------------------------------------

        summary["manual_paint_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
            and state["decision"] == DECISION_PAINT
        )

        summary["manual_no_paint_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
            and state["decision"] == DECISION_NO_PAINT
        )

        summary["manual_requires_inspection_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
            and state["decision"]
            == DECISION_REQUIRES_INSPECTION
        )

        summary["manual_uninspected_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if not state["inspected"]
        )

        summary["manual_inspected_patch_count"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
        )

        summary["manual_total_patch_count"] = len(
            self.patch_inspection
        )

        summary["manual_inspection_complete"] = (
            self._all_patches_inspected()
        )

        # Calculate the actual paintable area from source mesh
        # faces using the source coordinate unit.
        if (
            self.current_mesh is not None
            and self._source_unit_ready()
        ):
            face_areas = np.asarray(
                self.current_mesh.area_faces,
                dtype=float,
            )

            if len(face_areas) == len(labels):
                area_factor = (
                    MM_PER_UNIT[self.coordinate_unit] ** 2
                )

                candidate_area_mm2 = float(
                    np.sum(
                        face_areas[
                            labels == FACE_PAINTABLE_CANDIDATE
                        ]
                    ) * area_factor
                )

                summary["total_candidate_area_mm2"] = (
                    candidate_area_mm2
                )

    def _persist_patch_inspection(self):
        if (
            self.current_stage2_result is None
            or self.current_stl_path is None
        ):
            return

        try:
            output_path = self.result_manager.result_path(
                self.current_stl_path,
                stage_number=2,
                filename="surface_patch_inspection.json",
            )

            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            payload = {
                "source_file": str(self.current_stl_path),
                "coordinate_unit": self.coordinate_unit,
                "patch_count": len(self.patch_inspection),
                "inspected_count": sum(
                    1
                    for item in self.patch_inspection.values()
                    if item["inspected"]
                ),
                "all_inspected": self._all_patches_inspected(),
                "patches": {
                    str(patch_id): {
                        "inspected": bool(state["inspected"]),
                        "decision": state["decision"],
                    }
                    for patch_id, state in self.patch_inspection.items()
                },
                "face_status_values": {
                    "excluded": FACE_EXCLUDED,
                    "paintable_candidate": FACE_PAINTABLE_CANDIDATE,
                    "requires_inspection": FACE_REQUIRES_INSPECTION,
                    "invalid": FACE_INVALID,
                },
            }

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            self.patch_inspection_path = str(output_path)

            self.current_stage2_result[
                "manual_inspection_json_path"
            ] = str(output_path)

        except Exception as exc:
            traceback.print_exc()
            self.status_bar.showMessage(
                f"WARNING: PATCH INSPECTION SAVE FAILED: {exc}"
            )

    # ========================================================
    # UNIT STATE
    # ========================================================

    def _source_unit_ready(self):
        return (
            self.current_mesh is not None
            and self.coordinate_unit in MM_PER_UNIT
        )

    def _set_coordinate_unit_controls(self):
        has_product = self.current_mesh is not None
        unit_known = self._source_unit_ready()

        self.coordinate_unit_selector.setEnabled(
            has_product
            and not self._operation_running
            and self.source_unit_from_stl == "unknown"
        )

        self.unit_selector.setEnabled(
            has_product
            and unit_known
            and not self._operation_running
        )

    def _reset_downstream_results(self):
        self.current_stage1_result = None
        self.current_report = None
        self.current_stage2_result = None
        self.current_stage3_result = None

        self.current_stage3_parameters = None
        self.pending_stage3_parameters = None

        self.patch_inspection = {}
        self.patch_row_widgets = {}
        self.selected_patch_id = None

        # IMPORTANT:
        # The automatic Stage 2 baseline must never survive
        # loading a new product or changing coordinate units.
        self._stage2_automatic_face_status_labels = None
        self.patch_inspection_path = None

        if hasattr(self, "patch_inspection_panel"):
            self._clear_patch_rows()
            self.patch_inspection_panel.hide()

        self.stage_completed[STAGE_GEOMETRY_INSPECTION] = False
        self.stage_completed[STAGE_SURFACE_IDENTIFICATION] = False
        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = False

        if self.current_mesh is not None:
            self.active_stage = STAGE_GEOMETRY_INSPECTION

    def _on_coordinate_unit_changed(self, index):
        if self.current_mesh is None or self._operation_running:
            return

        selected_unit = normalize_unit(
            self.coordinate_unit_selector.currentData()
        )

        if selected_unit == self.coordinate_unit:
            self._set_coordinate_unit_controls()
            self._refresh_workflow_ui()
            return

        if self.source_unit_from_stl != "unknown":
            self.coordinate_unit_selector.blockSignals(True)

            restore_index = self.coordinate_unit_selector.findData(
                self.source_unit_from_stl
            )

            if restore_index >= 0:
                self.coordinate_unit_selector.setCurrentIndex(
                    restore_index
                )

            self.coordinate_unit_selector.blockSignals(False)

            QMessageBox.information(
                self,
                "STL Unit Detected",
                "A coordinate unit was detected in the STL metadata. "
                "The detected unit is used automatically.",
            )
            return

        self.coordinate_unit = selected_unit
        self.manual_unit_selected = (
            selected_unit != "unknown"
        )

        self._reset_downstream_results()

        self._set_coordinate_unit_controls()
        self._update_geometry_metrics()
        self._update_viewer_display_unit()
        self._refresh_workflow_ui()

        self.analysis_summary.setText(
            "COORDINATE UNIT CONFIGURATION UPDATED\n\n"
            f"ACTIVE COORDINATE UNIT: {selected_unit.upper()}\n\n"
            "PREVIOUS ANALYSIS RESULTS WERE INVALIDATED.\n"
            "RUN GEOMETRY INSPECTION AGAIN."
        )

    def _on_unit_changed(self, unit):
        if self.current_mesh is None:
            return

        self._update_geometry_metrics()
        self._update_viewer_display_unit()

        self.status_bar.showMessage(
            f"DISPLAY UNIT: {unit.upper()}  |  "
            f"STL COORDINATE UNIT: {self.coordinate_unit.upper()}  |  "
            "SOURCE COORDINATES UNCHANGED"
        )

    def _get_display_unit(self):
        return normalize_unit(
            self.unit_selector.currentText()
        )

    def _update_viewer_display_unit(self):
        if self.current_mesh is None:
            return

        self.viewer.set_source_unit(self.coordinate_unit)
        self.viewer.set_display_unit(
            self._get_display_unit()
        )

    def _display_length_factor(self):
        if not self._source_unit_ready():
            return None

        display_unit = self._get_display_unit()

        return (
            MM_PER_UNIT[self.coordinate_unit]
            / MM_PER_UNIT[display_unit]
        )

    def _update_unit_labels(self):
        # Geometry and dimension readouts were removed from the left panel.
        # Keep this method as a compatibility no-op for existing call sites.
        return

    # ========================================================
    # STAGE 3 PARAMETER STATE
    # ========================================================

    def _get_current_spray_parameters(self):
        profile = dict(self.sprayer_profile)
        coating_process = dict(self.coating_process)
        return {
            "standoff_mm": float(profile["standoff_mm"]),
            # Geometric spacing is calculated from the active circular footprint.
            "sprayer_profile": profile,
            "coating_process": coating_process,
            # Capture the operator-confirmed CAD-to-World pose before
            # the background Stage 3 worker starts.
            "product_transform_4x4": self.viewer.get_product_transform().tolist(),
        }

    def _open_robot_settings(self):
        if self._operation_running:
            return
        dialog = RobotSettingsDialog(self.robot_settings, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.robot_settings = dict(dialog.settings)
        save_robot_settings(self.robot_settings_path, self.robot_settings)
        self.viewer.set_robot_motion_settings(self.robot_settings)
        if len(getattr(self.viewer, "_interactive_spray_records", [])) >= 2:
            self.viewer._optimize_interactive_recorded_path()
            if self.interactive_spray_path is not None:
                self.interactive_spray_path.update({
                    "total_route_distance_mm": float(
                        self.viewer._interactive_recorded_replay_timing.get(
                            "total_route_distance_mm", 0.0
                        )
                    ) if self.viewer._interactive_recorded_replay_timing else 0.0,
                    "estimated_duration_s": float(
                        self.viewer._interactive_recorded_replay_timing.get(
                            "estimated_duration_s", 0.0
                        )
                    ) if self.viewer._interactive_recorded_replay_timing else 0.0,
                    "robot_motion_settings": dict(self.robot_settings),
                })
                self._show_interactive_route_summary(self.interactive_spray_path)
        self.status_bar.showMessage(
            "ROBOT SETTINGS SAVED | "
            f"PAINTING SPEED: {self.robot_settings['painting_speed_mm_s']:.1f} mm/s | "
            f"CONTROL: {self.robot_settings['control_frequency_hz']:.1f} Hz"
        )

    def _open_sprayer_settings(self):
        if self._operation_running:
            return

        dialog = SprayerSettingsDialog(
            config_path=self.sprayer_config_path,
            profiles=self.sprayer_profiles,
            current_profile=self.sprayer_profile,
            parent=self,
        )
        # The dialog stays open after Save/Save As. A saved profile is
        # applied to the main application when the user finally presses
        # the dialog's Cancel button to close it.
        dialog.exec()
        if dialog.selected_profile is None:
            return

        self.sprayer_profiles = dialog.profiles
        self.sprayer_profile = dict(dialog.selected_profile)
        self.active_sprayer_profile_name = self.sprayer_profile.get(
            "name", "default sprayer"
        )
        save_sprayer_profiles(
            self.sprayer_config_path,
            self.sprayer_profiles,
            active_profile_name=self.active_sprayer_profile_name,
        )

        self.sprayer_profile_label.setText(
            f"ACTIVE PROFILE: {self.sprayer_profile.get('name', 'Unnamed')}"
        )
        self.status_bar.showMessage(
            f"SPRAYER PROFILE LOADED: {self.sprayer_profile.get('name', 'Unnamed')}"
        )
        self._on_spray_parameter_changed()

    def _open_coating_process_settings(self):
        if self._operation_running:
            return
        dialog = CoatingProcessSettingsDialog(
            config_path=self.coating_process_config_path,
            current_process=self.coating_process,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted or dialog.selected_process is None:
            return
        self.coating_process = dict(dialog.selected_process)
        self.viewer.update_coating_process_preview(self.coating_process)
        if self.current_stage3_parameters is not None:
            self.current_stage3_parameters["coating_process"] = dict(self.coating_process)
            if isinstance(self.current_stage3_result, dict):
                trajectory = self.current_stage3_result.get("trajectory", {})
                if isinstance(trajectory, dict):
                    parameters = dict(trajectory.get("parameters", {}))
                    parameters["coating_process"] = dict(self.coating_process)
                    trajectory["parameters"] = parameters
        self.coating_process_label.setText(
            f"PROCESS: {self.coating_process.get('name', 'Unnamed')}  |  "
            f"TARGET: {self.coating_process['target_dry_film_thickness_um']:.1f} µm  |  "
            f"FLOW: {self.coating_process['fluid_flow_ml_min']:.1f} ml/min"
        )
        self.status_bar.showMessage(
            f"COATING PROCESS LOADED: {self.coating_process['name']}"
        )
        self._on_spray_parameter_changed()

    def _stage3_parameters_match(self):
        if self.current_stage3_parameters is None:
            return False

        current = self._get_current_spray_parameters()
        previous = self.current_stage3_parameters
        if current.keys() != previous.keys():
            return False

        for key, value in current.items():
            if key == "product_transform_4x4":
                if not np.allclose(
                    np.asarray(value, dtype=float),
                    np.asarray(previous[key], dtype=float),
                    rtol=0.0,
                    atol=1e-8,
                ):
                    return False
            elif key == "coating_process":
                # Coating material settings do not alter geometric spray spacing.
                continue
            elif value != previous[key]:
                return False

        return True

    def _on_spray_parameter_changed(self, value=None):
        if self._operation_running:
            return

        # Apply the current sprayer geometry to the live interactive route
        # preview as soon as the profile is accepted. Recalculate point spacing
        # from circular-cone footprint width and configured overlap.
        if getattr(self.viewer, "_interactive_spray_active", False):
            profile = dict(self.sprayer_profile)
            working_distance_mm = float(profile["standoff_mm"])
            footprint = _calculate_spray_footprint(
                profile, working_distance_mm
            )
            mm_per_unit = float(MM_PER_UNIT[self.coordinate_unit])
            spacing_mm = float(footprint["nominal_track_spacing_mm"])
            if not np.isfinite(spacing_mm) or spacing_mm <= 0.0:
                raise ValueError("Sprayer settings produced invalid spray point spacing.")
            self.viewer.update_interactive_sprayer_profile(
                profile,
                sampling_spacing=spacing_mm / mm_per_unit,
            )

        # Keep the visible Stage 3 nozzle-point stand-off synchronized without
        # rerunning the automatic trajectory calculation.
        if self.current_stage3_result is not None:
            self.viewer.update_spray_standoff_preview(
                float(self.sprayer_profile["standoff_mm"])
            )

        if self.current_stage3_result is None:
            if self.active_stage == STAGE_SPRAY_TRAJECTORY:
                self._show_stage3_ready_summary()
            return

        parameters_match = self._stage3_parameters_match()

        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = (
            parameters_match
        )

        if self.active_stage == STAGE_SPRAY_TRAJECTORY:
            if parameters_match:
                self._show_spray_trajectory_summary()
            else:
                self._show_stage3_outdated_summary()

        self._refresh_workflow_ui()

    # ========================================================
    # WORKFLOW STATE
    # ========================================================

    
    def _stage_available(self, stage_index):

        if stage_index == STAGE_PRODUCT_SETUP:
            return True

        if stage_index == STAGE_GEOMETRY_INSPECTION:
            return (
                self.current_mesh is not None
                and self._source_unit_ready()
            )

        if stage_index == STAGE_SURFACE_IDENTIFICATION:
            return (
                self.current_mesh is not None
                and self._source_unit_ready()
                and self.stage_completed[
                    STAGE_GEOMETRY_INSPECTION
                ]
                and self.current_stage1_result is not None
            )

        if stage_index == STAGE_SPRAY_TRAJECTORY:

            all_patches_inspected = (
                self.current_stage2_result is not None
                and bool(self.patch_inspection)
                and self._all_patches_inspected()
            )

            return (
                self.current_mesh is not None
                and self._source_unit_ready()
                and self.stage_completed[
                    STAGE_GEOMETRY_INSPECTION
                ]
                and self.current_stage1_result is not None
                and all_patches_inspected
            )

        if stage_index == STAGE_SPRAY_SIMULATION:
            return (
                self._stage_available(STAGE_SPRAY_TRAJECTORY)
                and self.stage_completed.get(STAGE_SPRAY_TRAJECTORY, False)
                and self.interactive_spray_path is not None
            )

        return False


    def _select_workflow_stage(self, stage_index):
        if self._operation_running:
            return

        if not 0 <= stage_index < len(WORKFLOW_STAGES):
            return

        if not self._stage_available(stage_index):
            return

        previous_stage = self.active_stage
        self.active_stage = stage_index
        self._refresh_workflow_ui()
        if stage_index == STAGE_SPRAY_SIMULATION:
            recorded_path_button = self._right_section_buttons.get("recorded_spray_path")
            if recorded_path_button is not None:
                recorded_path_button.setChecked(True)

        if stage_index == STAGE_SURFACE_IDENTIFICATION:
            self._show_surface_identification_summary()

        elif stage_index == STAGE_GEOMETRY_INSPECTION:
            if self.current_report is not None:
                self._show_analysis_summary(self.current_report)

        elif stage_index == STAGE_SPRAY_TRAJECTORY:
            self.viewer.set_recorded_path_page("route")
            if self.interactive_spray_path is None:
                self._show_stage3_ready_summary()
            else:
                self._show_interactive_route_summary(self.interactive_spray_path)

        elif stage_index == STAGE_SPRAY_SIMULATION:
            self.viewer.set_recorded_path_page("simulation")
            self._show_spray_simulation_summary()

        # Start recording only when entering route generation without an
        # existing route. Returning from simulation preserves the route.
        if (
            stage_index == STAGE_SPRAY_TRAJECTORY
            and previous_stage != STAGE_SPRAY_TRAJECTORY
            and self.interactive_spray_path is None
        ):
            self._start_interactive_spray_path()

    
    
    def _refresh_workflow_ui(self):
        route_generation_page = self.active_stage == STAGE_SPRAY_TRAJECTORY
        simulation_page = self.active_stage == STAGE_SPRAY_SIMULATION
        self.spray_settings_section.setVisible(route_generation_page)
        self.viewer.interaction_mode_panel.setVisible(route_generation_page)
        # The same Recorded Spray Path accordion is used on both route
        # generation and simulation pages; keep one authoritative group list.
        self.route_controls_section.setVisible(route_generation_page or simulation_page)
        self.simulation_playback_controls.setVisible(simulation_page)

        spray_settings_button = self._right_section_buttons.get("spray_settings")
        recorded_path_button = self._right_section_buttons.get("recorded_spray_path")
        # Before FINISH SPRAY PATH, open Spray Setting by default so the
        # operator can see the active spray parameters and mode instructions.
        # Once the route is finalized, make Recorded Spray Path the default.
        route_is_finalized = (
            self.interactive_spray_path is not None
            and self.stage_completed.get(STAGE_SPRAY_TRAJECTORY, False)
        )
        current_records = getattr(
            self.viewer, "_interactive_spray_records", []
        ) or []
        recorded_group_ids = {
            int(record.get(
                "group_id",
                int(record.get("stroke_id", 0)) + 1,
            ))
            for record in current_records
            if isinstance(record, dict)
            and not record.get("_stroke_break")
        }
        has_recorded_groups = bool(recorded_group_ids)

        # If any recorded groups remain (including after deleting one group),
        # keep the Recorded Spray Path accordion open. Do not force the user
        # back to Spray Settings merely because the finalized trajectory was
        # invalidated by a deletion.
        if route_generation_page and has_recorded_groups:
            if spray_settings_button is not None and spray_settings_button.isChecked():
                spray_settings_button.setChecked(False)
            if recorded_path_button is not None and not recorded_path_button.isChecked():
                recorded_path_button.setChecked(True)
        elif route_generation_page and not route_is_finalized:
            if spray_settings_button is not None and not spray_settings_button.isChecked():
                spray_settings_button.setChecked(True)
            if recorded_path_button is not None and recorded_path_button.isChecked():
                recorded_path_button.setChecked(False)
        elif route_generation_page or simulation_page:
            if spray_settings_button is not None and spray_settings_button.isChecked():
                spray_settings_button.setChecked(False)
            if recorded_path_button is not None and not recorded_path_button.isChecked():
                recorded_path_button.setChecked(True)
        self._refresh_recorded_spray_path_panel()
        self.simulation_csv_footer.setVisible(simulation_page)
        # Product setup loads the default model automatically when the
        # application starts, so the redundant LOAD STL PRODUCT action is
        # not shown. Keep this shared action button for later workflow stages.
        self.start_analysis_button.setVisible(
            self.active_stage not in (
                STAGE_PRODUCT_SETUP,
                STAGE_SPRAY_TRAJECTORY,
                STAGE_SPRAY_SIMULATION,
            )
        )

        stage = WORKFLOW_STAGES[self.active_stage]

        self.current_stage_label.setText(stage["name"])
        self.workflow_description.setText(stage["description"])

        self._set_coordinate_unit_controls()

        # ----------------------------------------------------
        # SYNCHRONIZE STAGE 2 COMPLETION
        # ----------------------------------------------------

        all_patches_inspected = (
            self.current_stage2_result is not None
            and bool(self.patch_inspection)
            and self._all_patches_inspected()
        )

        self.stage_completed[
            STAGE_SURFACE_IDENTIFICATION
        ] = all_patches_inspected

        # ----------------------------------------------------
        # WORKFLOW NAVIGATION BUTTONS
        # ----------------------------------------------------

        for index, button in enumerate(self.workflow_buttons):

            available = self._stage_available(index)

            button.setEnabled(
                available and not self._operation_running
            )

            button.setChecked(index == self.active_stage)

            if index == self.active_stage:
                button.setObjectName("StageActive")

            elif self.stage_completed.get(index, False):
                button.setObjectName("StageComplete")

            elif available:
                button.setObjectName("StageAvailable")

            else:
                button.setObjectName("StageDisabled")

            button.style().unpolish(button)
            button.style().polish(button)

        # ----------------------------------------------------
        # COLLAPSE PATCH INSPECTION WHEN ENTERING STAGE 3
        # ----------------------------------------------------

        if self.active_stage == STAGE_SPRAY_TRAJECTORY:
            self._collapse_patch_inspection_section()
            # Painting patch selection is locked once spray trajectory
            # simulation begins. Stage 2 remains the editing stage.
            self._set_patch_inspection_readonly(True)
        elif self.active_stage == STAGE_SURFACE_IDENTIFICATION:
            self._set_patch_inspection_readonly(False)

        # ----------------------------------------------------
        # MAIN ACTION BUTTON
        # ----------------------------------------------------

        if self.active_stage == STAGE_PRODUCT_SETUP:

            self.start_analysis_button.setText(
                "LOAD STL PRODUCT"
            )

            self.start_analysis_button.setEnabled(
                not self._operation_running
            )

        elif self.active_stage == STAGE_GEOMETRY_INSPECTION:

            if self.stage_completed[STAGE_GEOMETRY_INSPECTION]:

                self.start_analysis_button.setText(
                    "GEOMETRY INSPECTION COMPLETE"
                )

                self.start_analysis_button.setEnabled(False)

            else:

                self.start_analysis_button.setText(
                    "ANALYZE GEOMETRY"
                )

                self.start_analysis_button.setEnabled(
                    self.current_mesh is not None
                    and self._source_unit_ready()
                    and not self._operation_running
                )

        elif self.active_stage == STAGE_SURFACE_IDENTIFICATION:

            if self.current_stage2_result is None:

                self.start_analysis_button.setText(
                    "IDENTIFY PAINTABLE SURFACES"
                )

                self.start_analysis_button.setEnabled(
                    self._stage_available(
                        STAGE_SURFACE_IDENTIFICATION
                    )
                    and not self._operation_running
                )

            elif all_patches_inspected:

                # Keep the original trajectory entry button visible after
                # every surface patch has been inspected.
                self.start_analysis_button.setText(
                    "SPRAY TRAJECTORY"
                )
                self.start_analysis_button.setEnabled(
                    not self._operation_running
                    and self._stage_available(STAGE_SPRAY_TRAJECTORY)
                )

            else:

                self.start_analysis_button.setText(
                    "PATCH INSPECTION IN PROGRESS"
                )

                self.start_analysis_button.setEnabled(False)

        elif self.active_stage in (
            STAGE_SPRAY_TRAJECTORY, STAGE_SPRAY_SIMULATION
        ):
            self.start_analysis_button.setVisible(False)

        else:

            self.start_analysis_button.setText(
                stage["action"]
            )

            self.start_analysis_button.setEnabled(False)

        route_page = self.active_stage == STAGE_SPRAY_TRAJECTORY
        simulation_page = self.active_stage == STAGE_SPRAY_SIMULATION
        route_exists = (
            self.interactive_spray_path is not None
            and self.stage_completed.get(STAGE_SPRAY_TRAJECTORY, False)
        )
        recording = bool(self.viewer._interactive_spray_active)
        route_has_unfinished_samples = any(
            values is not None and len(values) > 0
            for values in (
                getattr(self.viewer, "_interactive_spray_records", None),
                getattr(self.viewer, "_interactive_pending_spray_strokes", None),
            )
        )
        can_generate_route = (
            route_page and recording and route_has_unfinished_samples
        )
        self.import_route_csv_button.setVisible(
            route_page and not route_exists and not route_has_unfinished_samples
        )
        self.import_route_csv_button.setEnabled(
            route_page and not self._operation_running
        )
        self.interactive_paint_color_button.setVisible(simulation_page)
        self.interactive_paint_color_button.setEnabled(
            simulation_page and not self._operation_running
        )
        self.finish_spray_path_button.setMinimumHeight(42)
        self.finish_spray_path_button.setVisible(route_page)
        self.finish_spray_path_button.setEnabled(
            can_generate_route and not self._operation_running
        )
        self.route_next_button.setMinimumHeight(42)
        self.route_next_button.setText(
            "LOADING SPRAY SIMULATION..." if self._simulation_loading
            else "SPRAY SIMULATION"
        )
        self.route_next_button.setEnabled(
            route_page and route_exists and not recording
            and not self._operation_running and not self._simulation_loading
        )
        self.simulation_loading_progress.setVisible(self._simulation_loading)
        self.route_next_button.setVisible(
            route_page and route_exists and not recording
        )
        self.route_next_button.setEnabled(
            route_page and route_exists and not recording
            and not self._operation_running and not self._simulation_loading
        )
        if route_page and route_exists and not recording:
            self.route_next_button.show()
            self.route_next_button.raise_()
        # Keep an entry point available after every recorded group has
        # been deleted. In that state this action starts a fresh route rather
        # than trying to resume a route that no longer exists.
        has_recorded_groups = any(
            isinstance(record, dict) and not record.get("_stroke_break")
            for record in (
                getattr(self.viewer, "_interactive_spray_records", []) or []
            )
        )
        self.route_edit_button.setText(
            "ADD TO EXISTING ROUTE"
            if route_exists or has_recorded_groups
            else "START NEW ROUTE"
        )
        self.route_edit_button.setToolTip(
            "Continue the remaining recorded spray groups."
            if route_exists or has_recorded_groups
            else "No recorded spray groups remain. Start a new spray-route recording session."
        )
        self.route_edit_button.setVisible(route_page and not recording)
        self.route_edit_button.setEnabled(
            route_page and not recording and not self._operation_running
        )
        self.simulation_back_button.setVisible(simulation_page)
        self.simulation_back_button.setEnabled(
            simulation_page and not self._operation_running
        )

        if self.active_stage in (
            STAGE_SPRAY_TRAJECTORY, STAGE_SPRAY_SIMULATION
        ):
            self.operation_panel.hide()

        # ----------------------------------------------------
        # WORKFLOW STATUS
        # ----------------------------------------------------

        if self._operation_running:

            self._set_workflow_status(
                "STATUS: OPERATION RUNNING",
                "#FFD740",
            )

        elif self.current_mesh is None:

            self._set_workflow_status(
                "STATUS: WAITING FOR PRODUCT",
                "#FFD740",
            )

        elif not self._source_unit_ready():

            self._set_workflow_status(
                "STATUS: COORDINATE UNIT REQUIRED",
                "#FFD740",
            )

        elif self.active_stage == STAGE_SPRAY_TRAJECTORY:

            self._set_workflow_status(
                "STATUS: ROUTE GENERATION",
                "#00E5FF",
            )

        elif self.active_stage == STAGE_SPRAY_SIMULATION:

            self._set_workflow_status(
                "STATUS: SPRAY ROUTE SIMULATION",
                "#00E676",
            )

        elif (
            self.active_stage == STAGE_SURFACE_IDENTIFICATION
            and all_patches_inspected
        ):

            self._set_workflow_status(
                "STATUS: ALL PATCHES INSPECTED",
                "#00E676",
            )

        elif (
            self.active_stage == STAGE_SURFACE_IDENTIFICATION
            and self.current_stage2_result is not None
        ):

            self._set_workflow_status(
                "STATUS: PATCH INSPECTION IN PROGRESS",
                "#FFD740",
            )

        elif (
            self.active_stage == STAGE_GEOMETRY_INSPECTION
            and self.stage_completed[STAGE_GEOMETRY_INSPECTION]
        ):

            self._set_workflow_status(
                "STATUS: GEOMETRY INSPECTION COMPLETE",
                "#00E676",
            )

        elif self.active_stage == STAGE_SURFACE_IDENTIFICATION:

            self._set_workflow_status(
                "STATUS: READY FOR SURFACE IDENTIFICATION",
                "#00E5FF",
            )

        elif self.active_stage == STAGE_GEOMETRY_INSPECTION:

            self._set_workflow_status(
                "STATUS: READY FOR GEOMETRY INSPECTION",
                "#00E5FF",
            )

        else:

            self._set_workflow_status(
                "STATUS: PRODUCT READY",
                "#00E5FF",
            )

        # ----------------------------------------------------
        # FINAL UI SYNCHRONIZATION
        # ----------------------------------------------------

        self._refresh_pipeline()
        self._refresh_stage2_action_visibility()



    def _refresh_stage2_action_visibility(self):
        """
        Refresh the Stage 2 inspection message and action button.

        If Stage 2 inspection is incomplete:
            - Show PATCH INSPECTION IN PROGRESS
            - Hide the main action button

        If all patches are inspected:
            - Hide PATCH INSPECTION IN PROGRESS
            - Show SPRAY TRAJECTORY
        """

        if not hasattr(
            self,
            "patch_inspection_status_label",
        ):
            return

        if not hasattr(
            self,
            "start_analysis_button",
        ):
            return

        if (
            self.active_stage
            != STAGE_SURFACE_IDENTIFICATION
        ):
            return

        if self.current_stage2_result is None:
            self.patch_inspection_status_label.hide()

            self.start_analysis_button.setVisible(
                True
            )

            return

        all_inspected = self._all_patches_inspected()

        if all_inspected:

            self.patch_inspection_status_label.hide()
            self.start_analysis_button.setVisible(True)
            self.start_analysis_button.setText("SPRAY TRAJECTORY")
            self.start_analysis_button.setEnabled(
                not self._operation_running
                and self._stage_available(STAGE_SPRAY_TRAJECTORY)
            )

        else:

            self.start_analysis_button.setText(
                "PATCH INSPECTION IN PROGRESS"
            )
            self.start_analysis_button.setEnabled(False)
            self.start_analysis_button.setVisible(True)

            self.patch_inspection_status_label.setText(
                "PATCH INSPECTION IN PROGRESS"
            )

            self.patch_inspection_status_label.setStyleSheet(
                "color: #FFD740; font-weight: bold;"
            )

            self.patch_inspection_status_label.show()
    def _set_workflow_status(self, text, color):
        self.workflow_status_label.setText(text)
        self.workflow_status_label.setStyleSheet(
            f"color: {color}; font-weight: bold;"
        )

    # ========================================================
    # PIPELINE STATUS
    # ========================================================

    def _refresh_pipeline(self):
        pipeline_states = [
            "COMPLETE" if self.current_mesh is not None else "ACTIVE"
        ]

        if self.stage_completed[STAGE_GEOMETRY_INSPECTION]:
            pipeline_states.append("COMPLETE")
        elif self.current_mesh is not None:
            pipeline_states.append(
                "ACTIVE" if self._source_unit_ready() else "WAITING"
            )
        else:
            pipeline_states.append("WAITING")

        if self.stage_completed[STAGE_SURFACE_IDENTIFICATION]:
            pipeline_states.append("COMPLETE")
        elif self.current_stage2_result is not None:
            pipeline_states.append("ACTIVE")
        elif self._stage_available(STAGE_SURFACE_IDENTIFICATION):
            pipeline_states.append("ACTIVE")
        else:
            pipeline_states.append("WAITING")

        if self.stage_completed[STAGE_SPRAY_TRAJECTORY]:
            pipeline_states.append("COMPLETE")
        elif self._stage_available(STAGE_SPRAY_TRAJECTORY):
            pipeline_states.append("ACTIVE")
        else:
            pipeline_states.append("WAITING")

        for label, state, item in zip(
            self.pipeline_labels,
            pipeline_states,
            PIPELINE_ITEMS,
        ):
            if state == "COMPLETE":
                label.setText("●  " + item + "  [COMPLETE]")
                label.setStyleSheet("color: #00E676;")
            elif state == "ACTIVE":
                label.setText("◉  " + item + "  [READY]")
                label.setStyleSheet("color: #00E5FF;")
            elif state == "WAITING":
                label.setText("○  " + item + "  [WAITING]")
                label.setStyleSheet("color: #FFD740;")
            else:
                label.setText("○  " + item + "  [NOT IMPLEMENTED]")
                label.setStyleSheet("color: #71818B;")

    # ========================================================
    # MAIN WORKFLOW ACTION
    # ========================================================

    def _handle_active_stage_action(self):
        """
        Handle the main workflow action button.

        Surface identification:
            Run identification, inspect patches, then enter Stage 3
            when all patches have been inspected.

        Stage 3 generates the initial circular-footprint preview.
        """

        if self._operation_running:
            return

        # ----------------------------------------------------
        # STAGE 1 - PRODUCT SETUP
        # ----------------------------------------------------

        if self.active_stage == STAGE_PRODUCT_SETUP:

            self.load_stl_dialog()
            return

        # ----------------------------------------------------
        # STAGE 2 - GEOMETRY INSPECTION
        # ----------------------------------------------------

        elif self.active_stage == STAGE_GEOMETRY_INSPECTION:

            self.run_geometry_analysis()
            return

        # ----------------------------------------------------
        # STAGE 3 - SURFACE IDENTIFICATION
        # ----------------------------------------------------

        elif self.active_stage == STAGE_SURFACE_IDENTIFICATION:

            # Stage 2 has not been executed yet.
            if self.current_stage2_result is None:

                self.run_surface_identification()
                return

            # Verify the actual inspection records.
            all_inspected = self._all_patches_inspected()

            if not all_inspected:

                self.stage_completed[
                    STAGE_SURFACE_IDENTIFICATION
                ] = False

                self._refresh_workflow_ui()

                QMessageBox.warning(
                    self,
                    "Patch Inspection Incomplete",
                    "All patches must be inspected before "
                    "generating the spray trajectory.",
                )

                return

            # Synchronize Stage 2 completion.
            self.stage_completed[
                STAGE_SURFACE_IDENTIFICATION
            ] = True

            # Stage 3 is now the interactive spray workspace.
            # Do not run the legacy automatic coverage planner, trajectory
            # inspection, JSON/NPY export or screenshot pipeline here.
            # Apply the operator's final Stage 2 decisions, then enter the
            # interactive stage; its entry handler starts recording once.
            self._apply_manual_patch_decisions()
            self._persist_patch_inspection()
            self._select_workflow_stage(STAGE_SPRAY_TRAJECTORY)
            return

        # ----------------------------------------------------
        # STAGE 3 - SPRAY TRAJECTORY
        # ----------------------------------------------------

        elif self.active_stage == STAGE_SPRAY_TRAJECTORY:
            # Stage 3 is interactive-only. Never invoke the legacy
            # automatic trajectory generator from the main action button.
            self._start_interactive_spray_path()
            return

    # ========================================================
    # OPERATION UI CONTROL
    # ========================================================

    def _start_operation_ui(self, operation_name):
        self._operation_running = True

        self.operation_status_label.setText(
            operation_name.upper() + "..."
        )
        self.operation_progress.setRange(0, 0)
        self.operation_panel.show()

        self.load_button.setEnabled(False)
        self.coordinate_unit_selector.setEnabled(False)
        self.unit_selector.setEnabled(False)

        self.sprayer_settings_button.setEnabled(False)

        self.system_status.setText("● PROCESSING")
        self.system_status.setStyleSheet(
            "color: #FFD740; font-weight: bold;"
        )

        self.status_bar.showMessage(
            operation_name.upper() + " IN PROGRESS..."
        )

        self._refresh_workflow_ui()

    def _finish_operation_ui(self, success):
        self._operation_running = False
        self.operation_panel.hide()

        self.load_button.setEnabled(True)

        self.sprayer_settings_button.setEnabled(True)

        self._set_coordinate_unit_controls()

        self.system_status.setText("● SYSTEM ONLINE")
        self.system_status.setStyleSheet(
            "color: #00E676; font-weight: bold;"
        )

        self._refresh_workflow_ui()

        if not success:
            self.status_bar.showMessage("OPERATION FAILED")

    # ========================================================
    # MANUAL REFERENCE POSE STORAGE
    # ========================================================

    def _save_reference_pose(self, pose):
        if self.current_stl_path is None:
            QMessageBox.warning(self, "Reference Pose", "Load a product CAD model before saving a pose.")
            return

        from datetime import datetime, timezone
        output_path = self.result_manager.result_path(
            self.current_stl_path, stage_number=1, filename="reference_poses.json"
        )
        records = []
        if output_path.is_file():
            try:
                with output_path.open("r", encoding="utf-8") as file:
                    existing = json.load(file)
                if isinstance(existing, dict):
                    records = existing.get("poses", [])
                elif isinstance(existing, list):
                    records = existing
            except (OSError, json.JSONDecodeError):
                QMessageBox.critical(
                    self, "Reference Pose",
                    "Existing reference_poses.json is invalid; no data was overwritten."
                )
                return

        record = dict(pose)
        record["pose_id"] = len(records) + 1
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        records.append(record)
        saved_path = self.result_manager.save_json(
            stl_path=self.current_stl_path,
            stage_number=1,
            filename="reference_poses.json",
            data={
                "schema_version": 1,
                "product_file": self.current_stl_path.name,
                "world_frame": "WORLD_XYZ",
                "pose_count": len(records),
                "poses": records,
            },
        )
        self.status_bar.showMessage(
            f"REFERENCE POSE {record['pose_id']} SAVED: {saved_path}", 10000
        )
        QMessageBox.information(
            self, "Reference Pose Saved",
            f"Pose {record['pose_id']} saved successfully.\n\n{saved_path}"
        )

    # ========================================================
    # STL LOADING
    # ========================================================

    def load_stl_dialog(self):
        """Compatibility alias: the primary product format is now STEP."""
        self.load_step_dialog()

    def load_step_dialog(self):
        if self._operation_running:
            return

        initial_dir = self.project_root / "models"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select STEP Product CAD",
            str(initial_dir),
            "STEP CAD Files (*.step *.stp *.STEP *.STP)",
        )
        if file_path:
            self.load_part(file_path)

    def load_part(self, file_path):
        if self._operation_running:
            return

        selected_path = Path(file_path).expanduser().resolve()
        if selected_path.suffix.lower() not in (".step", ".stp"):
            QMessageBox.warning(
                self,
                "Unsupported Product Format",
                "OpenArmX accepts STEP AP214 files only (.step / .stp).",
            )
            return

        try:
            mesh = load_step(selected_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "CAD Loading Error",
                str(exc),
            )
            return

        self.current_mesh = mesh
        self.current_stl_path = selected_path
        self.current_step_report = mesh.metadata.get("step_geometry_report")
        self.current_step_report_path = None
        if self.current_step_report is not None:
            self.current_step_report_path = self.result_manager.save_json(
                stl_path=self.current_stl_path,
                stage_number=1,
                filename="step_geometry_report.json",
                data=self.current_step_report,
            )

        self._reset_downstream_results()

        self.stage_completed = {
            STAGE_PRODUCT_SETUP: True,
            STAGE_GEOMETRY_INSPECTION: False,
            STAGE_SURFACE_IDENTIFICATION: False,
            STAGE_SPRAY_TRAJECTORY: False,
        }

        self.source_unit_from_stl = get_mesh_source_unit(mesh)

        if self.source_unit_from_stl != "unknown":
            self.coordinate_unit = self.source_unit_from_stl
            self.manual_unit_selected = False
        else:
            self.coordinate_unit = "unknown"
            self.manual_unit_selected = False

        selector_index = self.coordinate_unit_selector.findData(
            self.coordinate_unit
        )

        if selector_index >= 0:
            self.coordinate_unit_selector.blockSignals(True)
            self.coordinate_unit_selector.setCurrentIndex(
                selector_index
            )
            self.coordinate_unit_selector.blockSignals(False)

        self.coordinate_unit_selector.setEnabled(
            self.source_unit_from_stl == "unknown"
        )

        self.active_stage = STAGE_GEOMETRY_INSPECTION

        self.viewer.show_product(mesh)

        # STEP is translated into canonical millimetres by Open CASCADE.
        # Detect internal cylindrical surfaces immediately after import so
        # hole locations and CAD-origin distances are visible before Stage 1.
        hole_candidates = []
        try:
            hole_candidates = detect_hole_candidates(
                mesh,
                coordinate_unit="mm",
            )
        except Exception as exc:
            traceback.print_exc()
            self.status_bar.showMessage(
                f"STEP loaded, but preliminary hole detection failed: {exc}",
                12000,
            )
        self.viewer.set_hole_candidates(hole_candidates)
        diagnostics = mesh.metadata.get("hole_detection_diagnostics", {})
        if hole_candidates:
            self.status_bar.showMessage(
                f"STEP hole-axis candidates detected: {len(hole_candidates)}",
                10000,
            )
        else:
            self.status_bar.showMessage(
                "STEP loaded, but no internal-cylinder candidates were classified. "
                f"B-Rep cylinders: {diagnostics.get('cylindrical_face_count', 'unknown')}; "
                f"orientation counts: {diagnostics.get('cylinder_orientation_counts', {})}. "
                "See terminal output for exact B-Rep diagnostics.",
                15000,
            )

        self.part_name_label.setText(
            self.current_stl_path.name
        )

        self._set_coordinate_unit_controls()
        self._update_geometry_metrics()
        self._update_viewer_display_unit()

        part_dir = self.result_manager.part_dir(
            self.current_stl_path
        )

        self.analysis_summary.setStyleSheet("")

        if self.current_step_report is not None:
            report = self.current_step_report
            topo = report["topology"]

            self.analysis_summary.setText(
                f"STEP CAD LOADED\n"
                f"{topo['faces']:,} CAD FACES"
            )
        else:
            self.analysis_summary.setText(
                f"STEP LOADED\nUNIT: {self.coordinate_unit.upper()}"
            )

        self.summary_section.setText("STAGE SUMMARY")
        self._refresh_workflow_ui()

        self.status_bar.showMessage(
            f"PRODUCT LOADED  |  "
            f"{self.current_stl_path.name}  |  "
            f"COORDINATE UNIT: {self.coordinate_unit.upper()}  |  "
            f"DISPLAY UNIT: {self.unit_selector.currentText()}"
        )

    # ========================================================
    # GEOMETRY DISPLAY
    # ========================================================

    
    def _update_geometry_metrics(self):
        # Geometry and bounding-dimension values are no longer displayed
        # in the left sidebar. Analysis data and reports remain unchanged.
        return


    # ========================================================
    # STAGE 1 - GEOMETRY ANALYSIS
    # ========================================================

    def run_geometry_analysis(self):
        if self.current_mesh is None or self._operation_running:
            return

        if not self._source_unit_ready():
            QMessageBox.warning(
                self,
                "Coordinate Unit Required",
                "The STL file does not contain a recognized "
                "coordinate unit.\n\n"
                "Select the actual coordinate unit before "
                "running Geometry Inspection.",
            )
            return

        if self.stage_completed[STAGE_GEOMETRY_INSPECTION]:
            return

        manual_override = (
            self.coordinate_unit
            if self.manual_unit_selected
            else None
        )

        self._start_operation_ui("GEOMETRY INSPECTION")

        self._worker = OperationWorker(
            execute_stage1,
            mesh=self.current_mesh,
            stl_path=self.current_stl_path,
            result_manager=self.result_manager,
            source_unit=manual_override,
        )

        self._worker.succeeded.connect(
            self._on_geometry_analysis_complete
        )
        self._worker.failed.connect(
            self._on_geometry_analysis_failed
        )
        self._worker.finished.connect(
            self._on_worker_thread_finished
        )

        self._worker.start()

    
    def _on_geometry_analysis_complete(self, result):
        try:
            report = result["report"]
            labels = result["patch_labels"]

            self.viewer.show_surface_patches(
                self.current_mesh,
                labels,
            )

            self._update_viewer_display_unit()

            screenshot_path = self.result_manager.result_path(
                self.current_stl_path,
                stage_number=1,
                filename="surface_patches.png",
            )

            try:
                self.viewer.save_screenshot(screenshot_path)
            except Exception as exc:
                print("Screenshot export failed:", exc)

            self.current_stage1_result = result
            self.current_report = report

            # Refresh geometry metrics using the Stage 1 report.
            self._update_geometry_metrics()

            self.stage_completed[
                STAGE_GEOMETRY_INSPECTION
            ] = True

            self.active_stage = STAGE_SURFACE_IDENTIFICATION

            self._show_analysis_summary(report)
            self._expand_summary_section()
            self._finish_operation_ui(True)

            self.status_bar.showMessage(
                "GEOMETRY INSPECTION COMPLETE | "
                f"UNIT: {self.coordinate_unit.upper()} | "
                f"RESULT: {result['json_path']}"
            )

        except Exception as exc:
            traceback.print_exc()
            self._finish_operation_ui(False)

            QMessageBox.critical(
                self,
                "Geometry Display Error",
                str(exc),
            )


    def _on_geometry_analysis_failed(self, error_message):
        self._finish_operation_ui(False)

        self.workflow_status_label.setText(
            "STATUS: GEOMETRY ANALYSIS FAILED"
        )

        QMessageBox.critical(
            self,
            "Geometry Inspection Error",
            error_message,
        )

    # ========================================================
    # STAGE 2 - SURFACE IDENTIFICATION
    # ========================================================

    def run_surface_identification(self):
        if self.current_mesh is None or self._operation_running:
            return

        if self.current_stage1_result is None:
            QMessageBox.warning(
                self,
                "Stage 1 Required",
                "Run Geometry Inspection before "
                "Surface Identification.",
            )
            return

        if self.current_stage2_result is not None:
            return

        self._start_operation_ui(
            "AUTOMATIC SURFACE CLASSIFICATION"
        )

        self._worker = OperationWorker(
            execute_stage2,
            mesh=self.current_mesh,
            stl_path=self.current_stl_path,
            result_manager=self.result_manager,
            stage1_result=self.current_stage1_result,
            coordinate_unit=self.coordinate_unit,
        )

        self._worker.succeeded.connect(
            self._on_surface_identification_complete
        )
        self._worker.failed.connect(
            self._on_surface_identification_failed
        )
        self._worker.finished.connect(
            self._on_worker_thread_finished
        )

        self._worker.start()

    def _on_surface_identification_complete(self, result):
        try:
            self.current_stage2_result = result

            # ------------------------------------------------
            # CRITICAL:
            #
            # Save an immutable copy of the automatic Stage 2
            # face classification BEFORE any manual decisions
            # are applied.
            # ------------------------------------------------

            automatic_labels = np.asarray(
                result.get("face_status_labels", [])
            )

            if automatic_labels.ndim != 1:
                raise ValueError(
                    "Stage 2 face_status_labels must be a 1D array."
                )

            if (
                self.current_mesh is not None
                and len(automatic_labels)
                != len(self.current_mesh.faces)
            ):
                raise ValueError(
                    "Stage 2 face_status_labels count does not "
                    "match the STL face count."
                )

            self._stage2_automatic_face_status_labels = (
                automatic_labels.copy()
            )

            # Create a row for every actual Stage 1 patch.
            self._initialize_patch_inspection()

            # Keep the patch-color view while the operator
            # inspects individual patches.
            self.viewer.show_surface_patches(
                self.current_mesh,
                self.current_stage1_result["patch_labels"],
            )

            self._update_viewer_display_unit()

            self.active_stage = STAGE_SURFACE_IDENTIFICATION
            self.stage_completed[
                STAGE_SURFACE_IDENTIFICATION
            ] = False

            self._set_patch_inspection_readonly(False)
            self._expand_patch_inspection_section()
            # This now rebuilds the result from the untouched
            # automatic baseline. Since no patches are inspected
            # yet, the automatic labels remain unchanged.
            self._apply_manual_patch_decisions()
            self._persist_patch_inspection()

            self._show_surface_identification_summary()
            self._finish_operation_ui(True)

            self.status_bar.showMessage(
                f"STAGE 2 AUTOMATIC CLASSIFICATION COMPLETE  |  "
                f"{len(self.patch_inspection)} PATCHES REQUIRE REVIEW"
            )

        except Exception as exc:
            traceback.print_exc()
            self._finish_operation_ui(False)

            QMessageBox.critical(
                self,
                "Surface Display Error",
                str(exc),
            )

    def _on_surface_identification_failed(self, error_message):
        self._finish_operation_ui(False)

        self.workflow_status_label.setText(
            "STATUS: SURFACE IDENTIFICATION FAILED"
        )

        QMessageBox.critical(
            self,
            "Surface Identification Error",
            error_message,
        )

    # ========================================================
    # STAGE 3 - SURFACE TYPE MAPPING
    # ========================================================

    def _build_face_surface_types(self):
        if self.current_stage1_result is None:
            return None

        patch_labels = np.asarray(
            self.current_stage1_result.get(
                "patch_labels",
                [],
            )
        )

        if (
            patch_labels.ndim != 1
            or len(patch_labels) != len(self.current_mesh.faces)
        ):
            return None

        report = self.current_stage1_result.get("report", {})

        segmentation = report.get(
            "surface_segmentation",
            {},
        )

        patches = segmentation.get("patches", [])

        if not patches:
            return None

        patch_type_map = {}

        for patch in patches:
            patch_id = patch.get("patch_id")

            classification = str(
                patch.get("classification", "unknown")
            ).lower()

            if "planar" in classification:
                patch_type_map[patch_id] = "planar"
            elif "curved" in classification:
                patch_type_map[patch_id] = "curved"
            else:
                patch_type_map[patch_id] = "unknown"

        face_types = np.full(
            len(patch_labels),
            "unknown",
            dtype="<U16",
        )

        for patch_id, classification in patch_type_map.items():
            try:
                numeric_patch_id = int(patch_id)
            except (TypeError, ValueError):
                continue

            face_types[
                patch_labels == numeric_patch_id
            ] = classification

        return face_types

    # ========================================================
    # STAGE 3 - SPRAY TRAJECTORY GENERATION
    # ========================================================

    def run_spray_trajectory(self, force=False):
        """Generate the first circular-footprint coverage trajectory."""
        if self.viewer.is_trajectory_playing():
            return

        if not self._source_unit_ready():
            QMessageBox.warning(
                self,
                "Coordinate Unit Required",
                "Select the actual coordinate unit for the STEP model first.",
            )
            return

        if self.current_stage2_result is None:
            QMessageBox.warning(
                self,
                "Stage 2 Required",
                "Run Surface Identification before "
                "generating a spray trajectory.",
            )
            return

        if not self._all_patches_inspected():
            QMessageBox.warning(
                self,
                "Patch Inspection Incomplete",
                "Every Patch must be marked Inspected "
                "before Stage 3 can run.",
            )
            return

        if (
            not force
            and self.stage_completed[STAGE_SPRAY_TRAJECTORY]
            and self._stage3_parameters_match()
        ):
            return

        # Ensure the final manual decisions are reflected
        # in the face labels passed to Stage 3.
        self._apply_manual_patch_decisions()
        self._persist_patch_inspection()

        parameters = self._get_current_spray_parameters()
        self.pending_stage3_parameters = parameters.copy()

        self._start_operation_ui(
            "SPRAY TRAJECTORY GENERATION"
        )

        self._worker = OperationWorker(
            execute_stage3,
            mesh=self.current_mesh,
            stl_path=self.current_stl_path,
            result_manager=self.result_manager,
            stage2_result=self.current_stage2_result,
            stage1_report=self.current_report,
            coordinate_unit=self.coordinate_unit,
            standoff_mm=parameters["standoff_mm"],
            sprayer_profile=parameters["sprayer_profile"],
            coating_process=parameters["coating_process"],
            product_transform=parameters["product_transform_4x4"],
        )

        self._worker.succeeded.connect(
            self._on_spray_trajectory_complete
        )
        self._worker.failed.connect(
            self._on_spray_trajectory_failed
        )
        self._worker.finished.connect(
            self._on_worker_thread_finished
        )

        self._worker.start()

    def _choose_interactive_paint_color(self):
        current = getattr(self.viewer, "_trajectory_paint_color_hex", None) or "#D95B43"
        selected = QColorDialog.getColor(QColor(current), self, "Select spray paint color")
        if not selected.isValid():
            return
        self.viewer._trajectory_paint_color_hex = selected.name().upper()
        self.interactive_paint_color_button.setStyleSheet(
            "QPushButton { color: %s; font-weight: 700; }"
            % self.viewer._trajectory_paint_color_hex
        )
        self.interactive_paint_color_button.setText(
            "PAINT COLOR  " + self.viewer._trajectory_paint_color_hex
        )
        self.viewer._interactive_refresh_recorded_paint_color()

    def _import_interactive_spray_route_csv(self):
        """Import an existing CSV before drawing, then resume additive route editing."""
        if (
            self._operation_running
            or self.active_stage != STAGE_SPRAY_TRAJECTORY
            or self.current_mesh is None
            or not self._stage_available(STAGE_SPRAY_TRAJECTORY)
        ):
            return

        if self.interactive_spray_path is not None:
            QMessageBox.information(
                self,
                "Import spray route",
                "A finalized route already exists. Use ADD TO EXISTING to continue "
                "that route; import a CSV before drawing a new route.",
            )
            return

        has_unfinished_data = bool(
            getattr(self.viewer, "_interactive_spray_records", [])
            or getattr(self.viewer, "_interactive_pending_spray_strokes", [])
            or getattr(self.viewer, "_interactive_spray_stroke_input_poses", [])
        )
        if has_unfinished_data:
            QMessageBox.warning(
                self,
                "Import spray route",
                "Finish or undo the current drawing before importing a CSV. "
                "The import will not overwrite unfinished strokes.",
            )
            return

        # Stage 3 starts an empty recording session automatically. Stop that
        # empty session so the CSV importer can initialize its route records.
        if self.viewer._interactive_spray_active:
            try:
                self.viewer.finish_interactive_spray_path()
            except Exception as exc:
                QMessageBox.critical(
                    self, "Import spray route", f"Could not prepare route import:\n{exc}"
                )
                return

        if not self.viewer._import_interactive_trajectory_csv():
            # A cancelled or invalid import should leave Stage 3 ready for
            # ordinary drawing, rather than leaving mouse input disabled.
            self._start_interactive_spray_path()
            return

        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = False
        self.stage_completed[STAGE_SPRAY_SIMULATION] = False
        self.interactive_spray_path = None
        self.viewer.set_recorded_path_page("route")
        self._start_interactive_spray_path(resume_existing=True)
        self.status_bar.showMessage(
            "CSV ROUTE LOADED · CONTINUE DRAWING TO ADD MORE SPRAY GROUPS"
        )
        self._refresh_workflow_ui()

    def _start_interactive_spray_path(self, resume_existing=False):
        """Start or continue mouse-driven spray recording on Stage 3."""
        if (
            self._operation_running
            or self.current_mesh is None
            or self.viewer._interactive_spray_active
        ):
            return

        if not self._source_unit_ready():
            QMessageBox.warning(
                self,
                "Coordinate Unit Required",
                "Select the actual coordinate unit before interactive spraying.",
            )
            return

        parameters = self._get_current_spray_parameters()
        mm_per_unit = float(MM_PER_UNIT[self.coordinate_unit])
        footprint = _calculate_spray_footprint(
            parameters["sprayer_profile"],
            float(parameters["standoff_mm"]),
        )
        try:
            self.viewer.start_interactive_spray_path(
                standoff=float(parameters["standoff_mm"]) / mm_per_unit,
                cone_angle_deg=float(footprint["included_angle_deg"]),
                sampling_spacing=(
                    float(footprint["nominal_track_spacing_mm"])
                    if float(footprint["nominal_track_spacing_mm"]) > 0.0
                    else 1.0
                ) / mm_per_unit,
                sprayer_profile=parameters["sprayer_profile"],
                coating_process=parameters["coating_process"],
                excluded_face_mask=(
                    np.asarray(
                        self.current_stage2_result["face_status_labels"]
                    ) != FACE_PAINTABLE_CANDIDATE
                    if isinstance(self.current_stage2_result, dict)
                    and self.current_stage2_result.get("face_status_labels") is not None
                    else None
                ),
                resume_existing=bool(resume_existing),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Interactive Spray Path Error", str(exc))
            return
        self._refresh_workflow_ui()
        stage2_labels = np.asarray(
            self.current_stage2_result.get("face_status_labels", [])
            if isinstance(self.current_stage2_result, dict)
            else []
        )
        excluded_count = (
            int(np.count_nonzero(stage2_labels != FACE_PAINTABLE_CANDIDATE))
            if stage2_labels.shape == (len(self.current_mesh.faces),)
            else len(self.current_mesh.faces)
        )
        paintable_count = max(len(self.current_mesh.faces) - excluded_count, 0)
        action_text = (
            "CONTINUE SPRAYING"
            if resume_existing
            else "START SPRAYING"
        )
        self.status_bar.showMessage(
            f"STAGE 2: {excluded_count} NON-PAINTABLE FACES | "
            f"{action_text} | CAMERA VIEW ACTIVE | PRESS SPACE FOR SPRAY MODE"
        )
        self.analysis_summary.setText(
            "SPRAY TRAJECTORY\n\n"
            f"STAGE 2 PAINTABLE TRIANGLES: {paintable_count:,}\n"
            f"STAGE 2 EXCLUDED / UNCONFIRMED TRIANGLES: {excluded_count:,}\n"
            "Excluded and unconfirmed regions are highlighted RED.\n"
            "The sprayer turns RED and deposition is disabled over those regions.\n\n"
            "Camera navigation is active on entry.\n"
            "Press SPACE to switch to spray mode.\n"
            "Move the mouse over paintable Stage 2 surfaces.\n"
            "Hold the left mouse button to paint and record the path.\n"
            "When all strokes are drawn, select GENERATE SPRAY ROUTE to build the spray points."
        )

    def _on_spray_samples_changed(self):
        """Refresh route-generation controls after a completed mouse stroke."""
        self._refresh_recorded_spray_path_panel()
        self._refresh_workflow_ui()

    def _on_manual_route_readiness_changed(self, ready):
        """Enable the next stage when independent spray strokes are finalized."""
        if getattr(self, "_suppress_route_readiness_callback", False):
            return
        ready = bool(ready)
        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = ready
        if ready:
            timing = getattr(
                self.viewer, "_interactive_recorded_replay_timing", None
            ) or {}
            route = getattr(self, "interactive_spray_path", None)
            if isinstance(route, dict):
                route["spray_length_mm"] = float(timing.get("spray_distance_mm", 0.0))
                route["total_route_distance_mm"] = float(timing.get("spray_distance_mm", 0.0))
                route["estimated_duration_s"] = float(timing.get("estimated_duration_s", 0.0))
                route["route_timing"] = {
                    key: value for key, value in timing.items() if key != "timeline"
                }
                route["route_optimization"] = dict(
                    getattr(self.viewer, "_interactive_spray_route_optimization", {})
                )
                self._show_interactive_route_summary(route)
            self.status_bar.showMessage("INDEPENDENT SPRAY STROKES READY")
        else:
            self.status_bar.showMessage("DRAW SPRAY STROKES, THEN GENERATE THE ROUTE")
        self._refresh_workflow_ui()

    def _recorded_spray_group_metrics(self):
        """Return ordered per-group spray distance and time, excluding transfers."""
        records = list(getattr(self.viewer, "_interactive_spray_records", []))
        if not records:
            return []

        mm_factor = self.viewer._get_source_mm_factor()
        if mm_factor is None or not np.isfinite(mm_factor) or mm_factor <= 0.0:
            mm_factor = 1.0
        speed = max(float(self.robot_settings.get("painting_speed_mm_s", 150.0)), 1e-9)

        grouped = {}
        for record in records:
            if not bool(record.get("spray_on", True)):
                continue
            group_id = int(record.get("group_id", int(record.get("stroke_id", 0)) + 1))
            grouped.setdefault(group_id, []).append(record)

        result = []
        for group_id, group_records in grouped.items():
            # One mouse gesture can contain multiple safe contiguous runs.
            # Sum only within each run; never count a fictitious connector
            # between internal collision/excluded-face boundaries.
            runs = {}
            for record in group_records:
                segment_id = int(record.get("stroke_id", group_id - 1))
                runs.setdefault(segment_id, []).append(record)
            distance_units = 0.0
            for run_records in runs.values():
                points = np.asarray(
                    [record["nozzle_point_product"] for record in run_records],
                    dtype=np.float64,
                ).reshape(-1, 3)
                if len(points) > 1:
                    distance_units += float(
                        np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                    )
            distance_mm = distance_units * mm_factor
            result.append({
                "group_id": group_id,
                "distance_mm": distance_mm,
                "duration_s": distance_mm / speed,
            })
        return result

    def _refresh_recorded_spray_path_panel(self):
        """Refresh both route-generation and simulation views of spray group order."""
        if not hasattr(self, "recorded_spray_group_list"):
            return

        metrics = self._recorded_spray_group_metrics()
        total_distance = sum(item["distance_mm"] for item in metrics)
        total_duration = sum(item["duration_s"] for item in metrics)
        total_text = (
            f"TOTAL SPRAY LENGTH: {total_distance:.1f} mm\n"
            f"TOTAL EXECUTION TIME: {self.viewer._format_route_time(total_duration)}"
        )

        self._recorded_spray_list_updating = True
        try:
            list_widget = self.recorded_spray_group_list
            empty_label = self.recorded_spray_path_empty_label
            total_label = self.recorded_spray_path_total_label
            list_widget.clear()
            empty_label.setVisible(not metrics)
            list_widget.setVisible(bool(metrics))
            total_label.setText(total_text)

            for index, metric in enumerate(metrics, start=1):
                item = QListWidgetItem()
                item.setData(Qt.UserRole, int(metric["group_id"]))
                item.setSizeHint(QSize(350, 72))
                list_widget.addItem(item)

                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(8, 5, 8, 5)
                row_layout.setSpacing(8)

                drag_handle = QLabel("⋮⋮")
                drag_handle.setObjectName("Muted")
                row_layout.addWidget(drag_handle)

                order_label = QLabel(f"{index:02d}")
                order_label.setObjectName("MetricValue")
                order_label.setFixedWidth(22)
                row_layout.addWidget(order_label)

                group_label = QLabel(f"SPRAY GROUP {index}")
                group_label.setObjectName("SectionTitle")
                row_layout.addWidget(group_label, 1)

                metrics_label = QLabel(
                    f"LENGTH  {metric['distance_mm']:.1f} mm\n"
                    f"TIME  {self.viewer._format_route_time(metric['duration_s'])}"
                )
                metrics_label.setObjectName("MetricValue")
                metrics_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                row_layout.addWidget(metrics_label)

                delete_button = QPushButton("DELETE")
                delete_button.setFixedWidth(72)
                delete_button.setFixedHeight(34)
                delete_button.setSizePolicy(
                    QSizePolicy.Fixed,
                    QSizePolicy.Fixed,
                )
                delete_button.setStyleSheet(
                    "QPushButton {"
                    " background-color: #5A1E24;"
                    " color: #FFE8EA;"
                    " border: 1px solid #FF5252;"
                    " border-radius: 4px;"
                    " font-weight: bold;"
                    " padding: 3px 6px;"
                    "}"
                    "QPushButton:hover {"
                    " background-color: #7A2931;"
                    "}"
                    "QPushButton:pressed {"
                    " background-color: #40151A;"
                    "}"
                )
                delete_button.setToolTip(
                    f"Delete SPRAY GROUP {index}"
                )

                group_id = int(metric["group_id"])
                delete_button.clicked.connect(
                    lambda checked=False, gid=group_id:
                        self._confirm_delete_recorded_spray_group(gid)
                )
                row_layout.addWidget(
                    delete_button,
                    0,
                    Qt.AlignRight | Qt.AlignVCenter,
                )

                list_widget.setItemWidget(item, row)
        finally:
            self._recorded_spray_list_updating = False

    def _confirm_delete_recorded_spray_group(self, group_id):
        """Ask for confirmation before removing one recorded spray group."""
        if self._recorded_spray_list_updating or self._operation_running:
            return

        try:
            group_id = int(group_id)
        except (TypeError, ValueError):
            return

        answer = QMessageBox.question(
            self,
            "DELETE SPRAY GROUP",
            (
                f"Delete SPRAY GROUP {group_id}?\n\n"
                "Only the selected spray group will be deleted. "
                "The current page, camera, and view mode will remain unchanged.\n\n"
                "Use BACK when you want to navigate to another page."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self._delete_recorded_spray_group(group_id)

    def _delete_recorded_spray_group(self, group_id):
        """Delete the selected spray group without changing the current view."""
        if self._recorded_spray_list_updating or self._operation_running:
            return

        try:
            group_id = int(group_id)
        except (TypeError, ValueError):
            return

        records = list(getattr(self.viewer, "_interactive_spray_records", []))
        if not records:
            return

        # Remove only records belonging to the selected user gesture.
        # Keep every surviving group_id and stroke_id unchanged: these IDs
        # are also used by the route actor and playback segmentation, and
        # compacting group IDs here can make unrelated strokes collide.
        remaining_records = [
            record
            for record in records
            if int(record.get(
                "group_id",
                int(record.get("stroke_id", 0)) + 1,
            )) != group_id
        ]
        if len(remaining_records) == len(records):
            return

        remaining_group_ids = []
        for record in remaining_records:
            remaining_id = int(record.get(
                "group_id",
                int(record.get("stroke_id", 0)) + 1,
            ))
            if remaining_id not in remaining_group_ids:
                remaining_group_ids.append(remaining_id)

        self.viewer._interactive_spray_records = remaining_records

        # Rebuild route metrics/actors without allowing the readiness signal
        # to trigger a wider workflow refresh during this delete-only action.
        if remaining_records:
            was_suppressed = getattr(
                self, "_suppress_route_readiness_callback", False
            )
            self._suppress_route_readiness_callback = True
            try:
                self.viewer._optimize_interactive_recorded_path()
            finally:
                self._suppress_route_readiness_callback = was_suppressed
        else:
            self.viewer._interactive_spray_route_optimization = {}
            self.viewer._interactive_recorded_replay_timing = {}
            self.viewer._interactive_route_optimization_metrics = {}

        self.viewer._refresh_interactive_recorded_path_actor()

        # A generated trajectory based on the deleted group is stale, but
        # the active page, camera, view mode, and recording session are intact.
        self.interactive_spray_path = None
        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = False
        self.stage_completed[STAGE_SPRAY_SIMULATION] = False
        self._refresh_recorded_spray_path_panel()

        self._refresh_workflow_ui()
        self.status_bar.showMessage(
            f"SPRAY GROUP {group_id} DELETED · "
            f"{len(remaining_group_ids)} GROUP(S) REMAIN · "
            "CURRENT PAGE AND VIEW UNCHANGED"
        )

    def _on_recorded_spray_groups_reordered(self):
        """Apply the visible list order to the actual route records and replay."""
        if self._recorded_spray_list_updating or self._operation_running:
            return

        source_list = self.sender()
        list_widget = (
            source_list
            if source_list is self.recorded_spray_group_list
            else self.recorded_spray_group_list
        )
        ordered_group_ids = [
            int(list_widget.item(index).data(Qt.UserRole))
            for index in range(list_widget.count())
        ]
        if not ordered_group_ids:
            return

        records = list(getattr(self.viewer, "_interactive_spray_records", []))
        grouped_records = {}
        for record in records:
            group_id = int(record.get("group_id", int(record.get("stroke_id", 0)) + 1))
            grouped_records.setdefault(group_id, []).append(record)

        if set(ordered_group_ids) != {
            group_id for group_id, group_records in grouped_records.items()
            if any(bool(record.get("spray_on", True)) for record in group_records)
        }:
            self._refresh_recorded_spray_path_panel()
            return

        reordered_records = []
        for new_index, old_group_id in enumerate(ordered_group_ids):
            new_group_id = new_index + 1
            for record in grouped_records.get(old_group_id, []):
                updated = dict(record)
                updated["group_id"] = new_group_id
                updated["stroke_id"] = (
                    new_index if bool(updated.get("spray_on", True))
                    else -new_group_id
                )
                reordered_records.append(updated)

        self.viewer._interactive_spray_records = reordered_records
        self.viewer._optimize_interactive_recorded_path()
        self.viewer._interactive_spray_route_optimization["stroke_order"] = (
            "USER_DEFINED_DRAG_AND_DROP"
        )
        self.viewer._refresh_interactive_recorded_path_actor()

        route = self.interactive_spray_path
        if isinstance(route, dict):
            mm_factor = self.viewer._get_source_mm_factor() or 1.0
            route.update({
                "nozzle_points_product": np.asarray(
                    [item["nozzle_point_product"] for item in reordered_records], dtype=float
                ).reshape(-1, 3),
                "surface_points_product": np.asarray(
                    [item["surface_point_product"] for item in reordered_records], dtype=float
                ).reshape(-1, 3),
                "spray_directions_product": np.asarray(
                    [item["spray_direction_product"] for item in reordered_records], dtype=float
                ).reshape(-1, 3),
                "nozzle_points_world": np.asarray(
                    [item["nozzle_point_world"] for item in reordered_records], dtype=float
                ).reshape(-1, 3),
                "surface_points_world": np.asarray(
                    [item["surface_point_world"] for item in reordered_records], dtype=float
                ).reshape(-1, 3),
                "spray_directions_world": np.asarray(
                    [item["spray_direction_world"] for item in reordered_records], dtype=float
                ).reshape(-1, 3),
                "timestamps_monotonic_s": np.asarray(
                    [item["timestamp"] for item in reordered_records], dtype=float
                ),
                "spray_poses_product": np.asarray([
                    [
                        item.get("t_product_s", item["timestamp"]),
                        *np.asarray(item["nozzle_point_product"], dtype=float).tolist(),
                        *np.asarray(item["quaternion_product_xyzw"], dtype=float).tolist(),
                    ]
                    for item in reordered_records
                ], dtype=float).reshape(-1, 8),
                "face_indices": np.asarray(
                    [item["face_index"] for item in reordered_records], dtype=np.int64
                ),
                "stroke_ids": np.asarray(
                    [item["stroke_id"] for item in reordered_records], dtype=np.int64
                ),
                "group_ids": np.asarray(
                    [item["group_id"] for item in reordered_records], dtype=np.int64
                ),
                "route_timing": {
                    key: value
                    for key, value in (self.viewer._interactive_recorded_replay_timing or {}).items()
                    if key != "timeline"
                },
                "route_optimization": dict(
                    getattr(self.viewer, "_interactive_spray_route_optimization", {})
                ),
                "route_optimization_metrics": dict(
                    getattr(self.viewer, "_interactive_route_optimization_metrics", {})
                ),
            })
            timing = self.viewer._interactive_recorded_replay_timing or {}
            route["spray_length_mm"] = float(timing.get("spray_distance_mm", 0.0))
            route["total_route_distance_mm"] = float(timing.get("total_route_distance_mm", 0.0))
            route["estimated_duration_s"] = float(timing.get("estimated_duration_s", 0.0))
            route["route_length_product_units"] = (
                route["spray_length_mm"] / float(mm_factor)
            )

        self._show_interactive_route_summary(route)
        self._refresh_recorded_spray_path_panel()
        self.status_bar.showMessage(
            "SPRAY GROUP ORDER UPDATED · PATH SHAPES AND DIRECTIONS PRESERVED"
        )

    def _on_route_next_clicked(self):
        """Open spray simulation and show a loading state during initialization."""
        if (
            self._simulation_loading
            or self._operation_running
            or self.interactive_spray_path is None
            or not self.stage_completed.get(STAGE_SPRAY_TRAJECTORY, False)
        ):
            return

        self._simulation_loading = True
        self.route_next_button.setEnabled(False)
        self.route_next_button.setText("LOADING SPRAY SIMULATION...")
        self.simulation_loading_progress.show()
        self.status_bar.showMessage("LOADING SPRAY SIMULATION...")
        QApplication.processEvents()

        try:
            self._select_workflow_stage(STAGE_SPRAY_SIMULATION)
        except Exception as exc:
            traceback.print_exc()
            self.status_bar.showMessage(
                f"SPRAY SIMULATION COULD NOT BE OPENED: {exc}"
            )
            QMessageBox.critical(
                self, "Spray Simulation", str(exc)
            )
        finally:
            self._simulation_loading = False
            self.simulation_loading_progress.hide()
            self._refresh_workflow_ui()

    def _finish_interactive_spray_path(self):
        """Stop interactive input, optimize the recorded stroke route and report it."""
        if self._operation_running or not self.viewer._interactive_spray_active:
            return

        # Give Qt a chance to paint the loading state before route optimization.
        self.route_generation_progress.show()
        self.status_bar.showMessage(
            "GENERATING SPRAY ROUTE... PLEASE WAIT"
        )
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            result = self.viewer.finish_interactive_spray_path()
        except Exception as exc:
            self.route_generation_progress.hide()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Spray Route Generation Error", str(exc)
            )
            return

        self.route_generation_progress.hide()
        QApplication.restoreOverrideCursor()
        self.interactive_spray_path = result
        if result is None:
            self.status_bar.showMessage("INTERACTIVE SPRAY PATH CANCELLED")
            self._refresh_workflow_ui()
            return

        route_timing = getattr(
            self.viewer, "_interactive_recorded_replay_timing", None
        ) or {}
        route_ready = route_timing.get("safety_status") == "SAFE"
        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = bool(route_ready)
        self.stage_completed[STAGE_SPRAY_SIMULATION] = False
        self.viewer.set_recorded_path_page("route")
        self._refresh_workflow_ui()

        nozzle_points = np.asarray(result["nozzle_points_product"], dtype=float)
        spray_distance = float(result.get("spray_length_mm", 0.0))
        self.analysis_summary.setText(
            "INDEPENDENT INTERACTIVE SPRAY STROKES\n\n"
            f"RECORDED SPRAY POSES: {len(nozzle_points):,}\n"
            f"SPRAY GROUPS: {len(np.unique(result.get('group_ids', result.get('stroke_ids', [])))):,}\n"
            f"SPRAY DISTANCE: {spray_distance:.1f} mm\n"
            f"ESTIMATED SPRAY TIME: {self.viewer._format_route_time(float(result.get('estimated_duration_s', 0.0)))}\n"
            f"ROBOT SPEED: {float(self.robot_settings['painting_speed_mm_s']):.1f} mm/s\n\n"
            "Each group is an independent spray stroke.\n"
            "No line, sample, or interpolation is inserted between groups.\n"
            "MoveIt2 plans robot transfer motion between groups."
        )
        self.status_bar.showMessage(
            f"SPRAY ROUTE GENERATED | {len(np.unique(result.get('group_ids', result.get('stroke_ids', []))))} INDEPENDENT GROUPS | SPRAY {spray_distance:.1f} mm"
        )
        self.route_next_button.setVisible(True)
        self.route_next_button.setEnabled(True)
        self.route_edit_button.setVisible(True)
        self.route_edit_button.setEnabled(True)
        self._show_interactive_route_summary(result)
        self._refresh_recorded_spray_path_panel()
        self._refresh_workflow_ui()

    def _show_interactive_route_summary(self, result):
        """Summarize independent spray strokes without transfer geometry."""
        if not isinstance(result, dict):
            return
        spray_distance = float(result.get("spray_length_mm", 0.0))
        groups = len(np.unique(result.get("group_ids", result.get("stroke_ids", []))))
        self.summary_chart.hide()
        self.summary_section.setText("ROUTE GENERATION")
        self.analysis_summary.setStyleSheet("color: #00E676;")
        self.analysis_summary.setText(
            "INDEPENDENT SPRAY STROKES READY FOR SIMULATION\n\n"
            f"RECORDED SPRAY POSES: {len(result.get('nozzle_points_product', [])):,}\n"
            f"SPRAY GROUPS: {groups:,}\n"
            f"SPRAY DISTANCE: {spray_distance:.1f} mm\n"
            f"ROBOT SPEED: {float(self.robot_settings['painting_speed_mm_s']):.1f} mm/s\n\n"
            "Each group is a separate spray stroke.\n"
            "No inter-stroke connector or interpolated spray path exists.\n"
            "MoveIt2 plans transfer motion between groups."
        )

    def _show_spray_simulation_summary(self):
        self.summary_chart.hide()
        self.summary_section.setText("SPRAY SIMULATION")
        self.analysis_summary.setStyleSheet("color: #D8E5ED;")
        self.analysis_summary.setText(
            "SPRAY ROUTE SIMULATION\n\n"
            "The product starts in its unpainted state.\n"
            "Only the surface reached during SPRAY ON is coated.\n"
            "The spray sequence stops at every group boundary. MoveIt2 plans the transfer to the next group.\n\n"
            "Simulation playback uses the recorded path sequence.\n"
            "No robot execution speed or controller frequency is assumed."
        )

    def _continue_interactive_spray_path(self):
        """Resume an existing route, or start a fresh one when none remains."""
        if (
            self._operation_running
            or self.active_stage != STAGE_SPRAY_TRAJECTORY
            or self.current_mesh is None
        ):
            return

        existing_records = getattr(
            self.viewer, "_interactive_spray_records", []
        ) or []
        route_exists = bool(existing_records)
        if self.viewer._interactive_spray_active:
            return

        # Resume the surviving strokes when any recorded samples remain.
        # Only an empty record set (after deleting every group) starts fresh.
        self.stage_completed[STAGE_SPRAY_SIMULATION] = False
        if not route_exists:
            self.interactive_spray_path = None
            self.stage_completed[STAGE_SPRAY_TRAJECTORY] = False

        self.viewer.set_recorded_path_page("route")
        self._start_interactive_spray_path(resume_existing=route_exists)
        self._refresh_workflow_ui()

    def _display_current_trajectory(self):
        if self.current_stage3_result is None:
            return

        trajectory = self.current_stage3_result["trajectory"]
        face_surface_types = self._build_face_surface_types()
        face_status_labels = (
            self.current_stage2_result.get("face_status_labels")
            if isinstance(self.current_stage2_result, dict)
            else None
        )
        patch_labels = (
            self.current_stage1_result.get("patch_labels")
            if isinstance(self.current_stage1_result, dict)
            else None
        )

        self.viewer.show_spray_trajectory(
            self.current_mesh,
            trajectory,
            face_surface_types=face_surface_types,
            face_status_labels=face_status_labels,
            patch_labels=patch_labels,
        )

        # Display the generated trajectory as a static preview.
        self._update_viewer_display_unit()

    def _on_spray_trajectory_complete(self, result):
        try:
            self.current_stage3_result = result

            self.current_stage3_parameters = (
                self.pending_stage3_parameters.copy()
                if self.pending_stage3_parameters is not None
                else self._get_current_spray_parameters()
            )

            self.pending_stage3_parameters = None

            self.stage_completed[
                STAGE_SPRAY_TRAJECTORY
            ] = True

            self.active_stage = STAGE_SPRAY_TRAJECTORY

            self._display_current_trajectory()

            screenshot_path = self.result_manager.result_path(
                self.current_stl_path,
                stage_number=3,
                filename="spray_trajectory.png",
            )

            try:
                self.viewer.save_screenshot(screenshot_path)
            except Exception as exc:
                print(
                    "Stage 3 screenshot export failed:",
                    exc,
                )

            self._show_spray_trajectory_summary()
            self._finish_operation_ui(True)

            self.status_bar.showMessage(
                "SPRAY TRAJECTORY GENERATED  |  "
                f"SAMPLES: {result['sample_count']:,}  |  "
                f"RESULT: {result['json_path']}"
            )

        except Exception as exc:
            traceback.print_exc()
            self._finish_operation_ui(False)

            QMessageBox.critical(
                self,
                "Trajectory Display Error",
                str(exc),
            )

    def _on_spray_trajectory_failed(self, error_message):
        self.pending_stage3_parameters = None
        self._finish_operation_ui(False)

        self.workflow_status_label.setText(
            "STATUS: SPRAY TRAJECTORY GENERATION FAILED"
        )
        self.workflow_status_label.setStyleSheet(
            "color: #FF5252; font-weight: bold;"
        )

        QMessageBox.critical(
            self,
            "Spray Trajectory Error",
            error_message,
        )

    # ========================================================
    # STAGE 3 SUMMARIES
    # ========================================================

    @staticmethod
    def _format_status(value):
        if value is None:
            return "NOT AVAILABLE"

        return str(value).replace("_", " ").upper()

    def _show_stage3_ready_summary(self):
        self.summary_chart.hide()
        self.summary_section.setText("SPRAY TRAJECTORY")
        self.analysis_summary.setStyleSheet("color: #D8E5ED;")
        self.analysis_summary.setText("READY FOR SPRAY TRAJECTORY")

    def _show_stage3_outdated_summary(self):
        self.summary_chart.hide()
        self.summary_section.setText("SPRAY TRAJECTORY")
        self.analysis_summary.setStyleSheet("color: #FFD740;")
        self.analysis_summary.setText("TRAJECTORY PARAMETERS CHANGED")

    def _show_spray_trajectory_summary(self):
        # Keep the shared Stage 1/2 summary area, but do not expose
        # post-generation diagnostics or downstream workflow information.
        self.summary_chart.hide()
        self.summary_section.setText("SPRAY TRAJECTORY")
        self.analysis_summary.setStyleSheet("color: #00E676;")
        self.analysis_summary.setText("SPRAY TRAJECTORY GENERATED")

    # ========================================================
    # STAGE 1 SUMMARY
    # ========================================================

    def _show_analysis_summary(self, report):
        segmentation = report.get("surface_segmentation", {})
        patches = segmentation.get("patches", [])

        patch_count = int(
            segmentation.get("patch_count", len(patches))
        )

        counts = {
            "planar": 0,
            "curved": 0,
            "uncertain": 0,
            "invalid": 0,
        }
        cylinder_counts = {
            "internal": 0,
            "external": 0,
            "orientation_uncertain": 0,
        }

        for patch in patches:
            classification = str(
                patch.get("classification", "invalid")
            ).lower()

            if classification not in counts:
                classification = "invalid"

            counts[classification] += 1

            if classification == "curved":
                cylinder = patch.get("cylindrical_fit", {})
                if cylinder.get("status") == "candidate":
                    surface_type = cylinder.get("surface_type")
                    if surface_type == "internal_cylindrical_surface":
                        cylinder_counts["internal"] += 1
                    elif surface_type == "external_cylindrical_surface":
                        cylinder_counts["external"] += 1
                    elif surface_type == "cylindrical_surface_orientation_uncertain":
                        cylinder_counts["orientation_uncertain"] += 1

        # Keep report totals authoritative while safely handling older reports.
        classified_patch_records = sum(counts.values())
        if classified_patch_records < patch_count:
            counts["invalid"] += patch_count - classified_patch_records

        classified = counts["planar"] + counts["curved"]
        unresolved = counts["uncertain"] + counts["invalid"]
        curved_other = max(
            0,
            counts["curved"] - sum(cylinder_counts.values()),
        )

        def percentage(value):
            return (
                f"{100.0 * value / patch_count:.1f}%"
                if patch_count > 0
                else "0.0%"
            )

        mesh_info = report.get("mesh", {})
        valid_triangles = int(mesh_info.get("valid_faces", 0))
        invalid_triangles = int(mesh_info.get("invalid_faces", 0))
        degenerate_triangles = int(mesh_info.get("degenerate_faces", 0))
        zero_normal_triangles = int(mesh_info.get("zero_normal_faces", 0))

        self.summary_chart.set_values(counts)
        self.summary_chart.show()

        self.analysis_summary.setStyleSheet(
            "QLabel { color: #D8E5ED; font-size: 9pt; }"
        )
        self.analysis_summary.setText(
            f"<b style='color:#00E5FF'>{patch_count:,} TOTAL PATCHES</b><br>"
            f"<span style='color:#00E676'>● PLANAR: {counts['planar']:,} ({percentage(counts['planar'])})</span><br>"
            f"<span style='color:#00B8D9'>● CURVED: {counts['curved']:,} ({percentage(counts['curved'])})</span><br>"
            f"<span style='color:#FFD740'>● UNCERTAIN: {counts['uncertain']:,} ({percentage(counts['uncertain'])})</span><br>"
            f"<span style='color:#FF5252'>● INVALID: {counts['invalid']:,} ({percentage(counts['invalid'])})</span>"
            f"<br><br><b>CLASSIFIED:</b> <span style='color:#00E676'>{classified:,}</span> "
            f"<span style='color:#8195A3'>|</span> "
            f"<b>UNRESOLVED:</b> <span style='color:#FFD740'>{unresolved:,}</span>"
        )

    # ========================================================
    # STAGE 2 SUMMARY
    # ========================================================

    def _show_surface_identification_summary(self):
        self.summary_chart.hide()
        if self.current_stage2_result is None:
            if self.current_stage1_result is not None:
                self.summary_section.setText("STAGE 2 - READY")
                self.analysis_summary.setStyleSheet("")

                patch_count = len(self.patch_inspection)
                self.analysis_summary.setText(
                    f"{patch_count:,} PATCHES TO REVIEW"
                )
            return

        total = len(self.patch_inspection)

        inspected = sum(
            1
            for item in self.patch_inspection.values()
            if item["inspected"]
        )

        # ----------------------------------------------------
        # Only count actual completed manual decisions.
        # ----------------------------------------------------

        paint_count = sum(
            1
            for item in self.patch_inspection.values()
            if item["decision"] == DECISION_PAINT
        )

        no_paint_count = sum(
            1
            for item in self.patch_inspection.values()
            if item["decision"] == DECISION_NO_PAINT
        )

        # Every patch belongs to exactly one visible bucket.
        # Any patch without a completed Paint / No Paint decision
        # remains in REVIEW REQUIRED, including unchecked rows.
        requires_count = total - paint_count - no_paint_count

        self.analysis_summary.setStyleSheet(
            "color: #00E676;"
            if inspected == total and total > 0
            else "color: #FFD740;"
        )

        self.analysis_summary.setText(
            f"PATCH COVERAGE: {total:,} / {total:,} (100%)\n"
            f"PAINT: {paint_count:,}\n"
            f"UNPAINT: {no_paint_count:,}\n"
            f"REVIEW REQUIRED: {requires_count:,}\n"
            f"DECIDED: {inspected:,} / {total:,}"
        )


    # ========================================================
    # WORKER CLEANUP
    # ========================================================

    def _on_worker_thread_finished(self):
        self._worker = None

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def closeEvent(self, event):
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.wait()

        self.viewer.close()
        event.accept()