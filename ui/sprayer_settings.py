from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QScrollArea,
    QWidget,
    QInputDialog,
)


DEFAULT_SPRAYER_PROFILE = {
    "name": "default sprayer",
    "manufacturer": "Unspecified",
    "model": "Generic HVLP starting profile",
    "technology": "HVLP air spray",
    "pattern_type": "flat_fan",
    "nozzle_orifice_mm": 1.4,
    "nozzle_length_mm": 25.0,
    "gun_body_length_mm": 0.0,
    "gun_body_diameter_mm": 0.0,
    "tcp_to_nozzle_tip_mm": 0.0,
    "standoff_mm": 180.0,
    "spray_cone_angle_deg": 74.0,
    "fan_reference_distance_mm": 200.0,
    "overlap_percent": 50.0,
    "sampling_spacing_mm": 20.0,
    "travel_speed_mm_s": 0.0,
    "fan_air_pressure_bar": 1.7,
    "atomizing_air_pressure_bar": 1.7,
    "fluid_pressure_bar": 0.0,
    "fluid_flow_ml_min": 0.0,
    "nozzle_length_is_estimate": True,
    "profile_notes": (
        "Generic starting values, not a certified sprayer specification. "
        "Nozzle length, fluid pressure, fluid flow and travel speed must be "
        "confirmed from the actual sprayer and coating process."
    ),
}


def load_sprayer_profiles(config_path: str | Path) -> list[dict]:
    """Load local sprayer profiles; create the default config on first run."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        profiles = [deepcopy(DEFAULT_SPRAYER_PROFILE)]
        save_sprayer_profiles(path, profiles)
        return profiles

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
        if not isinstance(profiles, list):
            raise ValueError("The profiles entry must be a list.")
        valid = []
        for item in profiles:
            if not isinstance(item, dict):
                continue
            if not str(item.get("name", "")).strip():
                continue
            merged = deepcopy(DEFAULT_SPRAYER_PROFILE)
            merged.update(deepcopy(item))
            valid.append(merged)
        if not any(
            str(item.get("name", "")).strip().lower() == "default sprayer"
            for item in valid
        ):
            valid.insert(0, deepcopy(DEFAULT_SPRAYER_PROFILE))
            save_sprayer_profiles(path, valid)
        return valid or [deepcopy(DEFAULT_SPRAYER_PROFILE)]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not read sprayer profile configuration: {path}\n{exc}"
        ) from exc


def load_active_sprayer_profile_name(config_path: str | Path) -> str:
    path = Path(config_path)
    if not path.exists():
        return "default sprayer"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return str(payload.get("active_profile", "default sprayer"))
    except (OSError, json.JSONDecodeError):
        pass
    return "default sprayer"


def save_sprayer_profiles(
    config_path: str | Path,
    profiles: list[dict],
    active_profile_name: str | None = None,
) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if active_profile_name is None:
        active_profile_name = load_active_sprayer_profile_name(path)
    payload = {
        "schema_version": 1,
        "active_profile": str(active_profile_name),
        "profiles": profiles,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class SprayPatternPreview(QWidget):
    """Live schematic preview of nozzle length and spray envelope."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile = deepcopy(DEFAULT_SPRAYER_PROFILE)
        self.setMinimumSize(340, 400)

    def set_profile(self, profile: dict):
        self.profile = deepcopy(profile)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0A151E"))

        width = self.width()
        height = self.height()
        cx = width * 0.50
        tip_y = 76.0
        surface_y = max(tip_y + 70.0, height - 132.0)

        profile = self.profile
        standoff = max(1.0, float(profile.get("standoff_mm", 180.0)))
        angle = max(
            1.0,
            min(150.0, float(profile.get("spray_cone_angle_deg", 74.0))),
        )
        overlap = max(
            0.0, min(90.0, float(profile.get("overlap_percent", 50.0)))
        )
        sample_spacing = max(
            0.1, float(profile.get("sampling_spacing_mm", 20.0))
        )
        pattern_type = str(profile.get("pattern_type", "flat_fan"))

        spray_width_mm = (
            2.0 * standoff * math.tan(math.radians(angle * 0.5))
        )
        track_pitch_mm = spray_width_mm * (1.0 - overlap / 100.0)

        # Schematic scale: retain enough room to show the center pass and
        # its two neighboring passes without clipping the spray envelopes.
        max_half_width_px = max(20.0, min(width * 0.18, 78.0))
        visual_half_width = max_half_width_px
        pitch_px = visual_half_width * 2.0 * (1.0 - overlap / 100.0)
        visual_depth = surface_y - tip_y

        def envelope_polygon(center_x):
            if pattern_type == "round_cone":
                return QPolygon([
                    self._point(center_x, tip_y),
                    self._point(center_x - visual_half_width, surface_y),
                    self._point(center_x + visual_half_width, surface_y),
                ])
            return QPolygon([
                self._point(center_x - 5, tip_y),
                self._point(center_x + 5, tip_y),
                self._point(center_x + visual_half_width, surface_y),
                self._point(center_x - visual_half_width, surface_y),
            ])

        # Neighboring passes are outlines; the center pass is emphasized.
        for offset in (-pitch_px, pitch_px):
            painter.setPen(QPen(QColor("#4C9EAE"), 1.4, Qt.DashLine))
            painter.setBrush(QColor(0, 150, 180, 18))
            painter.drawPolygon(envelope_polygon(cx + offset))

        painter.setPen(QPen(QColor("#00E5FF"), 2))
        painter.setBrush(QColor(0, 190, 220, 62))
        painter.drawPolygon(envelope_polygon(cx))

        # Nozzle tip and spray centerline are references only; mechanical
        # nozzle dimensions do not affect the trajectory envelope.
        painter.setPen(QPen(QColor("#DCEAF1"), 2))
        painter.drawLine(int(cx), int(tip_y - 18), int(cx), int(tip_y - 2))
        painter.setBrush(QColor("#00E5FF"))
        painter.drawEllipse(int(cx - 5), int(tip_y - 5), 10, 10)
        painter.setPen(QPen(QColor("#A9C2D0"), 1, Qt.DashLine))
        painter.drawLine(int(cx), int(tip_y + 4), int(cx), int(surface_y))

        # Surface reference and sample-spacing markers.
        painter.setPen(QPen(QColor("#DCEAF1"), 2))
        painter.drawLine(12, int(surface_y), width - 12, int(surface_y))
        painter.setPen(QPen(QColor("#00F5A0"), 1.5))
        painter.setBrush(QColor("#00F5A0"))
        sample_count = max(
            2, min(14, int(round(spray_width_mm / sample_spacing)) + 1)
        )
        for index in range(sample_count):
            ratio = index / max(1, sample_count - 1)
            x = cx - visual_half_width + ratio * 2.0 * visual_half_width
            painter.drawEllipse(int(x - 2), int(surface_y - 3), 4, 6)

        # Dimension arrows for spray width and adjacent-track spacing.
        painter.setPen(QPen(QColor("#00E5FF"), 1.5))
        painter.drawLine(
            int(cx - visual_half_width), int(surface_y + 10),
            int(cx + visual_half_width), int(surface_y + 10),
        )
        painter.drawLine(
            int(cx - visual_half_width), int(surface_y + 5),
            int(cx - visual_half_width), int(surface_y + 15),
        )
        painter.drawLine(
            int(cx + visual_half_width), int(surface_y + 5),
            int(cx + visual_half_width), int(surface_y + 15),
        )

        pitch_start = cx + visual_half_width
        pitch_end = pitch_start + pitch_px
        painter.setPen(QPen(QColor("#FFB000"), 1.5))
        painter.drawLine(
            int(pitch_start), int(surface_y + 25),
            int(pitch_end), int(surface_y + 25),
        )
        painter.drawLine(
            int(pitch_start), int(surface_y + 20),
            int(pitch_start), int(surface_y + 30),
        )
        painter.drawLine(
            int(pitch_end), int(surface_y + 20),
            int(pitch_end), int(surface_y + 30),
        )

        painter.setPen(QColor("#EAF7FC"))
        painter.drawText(14, 24, "TRAJECTORY SPRAY ENVELOPE")
        painter.setPen(QColor("#00E5FF"))
        painter.drawText(14, 45, f"WORKING DISTANCE: {standoff:.1f} mm")
        painter.setPen(QColor("#EAF7FC"))
        painter.drawText(
            14, int(surface_y + 48),
            f"SPRAY WIDTH: {spray_width_mm:.1f} mm  |  ANGLE: {angle:.1f} deg"
        )
        painter.setPen(QColor("#FFB000"))
        painter.drawText(
            14, int(surface_y + 68),
            f"TRACK PITCH: {track_pitch_mm:.1f} mm  |  OVERLAP: {overlap:.1f}%"
        )
        painter.setPen(QColor("#00F5A0"))
        painter.drawText(
            14, int(surface_y + 88),
            f"SURFACE SAMPLE SPACING: {sample_spacing:.1f} mm"
        )
        painter.end()

    @staticmethod
    def _point(x: float, y: float):
        from PySide6.QtCore import QPoint
        return QPoint(int(round(x)), int(round(y)))


class SprayerSettingsDialog(QDialog):
    """Profile editor with immediate spray-envelope preview and JSON import/export."""

    def _cancel_dialog(self):
        self._cancel_requested = True
        super().reject()

    def reject(self):
        # Ignore Escape and implicit reject requests; the visible Cancel
        # button is the only control that is allowed to close this dialog.
        if self._cancel_requested:
            super().reject()

    def closeEvent(self, event):
        if self._cancel_requested:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def __init__(
        self,
        config_path: str | Path,
        profiles: list[dict],
        current_profile: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._cancel_requested = False
        self.config_path = Path(config_path)
        self.profiles = deepcopy(profiles)
        self.current_profile = deepcopy(current_profile)
        self.selected_profile = None
        self._loading = False
        self._build_ui()
        self._populate_profiles()
        active_index = self.profile_selector.findText(
            str(self.current_profile.get("name", "default sprayer"))
        )
        if active_index >= 0:
            self.profile_selector.setCurrentIndex(active_index)
        self._load_profile(self.current_profile)

    def _build_ui(self):
        self.setWindowTitle("SPRAYER TECHNICAL SPECIFICATIONS")
        self.setMinimumSize(980, 560)
        self.resize(1100, 650)
        root = QVBoxLayout(self)

        header = QLabel("SPRAYER PROFILE")
        header.setStyleSheet("color: #00E5FF; font-size: 15pt; font-weight: bold;")
        root.addWidget(header)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("PROFILE"))
        self.profile_selector = QComboBox()
        self.profile_selector.currentIndexChanged.connect(self._on_profile_changed)
        profile_row.addWidget(self.profile_selector, 1)
        root.addLayout(profile_row)

        body = QHBoxLayout()
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(7)

        self.name_edit = QLineEdit()
        self.manufacturer_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.technology_combo = QComboBox()
        self.technology_combo.addItems([
            "HVLP air spray",
            "Conventional air spray",
            "LVMP / compliant air spray",
            "Airless",
            "Air-assisted airless",
            "Electrostatic",
            "Other / unspecified",
        ])
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItem("Flat fan", "flat_fan")
        self.pattern_combo.addItem("Round cone", "round_cone")

        # Put the parameters most often adjusted for trajectory generation
        # at the top of the left-hand form for quick access.
        self.fields = {}
        field_specs = [
            ("standoff_mm", "Working distance (mm)", 20.0, 1000.0, 1),
            ("spray_cone_angle_deg", "Full spray angle (deg)", 1.0, 150.0, 1),
            ("overlap_percent", "Track overlap (%)", 0.0, 90.0, 1),
            ("sampling_spacing_mm", "Surface sample spacing (mm)", 1.0, 100.0, 1),
            ("nozzle_orifice_mm", "Nozzle orifice (mm)", 0.1, 10.0, 2),
            ("nozzle_length_mm", "Nozzle length (mm)", 0.0, 300.0, 1),
            ("gun_body_length_mm", "Gun body length (mm)", 0.0, 600.0, 1),
            ("gun_body_diameter_mm", "Gun body diameter (mm)", 0.0, 200.0, 1),
            ("tcp_to_nozzle_tip_mm", "TCP to nozzle tip (mm)", 0.0, 500.0, 1),
            ("fan_reference_distance_mm", "Reference distance (mm)", 10.0, 2000.0, 1),
            ("travel_speed_mm_s", "Gun travel speed (mm/s)", 0.0, 2000.0, 1),
            ("fan_air_pressure_bar", "Fan air pressure (bar)", 0.0, 10.0, 2),
            ("atomizing_air_pressure_bar", "Atomizing air pressure (bar)", 0.0, 10.0, 2),
            ("fluid_pressure_bar", "Fluid pressure (bar)", 0.0, 500.0, 2),
            ("fluid_flow_ml_min", "Fluid flow (ml/min)", 0.0, 10000.0, 1),
        ]

        trajectory_keys = {
            "standoff_mm",
            "spray_cone_angle_deg",
            "overlap_percent",
            "sampling_spacing_mm",
        }

        def add_profile_field(key, label, minimum, maximum, decimals):
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setKeyboardTracking(False)
            spin.setSingleStep(0.1 if decimals > 1 else 1.0)
            if key == "nozzle_length_mm":
                spin.setToolTip(
                    "This is a mechanical dimension. The default is illustrative; "
                    "replace it with the actual sprayer drawing."
                )
            elif key in (
                "gun_body_length_mm",
                "gun_body_diameter_mm",
                "tcp_to_nozzle_tip_mm",
                "travel_speed_mm_s",
                "fluid_pressure_bar",
                "fluid_flow_ml_min",
            ):
                spin.setToolTip(
                    "A value of 0 means not specified. Enter the value from "
                    "the sprayer data sheet or validated paint process."
                )
            elif key == "spray_cone_angle_deg":
                spin.setToolTip(
                    "Full included spray-pattern angle. This determines the "
                    "theoretical spray width at the working distance; it is "
                    "not the nozzle's surface-incidence angle."
                )
            elif key == "overlap_percent":
                spin.setToolTip(
                    "Target overlap used to calculate nominal track spacing. "
                    "Automatic raster-lane generation is not implemented yet."
                )
            spin.valueChanged.connect(self._refresh_preview)
            self.fields[key] = spin
            form.addRow(label, spin)

        trajectory_header = QLabel("TRAJECTORY GENERATION")
        trajectory_header.setStyleSheet(
            "color: #00E5FF; font-weight: bold; padding-top: 5px;"
        )
        form.addRow(trajectory_header)

        for key, label, minimum, maximum, decimals in field_specs:
            if key in trajectory_keys:
                add_profile_field(key, label, minimum, maximum, decimals)

        sprayer_header = QLabel("SPRAYER PROFILE / TECHNICAL SPECIFICATIONS")
        sprayer_header.setStyleSheet(
            "color: #00E5FF; font-weight: bold; padding-top: 8px;"
        )
        form.addRow(sprayer_header)
        form.addRow("Profile name", self.name_edit)
        form.addRow("Manufacturer", self.manufacturer_edit)
        form.addRow("Model / part number", self.model_edit)
        form.addRow("Atomization technology", self.technology_combo)
        form.addRow("Spray pattern", self.pattern_combo)

        for key, label, minimum, maximum, decimals in field_specs:
            if key not in trajectory_keys:
                add_profile_field(key, label, minimum, maximum, decimals)

        self.profile_notes = QLabel()
        self.profile_notes.setWordWrap(True)
        self.profile_notes.setStyleSheet("color: #9DB3C0; padding: 5px;")
        form.addRow("Notes", self.profile_notes)

        form_host.setMinimumWidth(470)

        # Keep the preview and action buttons visible while the technical
        # specification form scrolls independently in the shorter dialog.
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        form_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        form_scroll.setWidget(form_host)
        body.addWidget(form_scroll, 3)

        self.preview = SprayPatternPreview()
        body.addWidget(self.preview, 2)
        root.addLayout(body, 1)

        actions = QHBoxLayout()
        self.save_button = QPushButton("SAVE PROFILE")
        self.save_as_button = QPushButton("SAVE AS NEW")
        self.import_button = QPushButton("IMPORT")
        self.export_button = QPushButton("EXPORT")
        cancel_button = QPushButton("CANCEL")
        for button in (self.save_button, self.save_as_button, self.import_button,
                       self.export_button, cancel_button):
            button.setMinimumHeight(34)
            actions.addWidget(button)
        self.save_button.clicked.connect(self._save_and_accept)
        self.save_as_button.clicked.connect(self._save_as_new)
        self.import_button.clicked.connect(self._import_profile)
        self.export_button.clicked.connect(self._export_profile)
        cancel_button.clicked.connect(self._cancel_dialog)
        # Enter must never trigger QDialog's implicit accept behavior.
        for button in (
            self.save_button, self.save_as_button, self.import_button,
            self.export_button, cancel_button,
        ):
            button.setAutoDefault(False)
            button.setDefault(False)
        root.addLayout(actions)

        self.setStyleSheet("""
            QDialog { background-color: #101D27; color: #EAF2F7; }
            QLabel { color: #DCEAF1; }
            QLineEdit, QComboBox, QDoubleSpinBox {
                background-color: #172B38;
                color: #F2F8FB;
                border: 1px solid #365365;
                border-radius: 3px;
                padding: 5px;
                min-height: 23px;
            }
            QPushButton {
                background-color: #17465A;
                color: #EAFBFF;
                border: 1px solid #00B8D9;
                border-radius: 4px;
                padding: 6px 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #21627A; }
        """)

        for widget in (self.name_edit, self.manufacturer_edit, self.model_edit):
            widget.textChanged.connect(self._refresh_preview)
        self.technology_combo.currentTextChanged.connect(self._refresh_preview)
        self.pattern_combo.currentIndexChanged.connect(self._refresh_preview)

    def _populate_profiles(self):
        self._loading = True
        self.profile_selector.clear()
        for profile in self.profiles:
            self.profile_selector.addItem(
                str(profile.get("name", "Unnamed profile")),
                str(profile.get("name", "Unnamed profile")),
            )
        self._loading = False

    def _on_profile_changed(self, index):
        if self._loading or index < 0:
            return
        name = self.profile_selector.itemData(index)
        profile = next(
            (item for item in self.profiles if item.get("name") == name),
            None,
        )
        if profile is not None:
            self._load_profile(profile)

    def _load_profile(self, profile):
        self._loading = True
        data = deepcopy(DEFAULT_SPRAYER_PROFILE)
        data.update(deepcopy(profile))
        self.current_profile = deepcopy(data)
        self.name_edit.setText(str(data.get("name", "default sprayer")))
        self.manufacturer_edit.setText(str(data.get("manufacturer", "")))
        self.model_edit.setText(str(data.get("model", "")))
        self.technology_combo.setCurrentText(str(data.get("technology", "Other / unspecified")))
        pattern_index = self.pattern_combo.findData(data.get("pattern_type", "flat_fan"))
        self.pattern_combo.setCurrentIndex(max(0, pattern_index))
        for key, spin in self.fields.items():
            spin.setValue(float(data.get(key, 0.0) or 0.0))
        self.profile_notes.setText(str(data.get("profile_notes", "")))
        self._loading = False
        self._refresh_preview()

    def _collect_profile(self):
        profile = {
            "name": self.name_edit.text().strip(),
            "manufacturer": self.manufacturer_edit.text().strip(),
            "model": self.model_edit.text().strip(),
            "technology": self.technology_combo.currentText(),
            "pattern_type": self.pattern_combo.currentData(),
            "profile_notes": self.profile_notes.text(),
        }
        profile.update({
            key: float(spin.value())
            for key, spin in self.fields.items()
        })
        previous_nozzle_length = float(
            self.current_profile.get("nozzle_length_mm", 25.0)
        )
        profile["nozzle_length_is_estimate"] = bool(
            self.current_profile.get("nozzle_length_is_estimate", False)
            and math.isclose(
                profile["nozzle_length_mm"],
                previous_nozzle_length,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        return profile

    def _refresh_preview(self, *_):
        if self._loading:
            return
        profile = self._collect_profile()
        self.preview.set_profile(profile)

    def _save_current_to_config(self, profile):
        if not profile["name"]:
            raise ValueError("Enter a profile name.")
        replacement_index = next(
            (
                index for index, item in enumerate(self.profiles)
                if item.get("name", "").strip().lower()
                == profile["name"].strip().lower()
            ),
            None,
        )
        if replacement_index is None:
            self.profiles.append(profile)
        else:
            self.profiles[replacement_index] = profile
        save_sprayer_profiles(
            self.config_path,
            self.profiles,
            active_profile_name=profile["name"],
        )
        self.current_profile = deepcopy(profile)

    def _save_and_accept(self):
        try:
            profile = self._collect_profile()
            self._save_current_to_config(profile)
            self.selected_profile = deepcopy(profile)
            self._populate_profiles()
            self.profile_selector.setCurrentText(profile["name"])
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Sprayer Profile", str(exc))

    def _save_as_new(self):
        suggested = self.name_edit.text().strip() + " copy"
        name, accepted = QInputDialog.getText(
            self, "Save Sprayer As New", "New profile name:", text=suggested
        )
        if not accepted:
            return
        profile = self._collect_profile()
        profile["name"] = name.strip()
        try:
            if not profile["name"]:
                raise ValueError("Enter a profile name.")
            if any(
                item.get("name", "").strip().lower() == profile["name"].lower()
                for item in self.profiles
            ):
                raise ValueError("A profile with this name already exists.")
            self.profiles.append(profile)
            save_sprayer_profiles(
                self.config_path,
                self.profiles,
                active_profile_name=profile["name"],
            )
            self.current_profile = deepcopy(profile)
            self._populate_profiles()
            self.profile_selector.setCurrentText(profile["name"])
            self.name_edit.setText(profile["name"])
            self.current_profile = deepcopy(profile)
            self.selected_profile = deepcopy(profile)
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Sprayer Profile", str(exc))

    def _import_profile(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Sprayer Profile", "", "Sprayer JSON (*.json)"
        )
        if not filename:
            return
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
            profile = payload.get("profile", payload)
            if not isinstance(profile, dict) or not profile.get("name"):
                raise ValueError("The selected JSON does not contain a named profile.")
            self._load_profile(profile)
            self.name_edit.setText(str(profile["name"]))
            self._save_current_to_config(self._collect_profile())
            self._populate_profiles()
            self.profile_selector.setCurrentText(profile["name"])
            self.selected_profile = deepcopy(self.current_profile)
        except Exception as exc:
            QMessageBox.warning(self, "Import Sprayer Profile", str(exc))

    def _export_profile(self):
        profile = self._collect_profile()
        suggested = (profile["name"] or "sprayer").replace(" ", "_") + ".json"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Sprayer Profile", suggested, "Sprayer JSON (*.json)"
        )
        if not filename:
            return
        try:
            Path(filename).write_text(
                json.dumps({"schema_version": 1, "profile": profile},
                           indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Export Sprayer Profile", str(exc))
