"""Dialog for the minimal robot/product motion profile."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QFileDialog,
)

from core.robot_settings import (
    export_robot_profile, import_robot_profile, save_robot_settings,
    validate_robot_settings,
)


class RobotSettingsDialog(QDialog):
    def __init__(self, config_path: str | Path, current_settings: dict,
                 product_name: str = "Product", parent=None):
        super().__init__(parent)
        self.config_path = Path(config_path)
        initial = dict(current_settings)
        initial.setdefault("product_name", product_name)
        initial.setdefault("robot_name", "Robot")
        self.current_settings = validate_robot_settings(initial)
        self.selected_settings = None
        self.setWindowTitle("ROBOT SETTINGS")
        self.setMinimumWidth(540)
        root = QVBoxLayout(self)
        title = QLabel("ROBOT PROFILE")
        title.setStyleSheet("color:#00E5FF;font-size:15pt;font-weight:bold;")
        root.addWidget(title)

        form = QFormLayout()
        self.robot_name_edit = QLineEdit(self.current_settings["robot_name"])
        self.product_name_edit = QLineEdit(self.current_settings["product_name"])
        form.addRow("Robot name", self.robot_name_edit)
        form.addRow("Product name", self.product_name_edit)

        self.fields = {}
        for key, label, low, high in (
            ("base_travel_speed_mm_s", "Robot painting speed (mm/s)", 0.1, 2000.0),
            ("robot_control_frequency_hz", "Robot control frequency (Hz)", 1.0, 1000.0),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(1)
            spin.setKeyboardTracking(False)
            spin.setValue(float(self.current_settings[key]))
            self.fields[key] = spin
            form.addRow(label, spin)
        root.addLayout(form)

        self.profile_name_label = QLabel()
        self.profile_name_label.setStyleSheet("color:#00E5FF;font-weight:bold;padding:5px;")
        root.addWidget(self.profile_name_label)
        self.spacing_label = QLabel()
        self.spacing_label.setStyleSheet("color:#AFC5D1;padding:4px;")
        root.addWidget(self.spacing_label)

        file_actions = QHBoxLayout()
        load_button = QPushButton("LOAD PROFILE")
        export_button = QPushButton("EXPORT PROFILE")
        load_button.clicked.connect(self._load_profile)
        export_button.clicked.connect(self._export_profile)
        file_actions.addWidget(load_button)
        file_actions.addWidget(export_button)
        root.addLayout(file_actions)

        actions = QHBoxLayout()
        save_button = QPushButton("SAVE PROFILE")
        cancel_button = QPushButton("CANCEL")
        save_button.clicked.connect(self._save)
        cancel_button.clicked.connect(self.reject)
        actions.addWidget(save_button)
        actions.addWidget(cancel_button)
        root.addLayout(actions)
        self.setStyleSheet("""
            QDialog { background:#101D27;color:#EAF2F7; }
            QLabel { color:#DCEAF1; }
            QLineEdit,QDoubleSpinBox { background:#172B38;color:#F2F8FB;border:1px solid #365365;padding:5px; }
            QPushButton { background:#17465A;color:#EAFBFF;border:1px solid #00B8D9;padding:7px;font-weight:bold; }
        """)
        self.robot_name_edit.textChanged.connect(self._update_preview)
        self.product_name_edit.textChanged.connect(self._update_preview)
        for spin in self.fields.values():
            spin.valueChanged.connect(self._update_preview)
        self._update_preview()

    def _values(self):
        values = {
            "robot_name": self.robot_name_edit.text().strip(),
            "product_name": self.product_name_edit.text().strip(),
        }
        values.update({key: widget.value() for key, widget in self.fields.items()})
        return validate_robot_settings(values)

    def _update_preview(self, *_):
        try:
            values = self._values()
            self.profile_name_label.setText(f"PROFILE: {values['name']}")
            spacing = values["base_travel_speed_mm_s"] / values["robot_control_frequency_hz"]
            self.spacing_label.setText(f"WAYPOINT SPACING: {spacing:.3f} mm per controller update")
        except (ValueError, TypeError) as exc:
            self.profile_name_label.setText("PROFILE NAME: INVALID")
            self.spacing_label.setText(str(exc))

    def _apply_values(self, values):
        self.robot_name_edit.setText(values["robot_name"])
        self.product_name_edit.setText(values["product_name"])
        for key, widget in self.fields.items():
            widget.setValue(values[key])

    def _load_profile(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load robot profile", str(self.config_path.parent),
            "Robot profiles (*.json);;All files (*)",
        )
        if not file_name:
            return
        try:
            self._apply_values(import_robot_profile(file_name))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Could not load robot profile", str(exc))

    def _export_profile(self):
        try:
            values = self._values()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid robot profile", str(exc))
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export robot profile",
            str(self.config_path.parent / f"{values['name']}.json"),
            "Robot profiles (*.json)",
        )
        if not file_name:
            return
        path = Path(file_name)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            export_robot_profile(path, values)
        except OSError as exc:
            QMessageBox.warning(self, "Could not export robot profile", str(exc))

    def _save(self):
        try:
            values = self._values()
            save_robot_settings(self.config_path, values)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Invalid robot profile", str(exc))
            return
        self.selected_settings = values
        self.accept()
