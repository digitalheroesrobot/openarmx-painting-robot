"""Dialog for robot motion and controller configuration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from core.robot_settings import (
    save_robot_settings, validate_robot_settings,
)


class RobotSettingsDialog(QDialog):
    def __init__(self, config_path: str | Path, current_settings: dict, parent=None):
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.current_settings = validate_robot_settings(current_settings)
        self.selected_settings = None
        self.setWindowTitle("ROBOT SETTINGS")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        title = QLabel("ROBOT MOTION & CONTROLLER")
        title.setStyleSheet("color:#00E5FF;font-size:15pt;font-weight:bold;")
        root.addWidget(title)
        note = QLabel(
            "Enter limits and update frequency from the actual robot controller "
            "documentation. Defaults are provisional and are not hardware-verified."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#AFC5D1;padding:5px;")
        root.addWidget(note)
        form = QFormLayout()
        self.name_edit = QLineEdit(self.current_settings["name"])
        form.addRow("Robot profile name", self.name_edit)
        specs = [
            ("base_travel_speed_mm_s", "Painting speed (mm/s)", 0.1, 2000.0, 1),
            ("robot_control_frequency_hz", "Controller update frequency (Hz)", 1.0, 1000.0, 1),
            ("min_travel_speed_mm_s", "Minimum linear speed (mm/s)", 0.1, 2000.0, 1),
            ("max_linear_speed_mm_s", "Maximum linear speed (mm/s)", 0.1, 5000.0, 1),
            ("max_travel_acceleration_mm_s2", "Maximum acceleration (mm/s²)", 0.1, 10000.0, 1),
            ("max_angular_speed_deg_s", "Maximum angular speed (deg/s)", 0.1, 2000.0, 1),
        ]
        self.fields = {}
        for key, label, low, high, decimals in specs:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setKeyboardTracking(False)
            spin.setValue(float(self.current_settings[key]))
            self.fields[key] = spin
            form.addRow(label, spin)
        root.addLayout(form)
        self.spacing_label = QLabel()
        self.spacing_label.setStyleSheet("color:#00E5FF;padding:5px;font-weight:bold;")
        root.addWidget(self.spacing_label)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        actions = QHBoxLayout()
        save = QPushButton("SAVE ROBOT SETTINGS")
        cancel = QPushButton("CANCEL")
        save.clicked.connect(self._save)
        cancel.clicked.connect(self.reject)
        actions.addWidget(save)
        actions.addWidget(cancel)
        root.addLayout(actions)
        self.setStyleSheet("""
            QDialog { background:#101D27;color:#EAF2F7; }
            QLabel { color:#DCEAF1; }
            QLineEdit,QDoubleSpinBox { background:#172B38;color:#F2F8FB;border:1px solid #365365;padding:5px; }
            QPushButton { background:#17465A;color:#EAFBFF;border:1px solid #00B8D9;padding:7px;font-weight:bold; }
        """)
        for spin in self.fields.values():
            spin.valueChanged.connect(self._update_preview)
        self._update_preview()

    def _values(self):
        values = {"name": self.name_edit.text().strip()}
        values.update({key: widget.value() for key, widget in self.fields.items()})
        values["input_status"] = "UNVALIDATED_OPERATOR_INPUT"
        return validate_robot_settings(values)

    def _update_preview(self, *_):
        try:
            values = self._values()
            spacing = values["base_travel_speed_mm_s"] / values["robot_control_frequency_hz"]
            self.spacing_label.setText(f"WAYPOINT SPACING: {spacing:.3f} mm per controller update")
            self.status_label.setText(
                "Robot limits are operator inputs. Confirm them against the robot "
                "manufacturer/controller documentation before physical execution."
            )
            self.status_label.setStyleSheet("color:#FFD740;")
        except (ValueError, TypeError) as exc:
            self.spacing_label.setText("WAYPOINT SPACING: INVALID")
            self.status_label.setText(str(exc))
            self.status_label.setStyleSheet("color:#FF5252;")

    def _save(self):
        try:
            values = self._values()
            save_robot_settings(self.config_path, values)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Invalid robot settings", str(exc))
            return
        self.selected_settings = values
        self.accept()
