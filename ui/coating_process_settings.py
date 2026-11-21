"""Dialog for independent coating-process parameter profiles."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QComboBox,
)

from core.coating_process import (
    DEFAULT_COATING_PROCESS, coating_model_readiness,
    save_coating_process, validate_coating_process,
)


class CoatingProcessSettingsDialog(QDialog):
    def __init__(self, config_path: str | Path, current_process: dict, parent=None):
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.current_process = validate_coating_process(current_process)
        self.selected_process = None
        self.setWindowTitle("COATING PROCESS PARAMETERS")
        self.setMinimumWidth(560)
        root = QVBoxLayout(self)
        title = QLabel("COATING PROCESS PARAMETERS")
        title.setStyleSheet("color:#00E5FF;font-size:15pt;font-weight:bold;")
        root.addWidget(title)
        note = QLabel(
            "Enter values from the coating data sheet and validated production process. "
            "Unverified inputs produce estimates, never measured film thickness."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#AFC5D1;padding:5px;")
        root.addWidget(note)
        form = QFormLayout()
        self.name_edit = QLineEdit(self.current_process["name"])
        form.addRow("Process profile name", self.name_edit)
        self.fields = {}
        specs = [
            ("target_dry_film_thickness_um", "Target dry film thickness (µm)", 0.0, 2000.0, 1),
            ("fluid_flow_ml_min", "Fluid flow rate (ml/min)", 0.0, 10000.0, 1),
            ("transfer_efficiency_percent", "Transfer efficiency (%)", 0.0, 100.0, 1),
            ("volume_solids_percent", "Volume solids (%)", 0.0, 100.0, 1),
            ("base_travel_speed_mm_s", "Base travel speed (mm/s)", 0.1, 2000.0, 1),
            ("min_travel_speed_mm_s", "Minimum travel speed (mm/s)", 0.1, 2000.0, 1),
            ("max_travel_acceleration_mm_s2", "Maximum acceleration (mm/s²)", 0.1, 10000.0, 1),
            ("number_of_coats", "Number of coats", 1, 20, 0),
            ("spray_on_delay_ms", "Spray ON delay (ms)", 0.0, 10000.0, 1),
            ("spray_off_delay_ms", "Spray OFF delay (ms)", 0.0, 10000.0, 1),
        ]
        for key, label, low, high, decimals in specs:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setKeyboardTracking(False)
            spin.setValue(float(self.current_process[key]))
            if key in ("fluid_flow_ml_min", "transfer_efficiency_percent", "volume_solids_percent"):
                spin.setToolTip("Use a value confirmed for the actual paint and spray equipment. Zero means not configured.")
            self.fields[key] = spin
            form.addRow(label, spin)
        root.addLayout(form)
        self.readiness_label = QLabel()
        self.readiness_label.setWordWrap(True)
        root.addWidget(self.readiness_label)
        actions = QHBoxLayout()
        save = QPushButton("SAVE PROCESS")
        cancel = QPushButton("CANCEL")
        save.clicked.connect(self._save)
        cancel.clicked.connect(self.reject)
        for button in (save, cancel):
            button.setMinimumHeight(34)
            button.setAutoDefault(False)
            button.setDefault(False)
            actions.addWidget(button)
        root.addLayout(actions)
        self.setStyleSheet("""
            QDialog { background:#101D27;color:#EAF2F7; }
            QLabel { color:#DCEAF1; }
            QLineEdit,QDoubleSpinBox { background:#172B38;color:#F2F8FB;border:1px solid #365365;padding:5px; }
            QPushButton { background:#17465A;color:#EAFBFF;border:1px solid #00B8D9;padding:7px;font-weight:bold; }
        """)
        self._update_readiness()
        for spin in self.fields.values():
            spin.valueChanged.connect(self._update_readiness)

    def _values(self):
        values = {"name": self.name_edit.text().strip()}
        values.update({key: widget.value() for key, widget in self.fields.items()})
        values["number_of_coats"] = int(values["number_of_coats"])
        values["input_status"] = "UNVALIDATED_OPERATOR_INPUT"
        return validate_coating_process(values)

    def _update_readiness(self, *_):
        try:
            status, missing = coating_model_readiness(self._values())
            if missing:
                self.readiness_label.setText(
                    "MODEL STATUS: INPUTS INCOMPLETE — " + ", ".join(missing)
                    + ". Film thickness will remain N/A."
                )
                self.readiness_label.setStyleSheet("color:#FFD740;")
            else:
                self.readiness_label.setText(
                    "MODEL STATUS: INPUTS READY — UNCALIBRATED PREDICTION. "
                    "Calibration against measured coating coupons is still required."
                )
                self.readiness_label.setStyleSheet("color:#00E5FF;")
        except (ValueError, TypeError) as exc:
            self.readiness_label.setText(f"INVALID INPUT: {exc}")
            self.readiness_label.setStyleSheet("color:#FF5252;")

    def _save(self):
        try:
            values = self._values()
            save_coating_process(self.config_path, values)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Invalid coating process", str(exc))
            return
        self.selected_process = values
        self.accept()
