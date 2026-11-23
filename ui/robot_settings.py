from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout,
)


DEFAULT_ROBOT_SETTINGS = {
    "painting_speed_mm_s": 150.0,
    "control_frequency_hz": 100.0,
}


def load_robot_settings(path: str | Path) -> dict:
    config = Path(path)
    config.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(DEFAULT_ROBOT_SETTINGS)
    if config.exists():
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in settings:
                    value = float(payload.get(key, settings[key]))
                    if value > 0:
                        settings[key] = value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise RuntimeError(f"Could not read robot settings: {config}")
    else:
        save_robot_settings(config, settings)
    return settings


def save_robot_settings(path: str | Path, settings: dict) -> None:
    config = Path(path)
    config.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "painting_speed_mm_s": float(settings["painting_speed_mm_s"]),
        "control_frequency_hz": float(settings["control_frequency_hz"]),
    }
    temporary = config.with_suffix(config.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(config)


class RobotSettingsDialog(QDialog):
    """Robot arm motion settings; control frequency is stored for future export."""

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROBOT ARM SETTINGS")
        self.setMinimumWidth(420)
        self.settings = dict(DEFAULT_ROBOT_SETTINGS)
        self.settings.update(settings or {})
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel("PAINTING ROBOT MOTION")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00E5FF;")
        root.addWidget(title)

        form = QFormLayout()
        form.setVerticalSpacing(12)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 5000.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setSuffix(" mm/s")
        self.speed_spin.setValue(float(self.settings["painting_speed_mm_s"]))
        form.addRow("Painting travel speed", self.speed_spin)

        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(1.0, 2000.0)
        self.frequency_spin.setDecimals(1)
        self.frequency_spin.setSuffix(" Hz")
        self.frequency_spin.setValue(float(self.settings["control_frequency_hz"]))
        form.addRow("Control frequency", self.frequency_spin)

        root.addLayout(form)
        note = QLabel(
            "Painting speed determines the estimated total route time and simulation duration. "
            "Control frequency is saved for future spray-point export; it does not affect "
            "route geometry or simulation timing yet."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9FB3C1;")
        root.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("CANCEL")
        cancel.clicked.connect(self.reject)
        save = QPushButton("SAVE")
        save.setDefault(True)
        save.clicked.connect(self._accept_settings)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        root.addLayout(buttons)

    def _accept_settings(self):
        speed = float(self.speed_spin.value())
        frequency = float(self.frequency_spin.value())
        if speed <= 0.0 or frequency <= 0.0:
            QMessageBox.warning(self, "Invalid settings", "Both values must be greater than zero.")
            return
        self.settings = {
            "painting_speed_mm_s": speed,
            "control_frequency_hz": frequency,
        }
        self.accept()
