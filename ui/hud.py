
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase


COLORS = {
    "background": "#071018",
    "panel": "#0D1822",
    "panel_alt": "#132431",
    "border": "#203746",
    "text": "#D8E5ED",
    "muted": "#8195A3",
    "cyan": "#00E5FF",
    "green": "#00E676",
    "yellow": "#FFD740",
    "red": "#FF5252",
}


def load_local_fonts(project_root: Path) -> dict:
    font_dir = Path(project_root) / "fonts"

    font_files = [
        "Rajdhani-Regular.ttf",
        "Rajdhani-SemiBold.ttf",
        "Rajdhani-Bold.ttf",
        "JetBrainsMono-Regular.ttf",
        "JetBrainsMono-Bold.ttf",
    ]

    for filename in font_files:
        path = font_dir / filename

        if not path.is_file():
            continue

        font_id = QFontDatabase.addApplicationFont(str(path))

        if font_id < 0:
            print(f"[FONT WARNING] Failed to load: {path}")
        else:
            print(f"[FONT LOADED] {filename}")

    available = set(QFontDatabase.families())

    return {
        "heading": (
            "Rajdhani"
            if "Rajdhani" in available
            else "Arial"
        ),
        "body": (
            "Rajdhani"
            if "Rajdhani" in available
            else "Arial"
        ),
        "mono": (
            "JetBrains Mono"
            if "JetBrains Mono" in available
            else "Monospace"
        ),
    }


def apply_hud_style(app, fonts):
    app.setFont(QFont(fonts["body"], 10))

    app.setStyleSheet(f"""
        QWidget {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            font-family: "{fonts["body"]}";
            font-size: 11pt;
        }}

        QMainWindow {{
            background-color: {COLORS["background"]};
        }}

        QFrame#Panel {{
            background-color: {COLORS["panel"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 5px;
        }}

        QLabel#MainTitle {{
            color: {COLORS["cyan"]};
            font-family: "{fonts["heading"]}";
            font-size: 25pt;
            font-weight: bold;
            letter-spacing: 2px;
        }}

        QLabel#Subtitle {{
            color: {COLORS["muted"]};
            font-family: "{fonts["heading"]}";
            font-size: 11pt;
            letter-spacing: 1px;
        }}

        QLabel#SectionTitle {{
            color: {COLORS["cyan"]};
            font-family: "{fonts["heading"]}";
            font-size: 15pt;
            font-weight: bold;
        }}

        QLabel#MetricValue {{
            color: {COLORS["green"]};
            font-family: "{fonts["mono"]}";
            font-size: 10pt;
        }}

        QLabel#Muted {{
            color: {COLORS["muted"]};
        }}

        QLabel#StageNumber {{
            color: {COLORS["cyan"]};
            font-family: "{fonts["mono"]}";
            font-size: 10pt;
        }}

        QPushButton {{
            background-color: {COLORS["panel_alt"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            padding: 8px 10px;
            font-family: "{fonts["heading"]}";
            font-size: 12pt;
            font-weight: bold;
        }}

        QPushButton:hover {{
            border: 1px solid {COLORS["cyan"]};
            color: {COLORS["cyan"]};
        }}

        QPushButton:pressed {{
            background-color: #1B3443;
        }}

        QPushButton:disabled {{
            color: #50616D;
            background-color: #0A131B;
            border-color: #182833;
        }}

        QPushButton#PrimaryButton {{
            background-color: #07303A;
            color: {COLORS["cyan"]};
            border: 1px solid {COLORS["cyan"]};
            padding: 11px;
        }}

        QPushButton#StageActive {{
            background-color: #07303A;
            color: {COLORS["cyan"]};
            border: 1px solid {COLORS["cyan"]};
        }}

        QComboBox {{
            background-color: {COLORS["panel_alt"]};
            color: {COLORS["cyan"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            padding: 7px 10px;
            min-width: 75px;
            font-family: "{fonts["mono"]}";
            font-size: 10pt;
        }}

        QComboBox:hover {{
            border: 1px solid {COLORS["cyan"]};
        }}

        QComboBox QAbstractItemView {{
            background-color: {COLORS["panel"]};
            color: {COLORS["text"]};
            selection-background-color: #164052;
            selection-color: {COLORS["cyan"]};
            border: 1px solid {COLORS["cyan"]};
        }}

        QStatusBar {{
            background-color: {COLORS["panel"]};
            color: {COLORS["text"]};
            border-top: 1px solid {COLORS["border"]};
        }}

        /* High-contrast controls used throughout the right sidebar. */
        QToolButton {{
            background-color: #172B39;
            color: #F0F6FA;
            border: 1px solid #426174;
            border-radius: 4px;
            padding: 7px 8px;
            font-weight: bold;
        }}

        QToolButton:hover {{
            background-color: #204154;
            color: #FFFFFF;
            border: 1px solid {COLORS["cyan"]};
        }}

        QToolButton:checked {{
            background-color: #164052;
            color: {COLORS["cyan"]};
            border: 1px solid {COLORS["cyan"]};
        }}

        QToolButton:disabled {{
            color: #83939E;
            background-color: #111D26;
            border-color: #293B47;
        }}

        QTableWidget, QTreeWidget, QListWidget {{
            background-color: #101E29;
            alternate-background-color: #172936;
            color: #E7F0F5;
            gridline-color: #385160;
            border: 1px solid #385160;
            selection-background-color: #15516A;
            selection-color: #FFFFFF;
        }}

        QHeaderView::section {{
            background-color: #203746;
            color: #FFFFFF;
            border: 1px solid #385160;
            padding: 5px 4px;
            font-weight: bold;
        }}

        QCheckBox {{
            color: #E7F0F5;
            spacing: 7px;
        }}

        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid #78909F;
            border-radius: 3px;
            background-color: #132431;
        }}

        QCheckBox::indicator:checked {{
            background-color: #00AFC4;
            border: 1px solid #65F0FF;
        }}

        QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: #132431;
            color: #F0F6FA;
            border: 1px solid #496474;
            border-radius: 4px;
            padding: 5px 6px;
            selection-background-color: #15516A;
            selection-color: #FFFFFF;
        }}

        QSpinBox:focus, QDoubleSpinBox:focus,
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border: 1px solid {COLORS["cyan"]};
        }}

        QScrollBar:vertical {{
            background: #0D1822;
            width: 12px;
            margin: 2px;
        }}

        QScrollBar::handle:vertical {{
            background: #496474;
            min-height: 24px;
            border-radius: 5px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: #00AFC4;
        }}

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
    """)
