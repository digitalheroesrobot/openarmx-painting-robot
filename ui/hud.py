
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

        /* Improve readability only in data tables; keep the original HUD theme. */
        QTableWidget {{
            background-color: #0D1822;
            alternate-background-color: #132431;
            color: #F0F6FA;
            gridline-color: #29404D;
            selection-background-color: #164052;
            selection-color: #FFFFFF;
            font-size: 10pt;
        }}

        QTableWidget::item {{
            color: #F0F6FA;
            padding: 3px 5px;
        }}

        QHeaderView::section {{
            background-color: #132431;
            color: #00E5FF;
            border: 1px solid #29404D;
            padding: 5px 4px;
            font-weight: bold;
        }}
    """)
