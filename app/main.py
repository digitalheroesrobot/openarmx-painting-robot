
#!/usr/bin/env python3

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from ui.hud import load_local_fonts, apply_hud_style


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("OpenArmX Painting Control System")

    fonts = load_local_fonts(PROJECT_ROOT)
    apply_hud_style(app, fonts)

    window = MainWindow(
        project_root=PROJECT_ROOT,
        fonts=fonts,
    )

    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
