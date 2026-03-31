"""
DroCap Mission Control — entry point.

Usage:
    cd drocap_desktop
    poetry run python main.py
    # or if installed:
    drocap-desktop
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from app.theme import HUD_QSS
from app.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DroCap Mission Control")
    app.setOrganizationName("DroCap")
    app.setStyleSheet(HUD_QSS)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
