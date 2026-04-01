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

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from app.theme import HUD_QSS


def _make_splash_pixmap(w: int = 480, h: int = 260) -> QPixmap:
    """
    Draw the DroCap splash screen entirely in code — no image file needed.

    Layout:
        Navy header bar  →  "DroCap" wordmark
        White body       →  subtitle + status line
        Blue footer bar
    """
    px = QPixmap(w, h)
    px.fill(QColor("#FFFFFF"))

    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # ── Header bar ────────────────────────────────────────────────────────────
    header_h = 80
    p.fillRect(0, 0, w, header_h, QColor("#0747A6"))

    # DroCap wordmark
    font = QFont("Helvetica Neue", 28, QFont.Weight.Bold)
    p.setFont(font)
    p.setPen(QColor("#FFFFFF"))
    p.drawText(28, 54, "DroCap")

    # Tagline right-aligned in header
    font2 = QFont("Helvetica Neue", 9)
    p.setFont(font2)
    p.setPen(QColor("#B3D4FF"))
    p.drawText(0, 20, w - 20, 20, Qt.AlignmentFlag.AlignRight, "Schulich School of Engineering")
    p.drawText(0, 36, w - 20, 20, Qt.AlignmentFlag.AlignRight, "University of Calgary")

    # ── Body ─────────────────────────────────────────────────────────────────
    # Subtitle
    font3 = QFont("Helvetica Neue", 11)
    p.setFont(font3)
    p.setPen(QColor("#172B4D"))
    p.drawText(28, header_h + 30, "Drone-Based Markerless Motion Capture")

    # Description line
    font4 = QFont("Helvetica Neue", 9)
    p.setFont(font4)
    p.setPen(QColor("#5E6C84"))
    p.drawText(28, header_h + 52, "CSI Alberta  |  Sport Product Testing  |  Capstone 2025-2026")

    # ── Thin divider ─────────────────────────────────────────────────────────
    p.setPen(QColor("#DFE1E6"))
    div_y = header_h + 68
    p.drawLine(20, div_y, w - 20, div_y)

    # ── Loading bar (static; splash message updates below) ───────────────────
    bar_y = div_y + 18
    bar_h = 4
    bar_w = w - 56
    # Background track
    p.fillRect(28, bar_y, bar_w, bar_h, QColor("#EBECF0"))
    # Animated fill — roughly 30% shown on paint; message updates the label
    p.fillRect(28, bar_y, int(bar_w * 0.3), bar_h, QColor("#0052CC"))

    # ── Footer bar ────────────────────────────────────────────────────────────
    footer_h = 24
    p.fillRect(0, h - footer_h, w, footer_h, QColor("#F4F5F7"))
    p.setPen(QColor("#DFE1E6"))
    p.drawLine(0, h - footer_h, w, h - footer_h)
    font5 = QFont("Helvetica Neue", 8)
    p.setFont(font5)
    p.setPen(QColor("#A5ADBA"))
    p.drawText(0, h - footer_h, w - 12, footer_h,
               Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
               "Mission Control v1.0")

    # ── Border ────────────────────────────────────────────────────────────────
    p.setPen(QColor("#0052CC"))
    p.drawRect(0, 0, w - 1, h - 1)

    p.end()
    return px


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DroCap Mission Control")
    app.setOrganizationName("DroCap")
    app.setStyleSheet(HUD_QSS)

    # ── Splash screen ─────────────────────────────────────────────────────────
    splash_px = _make_splash_pixmap()
    splash = QSplashScreen(splash_px, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    _STEPS = [
        (300,  "Loading engine..."),
        (800,  "Initialising MediaPipe models..."),
        (1400, "Building UI components..."),
        (2000, "Starting Mission Control..."),
    ]

    def _update_splash(msg: str) -> None:
        splash.showMessage(
            msg,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            QColor("#0052CC"),
        )
        app.processEvents()

    for delay, msg in _STEPS:
        QTimer.singleShot(delay, lambda m=msg: _update_splash(m))

    # ── Import heavy modules while splash is visible ──────────────────────────
    from app.main_window import MainWindow  # noqa: PLC0415 — intentional late import

    window = MainWindow()

    # Show main window after 2.5 s, finish splash with a small overlap
    def _show_main() -> None:
        window.show()
        splash.finish(window)

    QTimer.singleShot(2500, _show_main)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
