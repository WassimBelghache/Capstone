"""Dark HUD stylesheet for DroCap Mission Control."""

HUD_QSS = """
QWidget {
    background-color: #0d0d0d;
    color: #e0e0e0;
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 12px;
}

QMainWindow {
    background-color: #0d0d0d;
}

/* Splitter handles */
QSplitter::handle {
    background-color: #1e1e2e;
    width: 3px;
    height: 3px;
}

/* Group boxes */
QGroupBox {
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    color: #7c7caa;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    left: 8px;
}

/* Buttons */
QPushButton {
    background-color: #1e1e2e;
    color: #c0c0e0;
    border: 1px solid #2a2a4a;
    border-radius: 3px;
    padding: 5px 14px;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #2a2a4a;
    border-color: #4a4a8a;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #0a84ff;
    border-color: #0a84ff;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #141420;
    color: #404060;
    border-color: #1a1a2e;
}
QPushButton#run_btn {
    background-color: #0a3a6a;
    border-color: #0a84ff;
    color: #5ac8fa;
    font-weight: bold;
    font-size: 13px;
    padding: 7px 20px;
}
QPushButton#run_btn:hover {
    background-color: #0a84ff;
    color: #ffffff;
}
QPushButton#run_btn:disabled {
    background-color: #0a1a2e;
    border-color: #0a2a4a;
    color: #2a4a6a;
}

/* Line edits / path displays */
QLineEdit {
    background-color: #141420;
    border: 1px solid #2a2a3e;
    border-radius: 3px;
    color: #a0a0c0;
    padding: 3px 6px;
    selection-background-color: #0a84ff;
}
QLineEdit:focus {
    border-color: #0a84ff;
}

/* Spin / double spin boxes */
QDoubleSpinBox, QSpinBox {
    background-color: #141420;
    border: 1px solid #2a2a3e;
    border-radius: 3px;
    color: #c0c0e0;
    padding: 2px 4px;
}
QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #0a84ff;
}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background-color: #1e1e2e;
    border: none;
    width: 16px;
}

/* Combo boxes */
QComboBox {
    background-color: #141420;
    border: 1px solid #2a2a3e;
    border-radius: 3px;
    color: #c0c0e0;
    padding: 3px 6px;
}
QComboBox:focus {
    border-color: #0a84ff;
}
QComboBox QAbstractItemView {
    background-color: #1a1a2a;
    border: 1px solid #2a2a4a;
    selection-background-color: #0a84ff;
    color: #c0c0e0;
}

/* Labels */
QLabel {
    color: #a0a0c0;
}
QLabel#section_title {
    color: #5ac8fa;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
}

/* Progress bar */
QProgressBar {
    background-color: #141420;
    border: 1px solid #2a2a3e;
    border-radius: 3px;
    text-align: center;
    color: #5ac8fa;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #0a84ff;
    border-radius: 2px;
}

/* Table widget */
QTableWidget {
    background-color: #0d0d18;
    gridline-color: #1e1e2e;
    border: 1px solid #2a2a3e;
    selection-background-color: #0a2a4a;
    color: #c0c0e0;
}
QTableWidget QHeaderView::section {
    background-color: #141420;
    color: #7c7caa;
    border: none;
    border-bottom: 1px solid #2a2a3e;
    padding: 4px;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
}
QTableWidget::item:selected {
    background-color: #0a2a4a;
    color: #5ac8fa;
}

/* Scroll bars */
QScrollBar:vertical {
    background: #0d0d0d;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #2a2a4a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #0d0d0d;
    height: 8px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #2a2a4a;
    border-radius: 4px;
    min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Status bar */
QStatusBar {
    background-color: #080810;
    color: #5a5a7a;
    border-top: 1px solid #1a1a2a;
    font-size: 11px;
}
"""

# pyqtgraph color palette
PG_COLORS = {
    "hip":        "#ff9f0a",   # amber
    "knee":       "#30d158",   # green
    "ankle":      "#5ac8fa",   # sky blue
    "cursor":     "#ff453a",   # red
    "gait_hs":    "#bf5af2",   # purple  (heel-strike)
    "gait_swing": "#ffd60a",   # yellow  (swing peak)
    "bg":         "#0d0d18",
    "grid":       "#1e1e2e",
}
