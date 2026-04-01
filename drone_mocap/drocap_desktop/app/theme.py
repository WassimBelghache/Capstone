"""Jira-style enterprise light theme for DroCap Mission Control."""

HUD_QSS = """
/* ── Global ─────────────────────────────────────────────────────────────── */
* {
    font-family: "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 12px;
    color: #172B4D;
}

QMainWindow, QDialog {
    background-color: #F4F5F7;
}

QWidget {
    background-color: #F4F5F7;
    color: #172B4D;
}

/* ── Cards / Group boxes ─────────────────────────────────────────────────── */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #DFE1E6;
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #5E6C84;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #FFFFFF;
    color: #5E6C84;
    text-transform: uppercase;
}

/* ── Panel card wrappers (QWidget children of splitter) ─────────────────── */
/* Each quadrant sits on the F4F5F7 window; the inner panels are white. */
ChartPanel, VideoPanel, MetricsPanel, ControlPanel {
    background-color: #FFFFFF;
    border: 1px solid #DFE1E6;
    border-radius: 6px;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #FFFFFF;
    color: #0052CC;
    border: 1.5px solid #0052CC;
    border-radius: 6px;
    padding: 6px 16px;
    min-width: 64px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #DEEBFF;
    border-color: #0065FF;
    color: #0065FF;
}
QPushButton:pressed {
    background-color: #B3D4FF;
    border-color: #0747A6;
    color: #0747A6;
}
QPushButton:disabled {
    background-color: #F4F5F7;
    border-color: #C1C7D0;
    color: #A5ADBA;
}

/* Browse buttons — compact, no min-width override needed, padding does the work */
QPushButton#browse_btn {
    padding: 5px 12px;
    min-width: 56px;
    border-radius: 6px;
}

QPushButton#run_btn {
    background-color: #0052CC;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 700;
    padding: 8px 20px;
    min-width: 120px;
    letter-spacing: 0.5px;
}
QPushButton#run_btn:hover {
    background-color: #0065FF;
}
QPushButton#run_btn:pressed {
    background-color: #0747A6;
}
QPushButton#run_btn:disabled {
    background-color: #B3D4FF;
    color: #FFFFFF;
}

QPushButton#abort_btn {
    background-color: #FFFFFF;
    color: #DE350B;
    border: 1.5px solid #DE350B;
    border-radius: 6px;
    padding: 8px 16px;
    min-width: 80px;
    font-weight: 600;
}
QPushButton#abort_btn:hover {
    background-color: #FFEBE6;
    border-color: #FF5630;
    color: #FF5630;
}
QPushButton#abort_btn:disabled {
    color: #FFBDAD;
    border-color: #FFBDAD;
    background-color: #FFFFFF;
}

QPushButton#export_btn {
    background-color: #FFFFFF;
    color: #00875A;
    border: 1.5px solid #00875A;
    border-radius: 6px;
    padding: 6px 16px;
    min-width: 100px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#export_btn:hover {
    background-color: #E3FCEF;
    border-color: #36B37E;
    color: #36B37E;
}
QPushButton#export_btn:pressed {
    background-color: #ABF5D1;
    color: #006644;
}
QPushButton#export_btn:disabled {
    color: #ABF5D1;
    border-color: #ABF5D1;
}

/* ── Text inputs ─────────────────────────────────────────────────────────── */
QLineEdit {
    background-color: #FAFBFC;
    border: 1.5px solid #DFE1E6;
    border-radius: 4px;
    color: #172B4D;
    padding: 5px 8px;
    selection-background-color: #B3D4FF;
}
QLineEdit:focus {
    border-color: #4C9AFF;
    background-color: #FFFFFF;
}
QLineEdit:disabled {
    background-color: #F4F5F7;
    color: #A5ADBA;
}

/* ── Spin boxes ──────────────────────────────────────────────────────────── */
QDoubleSpinBox, QSpinBox {
    background-color: #FAFBFC;
    border: 1.5px solid #DFE1E6;
    border-radius: 4px;
    color: #172B4D;
    padding: 4px 6px;
    selection-background-color: #B3D4FF;
}
QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #4C9AFF;
    background-color: #FFFFFF;
}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background-color: #F4F5F7;
    border: none;
    width: 18px;
    border-radius: 0px;
}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
    background-color: #DEEBFF;
}

/* ── Combo boxes ─────────────────────────────────────────────────────────── */
QComboBox {
    background-color: #FAFBFC;
    border: 1.5px solid #DFE1E6;
    border-radius: 4px;
    color: #172B4D;
    padding: 5px 8px;
}
QComboBox:focus {
    border-color: #4C9AFF;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #DFE1E6;
    selection-background-color: #DEEBFF;
    selection-color: #0052CC;
    color: #172B4D;
}

/* ── Labels ──────────────────────────────────────────────────────────────── */
QLabel {
    color: #5E6C84;
    background-color: transparent;
}
QLabel#section_title {
    color: #0747A6;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    background-color: transparent;
}
QLabel#status_label {
    color: #42526E;
    font-size: 11px;
    background-color: transparent;
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
QProgressBar {
    background-color: #EBECF0;
    border: none;
    border-radius: 3px;
    text-align: center;
    color: #0052CC;
    font-size: 10px;
    font-weight: 600;
}
QProgressBar::chunk {
    background-color: #0052CC;
    border-radius: 3px;
}

/* ── Table ───────────────────────────────────────────────────────────────── */
QTableWidget {
    background-color: #FFFFFF;
    gridline-color: #F4F5F7;
    border: 1px solid #DFE1E6;
    border-radius: 4px;
    selection-background-color: #DEEBFF;
    alternate-background-color: #F8F9FA;
    color: #172B4D;
}
QTableWidget QHeaderView::section {
    background-color: #F4F5F7;
    color: #5E6C84;
    border: none;
    border-bottom: 2px solid #DFE1E6;
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
QTableWidget::item {
    padding: 5px 8px;
    border-bottom: 1px solid #F4F5F7;
}
QTableWidget::item:selected {
    background-color: #DEEBFF;
    color: #0052CC;
}

/* ── Splitters ───────────────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #F4F5F7;
    width: 4px;
    height: 4px;
}
QSplitter::handle:hover {
    background-color: #B3D4FF;
}

/* ── Scroll bars ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #F4F5F7;
    width: 8px;
    border: none;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #C1C7D0;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #A5ADBA;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal {
    background: #F4F5F7;
    height: 8px;
    border: none;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #C1C7D0;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #A5ADBA; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

/* ── Status bar ──────────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #0747A6;
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 500;
    border-top: none;
    padding: 2px 8px;
}
QStatusBar QLabel {
    color: #FFFFFF;
    background-color: transparent;
}

/* ── Splash screen ───────────────────────────────────────────────────────── */
QSplashScreen {
    border: 2px solid #0052CC;
}
"""

# pyqtgraph color palette — optimized for white/light background
PG_COLORS = {
    "hip":        "#FF5630",   # Jira red-orange
    "knee":       "#00875A",   # Jira green
    "ankle":      "#0052CC",   # Jira blue
    "cursor":     "#FF991F",   # Jira amber
    "gait_hs":    "#6554C0",   # Jira purple (heel-strike)
    "gait_swing": "#FF991F",   # Jira amber  (swing peak)
    "bg":         "#FFFFFF",   # white plot area
    "grid":       "#DFE1E6",   # light grey grid
    "axis_text":  "#5E6C84",
}
