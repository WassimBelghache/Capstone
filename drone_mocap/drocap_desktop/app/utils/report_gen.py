"""
DroCap PDF Report Generator
===========================
Builds a branded one-page PDF report from a completed analysis run.

Public API
----------
generate_report(
    out_path    : str | Path,      # where to write the .pdf
    chart_panel : ChartPanel,      # live pyqtgraph widget for screenshot
    gait_cycles : GaitCycles | None,
    metrics     : dict | None,     # contents of metrics_mocap.json["metrics"]
    video_path  : str | None,      # source video filename (display only)
    athlete_height_m : float | None,
) -> Path                          # returns the written path

Dependencies
------------
reportlab >= 3.6   (pip install reportlab)
PyQt6              (already present — used for the chart screenshot)
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.panels.chart_panel import ChartPanel
    from app.utils.biomechanics import GaitCycles

# ── Colour palette (mirrors theme.py) ────────────────────────────────────────
_NAVY      = (7,   43, 77)      # #172B4D
_BLUE      = (0,   82, 204)     # #0052CC
_BLUE_DARK = (7,   71, 166)     # #0747A6
_GREEN     = (0,  135,  90)     # #00875A
_AMBER     = (255,153,  31)     # #FF991F
_ORANGE    = (255, 86,  48)     # #FF5630
_LIGHT_BG  = (244,245,247)      # #F4F5F7
_BORDER    = (223,225,230)      # #DFE1E6
_GREY_TEXT = (94, 108, 132)     # #5E6C84
_WHITE     = (255,255,255)


def _rgb01(rgb_int: tuple) -> tuple:
    """Convert 0-255 tuple to 0.0-1.0 tuple for reportlab."""
    return tuple(v / 255 for v in rgb_int)


def _capture_chart_png(chart_panel: "ChartPanel", tmp_path: Path) -> Path | None:
    """
    Grab a screenshot of the ChartPanel pyqtgraph widget and save as PNG.
    Must be called from the GUI thread.
    Returns the PNG path, or None on failure.
    """
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPixmap
        QApplication.processEvents()
        pixmap = chart_panel._gw.grab()
        png_path = tmp_path.with_suffix(".png")
        pixmap.save(str(png_path), "PNG")
        return png_path
    except Exception as exc:
        print(f"[report_gen] chart screenshot failed: {exc}")
        return None


def generate_report(
    out_path: "str | Path",
    chart_panel: "ChartPanel | None" = None,
    gait_cycles: "GaitCycles | None" = None,
    metrics: "dict | None" = None,
    video_path: "str | None" = None,
    athlete_height_m: "float | None" = None,
) -> Path:
    """
    Generate a branded DroCap PDF report.

    Parameters
    ----------
    out_path         : destination .pdf file path
    chart_panel      : live ChartPanel widget (for screenshot); None = skip
    gait_cycles      : GaitCycles from biomechanics.detect_gait_cycles
    metrics          : dict from metrics_mocap.json["metrics"] or None
    video_path       : source video filename (display only)
    athlete_height_m : athlete height in metres, or None

    Returns
    -------
    Path to the written PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.platypus import Table, TableStyle
    except ImportError as exc:
        raise ImportError(
            "reportlab is required for PDF export. "
            "Install it with: pip install reportlab"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = A4          # 595.28 x 841.89 pt
    M    = 20 * mm     # page margin
    cw   = W - 2 * M  # content width

    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle("DroCap Analysis Report")
    c.setAuthor("DroCap Mission Control")
    c.setSubject("Biomechanical Gait Analysis")

    # ── Helper: colour shortcuts ──────────────────────────────────────────────
    def set_fill(rgb: tuple)   -> None: c.setFillColorRGB(*_rgb01(rgb))
    def set_stroke(rgb: tuple) -> None: c.setStrokeColorRGB(*_rgb01(rgb))

    y = H  # current Y position (top-down)

    # ── Header bar ───────────────────────────────────────────────────────────
    BAR_H = 18 * mm
    set_fill(_BLUE_DARK)
    c.rect(0, H - BAR_H, W, BAR_H, fill=1, stroke=0)

    # Logo text — left aligned
    set_fill(_WHITE)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(M, H - 12 * mm, "DroCap")
    c.setFont("Helvetica", 10)
    c.drawString(M + 48, H - 12 * mm, "Drone-Based Markerless Motion Capture")

    # Date — right aligned
    c.setFont("Helvetica", 9)
    date_str = datetime.now().strftime("%B %d, %Y")
    c.drawRightString(W - M, H - 12 * mm, date_str)

    y = H - BAR_H - 6 * mm

    # ── Sub-header ────────────────────────────────────────────────────────────
    set_fill(_NAVY)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(M, y, "Gait Analysis Report")
    y -= 5 * mm

    # Thin divider
    set_stroke(_BORDER)
    c.setLineWidth(0.5)
    c.line(M, y, W - M, y)
    y -= 5 * mm

    # ── Meta row ─────────────────────────────────────────────────────────────
    set_fill(_GREY_TEXT)
    c.setFont("Helvetica", 8)
    meta_parts = []
    if video_path:
        meta_parts.append(f"Source: {Path(video_path).name}")
    if athlete_height_m:
        meta_parts.append(f"Height: {athlete_height_m:.2f} m")
    meta_parts.append(f"Generated: {datetime.now().strftime('%H:%M:%S')}")
    c.drawString(M, y, "   |   ".join(meta_parts))
    y -= 6 * mm

    # ── Partner note ─────────────────────────────────────────────────────────
    set_fill(_BLUE)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(M, y, "Developed in partnership with CSI Alberta / Sport Product Testing Lab")
    y -= 8 * mm

    # ── Gait Statistics card ──────────────────────────────────────────────────
    _draw_section_label(c, M, y, cw, "GAIT CYCLE STATISTICS", _BLUE)
    y -= 5 * mm

    if gait_cycles is not None and gait_cycles.n_strides > 0:
        gc = gait_cycles
        stats = [
            ("Strides Detected",    str(gc.n_strides)),
            ("Mean Stride Duration", f"{gc.mean_stride_s:.3f} s"
             if not math.isnan(gc.mean_stride_s) else "—"),
            ("Cadence",             f"{gc.cadence_steps_per_min:.1f} steps/min"
             if not math.isnan(gc.cadence_steps_per_min) else "—"),
            ("Heel-Strikes",        str(len(gc.heel_strike_times))),
            ("Swing Peaks",         str(len(gc.swing_peak_times))),
        ]
    else:
        stats = [
            ("Strides Detected",    "No gait cycles detected"),
            ("Mean Stride Duration", "—"),
            ("Cadence",             "—"),
        ]

    y = _draw_kv_table(c, M, y, cw, stats)
    y -= 6 * mm

    # ── Accuracy Metrics card ─────────────────────────────────────────────────
    _draw_section_label(c, M, y, cw, "SYSTEM ACCURACY METRICS", _BLUE)
    y -= 5 * mm

    if metrics:
        joint_rows = []
        for joint in ("HIP", "KNEE", "ANKLE"):
            jm = metrics.get(joint, {})
            rmse = _fmt(jm.get("rmse"))
            mae  = _fmt(jm.get("mae"))
            corr = _fmt(jm.get("corr"), 3)
            col  = str(jm.get("matched_col") or "—")
            joint_rows.append([joint, rmse, mae, corr, col])

        tbl_data = [["Joint", "RMSE (°)", "MAE (°)", "Pearson r", "MoCap Column"]]
        tbl_data += joint_rows

        tbl = Table(tbl_data, colWidths=[
            cw * 0.14, cw * 0.16, cw * 0.14, cw * 0.14, cw * 0.42
        ])
        _apply_table_style(tbl)
        tw, th = tbl.wrapOn(c, cw, 200)
        tbl.drawOn(c, M, y - th)
        y -= th + 6 * mm
    else:
        set_fill(_GREY_TEXT)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(M + 4, y, "No MoCap comparison data available for this run.")
        y -= 8 * mm

    # ── Kinematic Waveforms chart ─────────────────────────────────────────────
    _draw_section_label(c, M, y, cw, "KINEMATIC WAVEFORMS", _BLUE)
    y -= 5 * mm

    chart_h = 70 * mm
    if chart_panel is not None:
        png = _capture_chart_png(chart_panel, out_path.with_name("_chart_tmp"))
        if png and png.exists():
            try:
                c.drawImage(str(png), M, y - chart_h, width=cw, height=chart_h,
                            preserveAspectRatio=True, anchor="nw")
                try:
                    png.unlink()
                except Exception:
                    pass
            except Exception as exc:
                print(f"[report_gen] embed chart image failed: {exc}")
                _draw_placeholder(c, M, y - chart_h, cw, chart_h, "Chart image unavailable")
        else:
            _draw_placeholder(c, M, y - chart_h, cw, chart_h, "Chart capture unavailable")
    else:
        _draw_placeholder(c, M, y - chart_h, cw, chart_h,
                          "Chart panel not available (run analysis first)")
    y -= chart_h + 6 * mm

    # ── Footer ────────────────────────────────────────────────────────────────
    set_fill(_LIGHT_BG)
    c.rect(0, 0, W, 10 * mm, fill=1, stroke=0)
    set_stroke(_BORDER)
    c.setLineWidth(0.5)
    c.line(0, 10 * mm, W, 10 * mm)
    set_fill(_GREY_TEXT)
    c.setFont("Helvetica", 7.5)
    c.drawString(M, 3.5 * mm,
        "DroCap Mission Control  |  Schulich School of Engineering, University of Calgary  "
        "|  Capstone 2025–2026")
    c.drawRightString(W - M, 3.5 * mm, f"Page 1 of 1")

    c.save()
    return out_path


# ── Private drawing helpers ────────────────────────────────────────────────────

def _draw_section_label(c, x, y, w, text, color_rgb):
    from reportlab.lib.units import mm
    # Label pill background
    from reportlab.lib import colors as rlc
    c.setFillColorRGB(*_rgb01(_LIGHT_BG))
    c.roundRect(x, y - 5 * mm, w, 6.5 * mm, 3, fill=1, stroke=0)
    # Left accent bar
    c.setFillColorRGB(*_rgb01(color_rgb))
    c.rect(x, y - 5 * mm, 3, 6.5 * mm, fill=1, stroke=0)
    # Text
    c.setFillColorRGB(*_rgb01(color_rgb))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 8, y - 1.5 * mm, text)


def _draw_kv_table(c, x, y, w, rows: list[tuple[str, str]]) -> float:
    """Draw a simple two-column key-value table. Returns new y."""
    from reportlab.lib.units import mm
    row_h = 6.5 * mm
    col_k = w * 0.38
    col_v = w * 0.62

    for i, (k, v) in enumerate(rows):
        bg = _LIGHT_BG if i % 2 == 0 else _WHITE
        c.setFillColorRGB(*_rgb01(bg))
        c.rect(x, y - row_h, w, row_h, fill=1, stroke=0)

        c.setFillColorRGB(*_rgb01(_GREY_TEXT))
        c.setFont("Helvetica", 9)
        c.drawString(x + 4, y - row_h + 2 * mm, k)

        c.setFillColorRGB(*_rgb01(_NAVY))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x + col_k + 4, y - row_h + 2 * mm, v)

        y -= row_h

    return y


def _draw_placeholder(c, x, y, w, h, text):
    c.setFillColorRGB(*_rgb01(_LIGHT_BG))
    c.setStrokeColorRGB(*_rgb01(_BORDER))
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColorRGB(*_rgb01(_GREY_TEXT))
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(x + w / 2, y + h / 2 - 4, text)


def _apply_table_style(tbl):
    from reportlab.platypus import TableStyle
    from reportlab.lib import colors as rlc

    style = TableStyle([
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0), rlc.HexColor("#F4F5F7")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), rlc.HexColor("#5E6C84")),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING",    (0, 0), (-1, 0), 6),
        # Body rows
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rlc.HexColor("#FFFFFF"), rlc.HexColor("#F8F9FA")]),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        # Grid
        ("LINEBELOW",   (0, 0), (-1, 0), 1.5, rlc.HexColor("#DFE1E6")),
        ("LINEBELOW",   (0, 1), (-1, -1), 0.5, rlc.HexColor("#F4F5F7")),
        ("BOX",         (0, 0), (-1, -1), 1, rlc.HexColor("#DFE1E6")),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (4, 0), (4, -1), "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ])
    tbl.setStyle(style)


def _fmt(val, precision: int = 2) -> str:
    try:
        f = float(val)
        if f != f:
            return "—"
        return f"{f:.{precision}f}"
    except (TypeError, ValueError):
        return "—"
