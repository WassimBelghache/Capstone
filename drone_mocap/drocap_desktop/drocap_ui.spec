# -*- mode: python ; coding: utf-8 -*-
"""
DroCap Mission Control — PyInstaller spec file
===============================================
Build a single-folder distribution (onedir) that can be demonstrated on any
macOS laptop without requiring a terminal or Python installation.

Build command (run from drocap_desktop/):
    poetry run pyinstaller drocap_ui.spec --noconfirm

Output:
    dist/DroCap Mission Control/    ← folder to zip & distribute
    dist/DroCap Mission Control.app ← macOS app bundle (macOS only)

Notes
-----
- We use onedir (not onefile) so the app starts fast; the folder is zipped
  for distribution.
- MediaPipe ships its own .tflite model files inside the mediapipe package
  directory — we collect them as data so they are found at runtime.
- PyQt6 multimedia backends (for QMediaPlayer) live in Qt6/plugins/; they are
  collected automatically by collect_submodules('PyQt6').
- drone_mocap is a develop-install editable package; we collect it explicitly.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ── Resolve virtualenv site-packages ─────────────────────────────────────────
# Adjust this path if your poetry venv hash differs (check `poetry env info`).
_VENV_SITE = Path(
    "~/.cache/pypoetry/virtualenvs/"
    "drocap-desktop--Yz345_c-py3.11/lib/python3.11/site-packages"
).expanduser()

# ── Data files ───────────────────────────────────────────────────────────────
datas = []

# MediaPipe models & resources (*.tflite, *.pbtxt, *.binarypb …)
datas += collect_data_files("mediapipe", includes=["**/*.tflite",
                                                    "**/*.pbtxt",
                                                    "**/*.binarypb",
                                                    "**/*.task"])

# drone_mocap Python package
datas += collect_data_files("drone_mocap")

# PyQt6 Qt plugins (multimedia, platform, styles)
datas += collect_data_files("PyQt6", includes=["Qt6/plugins/**/*"])

# ── Hidden imports ────────────────────────────────────────────────────────────
hiddenimports = []
hiddenimports += collect_submodules("PyQt6")
hiddenimports += collect_submodules("pyqtgraph")
hiddenimports += collect_submodules("mediapipe")
hiddenimports += collect_submodules("drone_mocap")
hiddenimports += [
    "scipy.signal",
    "scipy.ndimage",
    "scipy.linalg",
    "sklearn",            # mediapipe may need it
    "reportlab",
    "reportlab.platypus",
    "reportlab.lib.colors",
    "reportlab.lib.pagesizes",
    "reportlab.lib.units",
    "reportlab.pdfgen.canvas",
    "cv2",
    "numpy",
    "pandas",
]

# ── Binaries ──────────────────────────────────────────────────────────────────
# mediapipe bundles a private opencv; include its .so/.dylib
binaries = []
try:
    import cv2
    _cv2_dir = Path(cv2.__file__).parent
    for lib in _cv2_dir.glob("*.so*"):
        binaries.append((str(lib), "cv2"))
    for lib in _cv2_dir.glob("*.dylib"):
        binaries.append((str(lib), "cv2"))
except Exception:
    pass

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["main.py"],
    pathex=[str(Path(".").resolve())],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter",
              "tornado", "zmq", "PyQt5"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,    # onedir mode — binaries go to COLLECT
    name="DroCap Mission Control",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,            # no terminal window on launch
    disable_windowed_traceback=False,
    # icon="assets/drocap.icns",  # uncomment if you add an icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DroCap Mission Control",
)

# ── macOS app bundle ──────────────────────────────────────────────────────────
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="DroCap Mission Control.app",
        # icon="assets/drocap.icns",
        bundle_identifier="ca.ucalgary.schulich.drocap",
        info_plist={
            "CFBundleName":             "DroCap Mission Control",
            "CFBundleDisplayName":      "DroCap Mission Control",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion":          "1.0.0",
            "NSHighResolutionCapable":  True,
            "NSCameraUsageDescription": "DroCap uses the camera for live pose estimation.",
            "LSMinimumSystemVersion":   "12.0",
        },
    )
