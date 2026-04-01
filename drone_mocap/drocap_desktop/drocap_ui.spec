# -*- mode: python ; coding: utf-8 -*-
"""
DroCap — PyInstaller spec file
===============================================
Produces a macOS onedir bundle (and .app) that can be demonstrated by
double-clicking on any macOS laptop without a terminal or Python install.

Build command (run from drocap_desktop/):
    poetry run pyinstaller drocap_ui.spec --clean --noconfirm

Output:
    dist/DroCap/            ← onedir folder
    dist/DroCap.app        ← macOS app bundle (macOS only)

Key design decisions
--------------------
1. onedir (not onefile) — fast cold-start; no temp-extract delay.
2. drone_mocap collected from its editable-install source tree, not from
   site-packages, because the .pth redirect is not preserved in the bundle.
3. MediaPipe .binarypb / .tflite / .pbtxt graph files are collected under
   the same relative path as in the venv (mediapipe/modules/...) so the
   C++ calculator framework can find them at _MEIPASS/mediapipe/modules/.
4. setup_bundle_env() (called first in main.py) sets CWD = _MEIPASS so
   MediaPipe's path resolution works without any monkey-patching.
5. console=False — no terminal window on double-click.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(".").resolve()                        # drocap_desktop/
_REPO_ROOT   = _HERE.parent                              # drone_mocap/
_SRC_ROOT    = _REPO_ROOT / "src"                        # contains drone_mocap/
_VENV_SITE   = Path(
    "~/.cache/pypoetry/virtualenvs/"
    "drocap-desktop--Yz345_c-py3.11/lib/python3.11/site-packages"
).expanduser()

# ── Data files ────────────────────────────────────────────────────────────────
datas = []

# 1. MediaPipe graph resources — must land at mediapipe/ inside _MEIPASS
#    so that mediapipe.__file__ → _MEIPASS/mediapipe/__init__.py and the
#    C++ framework finds mediapipe/modules/... relative to _MEIPASS.
datas += collect_data_files(
    "mediapipe",
    includes=[
        "**/*.tflite",
        "**/*.binarypb",
        "**/*.pbtxt",
        "**/*.task",
        "**/*.so",
    ],
)

# 2. drone_mocap — collected directly from the source tree so editable
#    install path (/…/drone_mocap/src) doesn't need to exist on target.
_dm_src = _SRC_ROOT / "drone_mocap"
if _dm_src.exists():
    datas.append((str(_dm_src), "drone_mocap"))

# 3. PyQt6 Qt plugins (multimedia/codecs, platform, imageformats, styles)
#    Required for QMediaPlayer/AVFoundation and the Fusion/macOS style.
datas += collect_data_files(
    "PyQt6",
    includes=[
        "Qt6/plugins/mediaservice/*",
        "Qt6/plugins/multimedia/*",
        "Qt6/plugins/platforms/*",
        "Qt6/plugins/imageformats/*",
        "Qt6/plugins/styles/*",
        "Qt6/plugins/audio/*",
    ],
)

# 4. pyqtgraph resources (colormaps etc.)
datas += collect_data_files("pyqtgraph")

# ── Hidden imports ────────────────────────────────────────────────────────────
hiddenimports = []
hiddenimports += collect_submodules("PyQt6")
hiddenimports += collect_submodules("pyqtgraph")
hiddenimports += collect_submodules("mediapipe")
hiddenimports += collect_submodules("drone_mocap")
hiddenimports += [
    # scipy
    "scipy.signal",
    "scipy.ndimage",
    "scipy.linalg",
    "scipy.sparse",
    "scipy.sparse.linalg",
    # data
    "numpy",
    "pandas",
    "pandas.core.frame",
    "pyarrow",             # for .parquet read/write
    "matplotlib",
    # PDF
    "reportlab",
    "reportlab.platypus",
    "reportlab.lib.colors",
    "reportlab.lib.pagesizes",
    "reportlab.lib.units",
    "reportlab.pdfgen.canvas",
    # OpenCV (mediapipe bundles its own)
    "cv2",
    # drone_mocap explicit
    "drone_mocap.pipeline.run",
    "drone_mocap.pose.mediapose",
    "drone_mocap.io.video",
    "drone_mocap.filters.smoothing",
    "drone_mocap.angles.saggital2D",
    "drone_mocap.io.mocap_txt",
    "drone_mocap.evaluation.compare_mocap",
    "drone_mocap.utils.scaling",
    # app bundle utility
    "app.utils.bundle",
]

# ── Binaries ──────────────────────────────────────────────────────────────────
binaries = []

# MediaPipe ships its own OpenCV dylib; include it so it shadows system cv2.
_mp_site = _VENV_SITE / "mediapipe"
for lib in _mp_site.glob("**/*.dylib"):
    dest = str(lib.parent.relative_to(_VENV_SITE))
    binaries.append((str(lib), dest))
for lib in _mp_site.glob("**/*.so"):
    dest = str(lib.parent.relative_to(_VENV_SITE))
    binaries.append((str(lib), dest))

# ── Runtime hooks ─────────────────────────────────────────────────────────────
# A runtime hook runs before any user code in the frozen process.
# We write it inline as a temp file name; PyInstaller evaluates it.
# (Alternatively, place as a .py file in a hooks/ directory.)
_RUNTIME_HOOK = str(_HERE / "_pyi_rthook_drocap.py")

import os, textwrap
_hook_src = textwrap.dedent("""\
    # DroCap PyInstaller runtime hook
    # Sets CWD = _MEIPASS so MediaPipe's C++ framework finds graph files.
    import os, sys
    if hasattr(sys, "_MEIPASS"):
        os.chdir(sys._MEIPASS)
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
""")
with open(_RUNTIME_HOOK, "w") as _f:
    _f.write(_hook_src)

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["main.py"],
    pathex=[
        str(_HERE),         # drocap_desktop/ — resolves app.*
        str(_SRC_ROOT),     # src/            — resolves drone_mocap.*
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[_RUNTIME_HOOK],
    excludes=[
        "tkinter", "IPython", "jupyter",
        "tornado", "zmq", "PyQt5", "PySide2", "PySide6",
        "wx", "gi",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DroCap",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no terminal window on double-click
    disable_windowed_traceback=False,
    argv_emulation=False,    # macOS: don't intercept Apple Events (prevents hang)
    target_arch=None,        # native arch; set "universal2" for fat binary
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/Logo.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DroCap",
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="DroCap.app",
        icon="assets/Logo.icns",
        bundle_identifier="ca.ucalgary.schulich.drocap",
        info_plist={
            # Required by macOS — prevents Gatekeeper hanging on missing keys
            "CFBundleName":               "DroCap",
            "CFBundleDisplayName":        "DroCap",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion":            "1.0.0",
            "CFBundlePackageType":        "APPL",
            "CFBundleSignature":          "????",
            # High-DPI Retina display support
            "NSHighResolutionCapable":    True,
            # Privacy descriptions — macOS 12+ requires these for entitlements
            "NSCameraUsageDescription":   "Motion analysis requires camera access.",
            "NSMicrophoneUsageDescription": "Required by AVFoundation for video processing.",
            # Minimum macOS version
            "LSMinimumSystemVersion":     "12.0",
            # Prevent Dock from showing app as unresponsive during first-launch
            # model loading (MediaPipe initialisation can take 2-3 s cold)
            "NSSupportsSuddenTermination": False,
            "LSUIElement":                False,
        },
    )
