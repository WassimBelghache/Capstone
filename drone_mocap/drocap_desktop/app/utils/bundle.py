"""
Bundle / resource-path utilities
=================================
Works identically in three launch modes:
  1. Terminal:   poetry run python main.py  (CWD = drocap_desktop/)
  2. PyInstaller bundle (onedir):  dist/DroCap Mission Control/
  3. macOS .app double-click:      bundle's MacOS/ directory

Public API
----------
resource_path(relative)  -> absolute path to a bundled asset
setup_bundle_env()       -> call once at process start (before any heavy import)
IS_BUNDLE                -> True when running from a PyInstaller bundle
"""
from __future__ import annotations

import os
import sys

# ── Bundle detection ──────────────────────────────────────────────────────────
IS_BUNDLE: bool = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

# PyInstaller 5: _MEIPASS == dist/<name>/
# PyInstaller 6: _MEIPASS == dist/<name>/_internal/
# Either way _MEIPASS is where bundled data files land.
if IS_BUNDLE:
    _BASE = sys._MEIPASS  # type: ignore[attr-defined]
else:
    # Walk up from app/utils/ → app/ → drocap_desktop/ (the project root)
    _BASE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def resource_path(relative: str) -> str:
    """
    Return the absolute path to a resource file.

    In dev mode:  <drocap_desktop>/<relative>
    In bundle:    <_MEIPASS>/<relative>

    Example
    -------
    resource_path("assets/logo.png")
    resource_path("drone_mocap/modules/pose_landmark/pose_landmark_full.tflite")
    """
    return os.path.join(_BASE, relative)


def setup_bundle_env() -> None:
    """
    Bootstrap the process environment for a PyInstaller bundle.

    Must be called at the very top of main.py, before importing any
    app-specific module.  Safe to call in dev mode (no-op for most steps).

    What it does
    ------------
    1. Injects _MEIPASS into sys.path[0] so that ``import app.*`` and
       ``import drone_mocap.*`` resolve against the bundle's extracted tree
       rather than whatever the OS has set as CWD.

    2. Changes the working directory to _MEIPASS so that MediaPipe's C++
       calculator framework (which resolves .binarypb / .tflite paths relative
       to CWD or the mediapipe package directory) can locate its graph files.
       This is the primary fix for the "infinite loading" symptom when
       double-clicking the macOS .app bundle.

    3. Sets QT_MAC_WANTS_LAYER=1 to prevent a common macOS Cocoa/Metal
       rendering stall that can look like an infinite load.

    4. On macOS, suppresses Gatekeeper-related quarantine slowdowns by
       telling AVFoundation to use a minimal plugin scan.
    """
    if IS_BUNDLE:
        meipass = sys._MEIPASS  # type: ignore[attr-defined]

        # 1. Ensure bundle root is first on sys.path
        if meipass not in sys.path:
            sys.path.insert(0, meipass)

        # 2. Change CWD so MediaPipe C++ graph resolution works
        os.chdir(meipass)

        # 3. Qt rendering layer hint (macOS Metal/Cocoa)
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

        # 4. Suppress slow AVFoundation plugin enumeration
        os.environ.setdefault("QT_AVFOUNDATION_OVERRIDE_PLUGINS", "0")

    else:
        # Dev mode: ensure drocap_desktop/ is on sys.path so ``app.*`` works
        # regardless of which directory the user launches from.
        dev_root = _BASE
        if dev_root not in sys.path:
            sys.path.insert(0, dev_root)
