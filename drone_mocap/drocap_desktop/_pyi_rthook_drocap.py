# DroCap PyInstaller runtime hook
# Sets CWD = _MEIPASS so MediaPipe's C++ framework finds graph files.
import os, sys
if hasattr(sys, "_MEIPASS"):
    os.chdir(sys._MEIPASS)
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
