"""
VideoPanel — right-top quadrant.

Responsibilities:
  • Live skeleton overlay (SkeletonPreview) during analysis Pass 1
  • After analysis: load and play diagnostic.mp4 via QMediaPlayer
  • Custom Jira-styled timeline scrubber
  • Bi-directional sync with ChartPanel:
      – Video playing → emit time_changed(float s) → chart cursor follows
      – Chart cursor dragged → receive seek_to_time(float s) → video jumps
  • Slider drag: instant seek + chart cursor update via time_changed
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPalette
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QStackedWidget,
    QStyle, QVBoxLayout, QWidget,
)

from app.panels.control_panel import SkeletonPreview


# ── Custom Jira-styled scrubber slider ────────────────────────────────────────

_SLIDER_QSS = """
QSlider::groove:horizontal {
    height: 4px;
    background: #DFE1E6;
    border-radius: 2px;
    margin: 0px;
}
QSlider::handle:horizontal {
    background: #0052CC;
    border: 2px solid #0052CC;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #0065FF;
    border-color: #0065FF;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #0052CC;
    border-radius: 2px;
    height: 4px;
}
QSlider::add-page:horizontal {
    background: #DFE1E6;
    border-radius: 2px;
    height: 4px;
}
"""


class VideoPanel(QWidget):
    time_changed = pyqtSignal(float)   # seconds — consumed by ChartPanel.set_cursor

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._duration_ms: int = 0
        self._seeking: bool = False
        self._fps: float = 30.0          # updated when diagnostic.mp4 loads

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Mode label ────────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        self._mode_label = QLabel("LIVE PREVIEW")
        self._mode_label.setObjectName("section_title")
        mode_row.addWidget(self._mode_label)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ── Stack: [0] skeleton preview   [1] video widget ───────────────────
        self._stack = QStackedWidget()

        self._skeleton_preview = SkeletonPreview()
        self._skeleton_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._video_widget = QVideoWidget()
        self._video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Ensure video background is black
        pal = self._video_widget.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#000000"))
        self._video_widget.setPalette(pal)

        self._stack.addWidget(self._skeleton_preview)   # index 0
        self._stack.addWidget(self._video_widget)       # index 1
        self._stack.setCurrentIndex(0)
        layout.addWidget(self._stack, stretch=1)

        # ── Transport bar ─────────────────────────────────────────────────────
        transport = QHBoxLayout()
        transport.setSpacing(8)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(32, 32)
        self._play_btn.setEnabled(False)
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._toggle_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 10000)
        self._slider.setValue(0)
        self._slider.setStyleSheet(_SLIDER_QSS)
        self._slider.sliderPressed.connect(self._on_slider_press)
        self._slider.sliderReleased.connect(self._on_slider_release)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._time_label = QLabel("0.00 / 0.00 s")
        self._time_label.setFixedWidth(96)
        self._time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._time_label.setStyleSheet("color: #5E6C84; font-size: 11px;")

        transport.addWidget(self._play_btn)
        transport.addWidget(self._slider, stretch=1)
        transport.addWidget(self._time_label)
        layout.addLayout(transport)

        # ── Media player ──────────────────────────────────────────────────────
        self._player = QMediaPlayer()
        self._audio  = QAudioOutput()
        self._audio.setVolume(0.0)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_video(self, path: str) -> None:
        """Switch from skeleton canvas to video player and load the file."""
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(path))
        self._play_btn.setEnabled(True)
        self._stack.setCurrentIndex(1)
        self._mode_label.setText("REVIEW")
        self._mode_label.setStyleSheet("color: #0052CC; font-weight: 700;")

    def enter_live_mode(self) -> None:
        """Switch back to skeleton canvas (called at start of new analysis)."""
        self._player.stop()
        self._player.setSource(QUrl())
        self._stack.setCurrentIndex(0)
        self._play_btn.setEnabled(False)
        self._slider.setValue(0)
        self._time_label.setText("0.00 / 0.00 s")
        self._duration_ms = 0
        self._mode_label.setText("LIVE PREVIEW")
        self._mode_label.setStyleSheet("")

    def update_skeleton(
        self,
        rgb_small: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
        orig_w: int = 0,
        orig_h: int = 0,
    ) -> None:
        """Push a pre-processed RGB frame+keypoints during Pass 1."""
        if self._stack.currentIndex() == 0:
            self._skeleton_preview.update_frame(rgb_small, xy, vis, orig_w, orig_h)

    def seek_to_time(self, time_s: float) -> None:
        """Seek the player to the given time — called by ChartPanel cursor drag."""
        if self._duration_ms <= 0:
            return
        ms = int(time_s * 1000)
        ms = max(0, min(ms, self._duration_ms))
        self._seeking = True
        self._player.setPosition(ms)
        # Update slider immediately for instant feel
        self._slider.blockSignals(True)
        self._slider.setValue(int(ms / self._duration_ms * 10000))
        self._slider.blockSignals(False)
        self._time_label.setText(
            f"{time_s:.2f} / {self._duration_ms / 1000:.2f} s"
        )
        self._seeking = False

    # Alias used by main_window Chart→Video wiring
    set_cursor = seek_to_time

    # ── Internal slots ────────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = duration_ms
        total_s = duration_ms / 1000.0
        self._time_label.setText(f"0.00 / {total_s:.2f} s")

    def _on_position_changed(self, pos_ms: int) -> None:
        if self._seeking or self._duration_ms <= 0:
            return
        t = pos_ms / 1000.0
        total_s = self._duration_ms / 1000.0
        self._time_label.setText(f"{t:.2f} / {total_s:.2f} s")

        self._slider.blockSignals(True)
        self._slider.setValue(int(pos_ms / self._duration_ms * 10000))
        self._slider.blockSignals(False)

        # Notify ChartPanel — throttle to avoid feedback loop while seeking
        self.time_changed.emit(t)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("⏸" if playing else "▶")

    def _on_slider_press(self) -> None:
        self._seeking = True
        # Pause during scrub for responsive feel
        self._player.pause()

    def _on_slider_release(self) -> None:
        if self._duration_ms > 0:
            pos_ms = int(self._slider.value() / 10000 * self._duration_ms)
            self._player.setPosition(pos_ms)
            t = pos_ms / 1000.0
            # Emit so chart cursor follows the scrub point
            self.time_changed.emit(t)
        self._seeking = False

    def _on_slider_moved(self, value: int) -> None:
        """Live update time label while dragging — visual only."""
        if self._duration_ms > 0:
            t = value / 10000 * self._duration_ms / 1000.0
            total_s = self._duration_ms / 1000.0
            self._time_label.setText(f"{t:.2f} / {total_s:.2f} s")
            # Emit as user drags so chart cursor tracks in real-time
            if not self._seeking:
                return
            self.time_changed.emit(t)
