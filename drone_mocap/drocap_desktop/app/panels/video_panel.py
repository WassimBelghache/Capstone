"""
VideoPanel — right-top quadrant.

Responsibilities:
  • Load and play back the source video via QMediaPlayer / QVideoWidget
  • Transport controls: play/pause, seek slider
  • Emit time_changed(float seconds) as the video plays
  • Receive seek_to_time(float seconds) from ChartPanel cursor
  • Show live skeleton overlay during analysis (before video is loaded)
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QTimer
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QStackedWidget,
    QSizePolicy,
)

from app.panels.control_panel import SkeletonPreview  # shared widget


class VideoPanel(QWidget):
    time_changed = pyqtSignal(float)   # emitted while playing

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._duration_ms: int = 0
        self._seeking = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Stacked: index 0 = skeleton preview, index 1 = video widget
        self._stack = QStackedWidget()
        self._skeleton_preview = SkeletonPreview()
        self._video_widget = QVideoWidget()
        self._video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._stack.addWidget(self._skeleton_preview)   # index 0
        self._stack.addWidget(self._video_widget)       # index 1
        self._stack.setCurrentIndex(0)
        layout.addWidget(self._stack, stretch=1)

        # Transport controls
        transport = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(60)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.sliderPressed.connect(self._on_slider_press)
        self._slider.sliderReleased.connect(self._on_slider_release)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._time_label = QLabel("0.00 s")
        self._time_label.setFixedWidth(60)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        transport.addWidget(self._play_btn)
        transport.addWidget(self._slider)
        transport.addWidget(self._time_label)
        layout.addLayout(transport)

        # Media player
        self._player = QMediaPlayer()
        self._audio  = QAudioOutput()
        self._audio.setVolume(0.0)       # muted by default (analysis videos have no useful audio)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_video(self, path: str) -> None:
        self._player.setSource(QUrl.fromLocalFile(path))
        self._play_btn.setEnabled(True)
        self._stack.setCurrentIndex(1)

    def update_skeleton(
        self,
        frame_bgr: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        """Called during analysis pass 1 — show live skeleton overlay on canvas."""
        if self._stack.currentIndex() == 0:
            self._skeleton_preview.update_frame(frame_bgr, xy, vis)

    def seek_to_time(self, time_s: float) -> None:
        """Seek the player to the given time (from ChartPanel cursor)."""
        if self._duration_ms <= 0:
            return
        ms = int(time_s * 1000)
        ms = max(0, min(ms, self._duration_ms))
        self._seeking = True
        self._player.setPosition(ms)
        self._seeking = False

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = duration_ms

    def _on_position_changed(self, pos_ms: int) -> None:
        if self._seeking or self._duration_ms <= 0:
            return
        t = pos_ms / 1000.0
        self._time_label.setText(f"{t:.2f} s")
        pct = int(pos_ms / self._duration_ms * 1000)
        self._slider.blockSignals(True)
        self._slider.setValue(pct)
        self._slider.blockSignals(False)
        if not self._seeking:
            self.time_changed.emit(t)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("Pause" if playing else "Play")

    def _on_slider_press(self) -> None:
        self._seeking = True

    def _on_slider_release(self) -> None:
        if self._duration_ms > 0:
            pos_ms = int(self._slider.value() / 1000 * self._duration_ms)
            self._player.setPosition(pos_ms)
        self._seeking = False

    def _on_slider_moved(self, value: int) -> None:
        if self._duration_ms > 0:
            t = value / 1000 * self._duration_ms / 1000.0
            self._time_label.setText(f"{t:.2f} s")

    def set_cursor(self, time_s: float) -> None:
        """Alias expected by main_window wiring."""
        self.seek_to_time(time_s)
