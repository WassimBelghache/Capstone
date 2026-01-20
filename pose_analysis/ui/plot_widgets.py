from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import List


class AnglePlotCanvas(FigureCanvas):
    """Base class for angle plotting with smoothing."""

    def __init__(self, title: str, ylabel: str, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.tight_layout(pad=3.0)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

        self.title = title
        self.ylabel = ylabel
        self.left_data: List[float] = []
        self.right_data: List[float] = []
        self.max_points = 500

    def update_plot(self, left_value: float, right_value: float, ylim: tuple = (0, 180)):
        """Update plot with new data point."""
        from config import ProcessingConfig

        self.left_data.append(left_value)
        self.right_data.append(right_value)

        # Limit data length
        if len(self.left_data) > self.max_points:
            self.left_data = self.left_data[-self.max_points:]
            self.right_data = self.right_data[-self.max_points:]

        # Smooth data
        left_smooth = self._smooth(
            self.left_data, ProcessingConfig.GRAPH_SMOOTHING_WINDOW
        )
        right_smooth = self._smooth(
            self.right_data, ProcessingConfig.GRAPH_SMOOTHING_WINDOW
        )

        # Redraw
        self.ax.cla()
        self.ax.plot(left_smooth, label="Left", color="red", linewidth=2)
        self.ax.plot(right_smooth, label="Right", color="magenta", linewidth=2)

        self.ax.set_xlabel("Frame", fontsize=11)
        self.ax.set_ylabel(self.ylabel, fontsize=11)
        self.ax.set_title(self.title, fontsize=13, fontweight="bold")
        self.ax.legend(loc="upper right", fontsize=9)
        self.ax.grid(True, alpha=0.3)

        # Dynamic y-limits
        all_data = left_smooth + right_smooth
        if all_data:
            ymin = max(ylim[0], min(all_data) - 5)
            ymax = min(ylim[1], max(all_data) + 5)
            self.ax.set_ylim(ymin, ymax)

        self.draw()

    def _smooth(self, data: List[float], window: int) -> List[float]:
        """Apply moving average smoothing."""
        if len(data) < 2 or window <= 1:
            return data

        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start : i + 1]
            smoothed.append(sum(window_data) / len(window_data))
        return smoothed

    def clear(self):
        """Clear all data."""
        self.left_data.clear()
        self.right_data.clear()
        self.ax.cla()
        self.draw()


class HipAngleCanvas(AnglePlotCanvas):
    """Canvas for hip angles (adduction or flexion)."""

    def __init__(self, parent=None, width=5, height=3, dpi=100):
        super().__init__(
            title="Hip Angles Over Time",
            ylabel="Angle (degrees)",
            parent=parent,
            width=width,
            height=height,
            dpi=dpi,
        )

    def update_frontal(self, left_adduction: float, right_adduction: float):
        """Update for frontal view."""
        self.title = "Hip Adduction Over Time"
        self.update_plot(left_adduction, right_adduction, ylim=(0, 60))

    def update_sagittal(self, left_flexion: float, right_flexion: float):
        """Update for sagittal view."""
        self.title = "Hip Flexion Over Time"
        self.update_plot(left_flexion, right_flexion, ylim=(0, 180))


class KneeAngleCanvas(AnglePlotCanvas):
    """Canvas for knee angles (valgus or flexion)."""

    def __init__(self, parent=None, width=5, height=3, dpi=100):
        super().__init__(
            title="Knee Angles Over Time",
            ylabel="Angle (degrees)",
            parent=parent,
            width=width,
            height=height,
            dpi=dpi,
        )

    def update_frontal(self, left_valgus: float, right_valgus: float):
        """Update for frontal view."""
        self.title = "Knee Valgus Over Time"
        self.update_plot(left_valgus, right_valgus, ylim=(0, 90))

    def update_sagittal(self, left_flexion: float, right_flexion: float):
        """Update for sagittal view."""
        self.title = "Knee Flexion Over Time"
        self.update_plot(left_flexion, right_flexion, ylim=(0, 180))


class AnkleAngleCanvas(AnglePlotCanvas):
    """Canvas for ankle flexion (sagittal view only)."""

    def __init__(self, parent=None, width=5, height=2.5, dpi=100):
        super().__init__(
            title="Ankle Flexion Over Time",
            ylabel="Angle (degrees)",
            parent=parent,
            width=width,
            height=height,
            dpi=dpi,
        )

    def update_sagittal(self, left_flexion: float, right_flexion: float):
        """Update for sagittal view."""
        self.update_plot(left_flexion, right_flexion, ylim=(0, 180))
