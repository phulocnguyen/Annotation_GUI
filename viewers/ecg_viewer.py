from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


class ECGViewerWidget(QtWidgets.QWidget):
    """Interactive ECG viewer with grid and single-lead modes."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.current_mode: str = "grid"
        self.selected_lead: Optional[int] = None
        self.signal: Optional[np.ndarray] = None
        self.max_time: int = 2000
        self._single_ax = None
        self._single_initial_xlim: Optional[tuple[float, float]] = None
        self._single_initial_ylim: Optional[tuple[float, float]] = None
        self._pan_anchor: Optional[tuple[QtCore.QPointF, tuple[float, float], tuple[float, float]]] = None

        self.figure = Figure(constrained_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMouseTracking(True)
        self.canvas.installEventFilter(self)

        self.back_button = QtWidgets.QPushButton("Return")
        self.back_button.setObjectName("ECGBackButton")
        self.back_button.setVisible(False)
        self.back_button.clicked.connect(self.show_grid_mode)

        self.zoom_label = QtWidgets.QLabel("Zoom: 100%")
        self.zoom_label.setVisible(False)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)
        top_bar.addWidget(self.back_button, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        top_bar.addStretch(1)
        top_bar.addWidget(self.zoom_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        top_bar.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(top_bar)
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self._grid_axes = []

    def set_signal(self, signal, max_time: int = 2000) -> None:
        if hasattr(signal, "detach"):
            signal = signal.detach().cpu().numpy()
        signal = np.asarray(signal)
        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError("signal must have shape (12, T)")

        self.signal = signal
        self.max_time = max_time
        self.show_grid_mode()

    def show_grid_mode(self) -> None:
        if self.signal is None:
            return
        self.current_mode = "grid"
        self.selected_lead = None
        self._single_ax = None
        self._single_initial_xlim = None
        self._single_initial_ylim = None
        self.back_button.setVisible(False)
        self.back_button.setEnabled(False)
        self.zoom_label.setVisible(False)
        self._pan_anchor = None

        self._render_grid()

    def show_single_mode(self, lead_index: int) -> None:
        if self.signal is None:
            return
        if lead_index < 0 or lead_index >= 12:
            return

        self.current_mode = "single"
        self.selected_lead = lead_index
        self.back_button.setVisible(True)
        self.back_button.setEnabled(True)
        self.zoom_label.setVisible(True)

        self._render_single()

    def on_click(self, event) -> None:
        if self.current_mode != "grid":
            return
        if event.inaxes is None:
            return

        for i, ax in enumerate(self._grid_axes):
            if event.inaxes == ax:
                self.show_single_mode(i)
                return

    def eventFilter(self, source, event):
        if source is self.canvas and self.current_mode == "single" and self._single_ax is not None:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton and self._point_in_single_axes(event.position()):
                    self._pan_anchor = (
                        event.position(),
                        self._single_ax.get_xlim(),
                        self._single_ax.get_ylim(),
                    )
                    self.canvas.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                if self._pan_anchor is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
                    self._pan_single_view(event.position())
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                if event.button() == QtCore.Qt.MouseButton.LeftButton and self._pan_anchor is not None:
                    self._pan_anchor = None
                    self.canvas.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                    return True
        return super().eventFilter(source, event)

    def _point_in_single_axes(self, position: QtCore.QPointF) -> bool:
        if self._single_ax is None:
            return False
        bbox = self._single_ax.bbox
        x = position.x()
        y = self.canvas.height() - position.y()
        return bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1

    def _pan_single_view(self, position: QtCore.QPointF) -> None:
        if (
            self._pan_anchor is None
            or self._single_ax is None
            or self._single_initial_xlim is None
            or self._single_initial_ylim is None
        ):
            return

        anchor_pos, anchor_xlim, anchor_ylim = self._pan_anchor
        bbox = self._single_ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return

        dx_pixels = position.x() - anchor_pos.x()
        dy_pixels = position.y() - anchor_pos.y()
        x_per_pixel = (anchor_xlim[1] - anchor_xlim[0]) / bbox.width
        y_per_pixel = (anchor_ylim[1] - anchor_ylim[0]) / bbox.height

        proposed_xlim = (
            anchor_xlim[0] - dx_pixels * x_per_pixel,
            anchor_xlim[1] - dx_pixels * x_per_pixel,
        )
        proposed_ylim = (
            anchor_ylim[0] + dy_pixels * y_per_pixel,
            anchor_ylim[1] + dy_pixels * y_per_pixel,
        )
        bounded_xlim = self._bounded_range(proposed_xlim, self._single_initial_xlim)
        bounded_ylim = self._bounded_range(proposed_ylim, self._single_initial_ylim)

        self._single_ax.set_xlim(*bounded_xlim)
        self._single_ax.set_ylim(*bounded_ylim)
        self._update_zoom_label()
        self.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        if self.current_mode != "single" or self._single_ax is None:
            return
        if event.inaxes != self._single_ax:
            return
        if self._single_initial_xlim is None or self._single_initial_ylim is None:
            return

        scale_factor = 1 / 1.2 if event.button == "up" else 1.2

        current_xlim = self._single_ax.get_xlim()
        current_ylim = self._single_ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else sum(current_xlim) / 2
        ydata = event.ydata if event.ydata is not None else sum(current_ylim) / 2

        new_xlim = self._zoom_range(current_xlim, xdata, scale_factor)
        new_ylim = self._zoom_range(current_ylim, ydata, scale_factor)
        bounded_xlim = self._bounded_range(new_xlim, self._single_initial_xlim)
        bounded_ylim = self._bounded_range(new_ylim, self._single_initial_ylim)

        self._single_ax.set_xlim(*bounded_xlim)
        self._single_ax.set_ylim(*bounded_ylim)
        self._update_zoom_label()
        self.canvas.draw_idle()

    def _zoom_range(
        self, current_range: tuple[float, float], center: float, scale_factor: float
    ) -> tuple[float, float]:
        start, end = current_range
        return (
            center - (center - start) * scale_factor,
            center + (end - center) * scale_factor,
        )

    def _bounded_range(
        self, new_range: tuple[float, float], initial_range: tuple[float, float]
    ) -> tuple[float, float]:
        new_start, new_end = new_range
        initial_start, initial_end = initial_range
        initial_span = initial_end - initial_start
        new_span = new_end - new_start

        min_span = initial_span / 20
        if new_span < min_span:
            center = (new_start + new_end) / 2
            new_start = center - min_span / 2
            new_end = center + min_span / 2
            new_span = min_span

        if new_span >= initial_span:
            return initial_range

        if new_start < initial_start:
            new_end += initial_start - new_start
            new_start = initial_start
        if new_end > initial_end:
            new_start -= new_end - initial_end
            new_end = initial_end

        return new_start, new_end

    def _update_zoom_label(self) -> None:
        if self._single_ax is None or self._single_initial_xlim is None:
            self.zoom_label.setText("Zoom: 100%")
            return
        initial_span = self._single_initial_xlim[1] - self._single_initial_xlim[0]
        current_xlim = self._single_ax.get_xlim()
        current_span = current_xlim[1] - current_xlim[0]
        if current_span <= 0:
            zoom_ratio = 1.0
        else:
            zoom_ratio = initial_span / current_span
        self.zoom_label.setText(f"Zoom: {zoom_ratio * 100:.0f}%")

    def _render_grid(self) -> None:
        if self.signal is None:
            return
        self.figure.clear()
        self._grid_axes = []

        t_len = min(self.signal.shape[1], self.max_time)
        axes = self.figure.subplots(6, 2, sharex=True)

        for i in range(12):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            ax.plot(self.signal[i, :t_len], linewidth=1.0)
            ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=25)
            ax.grid(alpha=0.3)
            self._grid_axes.append(ax)

        axes[-1, -1].set_xlabel("Time (samples)")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_single(self) -> None:
        if self.signal is None or self.selected_lead is None:
            return
        lead_index = self.selected_lead
        self.figure.clear()
        t_len = min(self.signal.shape[1], self.max_time)

        ax = self.figure.add_subplot(1, 1, 1)
        ax.plot(self.signal[lead_index, :t_len], linewidth=1.2)
        ax.set_title(f"Lead {LEAD_NAMES[lead_index]}")
        ax.set_ylabel(LEAD_NAMES[lead_index], rotation=0, labelpad=25)
        ax.set_xlabel("Time (samples)")
        ax.grid(alpha=0.3)
        self._single_ax = ax

        self.figure.tight_layout()
        self._single_initial_xlim = ax.get_xlim()
        self._single_initial_ylim = ax.get_ylim()
        self._update_zoom_label()
        self.canvas.draw_idle()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.signal is None:
            return
        if self.current_mode == "grid":
            QtCore.QTimer.singleShot(0, self._render_grid)
        elif self.current_mode == "single":
            QtCore.QTimer.singleShot(0, self._render_single)
