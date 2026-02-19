from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
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

        self.figure = Figure(constrained_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.toolbar.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        self._configure_toolbar()
        self.toolbar.setVisible(False)

        self.back_button = QtWidgets.QPushButton("Return")
        self.back_button.setObjectName("ECGBackButton")
        self.back_button.setVisible(False)
        self.back_button.clicked.connect(self.show_grid_mode)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)
        top_bar.addWidget(self.back_button, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        top_bar.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(top_bar)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self._grid_axes = []

    def _configure_toolbar(self) -> None:
        """Reduce toolbar size and hide unsupported actions."""
        for action in list(self.toolbar.actions()):
            text = (action.text() or "").lower()
            if any(keyword in text for keyword in ("save", "subplot", "customize", "edit")):
                self.toolbar.removeAction(action)

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
        self.back_button.setVisible(False)
        self.back_button.setEnabled(False)
        self.toolbar.setVisible(False)
        self.toolbar.setEnabled(False)

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
        self.toolbar.setVisible(True)
        self.toolbar.setEnabled(True)

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

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.signal is None:
            return
        if self.current_mode == "grid":
            QtCore.QTimer.singleShot(0, self._render_grid)
        elif self.current_mode == "single":
            QtCore.QTimer.singleShot(0, self._render_single)
