"""
Main GUI application for patient imaging navigation.
Multi-modality viewer for ECG, Angiography, and Echocardiography.
Includes interactive ECG 12-lead grid with single-lead mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from dataloader import PatientDataLoader
from visualizer import PatientVisualizer

LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


class SeparatorDelegate(QtWidgets.QStyledItemDelegate):
    """Custom delegate to render separator lines between patients."""

    def paint(self, painter, option, index):
        """Paint a separator line."""
        data = index.data(QtCore.Qt.ItemDataRole.UserRole)
        if data == "separator":
            rect = option.rect
            y = rect.top() + rect.height() // 2
            painter.setPen(QtGui.QPen(QtGui.QColor("#d0d0d0"), 1))
            painter.drawLine(rect.left() + 10, y, rect.right() - 10, y)


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
        """Reduce toolbar size, remove Save, and improve icon visibility."""
        for action in list(self.toolbar.actions()):
            text = (action.text() or "").lower()
            if "save" in text:
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

    def on_click(self, event) -> None:
        if self.current_mode != "grid":
            return
        if event.inaxes is None:
            return

        for i, ax in enumerate(self._grid_axes):
            if event.inaxes == ax:
                self.show_single_mode(i)
                return


class ImageViewer(QtWidgets.QGraphicsView):
    """Simple image viewer with wheel zoom."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self._zoom = 0

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._zoom = 0
        self.fitInView(self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        self._zoom += 1 if zoom_in else -1
        if self._zoom < -10:
            self._zoom = -10
            return
        if self._zoom > 30:
            self._zoom = 30
            return
        self.scale(factor, factor)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.setWindowTitle("Patient Imaging Navigation")
        self.setMinimumSize(900, 600)
        QtCore.QTimer.singleShot(0, self._fit_to_screen)

        self.data_loader = PatientDataLoader(data_dir)
        self.visualizer = PatientVisualizer()
        self.patients = ["p001", "p002", "p003", "p004"]
        self.current_patient = None
        self.temp_files = []
        self.current_echo_frames = None
        self.current_frame_index = 0
        self.video_timer = QtCore.QTimer()
        self.video_timer.timeout.connect(self.play_next_frame)
        self.is_video_playing = False
        self.autoplay_timer = QtCore.QTimer()
        self.autoplay_timer.setSingleShot(True)
        self.autoplay_timer.timeout.connect(self._start_echo_autoplay)
        self.current_echo_metadata = None
        self.current_modality_label = None

        self.formats = [
            {"label": "ECG", "detail": "Electrocardiogram"},
            {"label": "Cardiac Angiography", "detail": "X-ray imaging"},
            {"label": "Echocardiography", "detail": "Ultrasound imaging"},
        ]

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.patient_page = self.build_patient_page()
        self.viewer_page = self.build_viewer_page()
        self.stack.addWidget(self.patient_page)
        self.stack.addWidget(self.viewer_page)

        self.apply_theme()
        self.show_patients()

    def _fit_to_screen(self):
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.setGeometry(available)

    def apply_theme(self):
        """Apply custom stylesheet."""
        self.setStyleSheet(
            """
            QMainWindow { background: #ffffff; }
            QLabel#HeroTitle { font-size: 26px; font-weight: 600; color: #000000; }
            QLabel#HeroSubtitle { color: #000000; }
            QListWidget { background: #ffffff; border: 1px solid #cccccc; padding: 10px; }
            QListWidget::item { padding: 10px; margin-bottom: 6px; border-radius: 8px; color: #000000; }
            QListWidget::item:selected { background: #e0e0e0; color: #000000; }
            QPushButton#BackButton, QPushButton#ModalityButton { background: #f0f0f0; color: #000000; border: 1px solid #cccccc; padding: 6px 14px; border-radius: 14px; }
            QPushButton#ModalityButton:checked { background: #007AFF; color: #ffffff; border: 1px solid #007AFF; }
            QPushButton#PlayPauseButton, QPushButton#StopButton { background: #007AFF; color: #ffffff; border: none; padding: 6px 12px; border-radius: 6px; font-weight: 500; }
            QPushButton#PlayPauseButton:hover, QPushButton#StopButton:hover { background: #0051D5; }
            QPushButton#ECGBackButton { background: #0F766E; color: #ffffff; border: none; padding: 6px 12px; border-radius: 10px; font-weight: 600; }
            QPushButton#ECGBackButton:hover { background: #0B5D57; }
            QToolBar { background: #f7f7f7; border: 1px solid #d5d5d5; border-radius: 6px; }
            QToolButton { background: #0F766E; color: #ffffff; border: none; padding: 2px 6px; border-radius: 8px; font-weight: 600; }
            QToolButton:hover { background: #0B5D57; }
            QToolButton:checked { background: #0A4A46; }
            QSlider#FrameSlider::groove:horizontal { border: 1px solid #cccccc; height: 6px; background: #e8e8e8; border-radius: 3px; }
            QSlider#FrameSlider::handle:horizontal { background: #007AFF; width: 12px; margin: -3px 0; border-radius: 6px; }
            QLabel#FormatTitle { font-size: 18px; font-weight: 600; color: #000000; }
            QLabel#FormatDetail { color: #000000; }
            QFrame#Sidebar { background: #f9f9f9; border-right: 1px solid #cccccc; }
            QLabel#ImageFrame { background: #ffffff; border: 1px solid #cccccc; border-radius: 16px; }
            QLabel { color: #000000; }
            """
        )

    def build_patient_page(self):
        """Build patient selection page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Patient imaging navigation")
        title.setObjectName("HeroTitle")
        subtitle = QtWidgets.QLabel("Please select a patient.")
        subtitle.setObjectName("HeroSubtitle")

        self.patient_list = QtWidgets.QListWidget()
        self.patient_list.setSpacing(8)

        for i, patient_id in enumerate(self.patients):
            item = QtWidgets.QListWidgetItem(patient_id.upper())
            item.setSizeHint(QtCore.QSize(200, 50))
            item.setForeground(QtGui.QColor("#000000"))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, patient_id)
            self.patient_list.addItem(item)

            if i < len(self.patients) - 1:
                separator_item = QtWidgets.QListWidgetItem()
                separator_item.setSizeHint(QtCore.QSize(100, 2))
                separator_item.setFlags(separator_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                separator_item.setData(QtCore.Qt.ItemDataRole.UserRole, "separator")
                self.patient_list.addItem(separator_item)

        self.patient_list.itemClicked.connect(self.on_patient_selected)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.patient_list, 1)
        return page

    def build_viewer_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)

        self.patient_title = QtWidgets.QLabel("Patient")
        self.patient_title.setObjectName("HeroTitle")
        self.patient_title.setVisible(False)

        self.back_button = QtWidgets.QPushButton("Home")
        self.back_button.setObjectName("BackButton")
        self.back_button.clicked.connect(self.show_patients)
        header.addWidget(self.back_button)

        self.modality_buttons = []
        for i, format_data in enumerate(self.formats):
            button = QtWidgets.QPushButton(format_data["label"])
            button.setObjectName("ModalityButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked, idx=i: self.set_modality(idx))
            header.addWidget(button)
            self.modality_buttons.append(button)
        header.addStretch(1)

        viewer = QtWidgets.QVBoxLayout()
        viewer.setSpacing(8)

        self.format_title = QtWidgets.QLabel("Format")
        self.format_title.setObjectName("FormatTitle")
        self.format_title.setVisible(False)
        self.format_detail = QtWidgets.QLabel("Detail")
        self.format_detail.setObjectName("FormatDetail")
        self.format_detail.setVisible(False)

        self.format_list = QtWidgets.QListWidget()
        self.format_list.setSpacing(4)
        for format_data in self.formats:
            item = QtWidgets.QListWidgetItem(format_data["label"])
            item.setSizeHint(QtCore.QSize(180, 60))
            self.format_list.addItem(item)
        self.format_list.currentRowChanged.connect(self.on_format_changed)
        self.format_list.setVisible(False)

        self.image_viewer = ImageViewer()
        self.image_viewer.setObjectName("ImageFrame")
        self.image_viewer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.echo_label = QtWidgets.QLabel()
        self.echo_label.setObjectName("ImageFrame")
        self.echo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.echo_label.setScaledContents(False)
        self.echo_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.echo_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._echo_pixmap_orig = None

        self.ecg_viewer = ECGViewerWidget()
        self.ecg_viewer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.content_stack = QtWidgets.QStackedWidget()
        self.content_stack.addWidget(self.image_viewer)  # Angio (zoom)
        self.content_stack.addWidget(self.echo_label)    # Echo (no zoom)
        self.content_stack.addWidget(self.ecg_viewer)

        # Video control panel for echo
        self.controls_bar = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(self.controls_bar)
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.controls_bar.setMinimumHeight(44)

        self.autoplay_checkbox = QtWidgets.QCheckBox("Autoplay")
        self.autoplay_checkbox.setChecked(True)

        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        self.playback_speed = 1.0

        self.play_pause_button = QtWidgets.QPushButton("Play")
        self.play_pause_button.setObjectName("PlayPauseButton")
        self.play_pause_button.clicked.connect(self.toggle_video_playback)
        self.play_pause_button.setMaximumWidth(80)
        self.play_pause_button.setVisible(False)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setMaximumWidth(80)
        self.stop_button.setVisible(False)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setObjectName("FrameSlider")
        self.frame_slider.setVisible(False)
        self.frame_slider.sliderMoved.connect(self.on_frame_slider_moved)
        self.frame_slider.sliderPressed.connect(self.on_frame_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_frame_slider_released)

        self.frame_label = QtWidgets.QLabel("0/0")
        self.frame_label.setMaximumWidth(60)
        self.frame_label.setVisible(False)

        controls_layout.addWidget(self.autoplay_checkbox)
        controls_layout.addWidget(self.speed_combo)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_label)

        viewer.addWidget(self.content_stack, 1)
        viewer.addWidget(self.controls_bar)

        layout.addLayout(header)
        layout.addLayout(viewer, 1)
        return page

    def on_patient_selected(self, item):
        patient_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if patient_id is None:
            patient_id = item.text().replace("👤 ", "").strip()
        self.current_patient = patient_id
        self.patient_title.setText(f"Patient {patient_id}")
        self.stack.setCurrentWidget(self.viewer_page)
        self.set_modality(0)

    def set_modality(self, row: int):
        self.format_list.blockSignals(True)
        self.format_list.setCurrentRow(row)
        self.format_list.blockSignals(False)
        self.on_format_changed(row)

    def on_format_changed(self, row):
        if self.current_patient is None:
            return

        self.stop_video()

        format_data = self.formats[row]
        modality_label = format_data["label"]
        self.current_modality_label = modality_label
        self.format_title.setText(modality_label)
        self.format_detail.setText(format_data["detail"])
        if hasattr(self, "modality_buttons"):
            for i, button in enumerate(self.modality_buttons):
                button.setChecked(i == row)

        if modality_label == "ECG":
            self._hide_video_controls()
            self._display_ecg(self.current_patient)
        elif modality_label == "Cardiac Angiography":
            self._hide_video_controls()
            self._display_angio(self.current_patient)
        elif modality_label == "Echocardiography":
            self._show_video_controls()
            self._display_echo(self.current_patient)

    def _display_ecg(self, patient_id: str):
        self.current_echo_frames = None
        self.current_echo_metadata = None

        ecg_data = self.data_loader.load_ecg(patient_id)
        if ecg_data is None:
            self._show_placeholder("ECG data not found")
            return

        data, metadata = ecg_data
        try:
            # If ECG is a 12-lead signal, use interactive viewer
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == 12:
                self.ecg_viewer.set_signal(data, max_time=2000)
                self.content_stack.setCurrentWidget(self.ecg_viewer)
                size = metadata.get("shape", data.shape)
                self.format_detail.setText(f"12-lead ECG - {size[0]}×{size[1]} samples")
                return

            # Otherwise display as image
            temp_file = self.visualizer.frame_to_temp_file(data)
            self._display_angio_image_file(temp_file)
            self.content_stack.setCurrentWidget(self.image_viewer)
            self.temp_files.append(temp_file)

            size = metadata.get("size", metadata.get("shape", (0, 0)))
            self.format_detail.setText(f"12-lead ECG - {size[0]}×{size[1]} pixels")
        except Exception as e:
            self._show_placeholder(f"Error displaying ECG: {str(e)}")

    def _display_angio(self, patient_id: str):
        self.current_echo_frames = None
        self.current_echo_metadata = None

        angio_data = self.data_loader.load_angio(patient_id)
        if angio_data is None:
            self._show_placeholder("Angiography image not found")
            return

        data, metadata = angio_data
        try:
            temp_file = self.visualizer.frame_to_temp_file(data)
            self._display_angio_image_file(temp_file)
            self.content_stack.setCurrentWidget(self.image_viewer)
            self.temp_files.append(temp_file)
            self.format_detail.setText(
                f"X-ray imaging - {metadata['size'][0]}×{metadata['size'][1]} pixels"
            )
        except Exception as e:
            self._show_placeholder(f"Error displaying Angiography: {str(e)}")

    def _display_echo(self, patient_id: str):
        echo_data = self.data_loader.load_echo(patient_id)
        if echo_data is None:
            self._show_placeholder("Echocardiography video not found")
            self.play_pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            return

        frames, metadata = echo_data
        try:
            if frames and len(frames) > 0:
                self.current_echo_frames = frames
                self.current_echo_metadata = metadata
                self.current_frame_index = 0
                fps = metadata.get("fps", 30) or 30
                if fps <= 0:
                    fps = 30
                self.echo_fps = fps
                frame_interval = int(1000 / (fps * self.playback_speed))
                self.video_timer.setInterval(frame_interval)

                detail_text = f"Ultrasound imaging - {len(frames)} frames @ {fps:.1f} fps"

                if "filelist_data" in metadata:
                    filelist = metadata["filelist_data"]
                    detail_text += (
                        f"\nEF: {filelist['ef']:.2f}% | "
                        f"ESV: {filelist['esv']:.2f} | "
                        f"EDV: {filelist['edv']:.2f}"
                    )

                if "volume_tracings" in metadata:
                    tracings = metadata["volume_tracings"]
                    detail_text += f"\nVolume tracings: {len(tracings)} frames marked"

                self.format_detail.setText(detail_text)

                self.frame_slider.setMaximum(len(frames) - 1)
                self.frame_slider.setValue(0)
                self.update_frame_label()

                self.display_frame(0)
                self.content_stack.setCurrentWidget(self.echo_label)

                self.is_video_playing = False
                self.play_pause_button.setText("Play")
                self.autoplay_timer.stop()
                if self.autoplay_checkbox.isChecked():
                    self.autoplay_timer.start(1000)
                self.play_pause_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.frame_slider.setEnabled(True)
            else:
                self._show_placeholder("Echocardiography video empty")
                self.play_pause_button.setEnabled(False)
                self.stop_button.setEnabled(False)
                self.frame_slider.setEnabled(False)
        except Exception as e:
            self._show_placeholder(f"Error displaying Echo: {str(e)}")

    def play_next_frame(self):
        if self.current_echo_frames is None or len(self.current_echo_frames) == 0:
            self.video_timer.stop()
            self.is_video_playing = False
            self.play_pause_button.setText("Play")
            return

        self.display_frame(self.current_frame_index)

        self.current_frame_index += 1
        if self.current_frame_index >= len(self.current_echo_frames):
            self.current_frame_index = 0

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        self.update_frame_label()

    def display_frame(self, frame_index: int):
        if self.current_echo_frames is None or frame_index >= len(self.current_echo_frames):
            return

        frame = self.current_echo_frames[frame_index].copy()
        if self.current_echo_metadata and "volume_tracings" in self.current_echo_metadata:
            volume_tracings = self.current_echo_metadata["volume_tracings"]
            if frame_index in volume_tracings:
                frame = PatientVisualizer.draw_tracings_on_frame(
                    frame, volume_tracings[frame_index]
                )

        temp_file = self.visualizer.frame_to_temp_file(frame)
        self._display_echo_image_file(temp_file)
        if temp_file not in self.temp_files:
            self.temp_files.append(temp_file)

    def toggle_video_playback(self):
        if self.current_echo_frames is None or len(self.current_echo_frames) == 0:
            if self.current_patient:
                self._display_echo(self.current_patient)
            if self.current_echo_frames is None or len(self.current_echo_frames) == 0:
                return

        if self.is_video_playing:
            self.video_timer.stop()
            self.is_video_playing = False
            self.play_pause_button.setText("Play")
        else:
            self.autoplay_timer.stop()
            self._apply_playback_speed()
            self.video_timer.start()
            self.is_video_playing = True
            self.play_pause_button.setText("Pause")

    def _start_echo_autoplay(self):
        if self.current_modality_label != "Echocardiography":
            return
        if self.current_echo_frames is None or len(self.current_echo_frames) == 0:
            return
        if self.is_video_playing:
            return
        self.video_timer.start()
        self.is_video_playing = True
        self.play_pause_button.setText("Pause")

    def stop_video(self):
        if self.video_timer.isActive():
            self.video_timer.stop()
        self.autoplay_timer.stop()
        self.is_video_playing = False
        if self.current_echo_frames is not None and len(self.current_echo_frames) > 0:
            self.current_frame_index = 0
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.display_frame(0)
            self.play_pause_button.setText("Play")
            self.update_frame_label()

    def on_frame_slider_moved(self, value: int):
        if self.current_echo_frames is None:
            return
        self.current_frame_index = value
        self.display_frame(value)
        self.update_frame_label()

    def on_frame_slider_pressed(self):
        if self.is_video_playing:
            self.video_timer.stop()
            self.is_video_playing = False
            self.play_pause_button.setText("Play")

    def on_frame_slider_released(self):
        # Keep paused after scrubbing; user can press Play to resume
        pass

    def on_speed_changed(self, text: str):
        try:
            self.playback_speed = float(text.replace("x", ""))
        except ValueError:
            self.playback_speed = 1.0
        if self.is_video_playing:
            self._apply_playback_speed()

    def _apply_playback_speed(self):
        if hasattr(self, "echo_fps") and self.echo_fps:
            frame_interval = int(1000 / (self.echo_fps * self.playback_speed))
            self.video_timer.setInterval(frame_interval)

    def update_frame_label(self):
        if self.current_echo_frames is not None:
            self.frame_label.setText(
                f"{self.current_frame_index}/{len(self.current_echo_frames)}"
            )

    def _show_video_controls(self):
        self.autoplay_checkbox.setVisible(True)
        self.speed_combo.setVisible(True)
        self.play_pause_button.setVisible(True)
        self.stop_button.setVisible(True)
        self.frame_slider.setVisible(True)
        self.frame_label.setVisible(True)
        if hasattr(self, "controls_bar"):
            self.controls_bar.setVisible(True)
            self.controls_bar.raise_()

    def _hide_video_controls(self):
        self.autoplay_checkbox.setVisible(False)
        self.speed_combo.setVisible(False)
        self.play_pause_button.setVisible(False)
        self.stop_button.setVisible(False)
        self.frame_slider.setVisible(False)
        self.frame_label.setVisible(False)
        if hasattr(self, "controls_bar"):
            self.controls_bar.setVisible(False)

    def _display_angio_image_file(self, file_path: str):
        pixmap = QtGui.QPixmap(file_path)
        if not pixmap.isNull():
            self.image_viewer.set_image(pixmap)
        else:
            self._show_placeholder("Failed to load image")

    def _display_echo_image_file(self, file_path: str):
        pixmap = QtGui.QPixmap(file_path)
        if not pixmap.isNull():
            self._echo_pixmap_orig = pixmap
            self._apply_echo_pixmap()
        else:
            self._show_placeholder("Failed to load image")

    def _apply_echo_pixmap(self):
        if self._echo_pixmap_orig is None:
            return
        pixmap = self._echo_pixmap_orig
        label_size = self.echo_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return
        is_portrait = pixmap.height() >= pixmap.width()
        if is_portrait:
            scaled = pixmap.scaled(
                label_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        else:
            # Landscape: do not upscale, only scale down if needed
            if pixmap.width() > label_size.width() or pixmap.height() > label_size.height():
                scaled = pixmap.scaled(
                    label_size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            else:
                scaled = pixmap
        self.echo_label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if self.current_modality_label == "Echocardiography":
            self._apply_echo_pixmap()

    def _show_placeholder(self, message: str):
        pixmap = QtGui.QPixmap(520, 360)
        pixmap.fill(QtGui.QColor("#fffdf7"))
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor("#6b665e"))
        painter.setFont(QtGui.QFont("Arial", 10))
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, message)
        painter.end()
        if self.current_modality_label == "Cardiac Angiography":
            self.image_viewer.set_image(pixmap)
            self.content_stack.setCurrentWidget(self.image_viewer)
        else:
            self.echo_label.setPixmap(pixmap)
            self.content_stack.setCurrentWidget(self.echo_label)

    def show_patients(self):
        self.stack.setCurrentWidget(self.patient_page)
        self.stop_video()
        self.current_echo_frames = None
        self.current_echo_metadata = None

    def save_all_ecg_as_images(self, output_format: str = "png") -> dict:
        results = {}
        for patient_id in self.patients:
            try:
                ecg_data = self.data_loader.load_ecg(patient_id)
                if ecg_data is None:
                    print(f"⚠ ECG data not found for {patient_id}")
                    results[patient_id] = None
                    continue

                data, metadata = ecg_data
                patient_dir = Path(self.data_loader.data_dir) / patient_id
                output_filename = f"ecg.{output_format.lower()}"
                output_path = patient_dir / output_filename

                self.visualizer.save_ecg_as_image(
                    data,
                    str(output_path),
                    title=f"{patient_id} - 12-lead ECG",
                    format=output_format,
                )
                results[patient_id] = str(output_path)
            except Exception as e:
                print(f"Error saving ECG for {patient_id}: {str(e)}")
                results[patient_id] = None

        return results


def main():
    app = QtWidgets.QApplication([])
    data_dir = Path(__file__).resolve().parent / "data"
    window = MainWindow(data_dir=str(data_dir))
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
