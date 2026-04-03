from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from visualizer import PatientVisualizer


class EchoViewerWidget(QtWidgets.QWidget):
    """Echocardiography viewer with playback controls and overlay."""

    def __init__(self, visualizer: PatientVisualizer, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.visualizer = visualizer
        self.temp_files: list[str] = []

        self.current_echo_frames = None
        self.current_echo_metadata = None
        self.current_frame_index = 0

        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self.play_next_frame)
        self.is_video_playing = False

        self.playback_speed = 1.0
        self.echo_fps = 30.0

        self.echo_label = QtWidgets.QLabel()
        self.echo_label.setObjectName("EchoFrame")
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

        self.echo_container = QtWidgets.QWidget()
        echo_stack = QtWidgets.QStackedLayout(self.echo_container)
        echo_stack.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)
        echo_stack.setContentsMargins(0, 0, 0, 0)
        echo_stack.addWidget(self.echo_label)

        self.echo_overlay_root = QtWidgets.QWidget()
        overlay_layout = QtWidgets.QVBoxLayout(self.echo_overlay_root)
        overlay_layout.setContentsMargins(12, 12, 12, 12)
        overlay_layout.setSpacing(10)

        metrics_row = QtWidgets.QHBoxLayout()
        metrics_row.addStretch(1)
        self.echo_metrics_panel = QtWidgets.QFrame()
        self.echo_metrics_panel.setObjectName("EchoOverlayPanel")
        self.echo_metrics_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        metrics_panel_layout = QtWidgets.QVBoxLayout(self.echo_metrics_panel)
        metrics_panel_layout.setContentsMargins(12, 10, 12, 10)
        metrics_panel_layout.setSpacing(0)

        self.echo_metrics_label = QtWidgets.QLabel("Display : -")
        self.echo_metrics_label.setObjectName("EchoOverlayLabel")
        self.echo_metrics_label.setWordWrap(True)
        self.echo_metrics_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        metrics_font = self.echo_metrics_label.font()
        metrics_font.setPointSize(metrics_font.pointSize() + 2)
        self.echo_metrics_label.setFont(metrics_font)
        self.echo_metrics_label.setMinimumWidth(240)
        self.echo_metrics_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        metrics_panel_layout.addWidget(self.echo_metrics_label)
        metrics_row.addWidget(self.echo_metrics_panel)
        overlay_layout.addLayout(metrics_row)

        overlay_layout.addStretch(1)
        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.addStretch(1)
        self.echo_overlay_panel = QtWidgets.QFrame()
        self.echo_overlay_panel.setObjectName("EchoOverlayPanel")
        self.echo_overlay_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        panel_layout = QtWidgets.QGridLayout(self.echo_overlay_panel)
        panel_layout.setContentsMargins(8, 6, 8, 6)
        panel_layout.setHorizontalSpacing(8)
        panel_layout.setVerticalSpacing(6)

        self.echo_frame_title = QtWidgets.QLabel("Frame :")
        self.echo_frame_title.setObjectName("EchoOverlayLabel")
        self.echo_speed_title = QtWidgets.QLabel("Speed :")
        self.echo_speed_title.setObjectName("EchoOverlayLabel")
        overlay_title_font = self.echo_frame_title.font()
        overlay_title_font.setPointSize(overlay_title_font.pointSize() + 5)
        self.echo_frame_title.setFont(overlay_title_font)
        self.echo_speed_title.setFont(overlay_title_font)

        self.echo_frame_value = QtWidgets.QWidget()
        frame_value_layout = QtWidgets.QHBoxLayout(self.echo_frame_value)
        frame_value_layout.setContentsMargins(0, 0, 0, 0)
        frame_value_layout.setSpacing(2)
        self.echo_frame_current_label = QtWidgets.QLabel("0")
        self.echo_frame_current_label.setObjectName("EchoOverlayLabel")
        self.echo_frame_current_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.echo_frame_slash_label = QtWidgets.QLabel("/")
        self.echo_frame_slash_label.setObjectName("EchoOverlayLabel")
        self.echo_frame_total_label = QtWidgets.QLabel("0")
        self.echo_frame_total_label.setObjectName("EchoOverlayLabel")
        self.echo_frame_total_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        overlay_value_font = self.echo_frame_current_label.font()
        overlay_value_font.setPointSize(overlay_value_font.pointSize() + 5)
        self.echo_frame_current_label.setFont(overlay_value_font)
        self.echo_frame_slash_label.setFont(overlay_value_font)
        self.echo_frame_total_label.setFont(overlay_value_font)
        frame_value_layout.addWidget(self.echo_frame_current_label)
        frame_value_layout.addWidget(self.echo_frame_slash_label)
        frame_value_layout.addWidget(self.echo_frame_total_label)

        self.echo_speed_value_label = QtWidgets.QLabel("0x")
        self.echo_speed_value_label.setObjectName("EchoOverlayLabel")
        self.echo_speed_value_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.echo_speed_value_label.setFont(overlay_value_font)

        panel_layout.addWidget(self.echo_frame_title, 0, 0)
        panel_layout.addWidget(self.echo_frame_value, 0, 1)
        panel_layout.addWidget(self.echo_speed_title, 1, 0)
        panel_layout.addWidget(self.echo_speed_value_label, 1, 1)
        panel_layout.setColumnStretch(1, 1)
        overlay_row.addWidget(self.echo_overlay_panel)
        overlay_layout.addLayout(overlay_row)
        echo_stack.addWidget(self.echo_overlay_root)
        self.echo_overlay_root.setVisible(False)

        self.controls_bar = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(self.controls_bar)
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(0, 4, 0, 4)
        self.controls_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.controls_bar.setFixedHeight(44)
        self.controls_bar.setVisible(True)

        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.setObjectName("SpeedCombo")
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)

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

        controls_layout.addWidget(self.speed_combo)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.controls_bar)
        layout.addWidget(self.echo_container, 1)

    def reset(self) -> None:
        self.stop_video()
        self.current_echo_frames = None
        self.current_echo_metadata = None

    def set_echo_data(self, frames, metadata) -> None:
        if frames and len(frames) > 0:
            self.current_echo_frames = frames
            self.current_echo_metadata = metadata
            self.current_frame_index = 0

            fps = metadata.get("fps", 30) or 30
            if fps <= 0:
                fps = 30
            self.echo_fps = fps
            self._apply_playback_speed()

            self.frame_slider.setMaximum(len(frames) - 1)
            self.frame_slider.setValue(0)
            self._update_echo_overlay_widths(len(frames))
            self.update_frame_label()

            self.display_frame(0)

            self.is_video_playing = False
            self.play_pause_button.setText("Play")
            self.play_pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.frame_slider.setEnabled(True)
            self._show_video_controls()
        else:
            self.show_placeholder("Echocardiography video empty")
            self.play_pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            self._show_video_controls()

    def show_placeholder(self, message: str) -> None:
        pixmap = QtGui.QPixmap(520, 360)
        pixmap.fill(QtGui.QColor("#fffdf7"))
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor("#6b665e"))
        painter.setFont(QtGui.QFont("Arial", 10))
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, message)
        painter.end()
        self.echo_label.setPixmap(pixmap)
        self._echo_pixmap_orig = pixmap
        self._update_display_metrics()
        self._show_video_controls()

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

    def display_frame(self, frame_index: int) -> None:
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

    def toggle_video_playback(self) -> None:
        if self.current_echo_frames is None or len(self.current_echo_frames) == 0:
            return

        if self.is_video_playing:
            self.video_timer.stop()
            self.is_video_playing = False
            self.play_pause_button.setText("Play")
        else:
            self._apply_playback_speed()
            self.video_timer.start()
            self.is_video_playing = True
            self.play_pause_button.setText("Pause")

    def stop_video(self) -> None:
        if self.video_timer.isActive():
            self.video_timer.stop()
        self.is_video_playing = False
        if self.current_echo_frames is not None and len(self.current_echo_frames) > 0:
            self.current_frame_index = 0
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.display_frame(0)
            self.play_pause_button.setText("Play")
            self.update_frame_label()

    def on_frame_slider_moved(self, value: int) -> None:
        if self.current_echo_frames is None:
            return
        self.current_frame_index = value
        self.display_frame(value)
        self.update_frame_label()

    def on_frame_slider_pressed(self) -> None:
        if self.is_video_playing:
            self.video_timer.stop()
            self.is_video_playing = False
            self.play_pause_button.setText("Play")

    def on_frame_slider_released(self) -> None:
        pass

    def on_speed_changed(self, text: str) -> None:
        try:
            self.playback_speed = float(text.replace("x", ""))
        except ValueError:
            self.playback_speed = 1.0
        self._update_echo_overlay_widths(len(self.current_echo_frames) if self.current_echo_frames else 0)
        self._update_echo_overlay_text()
        if self.is_video_playing:
            self._apply_playback_speed()

    def _apply_playback_speed(self) -> None:
        if self.echo_fps:
            frame_interval = int(1000 / (self.echo_fps * self.playback_speed))
            self.video_timer.setInterval(frame_interval)

    def update_frame_label(self) -> None:
        if self.current_echo_frames is not None:
            display_index = min(self.current_frame_index + 1, len(self.current_echo_frames))
            self.frame_label.setText(
                f"{display_index}/{len(self.current_echo_frames)}"
            )
            self._update_echo_overlay_text()

    def _update_echo_overlay_text(self) -> None:
        total = len(self.current_echo_frames) if self.current_echo_frames else 0
        display_index = min(self.current_frame_index + 1, total) if total else 0
        speed_text = self.speed_combo.currentText() if hasattr(self, "speed_combo") else f"{self.playback_speed:g}x"
        self.echo_frame_current_label.setText(str(display_index))
        self.echo_frame_total_label.setText(str(total))
        self.echo_speed_value_label.setText(speed_text)

    def _update_echo_overlay_widths(self, total_frames: int) -> None:
        label_width = max(
            self.echo_frame_title.fontMetrics().horizontalAdvance("Frame :"),
            self.echo_speed_title.fontMetrics().horizontalAdvance("Speed :"),
        )
        self.echo_frame_title.setFixedWidth(label_width + 2)
        self.echo_speed_title.setFixedWidth(label_width + 2)

        total_digits = max(1, len(str(total_frames)))
        max_num = "9" * total_digits
        num_width = self.echo_frame_current_label.fontMetrics().horizontalAdvance(max_num)
        self.echo_frame_current_label.setFixedWidth(num_width + 2)
        self.echo_frame_total_label.setFixedWidth(num_width + 2)

        speed_items = [self.speed_combo.itemText(i) for i in range(self.speed_combo.count())]
        speed_longest = max(speed_items, key=len) if speed_items else f"{self.playback_speed:g}x"
        speed_width = self.echo_speed_value_label.fontMetrics().horizontalAdvance(speed_longest)
        self.echo_speed_value_label.setFixedWidth(speed_width + 2)

    def _show_video_controls(self) -> None:
        self.speed_combo.setVisible(True)
        self.play_pause_button.setVisible(True)
        self.stop_button.setVisible(True)
        self.frame_slider.setVisible(True)
        self.frame_label.setVisible(True)
        self.controls_bar.setVisible(True)
        self.controls_bar.raise_()
        self.echo_overlay_root.setVisible(True)
        self.echo_overlay_root.raise_()

    def _display_echo_image_file(self, file_path: str) -> None:
        pixmap = QtGui.QPixmap(file_path)
        if not pixmap.isNull():
            self._echo_pixmap_orig = pixmap
            self._apply_echo_pixmap()
            QtCore.QTimer.singleShot(0, self._apply_echo_pixmap)
        else:
            self.show_placeholder("Failed to load image")

    def _apply_echo_pixmap(self) -> None:
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
            if pixmap.width() > label_size.width() or pixmap.height() > label_size.height():
                scaled = pixmap.scaled(
                    label_size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            else:
                scaled = pixmap
        self.echo_label.setPixmap(scaled)
        self._update_display_metrics()

    def get_display_metrics(self) -> dict:
        content_rect = self.echo_label.contentsRect()
        pixmap = self._echo_pixmap_orig
        viewport_global = self.echo_label.mapToGlobal(content_rect.topLeft())
        metrics = {
            "viewport_size": (content_rect.width(), content_rect.height()),
            "viewport_global_offset": (viewport_global.x(), viewport_global.y()),
            "label_size": (content_rect.width(), content_rect.height()),
            "source_size": (pixmap.width(), pixmap.height()) if pixmap is not None else (0, 0),
            "display_rect": (0.0, 0.0, 0.0, 0.0),
            "offset": (0.0, 0.0),
            "scale": (0.0, 0.0),
        }
        if pixmap is None or content_rect.width() <= 0 or content_rect.height() <= 0:
            return metrics

        displayed = self.echo_label.pixmap()
        if displayed is None or displayed.isNull():
            return metrics

        disp_w = displayed.width()
        disp_h = displayed.height()
        offset_x = content_rect.x() + (content_rect.width() - disp_w) / 2
        offset_y = content_rect.y() + (content_rect.height() - disp_h) / 2
        metrics["display_rect"] = (float(offset_x), float(offset_y), float(disp_w), float(disp_h))
        metrics["offset"] = (float(offset_x), float(offset_y))
        if pixmap.width() > 0 and pixmap.height() > 0:
            metrics["scale"] = (disp_w / pixmap.width(), disp_h / pixmap.height())
        return metrics

    def _update_display_metrics(self) -> None:
        metrics = self.get_display_metrics()
        src_w, src_h = metrics["source_size"]
        vp_w, vp_h = metrics["viewport_size"]
        vp_gx, vp_gy = metrics["viewport_global_offset"]
        disp_x, disp_y, disp_w, disp_h = metrics["display_rect"]
        scale_x, _ = metrics["scale"]
        self.echo_metrics_label.setText(
            f"Source: {src_w} x {src_h} px\n"
            f"Viewport: {vp_w} x {vp_h} px\n"
            f"Viewport Offset: ({vp_gx}, {vp_gy})\n"
            f"Rect: ({disp_x:.1f}, {disp_y:.1f}, {disp_w:.1f}, {disp_h:.1f})\n"
            f"Offset: ({disp_x:.1f}, {disp_y:.1f})\n"
            f"Scale: {scale_x:.4f}"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_echo_pixmap()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._apply_echo_pixmap)
