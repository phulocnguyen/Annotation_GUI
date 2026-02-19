"""
Main GUI application for patient imaging navigation.
Multi-modality viewer for ECG, Angiography, and Echocardiography.
Includes interactive ECG 12-lead grid with single-lead mode.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from dataloader import PatientDataLoader
from visualizer import PatientVisualizer
from viewers.angio_viewer import AngioViewerWidget
from viewers.ecg_viewer import ECGViewerWidget
from viewers.echo_viewer import EchoViewerWidget


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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.setWindowTitle("Patient Imaging Navigation")
        self.setMinimumSize(900, 600)
        QtCore.QTimer.singleShot(0, self._fit_to_screen)

        self.data_loader = PatientDataLoader(data_dir)
        self.visualizer = PatientVisualizer()
        self.patients = self.data_loader.list_patients()
        self.current_patient = None
        self.current_modality_id = None
        self.formats = self.data_loader.get_modalities()

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
            QComboBox#SpeedCombo { background: #007AFF; color: #ffffff; border: 1px solid #007AFF; padding: 4px 7px; border-radius: 8px; min-width: 50px; }
            QComboBox#SpeedCombo::drop-down { border: none; width: 10px; }
            QComboBox#SpeedCombo QAbstractItemView { background: #0a57c2; color: #ffffff; selection-background-color: #0f66e0; selection-color: #ffffff; }
            QSlider#FrameSlider::groove:horizontal { border: 1px solid #cccccc; height: 6px; background: #e8e8e8; border-radius: 3px; }
            QSlider#FrameSlider::handle:horizontal { background: #007AFF; width: 12px; margin: -3px 0; border-radius: 6px; }
            QLabel#FormatTitle { font-size: 18px; font-weight: 600; color: #000000; }
            QLabel#FormatDetail { color: #000000; }
            QFrame#Sidebar { background: #f9f9f9; border-right: 1px solid #cccccc; }
            QGraphicsView#ImageFrame { background: #000000; border: 1px solid #222222; border-radius: 16px; }
            QLabel#ImageFrame { background: #ffffff; border: 1px solid #cccccc; border-radius: 16px; }
            QLabel#EchoFrame { background: #000000; border: 1px solid #222222; border-radius: 16px; }
            QFrame#EchoOverlayPanel { background: rgba(0, 0, 0, 170); border: 1px solid #111111; border-radius: 10px; }
            QLabel#EchoOverlayLabel { color: #ffffff; font-weight: 600; }
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
        self.patient_subtitle = QtWidgets.QLabel("Please select a patient.")
        self.patient_subtitle.setObjectName("HeroSubtitle")

        self.patient_list = QtWidgets.QListWidget()
        self.patient_list.setSpacing(8)

        self._populate_patient_list()

        self.patient_list.itemClicked.connect(self.on_patient_selected)
        layout.addWidget(title)
        layout.addWidget(self.patient_subtitle)
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
        header.addWidget(self.patient_title)

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
            item.setData(QtCore.Qt.ItemDataRole.UserRole, format_data["id"])
            self.format_list.addItem(item)
        self.format_list.currentRowChanged.connect(self.on_format_changed)
        self.format_list.setVisible(False)

        self.angio_viewer = AngioViewerWidget()
        self.echo_viewer = EchoViewerWidget(self.visualizer)
        self.ecg_viewer = ECGViewerWidget()
        self.ecg_viewer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.content_stack = QtWidgets.QStackedWidget()
        self.content_stack.addWidget(self.angio_viewer)  # Angio (zoom)
        self.content_stack.addWidget(self.echo_viewer)   # Echo (with controls)
        self.content_stack.addWidget(self.ecg_viewer)

        viewer.addWidget(self.content_stack, 1)

        layout.addLayout(header)
        layout.addLayout(viewer, 1)
        return page

    def _populate_patient_list(self) -> None:
        self.patients = self.data_loader.list_patients()
        self.patient_list.clear()
        if not self.patients:
            self.patient_subtitle.setText("No patient folders found in data directory.")
            return

        self.patient_subtitle.setText(f"Please select a patient ({len(self.patients)} found).")
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

    def _update_modality_availability(self, patient_id: str) -> None:
        available = set(self.data_loader.available_modalities(patient_id))
        for i, format_data in enumerate(self.formats):
            enabled = format_data["id"] in available
            self.modality_buttons[i].setEnabled(enabled)

    def on_patient_selected(self, item):
        patient_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if patient_id == "separator":
            return
        if patient_id is None:
            patient_id = item.text().replace("👤 ", "").strip()
        self.current_patient = patient_id
        self.patient_title.setText(f"Patient: {patient_id.upper()}")
        self.patient_title.setVisible(True)
        self._update_modality_availability(patient_id)
        self.stack.setCurrentWidget(self.viewer_page)
        selected_modality = False
        for i, button in enumerate(self.modality_buttons):
            if button.isEnabled():
                self.set_modality(i)
                selected_modality = True
                break
        if not selected_modality:
            self.format_title.setText("No available modality")
            self.format_detail.setText("No compatible data found for this patient.")
            self.angio_viewer.show_placeholder("No compatible data found for this patient")
            self.content_stack.setCurrentWidget(self.angio_viewer)

    def set_modality(self, row: int):
        if row < 0 or row >= len(self.formats):
            return
        if not self.modality_buttons[row].isEnabled():
            return
        self.format_list.blockSignals(True)
        self.format_list.setCurrentRow(row)
        self.format_list.blockSignals(False)
        self.on_format_changed(row)

    def on_format_changed(self, row):
        if self.current_patient is None:
            return
        if row < 0 or row >= len(self.formats):
            return

        self.echo_viewer.stop_video()

        format_data = self.formats[row]
        modality_id = format_data["id"]
        modality_label = format_data["label"]
        self.current_modality_id = modality_id
        self.format_title.setText(modality_label)
        self.format_detail.setText(format_data["detail"])
        if hasattr(self, "modality_buttons"):
            for i, button in enumerate(self.modality_buttons):
                button.setChecked(i == row)

        if modality_id == "ecg":
            self._display_ecg(self.current_patient)
        elif modality_id == "angio":
            self._display_angio(self.current_patient)
        elif modality_id == "echo":
            self._display_echo(self.current_patient)
        else:
            self._display_generic_modality(self.current_patient, modality_id, modality_label)

    def _display_ecg(self, patient_id: str):
        ecg_data = self.data_loader.load_ecg(patient_id)
        if ecg_data is None:
            self.angio_viewer.show_placeholder("ECG data not found")
            self.content_stack.setCurrentWidget(self.angio_viewer)
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
            self.angio_viewer.set_image_file(temp_file)
            self.content_stack.setCurrentWidget(self.angio_viewer)

            size = metadata.get("size", metadata.get("shape", (0, 0)))
            self.format_detail.setText(f"12-lead ECG - {size[0]}×{size[1]} pixels")
        except Exception as e:
            self.angio_viewer.show_placeholder(f"Error displaying ECG: {str(e)}")
            self.content_stack.setCurrentWidget(self.angio_viewer)

    def _display_angio(self, patient_id: str):
        angio_data = self.data_loader.load_angio(patient_id)
        if angio_data is None:
            self.angio_viewer.show_placeholder("Angiography image not found")
            self.content_stack.setCurrentWidget(self.angio_viewer)
            return

        data, metadata = angio_data
        try:
            temp_file = self.visualizer.frame_to_temp_file(data)
            self.angio_viewer.set_image_file(temp_file)
            self.content_stack.setCurrentWidget(self.angio_viewer)
            self.format_detail.setText(
                f"X-ray imaging - {metadata['size'][0]}×{metadata['size'][1]} pixels"
            )
        except Exception as e:
            self.angio_viewer.show_placeholder(f"Error displaying Angiography: {str(e)}")
            self.content_stack.setCurrentWidget(self.angio_viewer)

    def _display_echo(self, patient_id: str):
        echo_data = self.data_loader.load_echo(patient_id)
        if echo_data is None:
            self.echo_viewer.show_placeholder("Echocardiography video not found")
            self.content_stack.setCurrentWidget(self.echo_viewer)
            return

        frames, metadata = echo_data
        try:
            if frames and len(frames) > 0:
                fps = metadata.get("fps", 30) or 30
                if fps <= 0:
                    fps = 30
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
                self.echo_viewer.set_echo_data(frames, metadata)
                self.content_stack.setCurrentWidget(self.echo_viewer)
            else:
                self.echo_viewer.show_placeholder("Echocardiography video empty")
                self.content_stack.setCurrentWidget(self.echo_viewer)
        except Exception as e:
            self.echo_viewer.show_placeholder(f"Error displaying Echo: {str(e)}")
            self.content_stack.setCurrentWidget(self.echo_viewer)

    def _display_generic_modality(self, patient_id: str, modality_id: str, modality_label: str):
        payload = self.data_loader.load_modality(patient_id, modality_id)
        if payload is None:
            self.angio_viewer.show_placeholder(f"{modality_label} data not found")
            self.content_stack.setCurrentWidget(self.angio_viewer)
            return

        data, metadata = payload
        try:
            if isinstance(data, np.ndarray) and data.ndim in (2, 3):
                temp_file = self.visualizer.frame_to_temp_file(data)
                self.angio_viewer.set_image_file(temp_file)
                self.content_stack.setCurrentWidget(self.angio_viewer)
                size = metadata.get("size", metadata.get("shape", (0, 0)))
                self.format_detail.setText(f"{modality_label} - {size[0]}×{size[1]}")
                return

            if isinstance(data, list) and data and isinstance(data[0], np.ndarray):
                self.echo_viewer.set_echo_data(data, metadata)
                self.content_stack.setCurrentWidget(self.echo_viewer)
                fps = metadata.get("fps", 30) or 30
                self.format_detail.setText(f"{modality_label} - {len(data)} frames @ {fps:.1f} fps")
                return

            self.angio_viewer.show_placeholder(
                f"{modality_label} loaded, but no renderer is defined for type {type(data).__name__}"
            )
            self.content_stack.setCurrentWidget(self.angio_viewer)
        except Exception as e:
            self.angio_viewer.show_placeholder(f"Error displaying {modality_label}: {str(e)}")
            self.content_stack.setCurrentWidget(self.angio_viewer)

    def show_patients(self):
        self._populate_patient_list()
        self.stack.setCurrentWidget(self.patient_page)
        self.echo_viewer.reset()
        self.patient_title.setVisible(False)

    def save_all_ecg_as_images(self, output_format: str = "png") -> dict:
        results = {}
        for patient_id in self.data_loader.list_patients():
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
