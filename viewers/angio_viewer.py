from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets


class ImageViewer(QtWidgets.QGraphicsView):
    """Simple image viewer with wheel zoom."""
    zoom_changed = QtCore.pyqtSignal(float)

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
        self._zoom_ratio = 1.0
        self._zoom_step = 0.1
        self._min_zoom_ratio = 0.1
        self._max_zoom_ratio = 4.0

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._zoom_ratio = 1.0
        self.resetTransform()
        self.fitInView(self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(1.0)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        zoom_in = event.angleDelta().y() > 0
        old_ratio = self._zoom_ratio
        next_ratio = old_ratio + (self._zoom_step if zoom_in else -self._zoom_step)
        next_ratio = round(next_ratio, 2)
        if next_ratio < self._min_zoom_ratio or next_ratio > self._max_zoom_ratio:
            return
        factor = next_ratio / old_ratio
        self.scale(factor, factor)
        self._zoom_ratio = next_ratio
        self.zoom_changed.emit(self._zoom_ratio)


class AngioViewerWidget(QtWidgets.QWidget):
    """Angiography viewer wrapper around the zoomable ImageViewer."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.image_viewer = ImageViewer()
        self.image_viewer.setObjectName("ImageFrame")
        self.image_viewer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.zoom_label = QtWidgets.QLabel("Zoom: 1.00x")
        self.zoom_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.image_viewer.zoom_changed.connect(self._update_zoom_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.image_viewer)

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        if not pixmap.isNull():
            self.image_viewer.set_image(pixmap)
        else:
            self.show_placeholder("Failed to load image")

    def set_image_file(self, file_path: str) -> None:
        pixmap = QtGui.QPixmap(file_path)
        self.set_image(pixmap)

    def show_placeholder(self, message: str) -> None:
        pixmap = QtGui.QPixmap(520, 360)
        pixmap.fill(QtGui.QColor("#fffdf7"))
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor("#6b665e"))
        painter.setFont(QtGui.QFont("Arial", 10))
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, message)
        painter.end()
        self.image_viewer.set_image(pixmap)

    def _update_zoom_label(self, ratio: float) -> None:
        self.zoom_label.setText(f"Zoom: {ratio:.2f}x")
