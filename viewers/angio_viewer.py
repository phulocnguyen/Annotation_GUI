from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets


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

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
