from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets


class ImageViewer(QtWidgets.QGraphicsView):
    """Simple image viewer with wheel zoom."""
    zoom_changed = QtCore.pyqtSignal(float)
    metrics_changed = QtCore.pyqtSignal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.scene().setSceneRect(QtCore.QRectF(pixmap.rect()))
        self._zoom_ratio = 1.0
        self.resetTransform()
        self._fit_image()
        self.zoom_changed.emit(1.0)
        self._emit_metrics()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self._pixmap_item.pixmap().isNull() and self._zoom_ratio == 1.0:
            self._fit_image()
        self._emit_metrics()

    def _fit_image(self) -> None:
        self.fitInView(self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self._pixmap_item)
        self._emit_metrics()

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
        self._emit_metrics()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._emit_metrics()

    def get_display_metrics(self) -> dict:
        pixmap = self._pixmap_item.pixmap()
        viewport_rect = self.viewport().rect()
        viewport_global = self.viewport().mapToGlobal(viewport_rect.topLeft())
        metrics = {
            "viewport_size": (viewport_rect.width(), viewport_rect.height()),
            "viewport_global_offset": (viewport_global.x(), viewport_global.y()),
            "source_size": (pixmap.width(), pixmap.height()) if not pixmap.isNull() else (0, 0),
            "display_rect": (0.0, 0.0, 0.0, 0.0),
            "offset": (0.0, 0.0),
            "scene_rect": (0.0, 0.0, 0.0, 0.0),
            "scroll_offset": (
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            ),
            "zoom_ratio": self._zoom_ratio,
            "transform_scale": (self.transform().m11(), self.transform().m22()),
        }
        if pixmap.isNull():
            return metrics

        scene_rect = QtCore.QRectF(self._pixmap_item.sceneBoundingRect())
        view_polygon = self.mapFromScene(scene_rect)
        display_rect = view_polygon.boundingRect()
        metrics["display_rect"] = (
            float(display_rect.x()),
            float(display_rect.y()),
            float(display_rect.width()),
            float(display_rect.height()),
        )
        metrics["offset"] = (float(display_rect.x()), float(display_rect.y()))
        metrics["scene_rect"] = (
            float(scene_rect.x()),
            float(scene_rect.y()),
            float(scene_rect.width()),
            float(scene_rect.height()),
        )
        return metrics

    def _emit_metrics(self) -> None:
        self.metrics_changed.emit(self.get_display_metrics())


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
        self.top_bar = QtWidgets.QWidget()
        self.top_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.top_bar.setFixedHeight(44)
        self.metrics_label = QtWidgets.QLabel("Display: -")
        self.metrics_label.setObjectName("ImageMetricsLabel")
        self.metrics_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.metrics_label.setWordWrap(False)
        self.metrics_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.image_viewer.metrics_changed.connect(self._update_metrics_label)

        header = QtWidgets.QHBoxLayout(self.top_bar)
        header.setContentsMargins(0, 4, 0, 4)
        header.setSpacing(8)
        header.addWidget(self.metrics_label, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.top_bar)
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

    def get_display_metrics(self) -> dict:
        return self.image_viewer.get_display_metrics()

    def _update_metrics_label(self, metrics: dict) -> None:
        src_w, src_h = metrics["source_size"]
        vp_w, vp_h = metrics["viewport_size"]
        vp_gx, vp_gy = metrics["viewport_global_offset"]
        disp_x, disp_y, disp_w, disp_h = metrics["display_rect"]
        scroll_x, scroll_y = metrics["scroll_offset"]
        scale_x, _ = metrics["transform_scale"]
        self.metrics_label.setText(
            f"Source: {src_w} x {src_h} px    Viewport: {vp_w} x {vp_h} px    "
            f"Viewport Offset: ({vp_gx}, {vp_gy})    "
            f"Rect: ({disp_x:.1f}, {disp_y:.1f}, {disp_w:.1f}, {disp_h:.1f})    "
            f"Offset: ({disp_x:.1f}, {disp_y:.1f})    Scroll: ({scroll_x}, {scroll_y})    "
            f"Scale: {scale_x:.4f}"
        )
