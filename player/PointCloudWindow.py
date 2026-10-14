"""3D 主視窗：GUI 留在此類，場景功能組合 PointCloudScene。"""

import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, QSignalBlocker, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMainWindow,
    QSlider, QStyle, QToolButton, QVBoxLayout, QWidget,
)

from .PointCloudScene import PointCloudScene
from .SequenceSelector import SequenceSelector


class PointCloudWindow(QMainWindow):
    seek_requested = Signal(int)

    def __init__(self, controller, point_size=2.0):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("KITTI | 3D LiDAR")
        self.resize(1100, 760)
        self.setMinimumSize(640, 420)
        self._build_ui(point_size)
        self._connect_signals()
        self._show_playing(controller.playing)

    def _button(self, icon, tooltip):
        button = QToolButton(self)
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFixedSize(36, 36)
        return button

    def _build_ui(self, point_size):
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.sequence_combo = SequenceSelector(self.controller, self)
        if self.controller.sequence_ids:
            sequences = QHBoxLayout()
            sequences.addWidget(self.sequence_combo)
            sequences.addStretch()
            layout.addLayout(sequences)
        tools = QHBoxLayout()
        self.reset_button = self._button(QStyle.StandardPixmap.SP_BrowserReload, "Reset View")
        self.origin_button = self._button(QStyle.StandardPixmap.SP_DirHomeIcon, "Focus LiDAR Origin")
        tools.addWidget(self.reset_button)
        tools.addWidget(self.origin_button)
        self.axes_checkbox = QCheckBox("XYZ")
        self.axes_checkbox.setChecked(True)
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(True)
        self.ransac_checkbox = QCheckBox("RANSAC")
        self.ransac_checkbox.setToolTip("Highlight ground plane in green (distance <= 0.15 m)")
        tools.addWidget(self.axes_checkbox)
        tools.addWidget(self.grid_checkbox)
        tools.addWidget(self.ransac_checkbox)
        tools.addStretch()
        self.count_label = QLabel("0 points")
        tools.addWidget(self.count_label)
        tools.addWidget(QLabel("Size"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(1, 10)
        self.size_spin.setValue(point_size)
        self.size_spin.setSuffix(" px")
        tools.addWidget(self.size_spin)
        layout.addLayout(tools)
        self.view = gl.GLViewWidget()
        # 套件提供左鍵旋轉、中鍵或 Ctrl+左鍵平移、滾輪縮放。
        # OpenGL 資料更新也必須在 GUI thread 的 slot 內進行。
        self.scene = PointCloudScene(self.view, point_size, self)
        layout.addWidget(self.view, 1)
        playback = QHBoxLayout()
        self.previous_button = self._button(QStyle.StandardPixmap.SP_MediaSkipBackward, "Previous Frame")
        self.play_button = self._button(QStyle.StandardPixmap.SP_MediaPlay, "Play")
        self.next_button = self._button(QStyle.StandardPixmap.SP_MediaSkipForward, "Next Frame")
        for button in (self.previous_button, self.play_button, self.next_button):
            playback.addWidget(button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.controller.frame_ids) - 1)
        playback.addWidget(self.slider, 1)
        self.position_label = QLabel("Frame --")
        self.position_label.setMinimumWidth(165)
        playback.addWidget(self.position_label)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, max(60.0, self.controller.fps))
        self.fps_spin.setDecimals(1)
        self.fps_spin.setValue(self.controller.fps)
        self.fps_spin.setSuffix(" FPS")
        playback.addWidget(self.fps_spin)
        layout.addLayout(playback)
        self.statusBar().showMessage("LiDAR origin (0, 0, 0) | X forward, Y left, Z up | meters")

    def _connect_signals(self):
        self.reset_button.clicked.connect(self.scene.reset_view)
        self.origin_button.clicked.connect(self.scene.focus_origin)
        self.axes_checkbox.toggled.connect(self.scene.set_axes_visible)
        self.grid_checkbox.toggled.connect(self.scene.set_grid_visible)
        self.ransac_checkbox.toggled.connect(self.scene.set_ransac_enabled)
        self.size_spin.valueChanged.connect(self.scene.set_point_size)
        self.play_button.clicked.connect(self.controller.toggle)
        self.previous_button.clicked.connect(self.controller.previous)
        self.next_button.clicked.connect(self.controller.next)
        self.slider.sliderPressed.connect(self.controller.pause)
        self.slider.valueChanged.connect(self.seek_requested)
        self.seek_requested.connect(self.controller.seek)
        self.fps_spin.valueChanged.connect(self.controller.set_fps)
        self.controller.cloud_ready.connect(self.scene.set_cloud)
        self.controller.cloud_ready.connect(self._show_count)
        self.controller.position_changed.connect(self._show_position)
        self.controller.sequence_changed.connect(self._show_sequence)
        self.controller.playing_changed.connect(self._show_playing)
        self.controller.fps_changed.connect(self._show_fps)
        self.controller.error.connect(self._show_error)

    @Slot(object, object)
    def _show_count(self, points, colors):
        self.count_label.setText(f"{len(points):,} points")

    @Slot(float)
    def _show_fps(self, fps):
        with QSignalBlocker(self.fps_spin):
            self.fps_spin.setValue(fps)

    @Slot(int, int)
    def _show_position(self, index, frame_id):
        with QSignalBlocker(self.slider):
            if not self.slider.isSliderDown():
                self.slider.setValue(index)
        self.position_label.setText(f"Frame {frame_id:06d} | {index + 1}/{len(self.controller.frame_ids)}")
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.controller.frame_ids) - 1)
        # 另一個視窗也能調整 FPS；更新這邊時阻擋訊號，避免回送。
        with QSignalBlocker(self.fps_spin):
            self.fps_spin.setValue(self.controller.fps)

    @Slot(int)
    def _show_sequence(self, sequence_id):
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, len(self.controller.frame_ids) - 1)
            self.slider.setValue(0)
        self.position_label.setText("Frame --")
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(len(self.controller.frame_ids) > 1)

    @Slot(bool)
    def _show_playing(self, playing):
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))
        self.play_button.setToolTip("Pause" if playing else "Play")

    @Slot(str)
    def _show_error(self, message):
        # 兩個視窗共用 worker，避免同一錯誤彈出兩個對話框。
        self.statusBar().showMessage(message)
