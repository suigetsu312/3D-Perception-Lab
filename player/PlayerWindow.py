"""主類管理 GUI，組合 controller；GUI 不處理投影或播放計時。"""

from PySide6.QtCore import Qt, QSignalBlocker, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDoubleSpinBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QSizePolicy, QSlider, QStyle, QToolButton, QVBoxLayout, QWidget,
)

from .SequenceSelector import SequenceSelector


class PlayerWindow(QMainWindow):
    seek_requested = Signal(int)

    def __init__(self, controller, title="KITTI LiDAR Player", close_controller=True, image_signal=None, autoload=True):
        super().__init__()
        self.controller = controller
        self.close_controller = close_controller
        self.image_signal = image_signal if image_signal is not None else controller.frame_ready
        self._image = QImage()
        self.setWindowTitle(title)
        self.resize(1280, 600)
        self.setMinimumSize(640, 360)
        self._build_ui()
        self._connect_signals()
        self._set_playing(False)
        # 等訊號連線和 event loop 都準備好，才讀第一張影格。
        if autoload:
            QTimer.singleShot(0, controller.load)

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.sequence_combo = SequenceSelector(self.controller, self)
        if self.controller.sequence_ids:
            sequences = QHBoxLayout()
            sequences.addWidget(self.sequence_combo)
            sequences.addStretch()
            layout.addLayout(sequences)
        self.image_label = QLabel("Loading...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setStyleSheet("background: #141414; color: #eeeeee;")
        layout.addWidget(self.image_label, 1)
        controls = QHBoxLayout()
        self.previous_button = self._button(QStyle.StandardPixmap.SP_MediaSkipBackward, "Previous Frame")
        self.play_button = self._button(QStyle.StandardPixmap.SP_MediaPlay, "Play")
        self.next_button = self._button(QStyle.StandardPixmap.SP_MediaSkipForward, "Next Frame")
        for button in (self.previous_button, self.play_button, self.next_button):
            controls.addWidget(button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.controller.frame_ids) - 1)
        controls.addWidget(self.slider, 1)
        self.position_label = QLabel("Frame --")
        self.position_label.setMinimumWidth(170)
        controls.addWidget(self.position_label)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, max(60.0, self.controller.fps))
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSuffix(" FPS")
        self.fps_spin.setValue(self.controller.fps)
        controls.addWidget(self.fps_spin)
        layout.addLayout(controls)

    def _button(self, icon, tooltip):
        button = QToolButton(self)
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFixedSize(36, 36)
        return button

    def _connect_signals(self):
        # GUI 發出使用者意圖，controller 的 slot 處理播放邏輯。
        self.play_button.clicked.connect(self.controller.toggle)
        self.previous_button.clicked.connect(self.controller.previous)
        self.next_button.clicked.connect(self.controller.next)
        self.slider.sliderPressed.connect(self.controller.pause)
        self.slider.valueChanged.connect(self.seek_requested)
        self.seek_requested.connect(self.controller.seek)
        self.fps_spin.valueChanged.connect(self.controller.set_fps)
        self.image_signal.connect(self._show_frame)
        self.controller.fps_changed.connect(self._show_fps)
        self.controller.position_changed.connect(self._show_position)
        self.controller.sequence_changed.connect(self._show_sequence)
        self.controller.playing_changed.connect(self._set_playing)
        self.controller.error.connect(self._show_error)
        self._shortcuts = []
        for key, slot in (
            ("Space", self.controller.toggle), ("Left", self.controller.previous),
            ("Right", self.controller.next),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    @Slot(QImage)
    def _show_frame(self, image):
        self._image = image
        self._resize_image()

    @Slot(float)
    def _show_fps(self, fps):
        with QSignalBlocker(self.fps_spin):
            self.fps_spin.setValue(fps)

    def _resize_image(self):
        if not self._image.isNull():
            # QPixmap 與 QWidget 只能在 GUI thread 更新，縮放維持相機比例。
            pixmap = QPixmap.fromImage(self._image).scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(pixmap)

    @Slot(int, int)
    def _show_position(self, index, frame_id):
        # 防止程式更新 slider 又觸發 seek，造成自動播放被自己暫停。
        with QSignalBlocker(self.slider):
            if not self.slider.isSliderDown():
                self.slider.setValue(index)
        self.position_label.setText(f"Frame {frame_id:06d} | {index + 1}/{len(self.controller.frame_ids)}")
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.controller.frame_ids) - 1)

    @Slot(int)
    def _show_sequence(self, sequence_id):
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, len(self.controller.frame_ids) - 1)
            self.slider.setValue(0)
        self.position_label.setText("Frame --")
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(len(self.controller.frame_ids) > 1)

    @Slot(bool)
    def _set_playing(self, playing):
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))
        self.play_button.setToolTip("Pause" if playing else "Play")
        self.play_button.setAccessibleName(self.play_button.toolTip())

    @Slot(str)
    def _show_error(self, message):
        QMessageBox.critical(self, "Frame Loading Failed", message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_image()

    def closeEvent(self, event):
        # 共用 controller 時，關閉其中一個視窗不停止另一個視窗。
        if self.close_controller:
            self.controller.close()
        event.accept()
