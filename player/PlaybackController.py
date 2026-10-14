"""控制播放狀態；不建立 GUI 元件。"""

import math

from PySide6.QtCore import QObject, QElapsedTimer, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage


class RenderWorker(QObject):
    frame_ready = Signal(int, int, QImage, object, object, QImage)
    failed = Signal(int, int, str)

    def __init__(self, renderer, include_cloud=False, bev_renderer=None, include_image=True):
        super().__init__()
        self.renderer = renderer
        self.include_cloud = include_cloud
        self.bev_renderer = bev_renderer
        self.include_image = include_image

    @staticmethod
    def _to_qimage(image):
        if image is None:
            return QImage()
        height, width = image.shape[:2]
        # copy 讓 QImage 擁有資料，安全地跨執行緒傳送。
        return QImage(image.data, width, height, image.strides[0], QImage.Format.Format_BGR888).copy()

    @Slot(int, int, object)
    def render(self, generation, index, renderer):
        """在背景執行緒讀檔、投影，不接觸 QWidget/QPixmap。"""
        try:
            frame_id = renderer.frame_ids[index]
            if self.include_cloud or self.bev_renderer is not None:
                if self.include_image:
                    image, points, colors = renderer.render_with_cloud(frame_id)
                else:
                    image, points, colors = renderer.render_with_cloud(frame_id, include_image=False)
            else:
                image = renderer.render(frame_id)
                points, colors = None, None
            # QImage 原本引用 NumPy 記憶體；copy 讓它擁有資料，
            # 避免函式返回後陣列被回收，GUI 收到失效的影像。
            qt_image = self._to_qimage(image)
            bev_image = self._to_qimage(self.bev_renderer.render(points)) if self.bev_renderer is not None else QImage()
            # NumPy 陣列發送後不再修改；跨 thread 接收者與訊號持有資料參考。
            self.frame_ready.emit(generation, index, qt_image, points, colors, bev_image)
        except Exception as error:
            self.failed.emit(generation, index, str(error))


class PlaybackController(QObject):
    frame_ready = Signal(QImage)
    cloud_ready = Signal(object, object)  # N×3 xyz、N×4 RGBA
    bev_ready = Signal(QImage)
    fps_changed = Signal(float)
    position_changed = Signal(int, int)  # 序列索引、實際 frame ID
    playing_changed = Signal(bool)
    error = Signal(str)
    sequence_changed = Signal(int)
    render_requested = Signal(int, int, object)

    def __init__(self, renderer, fps=10.0, parent=None, include_cloud=False, bev_renderer=None, include_image=True,
                 sequence_ids=(), sequence_id=None, renderer_factory=None):
        super().__init__(parent)
        self._renderer = renderer
        self._generation = 0
        self.sequence_ids = list(sequence_ids)
        self.sequence_id = sequence_id
        self._renderer_factory = renderer_factory
        self.frame_ids = renderer.frame_ids.copy()
        self.index = 0
        self.playing = False
        self._busy = False
        self._closed = False
        self._displayed_index = None
        self._elapsed = QElapsedTimer()
        # 收到影格後才排下一張，慢機器也不會累積無限工作佇列。
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self.set_fps(fps)
        self._thread = QThread(self)
        self._worker = RenderWorker(renderer, include_cloud, bev_renderer, include_image)
        # worker 必須沒有 parent 才能搬移；QThread 物件本身仍屬於主執行緒。
        self._worker.moveToThread(self._thread)
        self.render_requested.connect(self._worker.render)
        self._worker.frame_ready.connect(self._receive_frame)
        self._worker.failed.connect(self._receive_error)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    @Slot()
    def load(self):
        self._request_frame()

    def _request_frame(self):
        if self._closed or self._busy:
            return
        self._busy = True
        self._elapsed.start()
        self.render_requested.emit(self._generation, self.index, self._renderer)

    @Slot(int, int, QImage, object, object, QImage)
    def _receive_frame(self, generation, index, image, points, colors, bev_image):
        self._busy = False
        if self._closed:
            return
        if generation != self._generation or index != self.index:
            # 拖曳只更新目標 index，丟棄舊結果，再處理最新目標。
            self._request_frame()
            return
        self._displayed_index = index
        if not image.isNull():
            self.frame_ready.emit(image)
        if points is not None:
            # 舊影格已在上方排除，2D 與 3D 只收到同一個被接受的影格。
            self.cloud_ready.emit(points, colors)
        if not bev_image.isNull():
            self.bev_ready.emit(bev_image)
        self.position_changed.emit(index, self.frame_ids[index])
        if self.playing:
            self._schedule_next()

    def _schedule_next(self):
        if self.index == len(self.frame_ids) - 1:
            self.pause()
        else:
            # 扣掉處理時間，讓間隔接近指定 FPS；處理太慢時不跳幀。
            self._timer.start(max(1, round(1000 / self.fps) - self._elapsed.elapsed()))

    @Slot(int, int, str)
    def _receive_error(self, generation, index, message):
        self._busy = False
        if self._closed:
            return
        if generation != self._generation or index != self.index:
            self._request_frame()
            return
        self.pause()
        self.error.emit(message)

    @Slot(int)
    def select_sequence(self, sequence_id):
        if self._closed or sequence_id == self.sequence_id:
            return
        self.pause()
        try:
            renderer = self._renderer_factory(sequence_id)
        except Exception as error:
            self.sequence_changed.emit(self.sequence_id)
            self.error.emit(str(error))
            return
        self._renderer = renderer
        self.frame_ids = renderer.frame_ids.copy()
        self.sequence_id = sequence_id
        self._generation += 1
        self.index = 0
        self._displayed_index = None
        self.sequence_changed.emit(sequence_id)
        self._request_frame()

    @Slot()
    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    @Slot()
    def play(self):
        if self._closed or self.playing:
            return
        self.playing = True
        self.playing_changed.emit(True)
        if self.index == len(self.frame_ids) - 1:
            self.index = 0
            self._request_frame()
        elif self._displayed_index == self.index and not self._busy:
            self._elapsed.start()
            self._schedule_next()
        else:
            self._request_frame()

    @Slot()
    def pause(self):
        self._timer.stop()
        if self.playing:
            self.playing = False
            self.playing_changed.emit(False)

    @Slot(int)
    def seek(self, index):
        if self._closed:
            return
        self.pause()
        self.index = max(0, min(index, len(self.frame_ids) - 1))
        self._request_frame()

    @Slot()
    def previous(self):
        self.seek(self.index - 1)

    @Slot()
    def next(self):
        self.seek(self.index + 1)

    @Slot()
    def _advance(self):
        if self.playing:
            self.index += 1
            self._request_frame()

    @Slot(float)
    def set_fps(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive")
        self.fps = fps
        self.fps_changed.emit(fps)
        if self.playing and self._timer.isActive():
            self._schedule_next()

    @Slot()
    def close(self):
        """停止 timer，等待當前工作完成，避免執行中的 thread 被銷毀。"""
        if self._closed:
            return
        self.pause()
        self._closed = True
        self._thread.quit()
        self._thread.wait()
