"""實際 OpenGL 驗證，必須在桌面環境執行：uv run python -m tests.check_3d_viewer。

不使用 offscreen：它無法證明點雲真的畫上 GPU framebuffer。
"""

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from kitti import KittiDataset
from kitti.KittiRenderer import KittiRenderer
from kitti.BEVRenderer import BEVRenderer
from player.BEVWindow import BEVWindow
from player.PlaybackController import PlaybackController
from player.PlayerWindow import PlayerWindow
from player.PointCloudWindow import PointCloudWindow


def pixels(image):
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.bits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


def main():
    app = QApplication([])
    renderer = KittiRenderer(KittiDataset("/mnt/d/mizu/dataset/kitti")[0])
    bev_renderer = BEVRenderer()
    controller = PlaybackController(renderer, include_cloud=True, bev_renderer=bev_renderer)
    window_2d = PlayerWindow(controller, close_controller=False)
    window = PointCloudWindow(controller)
    bev_window = BEVWindow(controller, bev_renderer)
    positions = []
    controller.position_changed.connect(lambda index, frame_id: positions.append(index))
    output = Path("outputs")
    output.mkdir(exist_ok=True)

    def wait_for(condition):
        deadline = time.monotonic() + 15
        while not condition() and time.monotonic() < deadline:
            app.processEvents()
            # 讓 Python worker 有機會取得 GIL，同時繼續處理 GUI 事件。
            time.sleep(0.01)
        assert condition(), "Timed out waiting for frame"

    def capture(name):
        app.processEvents()
        image = window.view.grabFramebuffer()
        assert not image.isNull(), "No OpenGL framebuffer"
        data = pixels(image)
        # 區分真正彩色點雲與只有暗背景／灰色格線的空畫面。
        colored = np.ptp(data[:, :, :3].astype(int), axis=2) > 40
        assert colored.sum() > 2000, "Point cloud is blank"
        assert window.grab().save(str(output / name))
        return data

    def drag(button, start, end):
        for event_type, position in (
            (QEvent.Type.MouseButtonPress, start), (QEvent.Type.MouseMove, end),
            (QEvent.Type.MouseButtonRelease, end),
        ):
            event = QMouseEvent(
                event_type, QPointF(*position), QPointF(*position),
                button if event_type != QEvent.Type.MouseMove else Qt.MouseButton.NoButton,
                button if event_type != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(window.view, event)

    window_2d.show()
    window.show()
    bev_window.show()
    try:
        wait_for(lambda: bool(positions))
        assert window.view.isValid(), "OpenGL context is invalid"
        assert len(window.scene.cloud.pos) > 100000
        initial = capture("qt_3d.png")
        azimuth = window.view.opts["azimuth"]
        drag(Qt.MouseButton.LeftButton, (250, 250), (300, 270))
        assert window.view.opts["azimuth"] != azimuth
        rotated = capture("qt_3d_rotated.png")
        assert np.mean(np.abs(initial.astype(float) - rotated.astype(float))) > 1
        center = window.view.opts["center"]
        drag(Qt.MouseButton.MiddleButton, (250, 250), (280, 270))
        assert window.view.opts["center"] != center
        window.origin_button.click()
        assert window.view.opts["center"].length() == 0
        distance = window.view.opts["distance"]
        wheel = QWheelEvent(
            QPointF(200, 200), QPointF(200, 200),
            QPoint(0, 0), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(window.view, wheel)
        assert window.view.opts["distance"] < distance
        window.reset_button.click()
        assert window.view.opts["distance"] == 65
        window.play_button.click()
        wait_for(lambda: positions[-1] >= 3)
        controller.pause()
        window.slider.setValue(11)
        wait_for(lambda: positions[-1] == 11)
        assert window_2d.slider.value() == 11
        assert bev_window.slider.value() == 11
        assert not bev_window._image.isNull()
        assert bev_window.grab().save(str(output / "qt_bev.png"))
        bev_window.resize(640, 420)
        app.processEvents()
        assert bev_window.grab().save(str(output / "qt_bev_small.png"))
        window.fps_spin.setValue(12)
        assert window_2d.fps_spin.value() == 12
        assert bev_window.fps_spin.value() == 12
        assert np.array_equal(window.scene.cloud.pos, renderer.render_with_cloud(11)[1])
        window.resize(640, 420)
        capture("qt_3d_small.png")
        assert window.slider.width() > 0
        window_2d.close()
        assert controller._thread.isRunning()
        window.next_button.click()
        wait_for(lambda: positions[-1] == 12)
        assert bev_window.slider.value() == 12
        print("3D pixels, controls, resize, and all three synchronized viewers passed")
    finally:
        window_2d.close()
        window.close()
        bev_window.close()
        controller.close()
    assert not controller._thread.isRunning()


if __name__ == "__main__":
    main()
