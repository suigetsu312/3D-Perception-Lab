"""用離屏 Qt 測試訊號與播放流程，不需要桌面或 KITTI 資料。"""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from player.PlaybackController import PlaybackController
from player.PlayerWindow import PlayerWindow
from player.PointCloudWindow import PointCloudWindow
from player.BEVWindow import BEVWindow
from player.ViewerSelectionDialog import ViewerSelectionDialog
from kitti.BEVRenderer import BEVRenderer
from PySide6.QtWidgets import QDialogButtonBox


class FakeRenderer:
    # 刻意不連續，確認播放器索引沒有被誤當成資料集的 frame ID。
    frame_ids = [0, 4, 11]

    def __init__(self):
        self.fail = False

    def render(self, frame_id):
        time.sleep(0.01)
        if self.fail:
            raise OSError("missing point cloud")
        return np.full((40, 80, 3), frame_id + 20, dtype=np.uint8)

    def render_with_cloud(self, frame_id, include_image=True):
        image = self.render(frame_id) if include_image else None
        return image, np.array([[frame_id, 0, 0]], dtype=np.float32), np.ones((1, 4), dtype=np.float32)


class PlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.renderer = FakeRenderer()
        self.controller = PlaybackController(self.renderer, fps=30)
        self.positions = []
        self.errors = []
        self.controller.position_changed.connect(lambda index, frame_id: self.positions.append((index, frame_id)))
        self.controller.error.connect(self.errors.append)

    def tearDown(self):
        self.controller.close()

    def wait_for(self, condition):
        deadline = time.monotonic() + 3
        while not condition() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(condition(), "Timed out waiting for Qt signals")

    def test_play_end_replay_and_pause(self):
        self.controller.load()
        self.wait_for(lambda: self.positions == [(0, 0)])
        self.controller.play()
        self.wait_for(lambda: self.positions[-1] == (2, 11) and not self.controller.playing)
        self.controller.play()
        self.wait_for(lambda: self.positions[-1] == (0, 0))
        self.controller.pause()
        QTest.qWait(100)
        self.assertEqual(self.positions[-1], (0, 0))

    def test_seek_coalesces_pending_frames(self):
        self.controller.load()
        self.controller.seek(1)
        self.controller.seek(2)
        self.wait_for(lambda: bool(self.positions))
        self.assertEqual(self.positions, [(2, 11)])
        self.assertFalse(self.controller.playing)

    def test_sequence_selection_syncs_views_and_resets_to_zero(self):
        self.controller.close()
        second = FakeRenderer()
        second.frame_ids = [20, 24]
        self.controller = PlaybackController(
            self.renderer, include_cloud=True, bev_renderer=BEVRenderer(),
            sequence_ids=[0, 3], sequence_id=0,
            renderer_factory=lambda sequence_id: {0: self.renderer, 3: second}[sequence_id],
        )
        camera = PlayerWindow(self.controller, close_controller=False, autoload=False)
        cloud = PointCloudWindow(self.controller)
        bev = BEVWindow(self.controller, BEVRenderer())
        windows = (camera, cloud, bev)
        positions = []
        self.controller.position_changed.connect(lambda index, frame: positions.append((index, frame)))
        try:
            self.controller.seek(2)
            self.wait_for(lambda: positions == [(2, 11)])
            self.controller.play()
            cloud.sequence_combo.setCurrentIndex(1)
            self.assertFalse(self.controller.playing)
            self.assertEqual(self.controller.index, 0)
            self.assertEqual(self.controller.fps, 10)
            for window in windows:
                self.assertEqual(window.sequence_combo.currentData(), 3)
                self.assertEqual(window.slider.value(), 0)
                self.assertEqual(window.slider.maximum(), 1)
            self.wait_for(lambda: positions[-1] == (0, 20))
            self.assertEqual(camera._image.pixelColor(0, 0).red(), 40)
            self.assertEqual(cloud.scene._points[0, 0], 20)
            self.assertFalse(bev._image.isNull())
            bev.sequence_combo.setCurrentIndex(0)
            self.wait_for(lambda: positions[-1] == (0, 0))
            for window in windows:
                self.assertEqual(window.sequence_combo.currentData(), 0)
                self.assertEqual(window.slider.maximum(), 2)
        finally:
            for window in windows:
                window.close()

    def test_sequence_switch_while_loading_uses_latest_sequence(self):
        self.controller.close()
        second = FakeRenderer()
        second.frame_ids = [30]
        third = FakeRenderer()
        third.frame_ids = [40, 41]
        self.controller = PlaybackController(
            self.renderer, sequence_ids=[0, 1, 2], sequence_id=0,
            renderer_factory=lambda sequence_id: {1: second, 2: third}[sequence_id],
        )
        positions, images = [], []
        self.controller.position_changed.connect(lambda index, frame: positions.append((index, frame)))
        self.controller.frame_ready.connect(images.append)
        self.controller.load()
        self.controller.select_sequence(1)
        self.controller.select_sequence(2)
        self.wait_for(lambda: bool(positions))
        self.assertEqual(positions, [(0, 40)])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].pixelColor(0, 0).red(), 60)

    def test_error_stops_playback_and_can_retry(self):
        self.renderer.fail = True
        self.controller.play()
        self.wait_for(lambda: bool(self.errors))
        self.assertFalse(self.controller.playing)
        self.renderer.fail = False
        self.controller.load()
        self.wait_for(lambda: bool(self.positions))

    def test_window_controls_and_thread_cleanup(self):
        window = PlayerWindow(self.controller)
        window.show()
        try:
            self.wait_for(lambda: not window._image.isNull())
            window.slider.setValue(1)
            self.wait_for(lambda: self.positions[-1] == (1, 4))
            self.assertEqual(window.slider.value(), 1)
            window.fps_spin.setValue(15)
            self.assertEqual(self.controller.fps, 15)
            window.play_button.click()
            self.wait_for(lambda: self.positions[-1] == (2, 11))
            self.assertFalse(self.controller.playing)
            window.resize(640, 360)
            QTest.qWait(20)
            self.assertFalse(window.image_label.pixmap().isNull())
        finally:
            window.close()
        self.assertFalse(self.controller._thread.isRunning())

    def test_cloud_and_image_use_same_latest_frame(self):
        self.controller.close()
        self.controller = PlaybackController(self.renderer, include_cloud=True)
        clouds, images, positions = [], [], []
        self.controller.cloud_ready.connect(lambda points, colors: clouds.append(points.copy()))
        self.controller.frame_ready.connect(lambda image: images.append(image.copy()))
        self.controller.position_changed.connect(lambda index, frame_id: positions.append(frame_id))
        self.controller.load()
        self.controller.seek(2)
        self.wait_for(lambda: bool(positions))
        self.assertEqual(positions, [11])
        self.assertEqual(len(images), 1)
        self.assertEqual(len(clouds), 1)
        self.assertEqual(clouds[0][0, 0], 11)
        self.assertEqual(images[0].pixelColor(0, 0).red(), 31)

    def test_ransac_switch_recolors_current_and_next_cloud(self):
        window = PointCloudWindow(self.controller)
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        ground = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.7)))
        points = np.vstack((ground, [[0, 0, 2], [1, 1, 3]])).astype(np.float32)
        colors = np.full((len(points), 4), 0.5, dtype=np.float32)
        colors[:, 3] = 1
        original = colors.copy()
        try:
            self.controller.cloud_ready.emit(points, colors)
            self.assertFalse(window.ransac_checkbox.isChecked())
            np.testing.assert_array_equal(window.scene.cloud.color, original)
            window.ransac_checkbox.setChecked(True)
            np.testing.assert_allclose(
                window.scene.cloud.color[:len(ground), :3],
                np.tile([0.2, 0.9, 0.35], (len(ground), 1)),
            )
            np.testing.assert_array_equal(window.scene.cloud.color[-2:], original[-2:])
            np.testing.assert_array_equal(colors, original)
            # 開啟中換到無法估平面的影格，不沿用上一張的 mask。
            self.controller.cloud_ready.emit(points[-2:], colors[-2:])
            np.testing.assert_array_equal(window.scene.cloud.color, original[-2:])
            self.controller.cloud_ready.emit(points, colors)
            window.ransac_checkbox.setChecked(False)
            np.testing.assert_array_equal(window.scene.cloud.color, original)
        finally:
            window.close()

    def test_ransac_rejects_wall_and_degenerate_cloud(self):
        window = PointCloudWindow(self.controller)
        y, z = np.meshgrid(np.arange(-3, 4), np.arange(-7, 0))
        wall = np.column_stack((np.full(y.size, 2), y.ravel(), z.ravel()))
        try:
            window.ransac_checkbox.setChecked(True)
            for points in (wall, np.zeros((5, 3)), np.empty((0, 3))):
                colors = np.ones((len(points), 4), dtype=np.float32)
                self.controller.cloud_ready.emit(points, colors)
                np.testing.assert_array_equal(window.scene.cloud.color, colors)
        finally:
            window.close()

    def test_shared_controller_survives_closing_one_window(self):
        window = PlayerWindow(self.controller, close_controller=False)
        window.show()
        self.wait_for(lambda: bool(self.positions))
        window.close()
        self.assertTrue(self.controller._thread.isRunning())
        self.controller.seek(2)
        self.wait_for(lambda: self.positions[-1] == (2, 11))

    def test_bev_only_load_seek_and_shared_fps(self):
        self.controller.close()
        bev = BEVRenderer()
        self.controller = PlaybackController(self.renderer, bev_renderer=bev, include_image=False)
        window = BEVWindow(self.controller, bev)
        camera_images = []
        self.controller.frame_ready.connect(camera_images.append)
        window.show()
        try:
            self.controller.load()
            self.wait_for(lambda: not window._image.isNull())
            self.assertEqual(window._image.width(), 800)
            window.slider.setValue(2)
            self.wait_for(lambda: "000011" in window.position_label.text())
            self.controller.set_fps(12)
            self.assertEqual(window.fps_spin.value(), 12)
            self.assertEqual(camera_images, [])
        finally:
            window.close()

    def test_selection_allows_combinations_and_rejects_empty(self):
        dialog = ViewerSelectionDialog()
        self.assertEqual(dialog.selected_viewers, {"camera"})
        dialog.options["camera"].setChecked(False)
        self.assertFalse(dialog.buttons.button(QDialogButtonBox.StandardButton.Open).isEnabled())
        dialog.accept()
        self.assertEqual(dialog.result(), 0)
        dialog.options["bev"].setChecked(True)
        self.assertEqual(dialog.selected_viewers, {"bev"})
        dialog.options["cloud"].setChecked(True)
        self.assertEqual(dialog.selected_viewers, {"bev", "cloud"})
        self.assertTrue(dialog.buttons.button(QDialogButtonBox.StandardButton.Open).isEnabled())


if __name__ == "__main__":
    unittest.main()
