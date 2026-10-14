"""3D 功能類：維護點雲、座標參考物與虛擬相機，不建立播放 GUI。"""

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QVector3D

from kitti.geometry import ransac_plane


class PointCloudScene(QObject):
    def __init__(self, view, point_size=2.0, parent=None):
        super().__init__(parent)
        self.view = view
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 4), dtype=np.float32)
        self._ground_mask = None
        self._ransac_enabled = False
        self.view.setBackgroundColor((22, 24, 27))
        # opaque 開啟深度測試，近處的點才會擋住同像素上的遠點。
        self.cloud = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32), size=point_size, pxMode=True,
            glOptions="opaque",
        )
        self.view.addItem(self.cloud)
        self.grid = gl.GLGridItem()
        self.grid.setSize(80, 80)
        self.grid.setSpacing(5, 5)
        self.grid.setColor((110, 115, 120, 80))
        # 格線位於 z=0，是感測器的水平參考面，不是實際路面。
        self.view.addItem(self.grid)
        self.axes = []
        for direction, color in (
            ([5, 0, 0], (1, 0.25, 0.25, 1)),
            ([0, 5, 0], (0.25, 1, 0.35, 1)),
            ([0, 0, 5], (0.3, 0.6, 1, 1)),
        ):
            axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], direction], dtype=np.float32),
                color=color, width=3, mode="lines", glOptions="additive",
            )
            axis.setDepthValue(90)
            self.axes.append(axis)
            self.view.addItem(axis)

        # 原始點雲不一定有 (0,0,0) 這個回波，所以另外加一個白色原點標記。
        # 導航標記關閉深度測試並最後繪製，避免被周圍點雲遮住。
        self.origin = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float32), color=(1, 1, 1, 1),
            size=16, pxMode=True, glOptions="additive",
        )
        self.origin.setDepthValue(100)
        self.view.addItem(self.origin)
        self.reset_view()

    @Slot(object, object)
    def set_cloud(self, points, colors):
        """只更新 GPU 點資料，保留滑鼠設定的相機位置與觀看方向。

        xyz 直接用 LiDAR 座標作為世界座標，Model 矩陣為單位矩陣。
        View/Projection 則由 GLViewWidget 的虛擬相機負責。
        """
        self._points = points
        self._colors = colors
        self._ground_mask = None
        self._update_cloud()

    @Slot(bool)
    def set_ransac_enabled(self, enabled):
        self._ransac_enabled = enabled
        self._update_cloud()

    def _update_cloud(self):
        colors = self._colors
        if self._ransac_enabled:
            if self._ground_mask is None:
                self._ground_mask = self._find_ground()
            colors = self._colors.copy()
            colors[self._ground_mask, :3] = (0.2, 0.9, 0.35)
        self.cloud.setData(pos=self._points, color=colors)

    def _find_ground(self):
        points = self._points
        mask = np.zeros(len(points), dtype=bool)
        # KITTI 的 LiDAR 在路面上方；限制候選區域，減少遠處牆面干擾。
        candidates = points[
            np.isfinite(points).all(axis=1)
            & (points[:, 2] < 0)
            & (np.linalg.norm(points[:, :2], axis=1) <= 40)
        ]
        if len(candidates) < 3:
            return mask
        if len(candidates) > 3000:
            indices = np.random.default_rng(7).choice(len(candidates), 3000, replace=False)
            candidates = candidates[indices]
        try:
            model = ransac_plane(candidates, threshold=0.15, iterations=100, rng=7)
        except ValueError:
            return mask
        # 法向量與 Z 軸夾角最多 20 度；垂直牆面不視為地面。
        if model.normal[2] >= np.cos(np.deg2rad(20)):
            mask = model.distances(points) <= 0.15
        return mask

    @Slot()
    def reset_view(self):
        self.view.setCameraPosition(
            pos=QVector3D(0, 0, 0), distance=65, elevation=35, azimuth=-135
        )

    @Slot()
    def focus_origin(self):
        # 只把 orbit 的中心移回原點，保留目前方向與縮放距離。
        self.view.setCameraPosition(pos=QVector3D(0, 0, 0))

    @Slot(bool)
    def set_axes_visible(self, visible):
        for item in self.axes:
            item.setVisible(visible)

    @Slot(bool)
    def set_grid_visible(self, visible):
        self.grid.setVisible(visible)

    @Slot(float)
    def set_point_size(self, size):
        self.cloud.setData(size=size)
