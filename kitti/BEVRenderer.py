"""BEV 正交投影與柵格化，不依賴 Qt。"""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import to_homogeneous


@dataclass
class BEVRenderer:
    x_range: tuple[float, float] = (-10, 70)
    y_range: tuple[float, float] = (-40, 40)
    z_range: tuple[float, float] = (-3, 3)
    resolution: float = 0.1  # 公尺／像素，不是像素／公尺。

    def __post_init__(self):
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        for bounds in (self.x_range, self.y_range, self.z_range):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError("ranges must contain finite increasing bounds")
        sizes = np.array([
            self.y_range[1] - self.y_range[0], self.x_range[1] - self.x_range[0]
        ]) / self.resolution
        if not np.allclose(sizes, np.rint(sizes)) or np.any(sizes < 1):
            raise ValueError("XY spans must be positive integer multiples of resolution")
        self.width, self.height = np.rint(sizes).astype(int)

    @property
    def projection(self):
        """[u,v,1] = A @ [x,y,z,1]，Z 欄為零，沒有透視深度除法。"""
        r = self.resolution
        return np.array([
            [0, -1 / r, 0, self.y_range[1] / r],
            [-1 / r, 0, 0, self.x_range[1] / r],
            [0, 0, 0, 1],
        ])

    def rasterize(self, points):
        """回傳每格最高 Z；沒有回波的格子是 -inf，不是高度零。

        多個點投到同格時，不能靠畫點的順序決定顏色；maximum.at
        對重複索引做 max reduction，因此打亂輸入順序也得到相同結果。
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        valid = np.isfinite(points).all(axis=1)
        valid &= (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
        points = points[valid]
        uv = (to_homogeneous(points) @ self.projection.T)[:, :2]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < self.width)
        inside &= (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        cells = np.floor(uv[inside]).astype(int)
        heights = np.full((self.height, self.width), -np.inf)
        np.maximum.at(heights, (cells[:, 1], cells[:, 0]), points[inside, 2])
        return heights

    def render(self, points):
        heights = self.rasterize(points)
        occupied = np.isfinite(heights)
        image = np.full((self.height, self.width, 3), (22, 24, 27), dtype=np.uint8)
        # 格線只表示 XY 的公尺刻度，每 5 m 一條，並不代表地面高度。
        for x in np.arange(np.ceil(self.x_range[0] / 5) * 5, self.x_range[1], 5):
            v = round((self.x_range[1] - x) / self.resolution)
            cv2.line(image, (0, v), (self.width - 1, v), (42, 45, 48), 1)
        for y in np.arange(np.ceil(self.y_range[0] / 5) * 5, self.y_range[1], 5):
            u = round((self.y_range[1] - y) / self.resolution)
            cv2.line(image, (u, 0), (u, self.height - 1), (42, 45, 48), 1)
        if occupied.any():
            values = np.rint(255 * (heights[occupied] - self.z_range[0]) / (self.z_range[1] - self.z_range[0]))
            image[occupied] = cv2.applyColorMap(values.astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        origin = (self.projection @ np.array([0, 0, 0, 1]))[:2]
        u, v = np.floor(origin).astype(int)
        if 0 <= u < self.width and 0 <= v < self.height:
            # 正 X 向圖片上方，正 Y 向左；原點是 LiDAR 而非全域地圖原點。
            length = max(1, round(5 / self.resolution))
            cv2.arrowedLine(image, (u, v), (u, max(0, v - length)), (70, 70, 255), 2)
            cv2.arrowedLine(image, (u, v), (max(0, u - length), v), (90, 255, 90), 2)
            cv2.circle(image, (u, v), 5, (255, 255, 255), -1)
            cv2.putText(image, "LiDAR (0,0)", (max(0, min(u + 8, self.width - 125)), max(15, min(v + 22, self.height - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return image
