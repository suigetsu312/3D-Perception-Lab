"""投影與畫點的功能類：不依賴 Qt，供播放器組合使用。"""

from functools import cached_property

import cv2
import numpy as np

from .KittiCamera import KittiCamera


class KittiRenderer:
    def __init__(self, sequence, point_radius=1, max_distance=50.0):
        if point_radius < 1 or not np.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("point_radius and max_distance must be positive")
        self.sequence = sequence
        self.frame_ids = sequence.frame_ids
        if not self.frame_ids:
            raise ValueError("Sequence has no frames")
        self.point_radius = point_radius
        self.max_distance = max_distance

    @cached_property
    def camera(self):
        # 只開 3D/BEV 時，不需要相機 calibration 或相機影像。
        return KittiCamera.from_file(self.sequence[self.frame_ids[0]].calib_path)

    def render(self, frame_id):
        """回傳 BGR 影像；只讀當前影格，避免整段點雲占滿記憶體。"""
        return self.render_with_cloud(frame_id)[0]

    def render_with_cloud(self, frame_id, include_image=True):
        """同次讀檔提供 2D 疊圖、原始 xyz 與 3D RGBA，保持兩個視窗同步。"""
        frame = self.sequence[frame_id]
        lidar = np.fromfile(frame.velodyne_path, dtype="<f4")
        if lidar.size % 4:
            raise ValueError(f"Invalid LiDAR file size: {frame.velodyne_path}")
        # 第四欄是反射強度，不能當齊次座標；只投影 xyz。
        points = lidar.reshape(-1, 4)[:, :3]
        points = np.ascontiguousarray(points[np.isfinite(points).all(axis=1)])
        # 3D 不裁掉相機視野外的點。單位仍是公尺，原點仍是 LiDAR 感測器。
        all_distances = np.linalg.norm(points, axis=1)
        all_values = np.rint(255 * (1 - np.clip(all_distances / self.max_distance, 0, 1)))
        rgba = np.ones((len(points), 4), dtype=np.float32)
        if len(points):
            bgr = cv2.applyColorMap(all_values.astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
            rgba[:, :3] = bgr[:, ::-1] / 255.0
        if not include_image:
            return None, points, rgba
        image = frame.image.copy()
        height, width = image.shape[:2]
        pixels, valid = self.camera.project_lidar(points, (width, height))
        pixels = pixels[valid]
        distances = np.linalg.norm(points[valid], axis=1)
        if not len(pixels):
            return image, points, rgba
        # 固定色階讓跨影格的距離可比較：近紅、遠藍。
        values = np.rint(255 * (1 - np.clip(distances / self.max_distance, 0, 1)))
        colors = cv2.applyColorMap(values.astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        positions = np.floor(pixels).astype(int)
        # 遠點先畫、近點後畫；這不是完整遮擋判定。
        for index in np.argsort(distances)[::-1]:
            u, v = positions[index]
            color = tuple(int(channel) for channel in colors[index])
            cv2.circle(image, (int(u), int(v)), self.point_radius, color, -1)
        return image, points, rgba
