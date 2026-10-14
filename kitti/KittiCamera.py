"""KITTI 校正後的相機模型；影像已 rectified，不再套用畸變校正。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import project_points, transform_points


@dataclass
class KittiCamera:
    # P2 包含內參與彩色相機相對參考座標系的平移，不能只取前三欄投影。
    projection: NDArray[np.float64]  # 3×4，預設讀左彩色相機 P2
    rectification: NDArray[np.float64]  # 4×4，參考相機 → 校正後相機
    lidar_to_camera: NDArray[np.float64]  # 4×4，LiDAR → 參考相機

    def __post_init__(self):
        for name, shape in (
            ("projection", (3, 4)),
            ("rectification", (4, 4)),
            ("lidar_to_camera", (4, 4)),
        ):
            matrix = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must be a finite matrix of shape {shape}")
            if shape == (4, 4) and not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError(f"{name} must have last row [0, 0, 0, 1]")
            setattr(self, name, matrix)

    @classmethod
    def from_file(cls, path: str | Path, camera_id: int = 2) -> "KittiCamera":
        """讀取序列的 calibration，P0～P3 分別對應四台相機。

        檔案矩陣採 row-major，數字依照每一列的順序排列。
        Tracking 檔案的 R_rect / Tr_velo_cam 欄位沒有冒號，P2 有冒號。
        """
        if camera_id not in (0, 1, 2, 3):
            raise ValueError("camera_id must be 0, 1, 2, or 3")
        matrices = {}
        for line in Path(path).read_text().splitlines():
            fields = line.split()
            if fields:
                matrices[fields[0].rstrip(":")] = np.array(
                    [float(value) for value in fields[1:]], dtype=np.float64
                )

        def read_matrix(key: str, shape: tuple[int, int]):
            if key not in matrices:
                raise ValueError(f"Missing calibration field: {key}")
            if matrices[key].size != shape[0] * shape[1]:
                raise ValueError(f"Invalid calibration matrix size: {key}")
            return matrices[key].reshape(shape)

        # 補最後一列 [0, 0, 0, 1]，才能把旋轉和平移串成齊次轉換。
        rectification = np.eye(4)
        rectification[:3, :3] = read_matrix("R_rect", (3, 3))
        lidar_to_camera = np.eye(4)
        lidar_to_camera[:3, :] = read_matrix("Tr_velo_cam", (3, 4))
        return cls(
            projection=read_matrix(f"P{camera_id}", (3, 4)),
            rectification=rectification,
            lidar_to_camera=lidar_to_camera,
        )

    @property
    def intrinsic(self) -> NDArray[np.float64]:
        """校正後的內參 K：焦距 fx、fy 與主點 cx、cy，單位為像素。"""
        return self.projection[:, :3].copy()

    def project_lidar(
        self, points: ArrayLike, image_size: tuple[int, int] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """LiDAR N×3 → 像素 N×2，以及與原始點一一對應的有效遮罩。

        依照 p_image ~ P2 @ R_rect @ Tr_velo_cam @ p_lidar 的順序。
        image_size 是 (width, height)，不是 NumPy image.shape 的順序。
        """
        camera_points = transform_points(points, self.lidar_to_camera)
        rectified_points = transform_points(camera_points, self.rectification)
        pixels, valid = project_points(rectified_points, self.projection)
        if image_size is not None:
            width, height = image_size
            if width <= 0 or height <= 0:
                raise ValueError("image_size must contain positive width and height")
            valid &= (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
        return pixels, valid
