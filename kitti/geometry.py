"""幾何運算：點以 N×3 儲存，公式則採用列向量 p' = T @ p。

NumPy 中每一列是一個點，因此批次運算寫成 points @ T.T。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PlaneModel:
    """平面模型 normal @ point + offset = 0。"""

    normal: NDArray[np.float64]
    offset: float
    inliers: NDArray[np.bool_]

    def distances(self, points: ArrayLike) -> NDArray[np.float64]:
        """回傳每個點到平面的垂直距離。"""
        points = _xyz(points)
        return np.abs(points @ self.normal + self.offset)


def _xyz(points: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points


def to_homogeneous(points: ArrayLike) -> NDArray[np.float64]:
    """將 (x, y, z) 補成 (x, y, z, 1)，讓矩陣也能表達平移。

    LiDAR 的 reflectance 不是齊次座標，呼叫前要先取 points[:, :3]。
    """
    points = _xyz(points)
    return np.column_stack((points, np.ones(len(points))))


def transform_points(points: ArrayLike, transform: ArrayLike) -> NDArray[np.float64]:
    """使用 4×4 剛體轉換 T = [[R, t], [0, 1]]，計算 R @ p + t。

    此處只接受仿射形式；透視投影的除法由 project_points 負責。
    """
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("transform must have last row [0, 0, 0, 1]")
    return (to_homogeneous(points) @ transform.T)[:, :3]


def project_points(
    points: ArrayLike, projection: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """以 3×4 投影矩陣計算 q = P @ [x, y, z, 1]，再做透視除法。

    像素為 (q_x / q_z, q_y / q_z)。只保留正深度、有限值的點；
    無效像素填 NaN，輸出仍有 N 列，方便對應原始點雲與顏色。
    此函式處理標準、校正後的相機投影，不處理鏡頭畸變。
    """
    points = _xyz(points)
    projection = np.asarray(projection, dtype=np.float64)
    if projection.shape != (3, 4):
        raise ValueError("projection must have shape (3, 4)")

    projected = to_homogeneous(points) @ projection.T
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(projected).all(axis=1)
        & (points[:, 2] > 0)
        & (projected[:, 2] > 0)
    )
    pixels = np.full((len(points), 2), np.nan)
    pixels[valid] = projected[valid, :2] / projected[valid, 2, None]
    valid &= np.isfinite(pixels).all(axis=1)
    pixels[~valid] = np.nan
    return pixels, valid


def _plane_from_three_points(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float] | None:
    a, b, c = points
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal)
    if length == 0 or not np.isfinite(length):
        return None
    normal = normal / length
    offset = -float(normal @ a)
    return normal, offset


def _fit_plane_least_squares(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    offset = -float(normal @ centroid)
    return normal, offset


def ransac_plane(
    points: ArrayLike,
    threshold: float,
    iterations: int = 100,
    min_inliers: int = 3,
    rng: np.random.Generator | int | None = None,
) -> PlaneModel:
    """用 RANSAC 從含外點的 3D 點雲估平面。

    流程是反覆抽 3 點建立候選平面，計算所有點到平面的距離，
    用 threshold 決定哪些點支持它，最後用最佳 inliers 重新做一次
    最小平方擬合，讓模型不只依賴那 3 個被抽到的點。
    """
    points = _xyz(points)
    if len(points) < 3:
        raise ValueError("at least 3 points are required")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if min_inliers < 3:
        raise ValueError("min_inliers must be at least 3")

    finite = np.isfinite(points).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices) < 3:
        raise ValueError("at least 3 finite points are required")

    generator = np.random.default_rng(rng)
    best_inliers = np.zeros(len(points), dtype=bool)
    best_error = np.inf

    for _ in range(iterations):
        sample_indices = generator.choice(finite_indices, size=3, replace=False)
        candidate = _plane_from_three_points(points[sample_indices])
        if candidate is None:
            continue
        normal, offset = candidate
        distances = np.full(len(points), np.inf)
        distances[finite] = np.abs(points[finite] @ normal + offset)
        inliers = distances <= threshold
        count = int(inliers.sum())
        error = float(distances[inliers].mean()) if count else np.inf

        best_count = int(best_inliers.sum())
        if count > best_count or (count == best_count and error < best_error):
            best_inliers = inliers
            best_error = error

    if best_inliers.sum() < min_inliers:
        raise ValueError("RANSAC could not find enough inliers")

    normal, offset = _fit_plane_least_squares(points[best_inliers])
    if normal[2] < 0:
        normal = -normal
        offset = -offset
    refined_distances = np.full(len(points), np.inf)
    refined_distances[finite] = np.abs(points[finite] @ normal + offset)
    refined_inliers = refined_distances <= threshold
    return PlaneModel(normal, offset, refined_inliers)
