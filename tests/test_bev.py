import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kitti.BEVRenderer import BEVRenderer
from kitti.KittiRenderer import KittiRenderer


class BEVTests(unittest.TestCase):
    def test_projection_orientation_origin_and_ignores_z(self):
        renderer = BEVRenderer()
        points = np.array([[0, 0, 0, 1], [20, 0, 0, 1], [20, 10, 3, 1], [20, 10, -3, 1]])
        projected = points @ renderer.projection.T
        np.testing.assert_allclose(projected, [[400, 700, 1], [400, 500, 1], [300, 500, 1], [300, 500, 1]])

    def test_highest_point_order_independent_and_empty_cells(self):
        renderer = BEVRenderer((-1, 1), (-1, 1), (-3, 3), 1)
        points = np.array([[0.5, 0.5, -2], [0.5, 0.5, 2], [0.5, 0.5, 0]])
        result = renderer.rasterize(points)
        self.assertEqual(result[0, 0], 2)
        self.assertEqual(np.isfinite(result).sum(), 1)
        self.assertTrue(np.isneginf(result[1, 1]))
        np.testing.assert_array_equal(result, renderer.rasterize(points[::-1]))

    def test_bounds_height_filter_and_nonfinite(self):
        renderer = BEVRenderer((-1, 1), (-1, 1), (-3, 3), 1)
        points = [[1, 1, -3], [-1, 0, 1], [0, -1, 1], [0, 0, 4], [np.nan, 0, 0]]
        result = renderer.rasterize(points)
        self.assertEqual(np.isfinite(result).sum(), 1)
        self.assertEqual(result[0, 0], -3)
        self.assertFalse(np.isfinite(renderer.rasterize(np.empty((0, 3)))).any())

    def test_invalid_configuration_and_image_size(self):
        with self.assertRaises(ValueError):
            BEVRenderer(resolution=0)
        with self.assertRaises(ValueError):
            BEVRenderer(x_range=(0, 1), resolution=0.3)
        renderer = BEVRenderer()
        self.assertEqual(renderer.render(np.empty((0, 3))).shape, (800, 800, 3))

    def test_cloud_only_does_not_load_camera_or_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000000.bin"
            np.array([[1, 2, 3, 0.5]], dtype="<f4").tofile(path)

            class Sequence:
                frame_ids = [0]

                def __getitem__(self, index):
                    # 刻意沒有 image 或 calib_path，單開 BEV/3D 不應讀它們。
                    return SimpleNamespace(velodyne_path=path)

            renderer = KittiRenderer(Sequence())
            image, points, colors = renderer.render_with_cloud(0, include_image=False)
        self.assertIsNone(image)
        np.testing.assert_array_equal(points, [[1, 2, 3]])
        self.assertEqual(colors.shape, (1, 4))
        self.assertNotIn("camera", renderer.__dict__)
