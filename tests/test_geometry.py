import tempfile
import unittest
from pathlib import Path

import numpy as np

from kitti import KittiCamera
from kitti.geometry import project_points, ransac_plane, to_homogeneous, transform_points


class GeometryTests(unittest.TestCase):
    def test_rotation_and_translation(self):
        transform = np.array([
            [0, -1, 0, 10],
            [1, 0, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ])
        np.testing.assert_allclose(transform_points([[1, 2, 3]], transform), [[8, 21, 33]])
        np.testing.assert_allclose(to_homogeneous([[1, 2, 3]]), [[1, 2, 3, 1]])

    def test_projection_keeps_fourth_column_and_rejects_invalid_depth(self):
        projection = np.array([[100, 0, 50, 20], [0, 100, 40, 0], [0, 0, 1, 0]])
        pixels, valid = project_points([[1, 2, 10], [1, 2, -10], [0, 0, 0]], projection)
        np.testing.assert_allclose(pixels[0], [62, 60])
        np.testing.assert_array_equal(valid, [True, False, False])
        self.assertTrue(np.isnan(pixels[1:]).all())

    def test_camera_transform_order_and_image_bounds(self):
        lidar_to_camera = np.eye(4)
        lidar_to_camera[0, 3] = 1
        rectification = np.array([
            [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        ])
        projection = np.array([[100, 0, 50, 0], [0, 100, 40, 0], [0, 0, 1, 0]])
        camera = KittiCamera(projection, rectification, lidar_to_camera)
        pixels, valid = camera.project_lidar([[0, 0, 10], [100, 0, 10]], (100, 80))
        np.testing.assert_allclose(pixels[0], [50, 50])
        np.testing.assert_array_equal(valid, [True, False])
        self.assertEqual(len(pixels), 2)

    def test_tracking_calibration_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0000.txt"
            path.write_text(
                "P2: 100 0 50 20 0 100 40 0 0 0 1 0\n"
                "R_rect 1 0 0 0 1 0 0 0 1\n"
                "Tr_velo_cam 1 0 0 0 0 1 0 0 0 0 1 0\n"
            )
            camera = KittiCamera.from_file(path)
        np.testing.assert_allclose(camera.intrinsic, [[100, 0, 50], [0, 100, 40], [0, 0, 1]])
        np.testing.assert_allclose(camera.lidar_to_camera, np.eye(4))

    def test_empty_points_and_wrong_shapes(self):
        pixels, valid = project_points(np.empty((0, 3)), np.eye(3, 4))
        self.assertEqual(pixels.shape, (0, 2))
        self.assertEqual(valid.shape, (0,))
        with self.assertRaises(ValueError):
            to_homogeneous([[1, 2, 3, 0.5]])
        with self.assertRaises(ValueError):
            transform_points([[1, 2, 3]], np.eye(3))

    def test_ransac_plane_ignores_outliers_and_refines_model(self):
        xs, ys = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        ground = np.column_stack((xs.ravel(), ys.ravel(), 0.5 * xs.ravel() - 0.25 * ys.ravel() + 1))
        ground[::3, 2] += 0.01
        outliers = np.array([
            [0, 0, 8],
            [1, -2, -7],
            [3, 3, 4],
            [-4, 1, -3],
        ])
        points = np.vstack((ground, outliers))

        model = ransac_plane(points, threshold=0.08, iterations=200, rng=7)

        expected_normal = np.array([-0.5, 0.25, 1.0])
        expected_normal /= np.linalg.norm(expected_normal)
        if model.normal @ expected_normal < 0:
            expected_normal = -expected_normal
        np.testing.assert_allclose(model.normal, expected_normal, atol=0.01)
        self.assertGreaterEqual(model.inliers[: len(ground)].sum(), len(ground) - 1)
        self.assertFalse(model.inliers[len(ground):].any())
        self.assertLess(model.distances(ground).max(), 0.04)

    def test_ransac_plane_validates_inputs(self):
        with self.assertRaises(ValueError):
            ransac_plane([[0, 0, 0], [1, 1, 1]], threshold=0.1)
        with self.assertRaises(ValueError):
            ransac_plane([[0, 0, 0], [1, 0, 0], [0, 1, 0]], threshold=0)


if __name__ == "__main__":
    unittest.main()
