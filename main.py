"""Qt 即時播放器入口：GUI 與功能類以組合和 signals/slots 串接。"""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

from kitti import KittiDataset, KittiSplit
from kitti.KittiRenderer import KittiRenderer
from kitti.BEVRenderer import BEVRenderer
from player.BEVWindow import BEVWindow
from player.PlaybackController import PlaybackController
from player.PlayerWindow import PlayerWindow
from player.PointCloudWindow import PointCloudWindow
from player.ViewerSelectionDialog import ViewerSelectionDialog


# 直接改參數，不使用 CLI；實際逐幀投影，不需要先產生影片。
DATASET_ROOT = "/mnt/d/mizu/dataset/kitti"
SPLIT = KittiSplit.TRAINING
SEQUENCE_ID = 0
FPS = 10.0
POINT_RADIUS = 1
COLOR_MAX_DISTANCE = 50.0
POINT_SIZE_3D = 2.0  # 3D 點大小以螢幕像素計算。
BEV_X_RANGE = (-10, 70)
BEV_Y_RANGE = (-40, 40)
BEV_Z_RANGE = (-3, 3)  # 篩選高度，同時用作顏色色階。
BEV_RESOLUTION = 0.1  # 公尺／像素。


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("KITTI LiDAR Player")
    selection = ViewerSelectionDialog()
    if selection.exec() != QDialog.DialogCode.Accepted:
        return 0
    selected = selection.selected_viewers
    dataset = KittiDataset(root=DATASET_ROOT, split=SPLIT)
    sequence_ids = [
        sequence_id for sequence_id in range(len(dataset))
        if dataset.paths.image_dir(sequence_id).is_dir()
        and any(dataset.paths.image_dir(sequence_id).glob("*.png"))
    ]
    initial_sequence = SEQUENCE_ID if not sequence_ids or SEQUENCE_ID in sequence_ids else sequence_ids[0]

    def make_renderer(sequence_id):
        return KittiRenderer(dataset[sequence_id], POINT_RADIUS, COLOR_MAX_DISTANCE)

    renderer = make_renderer(initial_sequence)
    bev_renderer = BEVRenderer(BEV_X_RANGE, BEV_Y_RANGE, BEV_Z_RANGE, BEV_RESOLUTION) if "bev" in selected else None
    # 同一次讀檔供給所選 viewer；未開 2D 就不讀相機影像或做相機投影。
    controller = PlaybackController(
        renderer, fps=FPS, include_cloud="cloud" in selected,
        bev_renderer=bev_renderer, include_image="camera" in selected,
        sequence_ids=sequence_ids, sequence_id=initial_sequence, renderer_factory=make_renderer,
    )
    windows = []  # 保留 Python 參考，避免視窗在 event loop 中被回收。
    if "camera" in selected:
        windows.append(PlayerWindow(
            controller, f"KITTI | 2D | {SPLIT.value}",
            close_controller=False, autoload=False,
        ))
    if "cloud" in selected:
        windows.append(PointCloudWindow(controller, POINT_SIZE_3D))
    if "bev" in selected:
        windows.append(BEVWindow(controller, bev_renderer))
    # 視窗關閉或應用退出時，都先清理背景 thread。
    app.aboutToQuit.connect(controller.close)
    for window in windows:
        window.show()
    # 所有視窗連好訊號後，只要求一次初始影格，單開 BEV/3D 也能載入。
    QTimer.singleShot(0, controller.load)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
