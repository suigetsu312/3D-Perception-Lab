"""BEV 主視窗重用 2D 播放 GUI，幾何與柵格化交給 BEVRenderer。"""

from .PlayerWindow import PlayerWindow


class BEVWindow(PlayerWindow):
    def __init__(self, controller, renderer):
        super().__init__(
            controller, "3D Perception Lab | BEV", close_controller=False,
            image_signal=controller.bev_ready, autoload=False,
        )
        self.resize(850, 920)
        self.statusBar().showMessage(
            f"Top height: {renderer.z_range[0]:g} to {renderer.z_range[1]:g} m | "
            f"{renderer.resolution:g} m/px | +X up, +Y left | white: LiDAR origin"
        )
