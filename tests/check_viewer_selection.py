"""自動操作真實啟動流程；VIEWER_CHECK 指定逗號分隔的 viewer 組合。"""

import os
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import main as entry
from player.BEVWindow import BEVWindow
from player.PlayerWindow import PlayerWindow
from player.PointCloudWindow import PointCloudWindow


selected = set(os.environ.get("VIEWER_CHECK", "bev").split(","))
assert selected <= {"camera", "cloud", "bev"} and selected
failures = []
checked = []


class AutomaticSelection(entry.ViewerSelectionDialog):
    def exec(self):
        def choose():
            for key, checkbox in self.options.items():
                checkbox.setChecked(key in selected)
            Path("outputs").mkdir(exist_ok=True)
            self.show()
            self.grab().save("outputs/viewer_selection.png")
            self.accept()
        QTimer.singleShot(0, choose)
        return super().exec()


class CheckedApplication(QApplication):
    def exec(self):
        deadline = time.monotonic() + 15
        timer = QTimer(self)
        timer.setInterval(50)

        def check():
            try:
                windows = [widget for widget in self.topLevelWidgets() if widget.isVisible()]
                expected = {"camera": PlayerWindow, "cloud": PointCloudWindow, "bev": BEVWindow}
                actual_types = {type(window) for window in windows}
                assert actual_types == {expected[key] for key in selected}, actual_types
                loaded = all(
                    (len(window.scene.cloud.pos) > 1000 if isinstance(window, PointCloudWindow)
                     else not window._image.isNull()) for window in windows
                )
                if not loaded:
                    assert time.monotonic() < deadline, "No initial frame"
                    return
                controller = windows[0].controller
                assert controller.index == 0
                if "camera" not in selected:
                    assert "camera" not in controller._worker.renderer.__dict__
                for window in windows:
                    assert window.grab().save(f"outputs/selected_{type(window).__name__}.png")
                checked.append(controller)
                timer.stop()
                self.quit()
            except Exception as error:
                failures.append(error)
                timer.stop()
                self.quit()

        timer.timeout.connect(check)
        timer.start()
        return super().exec()


if __name__ == "__main__":
    entry.ViewerSelectionDialog = AutomaticSelection
    entry.QApplication = CheckedApplication
    result = entry.main()
    assert result == 0 and not failures, failures
    assert checked, "Viewer check did not complete"
    assert not checked[0]._thread.isRunning()
    print("Startup selection and initial rendering passed:", sorted(selected))
