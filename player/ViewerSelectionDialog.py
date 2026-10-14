"""啟動時只建立使用者勾選的視窗；勾選屬於 GUI，不改資料參數。"""

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QVBoxLayout


class ViewerSelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Perception Lab | Select Viewers")
        self.setMinimumWidth(300)
        self.resize(340, 180)
        layout = QVBoxLayout(self)
        self.options = {}
        for key, label in (
            ("camera", "2D Camera + LiDAR"),
            ("cloud", "3D Point Cloud"),
            ("bev", "BEV / Bird's-Eye View"),
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(key == "camera")
            checkbox.toggled.connect(self._update_button)
            self.options[key] = checkbox
            layout.addWidget(checkbox)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_button()

    @property
    def selected_viewers(self):
        return {key for key, checkbox in self.options.items() if checkbox.isChecked()}

    @Slot()
    def _update_button(self):
        if hasattr(self, "buttons"):
            self.buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(bool(self.selected_viewers))

    def accept(self):
        if self.selected_viewers:
            super().accept()
