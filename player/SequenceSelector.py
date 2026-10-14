"""各 viewer 共用 controller 的序列選單。"""

from PySide6.QtCore import QSignalBlocker, Slot
from PySide6.QtWidgets import QComboBox


class SequenceSelector(QComboBox):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setToolTip("Sequence")
        self.setAccessibleName("Sequence")
        for sequence_id in controller.sequence_ids:
            self.addItem(f"Sequence {sequence_id:04d}", sequence_id)
        self.setVisible(bool(controller.sequence_ids))
        if controller.sequence_id is not None:
            self._show_sequence(controller.sequence_id)
        self.currentIndexChanged.connect(self._select_sequence)
        controller.sequence_changed.connect(self._show_sequence)

    @Slot(int)
    def _select_sequence(self, index):
        if index >= 0:
            self.controller.select_sequence(self.itemData(index))

    @Slot(int)
    def _show_sequence(self, sequence_id):
        with QSignalBlocker(self):
            self.setCurrentIndex(self.findData(sequence_id))
