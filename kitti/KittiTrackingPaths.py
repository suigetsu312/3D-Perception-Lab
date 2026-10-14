from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class KittiSplit(str, Enum):
    TRAINING = "training"
    TESTING = "testing"


KITTI_TRACKING_SEQUENCE_COUNT = {
    KittiSplit.TRAINING: 21,
    KittiSplit.TESTING: 29,
}


@dataclass(frozen=True)
class KittiTrackingPaths:
    root: Path
    split: KittiSplit = KittiSplit.TRAINING

    @property
    def num_sequences(self) -> int:
        return KITTI_TRACKING_SEQUENCE_COUNT[self.split]

    def image(self, sequence: int, frame: int) -> Path:
        return (
            self.root
            / "data_tracking_image_2"
            / self.split.value
            / "image_02"
            / f"{sequence:04d}"
            / f"{frame:06d}.png"
        )

    def image_dir(self, sequence: int) -> Path:
        return (
            self.root
            / "data_tracking_image_2"
            / self.split.value
            / "image_02"
            / f"{sequence:04d}"
        )

    def frame_ids(self, sequence: int) -> list[int]:
        image_dir = self.image_dir(sequence)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"KITTI image directory not found: {image_dir}")

        return sorted(int(path.stem) for path in image_dir.glob("*.png"))

    def velodyne(self, sequence: int, frame: int) -> Path:
        return (
            self.root
            / "data_tracking_velodyne"
            / self.split.value
            / "velodyne"
            / f"{sequence:04d}"
            / f"{frame:06d}.bin"
        )

    def calib(self, sequence: int) -> Path:
        return (
            self.root
            / "data_tracking_calib"
            / self.split.value
            / "calib"
            / f"{sequence:04d}.txt"
        )

    def oxts(self, sequence: int) -> Path:
        return (
            self.root
            / "data_tracking_oxts"
            / self.split.value
            / "oxts"
            / f"{sequence:04d}.txt"
        )

    def labels(self, sequence: int) -> Path:
        if self.split is not KittiSplit.TRAINING:
            raise ValueError("tracking labels are only available for training split")

        return (
            self.root
            / "data_tracking_label_2"
            / self.split.value
            / "label_02"
            / f"{sequence:04d}.txt"
        )
