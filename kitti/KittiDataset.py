# KittiDataset.py

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2

from .KittiTrackingPaths import (
    KittiTrackingPaths,
    KittiSplit,
    KITTI_TRACKING_SEQUENCE_COUNT,
)


@dataclass(frozen=True)
class KittiFrame:
    paths: KittiTrackingPaths
    sequence_id: int
    frame_id: int

    @property
    def image_path(self) -> Path:
        return self.paths.image(self.sequence_id, self.frame_id)

    @cached_property
    def image(self):
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"KITTI image not found: {self.image_path}")
        return image

    @property
    def velodyne_path(self) -> Path:
        return self.paths.velodyne(self.sequence_id, self.frame_id)

    @property
    def calib_path(self) -> Path:
        return self.paths.calib(self.sequence_id)

    @property
    def oxts_path(self) -> Path:
        return self.paths.oxts(self.sequence_id)

    @property
    def label_path(self) -> Path:
        return self.paths.labels(self.sequence_id)


class KittiSequence:
    def __init__(
        self,
        paths: KittiTrackingPaths,
        sequence_id: int,
    ):
        self.paths = paths
        self.sequence_id = sequence_id

        self._frame_ids = self.paths.frame_ids(sequence_id)
        self._frame_id_set = set(self._frame_ids)

    def __len__(self) -> int:
        return len(self._frame_ids)

    @property
    def frame_ids(self) -> list[int]:
        return self._frame_ids.copy()

    def __getitem__(self, frame_id: int) -> KittiFrame:
        if frame_id not in self._frame_id_set:
            raise IndexError(
                f"Frame {frame_id} does not exist "
                f"in sequence {self.sequence_id:04d}."
            )

        return KittiFrame(
            paths=self.paths,
            sequence_id=self.sequence_id,
            frame_id=frame_id,
        )


class KittiDataset:
    def __init__(
        self,
        root: str | Path,
        split: KittiSplit = KittiSplit.TRAINING,
    ):
        self.paths = KittiTrackingPaths(
            root=Path(root),
            split=split,
        )

    def __len__(self) -> int:
        return KITTI_TRACKING_SEQUENCE_COUNT[self.paths.split]

    def __getitem__(self, sequence_id: int) -> KittiSequence:
        if sequence_id < 0 or sequence_id >= len(self):
            raise IndexError(
                f"sequence_id out of range: {sequence_id}, "
                f"expected [0, {len(self) - 1}]"
            )

        return KittiSequence(
            paths=self.paths,
            sequence_id=sequence_id,
        )
