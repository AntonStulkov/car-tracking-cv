"""ByteTrack: присваивает машинам устойчивые ID между кадрами."""
from dataclasses import dataclass

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from .config import TrackerConfig


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    cls: int
    conf: float

    @property
    def center_bottom(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, self.box[3]


class ByteTrackWrapper:
    def __init__(self, cfg: TrackerConfig, fps: float = 30.0):
        self.cfg = cfg
        self.reset(fps)

    def reset(self, fps: float = 30.0) -> None:
        args = IterableSimpleNamespace(tracker_type="bytetrack", **vars(self.cfg))
        self._tracker = BYTETracker(args, frame_rate=int(round(fps)))

    def update(self, detections, frame: np.ndarray) -> list[Track]:
        if len(detections) == 0:
            return []
        out = self._tracker.update(detections, frame)
        # столбцы: x1, y1, x2, y2, track_id, score, cls, det_idx
        return [
            Track(id=int(r[4]), box=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                  cls=int(r[6]), conf=float(r[5]))
            for r in out
        ]
