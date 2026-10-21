"""Сборка пайплайна: (деблюр) -> детекция -> ByteTrack -> аналитика -> отрисовка."""
from collections import defaultdict, deque

import numpy as np

from .analytics import TrackLog, dominant_color_bgr
from .config import Config
from .deblur import Deblurrer
from .detector import Detector
from .tracker import ByteTrackWrapper, Track
from .visualization import draw_tracks


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.tracker = ByteTrackWrapper(cfg.tracker)
        self.deblur = Deblurrer(cfg.deblur_weights, cfg.device) if cfg.use_deblur else None
        self.log = TrackLog(cfg.meters_per_pixel)
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.trail_length))
        self.seen_ids: set[int] = set()

    def start(self, fps: float) -> None:
        """Вызывать при открытии нового источника."""
        self.tracker.reset(fps)
        self.log = TrackLog(self.cfg.meters_per_pixel)
        self.trails.clear()
        self.seen_ids.clear()

    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> tuple[np.ndarray, list[Track]]:
        if self.deblur is not None:
            frame = self.deblur(frame)

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        speeds = self.log.update(frame_idx, t, tracks)

        for tr in tracks:
            if tr.id not in self.seen_ids:
                self.seen_ids.add(tr.id)
                x1, y1, x2, y2 = map(int, tr.box)
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size:
                    self.log.colors[tr.id] = dominant_color_bgr(crop)
            self.trails[tr.id].append(tuple(map(int, tr.center_bottom)))

        annotated = draw_tracks(frame.copy(), tracks, self.detector.names, speeds, self.trails)
        return annotated, tracks
