"""Фоновый поток обработки видео, чтобы интерфейс не зависал."""
import time

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from ..config import Config
from ..pipeline import Pipeline
from ..sources import VideoSource


class VideoWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    stats = pyqtSignal(int, float)   # уникальных машин, FPS
    error = pyqtSignal(str)
    log_ready = pyqtSignal(object)   # TrackLog по завершении

    def __init__(self, source: str, cfg: Config):
        super().__init__()
        self.source = source
        self.cfg = cfg
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        src = None
        try:
            pipeline = Pipeline(self.cfg)
            src = VideoSource(self.source)
            src.open()
            fps_in = src.fps
            pipeline.start(fps_in)

            idx, t0, prev = 0, time.time(), time.time()
            while self._running:
                ok, frame = src.read()
                if not ok:
                    break
                # для файла - время по номеру кадра, для живых источников - по часам
                t = idx / fps_in if src.kind.value == "file" else time.time() - t0
                annotated, _ = pipeline.process(frame, idx, t)
                idx += 1

                now = time.time()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                self.stats.emit(len(pipeline.seen_ids), fps)

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                self.frame_ready.emit(
                    QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                )
            self.log_ready.emit(pipeline.log)
        except Exception as e:  # показываем пользователю, а не роняем приложение
            self.error.emit(str(e))
        finally:
            if src:
                src.release()
