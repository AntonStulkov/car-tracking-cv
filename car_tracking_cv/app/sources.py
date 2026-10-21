"""Источники видео: файл (mp4/mov), веб/USB-камера, RTSP-поток."""
import os

# До import cv2: RTSP по TCP - меньше артефактов и потерь пакетов
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


class SourceKind(str, Enum):
    FILE = "file"
    CAMERA = "camera"
    RTSP = "rtsp"


def detect_kind(source: str) -> SourceKind:
    if source.isdigit():
        return SourceKind.CAMERA
    if source.lower().startswith(("rtsp://", "rtsps://")):
        return SourceKind.RTSP
    if Path(source).suffix.lower() in VIDEO_EXT:
        return SourceKind.FILE
    raise ValueError(
        f"Неизвестный источник: {source!r}. Нужен индекс камеры, rtsp:// URL или видеофайл."
    )


class VideoSource:
    def __init__(self, source: str, max_reconnects: int = 10):
        self.source = source
        self.kind = detect_kind(source)
        self.max_reconnects = max_reconnects
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.kind is SourceKind.CAMERA:
            idx = int(self.source)
            self._cap = (
                cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
            )
        elif self.kind is SourceKind.RTSP:
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # минимальная задержка
        else:
            if not Path(self.source).exists():
                raise FileNotFoundError(f"Файл не найден: {self.source}")
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {self.source}")

    @property
    def fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
        return fps if fps and fps > 1 else 25.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if ok:
            return True, frame
        if self.kind is SourceKind.FILE:
            return False, None  # файл закончился

        # камера / RTSP: переподключаемся
        for attempt in range(1, self.max_reconnects + 1):
            print(f"Потеряно соединение, попытка {attempt}/{self.max_reconnects}...")
            self.release()
            time.sleep(2)
            try:
                self.open()
            except RuntimeError:
                continue
            ok, frame = self._cap.read()
            if ok:
                return True, frame
        return False, None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
