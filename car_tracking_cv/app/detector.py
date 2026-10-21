"""Детекция машин (YOLOv8 / другие версии Ultralytics, PyTorch)."""
import numpy as np
from ultralytics import YOLO

from .config import Config


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.names = self.model.names

    def detect(self, frame: np.ndarray):
        """Возвращает ultralytics Boxes (numpy-версию) с xywh/xyxy, conf, cls."""
        res = self.model.predict(
            frame,
            classes=self.cfg.classes,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )[0]
        return res.boxes.cpu().numpy()
