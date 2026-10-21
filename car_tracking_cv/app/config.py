"""Единая конфигурация проекта."""
from dataclasses import dataclass, field


@dataclass
class TrackerConfig:
    # Параметры совпадают с bytetrack.yaml из Ultralytics
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True


@dataclass
class Config:
    # --- детектор (YOLOv8; подойдут и другие версии Ultralytics: yolo11n.pt и т.д.) ---
    model_path: str = "yolov8n.pt"
    classes: list[int] = field(default_factory=lambda: [2])  # COCO: 2 car, 5 bus, 7 truck
    conf: float = 0.3
    imgsz: int = 640
    device: str | None = None  # None -> авто (cuda, если доступна)

    # --- деблюр (DeblurGAN) ---
    use_deblur: bool = False
    deblur_weights: str | None = None  # путь к TorchScript-генератору (.pt)

    # --- трекер (ByteTrack) ---
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # --- аналитика ---
    meters_per_pixel: float | None = None  # калибровка для скорости в км/ч
    trail_length: int = 40
    log_dir: str = "logs"
