"""Отрисовка рамок, ID, скорости и траекторий."""
import cv2
import numpy as np

from .tracker import Track


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return ((track_id * 37) % 256, (track_id * 91) % 256, (track_id * 151) % 256)


def draw_tracks(frame: np.ndarray, tracks: list[Track], names: dict, speeds: dict,
                trails: dict) -> np.ndarray:
    for tr in tracks:
        color = color_for_id(tr.id)
        x1, y1, x2, y2 = map(int, tr.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tr.id} {names.get(tr.cls, tr.cls)}"
        kmh = speeds.get(tr.id)
        if kmh is not None:
            label += f" {kmh:.0f} km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        pts = list(trails.get(tr.id, []))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)
    return frame
