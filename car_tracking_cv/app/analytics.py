"""Аналитика треков: скорость (NumPy), цвет (scikit-learn), лог и экспорт (Pandas/Polars)."""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans

from .tracker import Track


def dominant_color_bgr(crop: np.ndarray, k: int = 3) -> tuple[int, int, int]:
    """Доминирующий цвет области (KMeans по пикселям уменьшенного кропа)."""
    import cv2

    small = cv2.resize(crop, (32, 32)).reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(small)
    biggest = np.bincount(km.labels_).argmax()
    b, g, r = km.cluster_centers_[biggest]
    return int(b), int(g), int(r)


class TrackLog:
    def __init__(self, meters_per_pixel: float | None = None, ema_alpha: float = 0.3):
        self.mpp = meters_per_pixel
        self.alpha = ema_alpha
        self.rows: list[dict] = []
        self._last: dict[int, tuple[float, float, float]] = {}  # id -> (t, x, y)
        self._speed_px: dict[int, float] = {}
        self.colors: dict[int, tuple[int, int, int]] = {}

    def update(self, frame_idx: int, t: float, tracks: list[Track]) -> dict[int, float | None]:
        """Добавляет записи и возвращает скорость по ID (км/ч, либо None без калибровки)."""
        speeds: dict[int, float | None] = {}
        for tr in tracks:
            x, y = tr.center_bottom
            v_px = self._speed_px.get(tr.id, 0.0)
            if tr.id in self._last:
                t0, x0, y0 = self._last[tr.id]
                dt = max(t - t0, 1e-6)
                inst = float(np.hypot(x - x0, y - y0)) / dt
                v_px = self.alpha * inst + (1 - self.alpha) * v_px  # EMA-сглаживание
            self._last[tr.id] = (t, x, y)
            self._speed_px[tr.id] = v_px

            kmh = v_px * self.mpp * 3.6 if self.mpp else None
            speeds[tr.id] = kmh
            self.rows.append({
                "frame": frame_idx, "t": round(t, 3), "id": tr.id, "cls": tr.cls,
                "conf": round(tr.conf, 3),
                "x1": tr.box[0], "y1": tr.box[1], "x2": tr.box[2], "y2": tr.box[3],
                "speed_px_s": round(v_px, 2),
                "speed_kmh": None if kmh is None else round(kmh, 1),
            })
        return speeds

    def summary(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame(self.rows)
        return (
            df.groupby("id")
            .agg(first_frame=("frame", "min"), last_frame=("frame", "max"),
                 frames=("frame", "count"), mean_speed_px_s=("speed_px_s", "mean"))
            .reset_index()
        )

    def export(self, directory: str, name: str = "tracks") -> list[Path]:
        """Сохраняет полный лог (Polars -> CSV + Parquet) и сводку по ID (Pandas -> CSV)."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        if not self.rows:
            return paths
        df = pl.DataFrame(self.rows)
        p_csv, p_pq, p_sum = out / f"{name}.csv", out / f"{name}.parquet", out / f"{name}_summary.csv"
        df.write_csv(p_csv)
        df.write_parquet(p_pq)
        self.summary().to_csv(p_sum, index=False)
        return [p_csv, p_pq, p_sum]
