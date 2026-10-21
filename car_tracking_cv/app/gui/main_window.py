"""Главное окно PyQt6."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from ..config import Config
from .worker import VideoWorker

KINDS = ["Видеофайл (mp4, mov)", "Веб/USB-камера", "RTSP-поток"]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car tracking")
        self.resize(1100, 760)
        self.worker: VideoWorker | None = None
        self.last_log = None

        self.kind = QComboBox()
        self.kind.addItems(KINDS)
        self.kind.currentIndexChanged.connect(self._on_kind)
        self.source_edit = QLineEdit()
        self.browse_btn = QPushButton("Обзор…")
        self.browse_btn.clicked.connect(self._browse)
        self.deblur_cb = QCheckBox("Деблюр")
        self.start_btn = QPushButton("Старт")
        self.stop_btn = QPushButton("Стоп")
        self.export_btn = QPushButton("Экспорт логов")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.export_btn.clicked.connect(self.export)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        top = QHBoxLayout()
        for w in (self.kind, self.source_edit, self.browse_btn, self.deblur_cb,
                  self.start_btn, self.stop_btn, self.export_btn):
            top.addWidget(w)
        top.setStretch(1, 1)

        self.video = QLabel("Выберите источник и нажмите «Старт»")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#111;color:#aaa;")
        self.status = QLabel("Уникальных машин: 0   FPS: 0")

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.video, 1)
        lay.addWidget(self.status)
        self._on_kind(0)

    def _on_kind(self, i: int) -> None:
        self.browse_btn.setVisible(i == 0)
        self.source_edit.setPlaceholderText(
            ["Путь к видеофайлу", "Индекс камеры, например 0", "rtsp://user:pass@host:554/stream"][i]
        )

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Видео", "", "Video (*.mp4 *.mov *.avi *.mkv *.m4v *.webm)")
        if path:
            self.source_edit.setText(path)

    def start(self) -> None:
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "Источник", "Укажите источник видео")
            return
        cfg = Config(use_deblur=self.deblur_cb.isChecked())
        self.worker = VideoWorker(source, cfg)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.stats.connect(lambda n, f: self.status.setText(
            f"Уникальных машин: {n}   FPS: {f:.1f}"))
        self.worker.error.connect(lambda m: QMessageBox.critical(self, "Ошибка", m))
        self.worker.log_ready.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()

    def _show_frame(self, img: QImage) -> None:
        pm = QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.video.setPixmap(pm)

    def _on_log(self, log) -> None:
        self.last_log = log
        self.export_btn.setEnabled(bool(log.rows))

    def _on_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def export(self) -> None:
        if self.last_log:
            paths = self.last_log.export(Config().log_dir)
            QMessageBox.information(self, "Экспорт", "Сохранено:\n" + "\n".join(map(str, paths)))

    def closeEvent(self, e) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(e)
