"""Точка входа.

  python main.py                      - окно PyQt6
  python main.py --source <src>       - консольный режим (src: файл / индекс камеры / rtsp://...)
"""
import argparse
import sys


def run_gui() -> None:
    from PyQt6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


def run_cli(args) -> None:
    import time

    import cv2

    from app.config import Config
    from app.pipeline import Pipeline
    from app.sources import VideoSource

    cfg = Config(model_path=args.model, classes=args.classes, conf=args.conf,
                 use_deblur=args.deblur, deblur_weights=args.deblur_weights,
                 meters_per_pixel=args.mpp)
    pipeline = Pipeline(cfg)
    src = VideoSource(args.source)
    src.open()
    pipeline.start(src.fps)

    writer, idx, t0 = None, 0, time.time()
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            t = idx / src.fps if src.kind.value == "file" else time.time() - t0
            annotated, _ = pipeline.process(frame, idx, t)
            idx += 1

            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src.fps, (w, h))
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow("Car tracking", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        src.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    paths = pipeline.log.export(cfg.log_dir)
    print(f"Уникальных машин (ID): {len(pipeline.seen_ids)}")
    for p in paths:
        print("Сохранено:", p)


def parse_args():
    p = argparse.ArgumentParser(description="Трекинг машин с ID")
    p.add_argument("--source", help="файл (mp4/mov), индекс камеры или rtsp://. Без него - GUI")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--classes", type=int, nargs="+", default=[2])
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--deblur", action="store_true")
    p.add_argument("--deblur-weights", default=None)
    p.add_argument("--mpp", type=float, default=None, help="метров на пиксель (для км/ч)")
    p.add_argument("--output", default=None, help="сохранить видео с разметкой")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_cli(a) if a.source else run_gui()
