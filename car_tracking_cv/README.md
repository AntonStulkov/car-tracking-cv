# car-tracking-cv

Детекция и трекинг машин с присвоением ID. Источники: видеофайл (mp4, mov), веб/USB-камера, RTSP.

## Стек
Python, YOLOv8 (Ultralytics), ByteTrack, DeblurGAN, PyTorch, scikit-learn, OpenCV, PyQt6,
NumPy, Pandas, Polars, Visual Studio Code.

## Структура
```
main.py                  точка входа (GUI / CLI)
app/
  config.py              конфигурация (dataclass)
  sources.py             файл / камера / RTSP + переподключение
  deblur.py              DeblurGAN (TorchScript) или unsharp mask
  detector.py            YOLOv8 - детекция
  tracker.py             ByteTrack - присвоение ID
  analytics.py           скорость (NumPy), цвет (sklearn), лог и экспорт (Pandas/Polars)
  visualization.py       рамки, ID, траектории
  pipeline.py            сборка всех шагов
  gui/
    main_window.py       окно PyQt6
    worker.py            QThread для обработки видео
.vscode/launch.json      конфигурации запуска для VS Code
```

## Запуск
```bash
pip install -r requirements.txt
python main.py                                   # окно PyQt6
python main.py --source video.mp4 --output out.mp4
python main.py --source 0                        # веб/USB-камера
python main.py --source rtsp://user:pass@192.168.1.10:554/stream
```
Логи (CSV и Parquet) сохраняются в `logs/`.
