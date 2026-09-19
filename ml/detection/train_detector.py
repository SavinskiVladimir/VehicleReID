"""
Дообучение YOLO11 (YOLO11s / YOLO11m) на задаче детекции единого класса
"vehicle", согласно этапу 2 порядка разработки АИС из ТЗ:
"Разработка и обучение модели детекции — дообучение модели семейства
YOLOv11 на задаче обнаружения ТС на изображении".

Модель стартует с ImageNet/COCO-претрейна ultralytics (YOLO11 уже умеет
находить car/truck/bus "из коробки") и дообучается на CCTV-специфичных
кадрах UA-DETRAC, подготовленных ml/detection/prepare_dataset.py.

Пример запуска:

    python ml/detection/train_detector.py \
        --data data/detection_yolo/data.yaml \
        --model yolo11s.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16 \
        --project ml/detection/runs \
        --name vehicle_detector

Возобновление после прерывания (ultralytics хранит last.pt в
<project>/<name>/weights/last.pt и сам восстанавливает состояние
оптимизатора/эпохи):

    python ml/detection/train_detector.py \
        --resume ml/detection/runs/vehicle_detector/weights/last.pt
"""

from __future__ import annotations

import argparse
import os

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Дообучение YOLO11 на детекции vehicle")

    parser.add_argument("--data", type=str,
                         help="Путь к data.yaml (из prepare_dataset.py). "
                              "Не требуется при --resume.")
    parser.add_argument("--model", type=str, default="yolo11s.pt",
                         help="Стартовый чекпоинт: yolo11n.pt (самый лёгкий) / "
                              "yolo11s.pt / yolo11m.pt (по архитектуре проекта - "
                              "s или m). Скачивается автоматически ultralytics "
                              "при первом запуске.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15,
                         help="Остановка, если mAP не растёт N эпох подряд (early stopping)")
    parser.add_argument("--device", type=str, default=None,
                         help="'0' для первого GPU, 'cpu' для CPU. По умолчанию "
                              "ultralytics сам выбирает GPU, если доступен.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="ml/detection/runs",
                         help="Куда сохранять результаты обучения (веса, графики, логи)")
    parser.add_argument("--name", type=str, default="vehicle_detector",
                         help="Имя конкретного запуска (подпапка внутри --project)")
    parser.add_argument("--resume", type=str, default=None,
                         help="Путь к last.pt для возобновления прерванного обучения "
                              "(если указан, --data/--model/--project/--name не нужны - "
                              "ultralytics берёт их из сохранённой конфигурации запуска)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.resume is None and args.data is None:
        parser.error("Нужно указать --data (для нового обучения) или --resume (для продолжения)")
    return args


def main() -> None:
    args = parse_args()

    if args.resume:
        print(f"Возобновляю обучение из: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"Не найден {args.data}. Сначала запустите "
            f"ml/detection/prepare_dataset.py, чтобы получить data.yaml."
        )

    print(f"Загружаю стартовые веса: {args.model}")
    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        # Единственный класс "vehicle" - боксы разного типа транспорта
        # уже слиты в один класс на этапе prepare_dataset.py, здесь
        # никаких дополнительных настроек под multi-class не нужно.
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    print()
    print("Обучение завершено.")
    print(f"Лучший чекпоинт: {best_path}")
    print(f"Метрики финального запуска: {results}")


if __name__ == "__main__":
    main()
