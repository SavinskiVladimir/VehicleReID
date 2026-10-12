"""
Оценка качества модели детекции ТС на валидационной выборке (метрики Precision, Recall,
mAP@50, mAP@50-95), согласно разделу об оценке качества моделей.

Это отдельный повторный расчёт по готовому чекпоинту (best.pt или .onnx), без дообучения.
Ultralytics сохраняет графики (кривые PR/F1, матрицу ошибок) рядом с результатами.

    python ml/detection/evaluate_detector.py \
        --model ml/detection/runs/vehicle_detector/weights/best.pt \
        --data data/detection_yolo/data.yaml
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оценка качества детектора ТС (mAP@50, mAP@50-95)")
    p.add_argument("--model", type=str, required=True, help="Путь к best.pt (или .onnx)")
    p.add_argument("--data", type=str, required=True, help="data.yaml из prepare_dataset.py")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--project", type=str, default="ml/detection/runs")
    p.add_argument("--name", type=str, default="evaluate")
    p.add_argument("--workers", type=int, default=4, help="Число воркеров DataLoader (0 - без многопроцессности)")
    return p.parse_args()


def extract_metrics(box) -> dict:
    """Достаёт четыре метрики из results.box (Ultralytics) в процентах."""
    return {
        "precision": float(box.mp) * 100,
        "recall": float(box.mr) * 100,
        "map50": float(box.map50) * 100,
        "map50_95": float(box.map) * 100,
    }


def format_metrics(m: dict) -> str:
    return (
        f"Precision:  {m['precision']:.1f} %\n"
        f"Recall:     {m['recall']:.1f} %\n"
        f"mAP@50:     {m['map50']:.1f} %\n"
        f"mAP@50-95:  {m['map50_95']:.1f} %"
    )


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Не найден файл модели: {args.model}")
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Не найден {args.data}. Сначала запустите prepare_dataset.py")

    from ultralytics import YOLO

    kwargs = dict(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                  workers=args.workers, project=args.project, name=args.name, plots=True, exist_ok=True)
    if args.device is not None:
        kwargs["device"] = args.device

    results = YOLO(args.model).val(**kwargs)
    metrics = extract_metrics(results.box)

    print("\n" + format_metrics(metrics))
    out_dir = os.path.join(args.project, args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nМетрики сохранены: {out_json}; графики (PR/F1-кривые, матрица ошибок): {out_dir}")


if __name__ == "__main__":
    main()