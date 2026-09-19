"""
Экспорт обученной YOLO11-модели детекции в ONNX, согласно этапу 4
порядка разработки АИС из ТЗ: "Экспорт и оптимизация моделей —
подготовка обученных моделей к использованию в составе АИС".

Зачем ONNX, а не просто .pt: ТЗ явно требует (п. 1.3.2.3), чтобы
серверная часть могла работать "как с использованием GPU, так и на CPU
при отсутствии ускорителя". Экспорт в ONNX даёт переносимый формат,
который ultralytics и onnxruntime одинаково умеют запускать на обоих
устройствах, и который проще версионировать/деплоить в Docker-образе
сервера, не таская за собой полный вес обучающего фреймворка.

Пример запуска:

    python ml/detection/export_detector.py \
        --checkpoint ml/detection/runs/vehicle_detector/weights/best.pt \
        --output ml/detection/exported \
        --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import shutil

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт YOLO11 детектора в ONNX")

    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Путь к обученному best.pt")
    parser.add_argument("--output", type=str, required=True,
                         help="Папка, куда положить итоговый .onnx файл")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Размер входа, ЗАФИКСИРОВАННЫЙ в графе ONNX. "
                              "Должен совпадать с --imgsz, использованным при "
                              "обучении (train_detector.py), иначе точность "
                              "детекции на инференсе может ухудшиться.")
    parser.add_argument("--dynamic", action="store_true",
                         help="Разрешить переменный размер батча/входа в ONNX-графе "
                              "(немного медленнее, но гибче для разных клиентов)")
    parser.add_argument("--simplify", action="store_true", default=True,
                         help="Упростить ONNX-граф через onnxsim (обычно ускоряет "
                              "инференс без потери точности)")
    parser.add_argument("--opset", type=int, default=17)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Не найден чекпоинт: {args.checkpoint}")

    os.makedirs(args.output, exist_ok=True)

    print(f"Загружаю чекпоинт: {args.checkpoint}")
    model = YOLO(args.checkpoint)

    print(f"Экспортирую в ONNX (imgsz={args.imgsz}, opset={args.opset})...")
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )

    final_path = os.path.join(args.output, "vehicle_detector.onnx")
    shutil.move(str(exported_path), final_path)

    print()
    print("Готово.")
    print(f"ONNX-модель сохранена: {final_path}")
    print(
        "Используйте этот путь в backend/app/services/detector.py "
        "(YOLO(path_to_onnx) работает точно так же, как YOLO(path_to_pt), "
        "ultralytics сам определяет формат по расширению файла)."
    )


if __name__ == "__main__":
    main()
