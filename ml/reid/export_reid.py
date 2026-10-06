"""
Экспорт обученной ReID-модели (VehicleReIDNet, best_checkpoint.pt) в ONNX,
согласно этапу 4 порядка разработки АИС из ТЗ: "Экспорт и оптимизация
моделей — подготовка обученных моделей к использованию в составе АИС".

Симметричен ml/detection/export_detector.py: та же логика (переносимый
формат для CPU/GPU-гибкости, независимость от полного веса обучающего
фреймворка при развёртывании), только для другой модели.

Зачем нужна обёртка ReIDEmbedderWrapper: VehicleReIDNet.forward() в
eval-режиме возвращает (embedding, None, normalized_embedding) - поле
logits становится None, т.к. classification head на инференсе не нужен
(см. ml/reid/models/efficientnetv2_reid.py). torch.onnx.export не умеет
сериализовать None среди выходов графа, поэтому наружу отдаём только
единственный тензор, который реально нужен на инференсе -
L2-нормализованный эмбеддинг (именно его кладём в pgvector/FAISS и
сравниваем по косинусной близости).

Пример запуска:

    python ml/reid/export_reid.py \
        --checkpoint ml/reid/checkpoints/best_checkpoint.pt \
        --output ml/reid/exported \
        --imgsz 224 224
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from models.efficientnetv2_reid import VehicleReIDNet


class ReIDEmbedderWrapper(nn.Module):
    """Оборачивает VehicleReIDNet для экспорта: наружу отдаём только
    L2-нормализованный эмбеддинг - единственное, что нужно на инференсе."""

    def __init__(self, model: VehicleReIDNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, normalized_embedding = self.model(x)
        return normalized_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт VehicleReIDNet в ONNX")

    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Путь к обученному best_checkpoint.pt")
    parser.add_argument("--output", type=str, required=True,
                         help="Папка, куда положить итоговый .onnx файл")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[224, 224],
                         help="Размер входа (H W), ДОЛЖЕН совпадать с --image-size, "
                              "использованным при обучении (train.py), иначе точность "
                              "эмбеддингов на инференсе может ухудшиться.")
    parser.add_argument("--dynamic-batch", action="store_true",
                         help="Разрешить переменный размер батча в ONNX-графе "
                              "(полезно, если сервис будет обрабатывать несколько "
                              "ТС с одного кадра одним батчем, а не по одному)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-verify", action="store_true",
                         help="Пропустить сверку ONNX Runtime с PyTorch после экспорта")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Не найден чекпоинт: {args.checkpoint}")

    os.makedirs(args.output, exist_ok=True)

    print(f"Загружаю чекпоинт: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    num_train_pids = checkpoint["num_train_pids"]
    embedding_dim = checkpoint.get("embedding_dim", 512)

    model = VehicleReIDNet(
        num_classes=num_train_pids, embedding_dim=embedding_dim, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Эпоха чекпоинта: {checkpoint.get('epoch', '?')}, embedding_dim={embedding_dim}")

    wrapper = ReIDEmbedderWrapper(model)
    wrapper.eval()

    h, w = args.imgsz
    dummy_input = torch.randn(1, 3, h, w)

    output_path = os.path.join(args.output, "reid_embedder.onnx")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        }

    print(f"Экспортирую в ONNX (imgsz={h}x{w}, opset={args.opset}, "
          f"dynamic_batch={args.dynamic_batch})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["embedding"],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            # Явно используем классический (TorchScript-based) экспортёр,
            # а не новый dynamo-based (torch >= 2.x делает его дефолтным) -
            # тот требует дополнительную зависимость onnxscript, которой
            # может не быть в окружении, и пока менее предсказуемо ведёт
            # себя на нестандартных модулях вроде обёрток с распаковкой
            # кортежа внутри forward.
            dynamo=False,
        )

    print(f"Сохранено: {output_path}")

    if not args.skip_verify:
        _verify_and_benchmark(wrapper, output_path, dummy_input)

    print()
    print("Готово.")
    print(
        "Используйте этот путь в backend/app/services/reid_embedder.py "
        "(onnxruntime.InferenceSession напрямую, т.к. в отличие от "
        "ultralytics.YOLO, у ReID-модели нет обёртки, понимающей .onnx "
        "прозрачно - нужен обычный onnxruntime inference)."
    )


def _verify_and_benchmark(
    wrapper: nn.Module, onnx_path: str, dummy_input: torch.Tensor
) -> None:
    """
    Две проверки после экспорта:
      1. Совпадение выхода ONNX Runtime и PyTorch на одном и том же входе
         (защита от тихой ошибки конвертации графа).
      2. Замер задержки инференса на CPU - прямая проверка требования
         ТЗ (п.1.3.2.2): "время формирования признакового описания
         одного ТС не должно превышать 300 мс на CPU".
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "  ПРЕДУПРЕЖДЕНИЕ: onnxruntime не установлен, пропускаю проверку "
            "и замер задержки (pip install onnxruntime)."
        )
        return

    print("Проверяю совпадение выходов PyTorch и ONNX Runtime...")
    with torch.no_grad():
        torch_output = wrapper(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    print(f"  Максимальное расхождение PyTorch vs ONNX: {max_abs_diff:.2e}")
    if max_abs_diff > 1e-3:
        print(
            "  ПРЕДУПРЕЖДЕНИЕ: расхождение больше ожидаемого для fp32-экспорта. "
            "Проверьте версию opset / операции модели на совместимость."
        )
    else:
        print("  OK - выходы практически идентичны.")

    print("Замеряю задержку инференса на CPU (требование ТЗ: <= 300 мс)...")
    num_warmup, num_runs = 3, 20
    for _ in range(num_warmup):
        session.run(None, {"input": dummy_input.numpy()})

    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {"input": dummy_input.numpy()})
        timings.append((time.perf_counter() - start) * 1000)

    mean_ms = float(np.mean(timings))
    p95_ms = float(np.percentile(timings, 95))
    print(f"  Среднее время: {mean_ms:.1f} мс, p95: {p95_ms:.1f} мс (batch=1, CPU)")
    if mean_ms > 300:
        print(
            "  ПРЕДУПРЕЖДЕНИЕ: среднее время превышает лимит ТЗ (300 мс на CPU). "
            "Рассмотрите более лёгкий backbone или квантизацию модели."
        )
    else:
        print("  Требование ТЗ по задержке на CPU выполняется.")


if __name__ == "__main__":
    main()
