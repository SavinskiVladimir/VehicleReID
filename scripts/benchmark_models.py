"""
Изолированный замер ONNX-моделей БЕЗ сервера: чистое время инференса детектора и ReID при разном числе
потоков onnxruntime. Отвечает на вопрос "медленно само железо/настройки потоков или что-то в сервере".
Ориентир для CPU-профиля ТЗ (>= 4 ядра): детектор около 100 мс, ReID около 50-150 мс на кроп 224x224.

    python scripts/benchmark_models.py                       # обе модели из ml/*/exported
    python scripts/benchmark_models.py --image data/frames/1.jpg   # + весь DetectorService.detect на реальном кадре
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def percentile95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def time_session(model_path: str, threads: int, runs: int, warmup: int, hw: int = 224, opt_level: str = "all"):
    """Время одного прогона модели (мс): (медиана, p95). threads=0 - настройка onnxruntime по умолчанию."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    levels = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,            # + перекладка в NCHWc
              "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
              "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC}
    options.graph_optimization_level = levels[opt_level]
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else (1 if i == 0 else hw) for i, d in enumerate(spec.shape)]
    batch = {spec.name: np.random.rand(*shape).astype(np.float32)}
    for _ in range(warmup):
        session.run(None, batch)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, batch)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), percentile95(times)


def time_torch(checkpoint: str, threads: int, runs: int, warmup: int, channels_last: bool = False, hw: int = 224):
    """Тот же ReID из чекпоинта в PyTorch (eager, CPU): (медиана, p95) в мс. Эталон для сравнения с onnxruntime."""
    import torch

    sys.path.insert(0, str(ROOT / "ml" / "reid"))
    from models.efficientnetv2_reid import VehicleReIDNet

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # свой чекпоинт, ему доверяем
    model = VehicleReIDNet(num_classes=state["num_train_pids"], embedding_dim=state.get("embedding_dim", 512), pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    x = torch.rand(1, 3, hw, hw)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    if threads:
        torch.set_num_threads(threads)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), percentile95(times)


def build_parser():
    p = argparse.ArgumentParser(description="Изолированный замер ONNX-моделей")
    p.add_argument("--detector", default=str(ROOT / "ml/detection/exported/vehicle_detector.onnx"))
    p.add_argument("--reid", default=str(ROOT / "ml/reid/exported/reid_embedder.onnx"))
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, nargs="*", default=[0, 1, 2, 4, 8], help="0 = по умолчанию")
    p.add_argument("--opt", choices=["all", "extended", "basic"], default="all", help="Уровень оптимизации графа onnxruntime (all включает перекладку в NCHWc)")
    p.add_argument("--torch", default=None, metavar="CHECKPOINT", help="Чекпоинт ReID: замерить его же в PyTorch для сравнения")
    p.add_argument("--image", default=None, help="Кадр: замерить ещё и весь DetectorService.detect (с ultralytics)")
    return p


def run(args) -> int:
    import onnxruntime as ort

    print(f"CPU: {platform.processor() or 'н/д'}, логических ядер: {os.cpu_count()}, onnxruntime {ort.__version__}")
    for title, path in (("Детектор", args.detector), ("ReID", args.reid)):
        if not Path(path).is_file():
            print(f"\n{title}: файл не найден ({path}), пропуск")
            continue
        print(f"\n{title} ({Path(path).name}), чистый инференс, мс:")
        for threads in args.threads:
            median, p95 = time_session(path, threads, args.runs, args.warmup, opt_level=args.opt)
            print(f"  потоков {'по умолчанию' if not threads else threads:>12}: медиана {median:7.0f}   p95 {p95:7.0f}")

    if args.torch:
        print(f"\nReID в PyTorch ({Path(args.torch).name}, eager, CPU), мс:")
        for channels_last in (False, True):
            for threads in args.threads:
                median, p95 = time_torch(args.torch, threads, args.runs, args.warmup, channels_last)
                label = "channels_last" if channels_last else "обычная раскладка"
                print(f"  {label:>17}, потоков {'по умолчанию' if not threads else threads:>12}: медиана {median:7.0f}   p95 {p95:7.0f}")

    if args.image and Path(args.detector).is_file():
        sys.path.insert(0, str(ROOT))
        from PIL import Image

        from backend.app.services.detector import DetectorService

        image = Image.open(args.image).convert("RGB")
        detector = DetectorService(model_path=args.detector)
        for _ in range(args.warmup):
            detector.detect(image)
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            detector.detect(image)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"\nDetectorService.detect на кадре {image.width}x{image.height}: медиана {statistics.median(times):.0f} мс, "
              f"p95 {percentile95(times):.0f} мс (разница с чистым инференсом - предобработка и постобработка ultralytics)")
    return 0


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))