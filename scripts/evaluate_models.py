"""
Оценка качества моделей на тестовой выборке (этап 8 плана): ReID на VeRi-776 в трёх конфигурациях
(базовая, + flip-TTA, + flip-TTA + re-ranking) и, если указан --detector-data, детектор на валидации
(mAP50, mAP50-95, точность, полнота). Результаты - таблицей и в runs/evaluation/metrics.json.

    python scripts/evaluate_models.py --veri-root data/VeRi
    python scripts/evaluate_models.py --veri-root data/VeRi --detector-data path/to/data.yaml
"""

from __future__ import annotations

import os
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [("базовая", []), ("+ flip-TTA", ["--flip-tta"]), ("+ flip-TTA + re-ranking", ["--flip-tta", "--rerank"])]
PATTERNS = {"mAP": r"mAP:\s+([\d.]+)%", "Rank-1": r"Rank-1:\s+([\d.]+)%", "Rank-5": r"Rank-5:\s+([\d.]+)%", "Rank-10": r"Rank-10:\s+([\d.]+)%"}


def parse_metrics(output: str):
    """Достаёт mAP и Rank-k (в процентах) из вывода ml/reid/evaluate.py; None, если чего-то нет."""
    found = {}
    for name, pattern in PATTERNS.items():
        match = re.search(pattern, output)
        if not match:
            return None
        found[name] = float(match.group(1))
    return found


def evaluate_reid(args):
    rows = []
    for title, extra in CONFIGS:
        cmd = [sys.executable, str(ROOT / "ml/reid/evaluate.py"), "--data-root", str(Path(args.veri_root).resolve()),
               "--checkpoint", str(Path(args.checkpoint).resolve()), "--image-size", "224", "224", *extra]
        if args.device:
            cmd += ["--device", args.device]
        print(f"ReID: {title} ...", flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + env["PYTHONPATH"]
            if env.get("PYTHONPATH")
            else ""
        )
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        metrics = parse_metrics(result.stdout)
        if result.returncode != 0 or metrics is None:
            print(result.stdout[-1500:], result.stderr[-1500:], file=sys.stderr)
            raise RuntimeError(f"evaluate.py не отработал для конфигурации «{title}»")
        rows.append({"config": title, **metrics})
    return rows


def evaluate_detector(args):
    from ultralytics import YOLO

    m = YOLO(args.detector).val(data=args.detector_data, imgsz=640, device=args.device or "cpu", verbose=False).box
    return {"mAP50": float(m.map50), "mAP50-95": float(m.map), "precision": float(m.mp), "recall": float(m.mr)}


def build_parser():
    p = argparse.ArgumentParser(description="Метрики моделей на тестовой выборке")
    p.add_argument("--veri-root", default="data/VeRi")
    p.add_argument("--checkpoint", default="ml/reid/checkpoints/best_checkpoint.pt")
    p.add_argument("--detector", default="ml/detection/runs/vehicle_detector/weights/best.pt")
    p.add_argument("--detector-data", default=None, help="data.yaml YOLO-датасета: включает оценку детектора")
    p.add_argument("--device", default=None, help="cpu или cuda (по умолчанию выбирает evaluate.py)")
    p.add_argument("--out", default="runs/evaluation")
    return p


def run(args) -> int:
    result = {"reid": evaluate_reid(args)}
    print(f"\n{'ReID, VeRi-776 (1678 query)':<32}{'mAP':>8}{'Rank-1':>9}{'Rank-5':>9}{'Rank-10':>9}")
    for r in result["reid"]:
        print(f"{r['config']:<32}{r['mAP']:>7.2f}%{r['Rank-1']:>8.2f}%{r['Rank-5']:>8.2f}%{r['Rank-10']:>8.2f}%")
    if args.detector_data:
        result["detector"] = evaluate_detector(args)
        d = result["detector"]
        print(f"\nДетектор: mAP50 {d['mAP50']:.3f}, mAP50-95 {d['mAP50-95']:.3f}, точность {d['precision']:.3f}, полнота {d['recall']:.3f}")
    else:
        print("\nДетектор: не оценивался (нужен --detector-data с data.yaml валидационной выборки).")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))