"""
Замер быстродействия API по бюджетам ТЗ п.1.1.2.2: прогрев, N запросов, медиана и p95 каждого
этапа рядом с бюджетом и итогом (выполнено / на грани / не выполнено).

Этапы берутся из заголовка Server-Timing (backend/app/timing.py), общее время - со стороны клиента
(так, как в ТЗ: вместе с сетью). "Запрос" - цикл /api/search: он начинается с выбора ТС пользователем и включает детекцию, кроп,
эмбеддинг и поиск. Время /api/detect и ожидание выбора пользователем в бюджет не входят.

    Кроп ТС        detect + crop из /api/search       <= 100 мс (CPU) / 3 мс (GPU)
    Эмбеддинг      embed                              <= 800 мс (CPU) / 250 мс (GPU)
    Поиск в БД     search                             <= 30 мс при 100 000 записей
    Весь запрос    весь /api/search, замер клиента    <= 1000 мс (CPU) / 500 мс (GPU)

Сервер должен быть запущен. Берите КАДРЫ с машинами, а не готовые кропы. Каждый запуск /api/search
пишет запись в историю поиска (журнал ограничен SEARCH_HISTORY_MAX_ITEMS).

    python scripts/benchmark_api.py data/frames -n 20 --profile cpu
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

BUDGETS_MS = {
    "cpu": {"crop": 100, "embed": 800, "search": 30, "total": 1000},
    "gpu": {"crop": 3, "embed": 250, "search": 30, "total": 500},
}
LABELS = {
    "crop": "Кроп ТС (detect + crop)",
    "embed": "Эмбеддинг (embed)",
    "search": "Поиск в БД (search)",
    "total": "Весь запрос (/api/search, с сетью)",
}
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def parse_server_timing(value: str) -> Dict[str, float]:
    """'detect;dur=41.2, embed;dur=88.0' -> {'detect': 41.2, 'embed': 88.0}."""
    return {name: float(ms) for name, ms in re.findall(r"([A-Za-z_]+);dur=([\d.]+)", value or "")}


def percentile(values: List[float], q: float) -> float:
    """Метод ближайшего ранга: p95 из 20 замеров - 19-й по возрастанию."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def verdict(median: float, p95: float, budget: float) -> str:
    if p95 <= budget:
        return "выполнено"
    return "на грани (p95 выше)" if median <= budget else "НЕ выполнено"


def collect_metrics(run: dict) -> Dict[str, float]:
    s = run["search_stages"]
    return {
        "crop": s.get("detect", 0.0) + s.get("crop", 0.0),
        "embed": s.get("embed", 0.0),
        "search": s.get("search", 0.0),
        "total": run["search_wall"],
    }


def summarize(runs: List[dict], profile: str):
    """Возвращает (строки с итогами по четырём бюджетам ТЗ, предупреждения)."""
    metrics = [collect_metrics(r) for r in runs]
    rows = []
    for key, budget in BUDGETS_MS[profile].items():
        values = [m[key] for m in metrics]
        median, p95 = statistics.median(values), percentile(values, 0.95)
        rows.append({"key": key, "name": LABELS[key], "median": median, "p95": p95,
                     "budget": budget, "verdict": verdict(median, p95, budget)})
    warnings = []
    if any("detect" not in r["search_stages"] for r in runs):
        warnings.append("В Server-Timing ответа /api/search нет этапа detect: детекция не попала в замер. "
                        "Проверьте, что в search_vehicle в _get_selected_detection передан timer.")
    return rows, warnings


def stage_medians(runs: List[dict], endpoint: str) -> Dict[str, float]:
    names = {n for r in runs for n in r[f"{endpoint}_stages"]}
    return {n: statistics.median(r[f"{endpoint}_stages"].get(n, 0.0) for r in runs) for n in sorted(names)}


def one_run(client, image_path: Path, top_k: int) -> Optional[dict]:
    """Один запрос пользователя; None, если на кадре не найдено ни одного ТС."""
    data = image_path.read_bytes()
    t0 = time.perf_counter()
    detect = client.post("/api/detect", files={"file": (image_path.name, data, IMAGE_TYPES[image_path.suffix.lower()])})
    detect_wall = (time.perf_counter() - t0) * 1000
    detect.raise_for_status()
    detections = detect.json()["detections"]
    if not detections:
        return None
    t1 = time.perf_counter()
    search = client.post("/api/search", json={"upload_id": detect.json()["upload_id"], "detection_index": 0, "top_k": top_k})
    search_wall = (time.perf_counter() - t1) * 1000
    search.raise_for_status()
    return {"detect_wall": detect_wall, "search_wall": search_wall,
            "detect_stages": parse_server_timing(detect.headers.get("server-timing", "")),
            "search_stages": parse_server_timing(search.headers.get("server-timing", ""))}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Замер быстродействия API по бюджетам ТЗ")
    p.add_argument("images", help="Папка с кадрами или один файл")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Адрес сервера")
    p.add_argument("-n", type=int, default=20, help="Число замеряемых запросов (по умолчанию 20)")
    p.add_argument("--warmup", type=int, default=3, help="Прогревочные запросы, не входят в статистику")
    p.add_argument("--profile", choices=list(BUDGETS_MS), default="cpu", help="Бюджеты ТЗ: cpu или gpu")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--timeout", type=float, default=120, help="Таймаут одного запроса, с")
    p.add_argument("--insecure", action="store_true", help="Не проверять TLS-сертификат")
    return p


def run(args: argparse.Namespace, client=None) -> int:
    path = Path(args.images)
    files = [path] if path.is_file() else sorted(p for p in path.glob("*") if p.suffix.lower() in IMAGE_TYPES) if path.is_dir() else []
    if not files:
        print(f"ОШИБКА: изображения не найдены: {path}", file=sys.stderr)
        return 1
    if client is None:
        try:
            import httpx
        except ImportError:
            print("ОШИБКА: нужен пакет httpx (pip install httpx)", file=sys.stderr)
            return 1
        client = httpx.Client(base_url=args.url, timeout=args.timeout, verify=not args.insecure)

    try:
        records = client.get("/api/observations", params={"limit": 1}).json()["total"]
        print(f"Сервер {args.url}, записей в базе: {records}, профиль {args.profile}. Прогрев: {args.warmup}, замеров: {args.n}...")
        runs, skipped = [], 0
        for i in range(args.warmup + args.n):
            t_req = time.perf_counter()
            result = one_run(client, files[i % len(files)], args.top_k)
            print(f"  {i + 1}/{args.warmup + args.n}: {files[i % len(files)].name} "
                  f"{(time.perf_counter() - t_req) * 1000:.0f} мс", flush=True)
            if result is None:
                skipped += 1
            elif i >= args.warmup:
                runs.append(result)
    except Exception as exc:
        print(f"ОШИБКА при обращении к серверу: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not runs:
        print("ОШИБКА: ни на одном кадре не найдено ТС, замерять нечего.", file=sys.stderr)
        return 1

    rows, warnings = summarize(runs, args.profile)
    print(f"\nЗамеров: {len(runs)}" + (f" (кадров без ТС пропущено: {skipped})" if skipped else "") + "\n")
    print(f"{'Метрика (ТЗ п.1.1.2.2)':<42}{'медиана':>10}{'p95':>10}{'бюджет':>9}   итог")
    for r in rows:
        print(f"{r['name']:<42}{r['median']:>8.0f}мс{r['p95']:>8.0f}мс{r['budget']:>7}мс   {r['verdict']}")
    print("\nМедианы всех этапов, мс:")
    for endpoint in ("detect", "search"):
        print(f"  /api/{endpoint}: " + ", ".join(f"{n}={ms:.0f}" for n, ms in stage_medians(runs, endpoint).items()))
    if records < 100_000:
        print(f"\nПоиск замерен на {records} записях; требование ТЗ - до 100 000. Для проверки нужна база такого размера.")
    for w in warnings:
        print(f"\nПРЕДУПРЕЖДЕНИЕ: {w}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())