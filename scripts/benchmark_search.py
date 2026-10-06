"""
Замер поиска по вектору на базе из N записей (ТЗ п.1.1.2.2: <= 30 мс при 100 000 записей).
Реальные изображения не нужны: во ВРЕМЕННУЮ таблицу benchmark_vectors кладутся N случайных векторов,
строится тот же HNSW-индекс, что в проде (m=16, ef_construction=64), и замеряется запрос top-k
вместе с обращением к БД. Таблица в конце удаляется; рабочие таблицы не затрагиваются.

    python scripts/benchmark_search.py --n 100000
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import statistics
import sys
import time

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

TABLE = "benchmark_vectors"
BUDGET_MS = 30


def percentile95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def build_parser():
    p = argparse.ArgumentParser(description="Замер поиска на N записях")
    p.add_argument("--n", type=int, default=100_000, help="Число записей (по умолчанию 100 000, как в ТЗ)")
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ef-search", type=int, default=40, help="hnsw.ef_search (по умолчанию как в приложении)")
    p.add_argument("--database-url", default=None, help="По умолчанию DATABASE_URL из .env")
    p.add_argument("--keep", action="store_true", help="Не удалять таблицу после замера")
    return p


async def run(args) -> int:
    load_dotenv()
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("ОШИБКА: не задан DATABASE_URL", file=sys.stderr)
        return 1
    engine = create_async_engine(url)
    rng = np.random.default_rng(42)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
            await conn.execute(text(f"CREATE TABLE {TABLE} (id serial PRIMARY KEY, embedding vector({args.dim}))"))
            print(f"Генерирую {args.n} случайных векторов (на стороне БД)...", flush=True)
            await conn.execute(text(f"INSERT INTO {TABLE} (embedding) SELECT ARRAY(SELECT random() - 0.5 "
                                    f"FROM generate_series(1, {args.dim}))::vector FROM generate_series(1, {args.n})"))
            print("Строю HNSW-индекс (m=16, ef_construction=64)...", flush=True)
            t0 = time.perf_counter()
            await conn.execute(text(f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"))
            build_s = time.perf_counter() - t0
            await conn.execute(text(f"ANALYZE {TABLE}"))

        query = f"SELECT id FROM {TABLE} ORDER BY embedding <=> CAST(:q AS vector) LIMIT {int(args.top_k)}"
        vectors = ["[" + ",".join(f"{x:.5f}" for x in rng.standard_normal(args.dim)) + "]" for _ in range(args.queries + 3)]
        times = []
        async with engine.connect() as conn:
            await conn.execute(text(f"SET hnsw.ef_search = {int(args.ef_search)}"))
            plan = "\n".join(r[0] for r in (await conn.execute(text("EXPLAIN " + query), {"q": vectors[0]})).all())
            for i, vec in enumerate(vectors):
                t0 = time.perf_counter()
                (await conn.execute(text(query), {"q": vec})).all()
                if i >= 3:  # первые запросы - прогрев
                    times.append((time.perf_counter() - t0) * 1000)
        median, p95 = statistics.median(times), percentile95(times)
        verdict = "выполнено" if p95 <= BUDGET_MS else ("на грани (p95 выше)" if median <= BUDGET_MS else "НЕ выполнено")
        print(f"\nЗаписей: {args.n}, индекс построен за {build_s:.0f} с, индекс в плане запроса: {'да' if 'index scan' in plan.lower() else 'НЕТ'}")
        print(f"Поиск top-{args.top_k} (ef_search={args.ef_search}), {len(times)} запросов, с обращением к БД: "
              f"медиана {median:.1f} мс, p95 {p95:.1f} мс, бюджет {BUDGET_MS} мс - {verdict}")
        return 0
    finally:
        if not args.keep:
            async with engine.begin() as conn:
                await conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(build_parser().parse_args())))