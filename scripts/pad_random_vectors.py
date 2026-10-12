"""
Временное дополнение РАБОЧЕЙ таблицы observations случайными векторами до заданного объёма
(ТЗ п.1.1.2.2: поиск <= 30 мс при 100 000 записей) - чтобы замерить быстродействие всего API
(scripts/benchmark_api.py) на базе нужного размера, а не только изолированный запрос в БД.

Синтетические записи помечаются путём image_path/crop_path вида '__synthetic__/<номер>',
по этой метке их можно полностью удалить, не затрагивая реальные наблюдения:

    python scripts/pad_random_vectors.py --target 100000    # дополнить до 100 000 записей всего
    python scripts/pad_random_vectors.py --status           # сколько реальных и синтетических записей
    python scripts/pad_random_vectors.py --cleanup          # удалить ВСЕ синтетические записи

Повторный запуск с тем же --target ничего не добавит: учитываются уже имеющиеся синтетические
записи. После --cleanup выполняется VACUUM ANALYZE (убирает следы удалённых векторов из
HNSW-индекса); для полного перестроения индекса добавьте --reindex.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

MARKER = "__synthetic__/"
DIM = 512
CHUNK = 10_000


def rows_to_add(total: int, synthetic: int, target: int) -> int:
    """Сколько синтетических записей нужно добавить, чтобы всего стало target.

    total - всего записей в таблице (вместе с синтетическими), synthetic - из них синтетических.
    Если записей уже не меньше target, добавлять нечего (реальные записи никогда не удаляются).
    """
    if target < 0:
        raise ValueError("target должен быть неотрицательным")
    return max(0, target - total)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Временное дополнение observations случайными векторами")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", type=int, help="Сколько записей должно быть в таблице всего")
    group.add_argument("--cleanup", action="store_true", help="Удалить все синтетические записи")
    group.add_argument("--status", action="store_true", help="Показать число реальных и синтетических записей")
    p.add_argument("--reindex", action="store_true", help="С --cleanup: перестроить индексы таблицы")
    p.add_argument("--database-url", default=None, help="По умолчанию DATABASE_URL из .env")
    return p


async def counts(conn) -> tuple[int, int]:
    row = (await conn.execute(text(
        "SELECT count(*), count(*) FILTER (WHERE starts_with(image_path, :m)) FROM observations"
    ), {"m": MARKER})).one()
    return int(row[0]), int(row[1])


async def add_synthetic(conn, n: int, start_number: int) -> None:
    # '0 * g' внутри подзапроса делает его коррелированным: иначе PostgreSQL может вычислить
    # ARRAY(...) один раз и выдать одинаковый вектор во всех строках.
    await conn.execute(text(f"""
        INSERT INTO observations (id, image_path, crop_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding, created_at)
        SELECT gen_random_uuid(),
               CAST(:m AS text) || g::text, CAST(:m AS text) || g::text, 0, 0, 100, 100,
               ARRAY(SELECT random() - 0.5 + 0 * g FROM generate_series(1, {DIM}))::vector,
               now()
        FROM generate_series(CAST(:a AS bigint), CAST(:b AS bigint)) AS g
    """), {"m": MARKER, "a": start_number, "b": start_number + n - 1})


async def run(args) -> int:
    load_dotenv()
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("ОШИБКА: не задан DATABASE_URL", file=sys.stderr)
        return 1
    engine = create_async_engine(url)
    try:
        if args.status:
            async with engine.connect() as conn:
                total, synthetic = await counts(conn)
            print(f"Всего записей: {total}; реальных: {total - synthetic}; синтетических: {synthetic}")
            return 0

        if args.cleanup:
            async with engine.begin() as conn:
                result = await conn.execute(text("DELETE FROM observations WHERE starts_with(image_path, :m)"),
                                            {"m": MARKER})
                print(f"Удалено синтетических записей: {result.rowcount}")
            async with engine.connect() as conn:
                conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                await conn.execute(text("VACUUM ANALYZE observations"))
                if args.reindex:
                    print("Перестраиваю индексы...", flush=True)
                    await conn.execute(text("REINDEX TABLE observations"))
                total, synthetic = await counts(conn)
            print(f"Готово. Всего записей: {total}; синтетических: {synthetic}")
            return 0

        async with engine.connect() as conn:
            total, synthetic = await counts(conn)
        need = rows_to_add(total, synthetic, args.target)
        print(f"Сейчас записей: {total} (синтетических {synthetic}); цель {args.target}; добавляю {need}.")
        # Номер должен быть уникален среди синтетических: берём следующий после максимального.
        async with engine.connect() as conn:
            mx = (await conn.execute(text(
                "SELECT coalesce(max(substr(image_path, CAST(:l AS integer))::bigint), 0) FROM observations "
                "WHERE starts_with(image_path, :m)"), {"m": MARKER, "l": len(MARKER) + 1})).scalar_one()
        number = int(mx) + 1
        done = 0
        while done < need:
            n = min(CHUNK, need - done)
            async with engine.begin() as conn:
                await add_synthetic(conn, n, number + done)
            done += n
            print(f"  добавлено {done}/{need}", flush=True)
        async with engine.connect() as conn:
            conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            await conn.execute(text("ANALYZE observations"))
            total, synthetic = await counts(conn)
        print(f"Готово. Всего записей: {total}; из них синтетических: {synthetic}. "
              f"Для удаления: python scripts/pad_random_vectors.py --cleanup")
        return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(build_parser().parse_args())))