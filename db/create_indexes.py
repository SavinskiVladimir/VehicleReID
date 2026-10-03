"""
Создаёт HNSW-индекс на колонке embedding таблицы observations.

Согласно ТЗ, поиск по векторному сходству должен укладываться в ≤30 мс
при 100 000 записей (со ссылкой на бенчмарк векторных БД). Без индекса
pgvector выполняет точный перебор (sequential scan) по ВСЕМ строкам при
каждом запросе - на 100k записей это заведомо не уложится в требование.
HNSW (Hierarchical Navigable Small World) - метод приближённого поиска
ближайших соседей, который pgvector поддерживает "из коробки" (начиная
с версии 0.5.0) - именно он выбран в аналитическом разделе работы
(п.1.1.4) как метод индексации.

Запускать ПОСЛЕ db/create_table.py (таблица observations должна уже
существовать). Идемпотентен - повторный запуск безопасен благодаря
CREATE INDEX IF NOT EXISTS.

Параметры m / ef_construction влияют на компромисс между точностью
поиска, скоростью построения индекса и объёмом памяти:
    m - макс. число связей на узел графа (больше = точнее поиск, но
        больше памяти и медленнее построение индекса). 16 -
        рекомендованное документацией pgvector значение по умолчанию.
    ef_construction - размер списка кандидатов при ПОСТРОЕНИИ индекса
        (больше = точнее индекс, но медленнее его строить). 64 -
        типичное значение по умолчанию.

Отдельный параметр ef_search (точность/скорость уже на этапе ПОИСКА,
не построения) настраивается не здесь, а за запрос - см.
repository.search_similar(..., ef_search=...).

Запуск (из корня проекта):
    python db/create_indexes.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

INDEX_NAME = "observations_embedding_hnsw_idx"


async def main() -> None:
    engine = create_async_engine(DATABASE_URL, echo=True)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS {INDEX_NAME}
                ON observations
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        )

    await engine.dispose()
    print(f"Индекс {INDEX_NAME} создан (или уже существовал).")


if __name__ == "__main__":
    asyncio.run(main())