"""
Интеграционные тесты для db/create_indexes.py.

Как и tests/test_repository.py, требует настоящего PostgreSQL с
расширением pgvector - проверяется, что HNSW-индекс реально создаётся,
используется планировщиком запросов, и что поиск продолжает давать
корректные результаты при его наличии (а не просто "индекс есть, но
ничего не проверено").

Если TEST_DATABASE_URL не задан - тесты этого файла аккуратно
пропускаются, как и в test_repository.py.

Запуск: из корня проекта  ->  pytest tests/test_create_indexes.py
"""

import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db import repository
from backend.app.db.models import Base

load_dotenv()

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL не задан - пропускаю интеграционные тесты с реальным PostgreSQL+pgvector",
)

INDEX_NAME = "observations_embedding_hnsw_idx"


async def _create_index(engine):
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


@pytest_asyncio.fixture
async def indexed_session():
    """Чистая таблица + реально созданный HNSW-индекс перед каждым тестом."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await _create_index(engine)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s, engine

    await engine.dispose()


def _make_embedding(dominant_index: int, dim: int = 512) -> list:
    vec = [0.0] * dim
    vec[dominant_index] = 1.0
    return vec


class TestIndexCreation:
    @pytest.mark.asyncio
    async def test_index_is_created_idempotently(self, indexed_session):
        session, engine = indexed_session

        # Повторный вызов (IF NOT EXISTS) не должен падать.
        await _create_index(engine)

        result = await session.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'observations' AND indexname = :name"
            ),
            {"name": INDEX_NAME},
        )
        row = result.first()
        assert row is not None, "Индекс не найден в pg_indexes после создания"
        assert "hnsw" in row[0].lower()
        assert "vector_cosine_ops" in row[0]

    @pytest.mark.asyncio
    async def test_query_plan_uses_index_scan(self, indexed_session):
        session, _ = indexed_session

        for i in range(50):
            await repository.add_observation(
                session, f"obs_{i}.jpg", f"obs_{i}_c.jpg", (0, 0, 10, 10),
                _make_embedding(i % 512),
            )

        # На небольшом числе строк планировщик Postgres МОЖЕТ (и
        # абсолютно корректно) выбрать Seq Scan вместо индекса - для
        # таблицы из полусотни строк полный перебор реально дешевле,
        # чем обход графа HNSW. Это не говорит о проблеме с индексом,
        # просто означает, что тест на "индекс физически пригоден к
        # использованию" не должен зависеть от того, дешевле ли он на
        # ЭТОМ конкретном маленьком датасете по мнению оптимизатора.
        #
        # SET LOCAL enable_seqscan = off заставляет планировщик
        # использовать индекс, если он вообще применим к запросу -
        # детерминированная проверка "индекс пригоден", не зависящая
        # от эвристик стоимости на небольшом объёме тестовых данных.
        await session.execute(text("SET LOCAL enable_seqscan = off"))

        query_vec_literal = str(_make_embedding(0))
        result = await session.execute(
            text(
                "EXPLAIN SELECT id FROM observations "
                "ORDER BY embedding <=> (:query_vec)::vector LIMIT 5"
            ),
            {"query_vec": query_vec_literal},
        )
        plan = "\n".join(row[0] for row in result.fetchall())
        assert "hnsw" in plan.lower() or "Index Scan" in plan, (
            f"HNSW-индекс не используется в плане запроса даже с "
            f"отключённым Seq Scan - индекс физически непригоден для "
            f"этого запроса:\n{plan}"
        )


class TestSearchWithIndex:
    @pytest.mark.asyncio
    async def test_search_still_correct_with_index_present(self, indexed_session):
        """
        Главная проверка: HNSW - ПРИБЛИЖЁННЫЙ метод поиска, поэтому важно
        убедиться, что он не ломает корректность на обычных объёмах
        данных (recall по умолчанию близок к 100% при разумном числе
        записей и ef_search по умолчанию).
        """
        session, _ = indexed_session

        await repository.add_observation(
            session, "exact.jpg", "exact_c.jpg", (0, 0, 10, 10), _make_embedding(0)
        )
        for i in range(1, 30):
            await repository.add_observation(
                session, f"other_{i}.jpg", f"other_{i}_c.jpg", (0, 0, 10, 10),
                _make_embedding(i),
            )

        results = await repository.search_similar(session, _make_embedding(0), top_k=5)
        assert results[0][0].image_path == "exact.jpg"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.asyncio
    async def test_custom_ef_search_does_not_break_results(self, indexed_session):
        session, _ = indexed_session

        await repository.add_observation(
            session, "exact.jpg", "exact_c.jpg", (0, 0, 10, 10), _make_embedding(0)
        )
        for i in range(1, 20):
            await repository.add_observation(
                session, f"other_{i}.jpg", f"other_{i}_c.jpg", (0, 0, 10, 10),
                _make_embedding(i),
            )

        for ef_search in (10, 40, 200):
            results = await repository.search_similar(
                session, _make_embedding(0), top_k=3, ef_search=ef_search
            )
            assert results[0][0].image_path == "exact.jpg", (
                f"Неверный топ-1 при ef_search={ef_search}"
            )