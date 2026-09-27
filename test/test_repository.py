"""
Интеграционные тесты для backend/app/db/repository.py.

В ОТЛИЧИЕ от остальных тестов проекта, здесь нужен настоящий
PostgreSQL с расширением pgvector - оператор косинусного расстояния
(`<=>`) не эмулируется никаким мок-объектом, это реальная SQL-логика.

Если переменная окружения TEST_DATABASE_URL не задана (ни в shell, ни в
.env в корне проекта) или подключение не удаётся - все тесты этого
файла аккуратно пропускаются (а не падают) с понятным сообщением,
чтобы `pytest` не требовал поднятого Postgres для обычного локального
запуска. В CI (см. .github/workflows/tests.yml) эта переменная задана и
указывает на сервис-контейнер postgres+pgvector, поэтому там тесты
реально выполняются.

ВАЖНО: TEST_DATABASE_URL должен указывать на ОТДЕЛЬНУЮ базу, не на ту,
что использует приложение (DATABASE_URL) - фикстура ниже на каждом
тесте удаляет и заново создаёт таблицу (drop_all/create_all), поэтому
указание сюда рабочей базы с реальными данными их уничтожит.

Запуск локально с реальной БД:
    # внутри контейнера, который поднят docker-compose (сервис db):
    docker compose exec db psql -U <user> -d <db> -c "CREATE DATABASE <db>_test;"
    docker compose exec db psql -U <user> -d <db>_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

    # добавить в .env (НЕ совпадает с DATABASE_URL):
    # TEST_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:PORT/<db>_test

    pytest tests/test_repository.py
"""

import os
import uuid

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db import repository
from backend.app.db.models import Base

# Подхватывает TEST_DATABASE_URL из .env в корне проекта, если он там
# задан. load_dotenv() НЕ перезаписывает переменную, если она уже
# установлена в окружении shell'а - так что явный `$env:TEST_DATABASE_URL`
# (например, для CI с другим значением) по-прежнему имеет приоритет.
load_dotenv()

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL не задан - пропускаю интеграционные тесты с реальным PostgreSQL+pgvector",
)


@pytest_asyncio.fixture
async def session():
    """
    Создаёт таблицы заново перед каждым тестом и удаляет после - каждый
    тест видит чистую базу, независимо от порядка выполнения.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    await engine.dispose()


def _make_embedding(dominant_index: int, dim: int = 512) -> list:
    """Единичный вектор с 1.0 в позиции dominant_index, иначе 0.0."""
    vec = [0.0] * dim
    vec[dominant_index] = 1.0
    return vec


class TestAddObservation:
    @pytest.mark.asyncio
    async def test_add_and_retrieve(self, session):
        obs = await repository.add_observation(
            session, image_path="a.jpg", crop_path="a_crop.jpg",
            bbox=(10, 20, 100, 150), embedding=_make_embedding(0),
        )
        assert obs.id is not None
        assert obs.bbox_x1 == 10 and obs.bbox_y2 == 150
        assert obs.created_at is not None


class TestSearchSimilar:
    @pytest.mark.asyncio
    async def test_exact_match_ranks_first_with_similarity_near_one(self, session):
        await repository.add_observation(
            session, "exact.jpg", "exact_c.jpg", (0, 0, 10, 10), _make_embedding(0)
        )
        await repository.add_observation(
            session, "close.jpg", "close_c.jpg", (0, 0, 10, 10),
            [0.99 if i == 0 else (0.01 if i == 1 else 0.0) for i in range(512)],
        )
        await repository.add_observation(
            session, "far.jpg", "far_c.jpg", (0, 0, 10, 10), _make_embedding(5)
        )

        query = _make_embedding(0)
        results = await repository.search_similar(session, query, top_k=10)

        assert [obs.image_path for obs, _ in results] == ["exact.jpg", "close.jpg", "far.jpg"]
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)
        assert results[0][1] > results[1][1] > results[2][1]

    @pytest.mark.asyncio
    async def test_top_k_limits_number_of_results(self, session):
        for i in range(5):
            await repository.add_observation(
                session, f"obs_{i}.jpg", f"obs_{i}_c.jpg", (0, 0, 10, 10),
                _make_embedding(i % 512),
            )
        results = await repository.search_similar(session, _make_embedding(0), top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_empty_database_returns_empty_list(self, session):
        results = await repository.search_similar(session, _make_embedding(0), top_k=10)
        assert results == []


class TestGetObservation:
    @pytest.mark.asyncio
    async def test_existing_id(self, session):
        obs = await repository.add_observation(
            session, "a.jpg", "a_c.jpg", (0, 0, 10, 10), _make_embedding(0)
        )
        fetched = await repository.get_observation(session, obs.id)
        assert fetched is not None
        assert fetched.image_path == "a.jpg"

    @pytest.mark.asyncio
    async def test_nonexistent_id_returns_none(self, session):
        fetched = await repository.get_observation(session, uuid.uuid4())
        assert fetched is None


class TestListObservations:
    @pytest.mark.asyncio
    async def test_pagination(self, session):
        for i in range(5):
            await repository.add_observation(
                session, f"obs_{i}.jpg", f"obs_{i}_c.jpg", (0, 0, 10, 10),
                _make_embedding(i % 512),
            )

        page1, total1 = await repository.list_observations(session, limit=2, offset=0)
        page2, total2 = await repository.list_observations(session, limit=2, offset=2)

        assert total1 == 5 and total2 == 5
        assert len(page1) == 2 and len(page2) == 2
        page1_ids = {obs.id for obs in page1}
        page2_ids = {obs.id for obs in page2}
        assert page1_ids.isdisjoint(page2_ids), "Страницы не должны пересекаться"

    @pytest.mark.asyncio
    async def test_ordered_by_created_at_descending(self, session):
        first = await repository.add_observation(
            session, "first.jpg", "first_c.jpg", (0, 0, 10, 10), _make_embedding(0)
        )
        second = await repository.add_observation(
            session, "second.jpg", "second_c.jpg", (0, 0, 10, 10), _make_embedding(1)
        )
        items, _ = await repository.list_observations(session)
        assert items[0].id == second.id  # самая новая - первая
        assert items[1].id == first.id


class TestDeleteObservation:
    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self, session):
        obs = await repository.add_observation(
            session, "a.jpg", "a_c.jpg", (0, 0, 10, 10), _make_embedding(0)
        )
        result = await repository.delete_observation(session, obs.id)
        assert result is True

        fetched = await repository.get_observation(session, obs.id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, session):
        result = await repository.delete_observation(session, uuid.uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_deleted_observation_excluded_from_search(self, session):
        obs = await repository.add_observation(
            session, "a.jpg", "a_c.jpg", (0, 0, 10, 10), _make_embedding(0)
        )
        await repository.delete_observation(session, obs.id)

        results = await repository.search_similar(session, _make_embedding(0), top_k=10)
        assert results == []