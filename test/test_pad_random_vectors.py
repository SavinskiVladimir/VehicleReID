"""
Тесты scripts/pad_random_vectors.py.

Чистая логика (расчёт числа добавляемых записей, разбор аргументов) проверяется всегда.
Работа с настоящим PostgreSQL+pgvector - только при заданном TEST_DATABASE_URL.

Запуск: из корня проекта  ->  pytest tests/test_pad_random_vectors.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db import repository
from backend.app.db.models import Base

load_dotenv()
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

_SPEC = importlib.util.spec_from_file_location(
    "pad_random_vectors", Path(__file__).resolve().parents[1] / "scripts" / "pad_random_vectors.py")
pad = importlib.util.module_from_spec(_SPEC)
sys.modules["pad_random_vectors"] = pad
_SPEC.loader.exec_module(pad)


class TestRowsToAdd:
    def test_adds_difference(self):
        assert pad.rows_to_add(total=49_357, synthetic=0, target=100_000) == 50_643

    def test_already_padded_adds_nothing(self):
        assert pad.rows_to_add(total=100_000, synthetic=50_643, target=100_000) == 0

    def test_never_negative_when_target_below_total(self):
        assert pad.rows_to_add(total=49_357, synthetic=0, target=1_000) == 0

    def test_negative_target_rejected(self):
        with pytest.raises(ValueError):
            pad.rows_to_add(0, 0, -1)


class TestParser:
    def test_requires_one_mode(self):
        with pytest.raises(SystemExit):
            pad.build_parser().parse_args([])

    def test_modes_are_exclusive(self):
        with pytest.raises(SystemExit):
            pad.build_parser().parse_args(["--target", "10", "--cleanup"])

    def test_target_parsed(self):
        assert pad.build_parser().parse_args(["--target", "100000"]).target == 100000


needs_db = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL не задан")


@pytest_asyncio.fixture
async def clean_db():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        await repository.add_observation(s, "real.jpg", "real_c.jpg", (0, 0, 10, 10), [1.0] + [0.0] * 511)
    await engine.dispose()
    yield TEST_DATABASE_URL


def _args(url, *extra):
    return pad.build_parser().parse_args(["--database-url", url, *extra])


async def _stats(url):
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            total, synthetic = await pad.counts(conn)
            distinct = (await conn.execute(text(
                "SELECT count(DISTINCT embedding::text) FROM observations WHERE starts_with(image_path, :m)"),
                {"m": pad.MARKER})).scalar_one()
        return total, synthetic, distinct
    finally:
        await engine.dispose()


@needs_db
@pytest.mark.asyncio
async def test_pad_is_idempotent_and_vectors_are_distinct(clean_db):
    assert await pad.run(_args(clean_db, "--target", "301")) == 0
    total, synthetic, distinct = await _stats(clean_db)
    assert (total, synthetic) == (301, 300)
    assert distinct == 300  # каждая строка получила свой вектор

    assert await pad.run(_args(clean_db, "--target", "301")) == 0
    assert (await _stats(clean_db))[:2] == (301, 300)


@needs_db
@pytest.mark.asyncio
async def test_cleanup_removes_only_synthetic(clean_db):
    await pad.run(_args(clean_db, "--target", "50"))
    assert await pad.run(_args(clean_db, "--cleanup")) == 0
    total, synthetic, _ = await _stats(clean_db)
    assert (total, synthetic) == (1, 0)


@needs_db
@pytest.mark.asyncio
async def test_numbering_continues_after_partial_cleanup(clean_db):
    await pad.run(_args(clean_db, "--target", "11"))
    await pad.run(_args(clean_db, "--target", "21"))
    total, synthetic, distinct = await _stats(clean_db)
    assert (total, synthetic, distinct) == (21, 20, 20)