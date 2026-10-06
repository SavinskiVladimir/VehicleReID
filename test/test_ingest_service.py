"""
Тесты пакетной загрузки изображений в базу наблюдений
(backend/app/services/ingest_service.py).

Поиск файлов и статистика проверяются без БД. Сама загрузка - на НАСТОЯЩЕМ
стеке: PostgreSQL с pgvector, onnxruntime с крошечной ONNX-моделью (фабрика
build_toy_embedder из conftest.py), настоящее файловое хранилище. Подменён
только детектор (в тестах нет весов YOLO): число "найденных ТС" кодируется
шириной картинки (ширина - 64), чтобы каждый тест сам решал, сколько машин на кадре.

Что проверяется - то, ради чего эта функция существует и что моками не
проверишь: идемпотентность (повтор, дубликаты под другим именем), атомарность
файла (сбой модели или БД посреди кадра не оставляет ни записей без файлов, ни
файлов без записей), replace (заменяет, а не плодит) и что загруженное
действительно находится поиском.

Требуется TEST_DATABASE_URL (и пакет onnx) - иначе соответствующие тесты пропускаются.

Запуск: из корня проекта  ->  pytest tests/test_ingest_service.py
"""

import os
import shutil
import types
import uuid
from hashlib import sha256
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
load_dotenv()

import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db import repository
from backend.app.db.models import Base
from backend.app.services.image_service import ImageService
from backend.app.services.ingest_service import (
    IngestResult,
    IngestService,
    IngestStats,
    iter_image_files,
)
from backend.app.services.reid_embedder import ReIDEmbedderService
from backend.app.services.search_service import SearchService
from backend.app.storage import FileStorage

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
needs_database = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL не задан")

RED, GREEN, BLUE, YELLOW = (230, 20, 20), (20, 230, 20), (20, 20, 230), (230, 230, 20)


def make_image(path: Path, color=RED, vehicles: int = 1) -> Path:
    """PNG (цвет сохраняется точно); число ТС для детектора-заглушки = ширина - 64."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64 + vehicles, 64), color).save(path)
    return path


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


# ===========================================================================
# Поиск файлов
# ===========================================================================

class TestIterImageFiles:
    def test_finds_images_recursively_in_stable_order(self, tmp_path):
        make_image(tmp_path / "b.png")
        make_image(tmp_path / "a.png", GREEN)
        make_image(tmp_path / "sub" / "deep" / "c.png", BLUE)

        found = iter_image_files([tmp_path])

        assert [p.relative_to(tmp_path).as_posix() for p in found] == ["a.png", "b.png", "sub/deep/c.png"]

    def test_order_is_reproducible(self, tmp_path):
        for name in ("z.png", "m.png", "a.png"):
            make_image(tmp_path / name, (len(name) * 20, 10, 10))
        assert iter_image_files([tmp_path]) == iter_image_files([tmp_path])

    def test_non_recursive_ignores_subfolders(self, tmp_path):
        make_image(tmp_path / "top.png")
        make_image(tmp_path / "sub" / "inner.png", GREEN)

        found = iter_image_files([tmp_path], recursive=False)

        assert [p.name for p in found] == ["top.png"]

    def test_extension_filter_is_case_insensitive_and_skips_other_files(self, tmp_path):
        make_image(tmp_path / "ok.PNG")
        Image.new("RGB", (8, 8)).save(tmp_path / "photo.JPG")
        Image.new("RGB", (8, 8)).save(tmp_path / "pic.webp")
        (tmp_path / "notes.txt").write_text("не изображение")
        (tmp_path / "data.csv").write_text("1,2")

        names = sorted(p.name for p in iter_image_files([tmp_path]))

        assert names == ["ok.PNG", "photo.JPG", "pic.webp"]

    def test_hidden_files_are_skipped(self, tmp_path):
        # "._x.jpg" - служебные файлы macOS, это не изображения
        make_image(tmp_path / "real.png")
        (tmp_path / "._real.png").write_bytes(b"\x00\x05\x16\x07 resource fork")
        (tmp_path / ".hidden.png").write_bytes(b"x")

        assert [p.name for p in iter_image_files([tmp_path])] == ["real.png"]

    def test_accepts_files_and_folders_without_duplicates(self, tmp_path):
        inside = make_image(tmp_path / "folder" / "one.png")
        make_image(tmp_path / "folder" / "two.png", GREEN)

        found = iter_image_files([tmp_path / "folder", inside])  # файл указан и сам, и через папку

        assert len(found) == 2

    def test_missing_path_raises_instead_of_loading_nothing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="не найден"):
            iter_image_files([tmp_path / "опечатка"])

    def test_empty_folder(self, tmp_path):
        assert iter_image_files([tmp_path]) == []


class TestIngestStats:
    def test_counts_every_outcome(self):
        stats = IngestStats()
        stats.record("a", IngestResult("added", observations=3, replaced=1))
        stats.record("b", IngestResult("added", observations=2))
        stats.record("c", IngestResult("skipped", observations=5))   # пропущенное в "добавлено" не входит
        stats.record("d", IngestResult("no_vehicles"))
        stats.record("e", IngestResult("failed", error="битый файл"))

        assert (stats.added_files, stats.observations, stats.replaced) == (2, 5, 1)
        assert (stats.skipped, stats.no_vehicles, stats.failed) == (1, 1, 1)
        assert stats.processed == 5
        assert stats.errors == [("e", "битый файл")]


# ===========================================================================
# Загрузка на настоящем стеке
# ===========================================================================

class CountingDetector:
    """Вместо YOLO: число ТС = ширина кадра - 64 (см. make_image)."""

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [
            types.SimpleNamespace(bbox=(i * 10, 0, i * 10 + 20, 30), confidence=round(0.9 - 0.1 * i, 2))
            for i in range(image.width - 64)
        ]

    def crop_with_padding(self, image, bbox):
        return image.crop(bbox)


class ForbiddenDetector:
    def detect(self, image):
        raise AssertionError("режим whole не должен вызывать детектор")

    crop_with_padding = detect


class Env:
    def __init__(self, ingest, session_factory, storage, search_service, detector, tmp_path, engine):
        self.ingest = ingest
        self.session_factory = session_factory
        self.storage = storage
        self.search_service = search_service
        self.detector = detector
        self.tmp_path = tmp_path
        self.engine = engine

    async def run(self, path, **kwargs):
        async with self.session_factory() as session:
            return await self.ingest.ingest_file(session, path, **kwargs)

    async def count(self, table):
        async with self.engine.connect() as conn:
            return (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar()

    async def observation_ids(self):
        async with self.engine.connect() as conn:
            rows = (await conn.execute(text("SELECT id FROM observations ORDER BY created_at"))).scalars().all()
        return list(rows)

    def files(self, subdir):
        directory = self.tmp_path / "permanent" / subdir
        return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


@pytest_asyncio.fixture
async def env(tmp_path, build_toy_embedder):
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL не задан")

    toy = build_toy_embedder(tmp_path / "reid.onnx", out_dim=512, size=64)
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    storage = FileStorage(temp_dir=str(tmp_path / "temp"), permanent_dir=str(tmp_path / "permanent"))
    search_service = SearchService(ImageService(image_size=(64, 64)), ReIDEmbedderService(str(toy)))
    detector = CountingDetector()
    ingest = IngestService(search_service=search_service, storage=storage, detector=detector)

    yield Env(ingest, async_sessionmaker(engine, expire_on_commit=False), storage,
              search_service, detector, tmp_path, engine)
    await engine.dispose()


@needs_database
class TestModes:
    @pytest.mark.asyncio
    async def test_all_creates_an_observation_per_vehicle(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)

        result = await env.run(frame, mode="all")

        assert (result.status, result.observations) == ("added", 2)
        assert await env.count("observations") == 2
        assert len(env.files("images")) == 1   # кадр целиком - ОДИН раз, а не по копии на каждое ТС
        assert len(env.files("crops")) == 2    # кроп - по одному на ТС

    @pytest.mark.asyncio
    async def test_vehicles_of_one_frame_share_the_full_image_but_not_crops(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=3)
        await env.run(frame, mode="all")

        async with env.session_factory() as session:
            items, _ = await repository.list_observations(session, limit=10)

        assert len({o.image_path for o in items}) == 1
        assert len({o.crop_path for o in items}) == 3
        assert all(Path(o.image_path).is_file() and Path(o.crop_path).is_file() for o in items)

    @pytest.mark.asyncio
    async def test_bbox_of_each_vehicle_is_stored(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)
        await env.run(frame, mode="all")

        async with env.session_factory() as session:
            items, _ = await repository.list_observations(session, limit=10)

        assert sorted((o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2) for o in items) == [
            (0, 0, 20, 30), (10, 0, 30, 30),
        ]

    @pytest.mark.asyncio
    async def test_best_keeps_only_the_most_confident_vehicle(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=3)

        result = await env.run(frame, mode="best")

        assert result.observations == 1
        async with env.session_factory() as session:
            (only,), _ = await repository.list_observations(session, limit=10)
        assert (only.bbox_x1, only.bbox_x2) == (0, 20)  # первое = самое уверенное

    @pytest.mark.asyncio
    async def test_whole_uses_the_full_frame_and_never_calls_the_detector(self, env, tmp_path):
        env.ingest.detector = ForbiddenDetector()
        image = make_image(tmp_path / "in" / "crop.png", GREEN, vehicles=0)

        result = await env.run(image, mode="whole")

        assert (result.status, result.observations) == ("added", 1)
        async with env.session_factory() as session:
            (obs,), _ = await repository.list_observations(session, limit=10)
        assert (obs.bbox_x1, obs.bbox_y1, obs.bbox_x2, obs.bbox_y2) == (0, 0, 64, 64)
        assert Image.open(obs.crop_path).size == (64, 64)  # кадр = кроп, без отступа и обрезки

    @pytest.mark.asyncio
    async def test_whole_works_without_any_detector(self, env, tmp_path):
        env.ingest.detector = None
        image = make_image(tmp_path / "in" / "crop.png", GREEN, vehicles=0)
        assert (await env.run(image, mode="whole")).status == "added"

    @pytest.mark.asyncio
    async def test_modes_that_need_a_detector_refuse_to_run_without_one(self, env, tmp_path):
        env.ingest.detector = None
        image = make_image(tmp_path / "in" / "frame.png")
        for mode in ("all", "best"):
            with pytest.raises(ValueError, match="детектор"):
                await env.run(image, mode=mode)

    @pytest.mark.asyncio
    async def test_unknown_mode_is_rejected(self, env, tmp_path):
        with pytest.raises(ValueError, match="режим"):
            await env.run(make_image(tmp_path / "x.png"), mode="everything")

    @pytest.mark.asyncio
    async def test_frame_without_vehicles_saves_nothing_but_is_remembered(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "empty.png", RED, vehicles=0)

        first = await env.run(frame, mode="all")
        second = await env.run(frame, mode="all")

        assert first.status == "no_vehicles"
        assert await env.count("observations") == 0
        assert env.files("images") == [] and env.files("crops") == []   # пустой кадр на диск не пишется
        assert await env.count("ingest_sources") == 1
        assert second.status == "skipped"           # и детектор по нему заново не гоняется
        assert env.detector.calls == 1


@needs_database
class TestIdempotence:
    @pytest.mark.asyncio
    async def test_second_run_adds_nothing(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)
        await env.run(frame)

        again = await env.run(frame)

        assert again.status == "skipped" and again.observations == 2
        assert await env.count("observations") == 2
        assert len(env.files("images")) == 1 and len(env.files("crops")) == 2
        assert env.detector.calls == 1   # повторно ни детектор, ни модель не запускались

    @pytest.mark.asyncio
    async def test_renamed_or_copied_file_is_recognised_by_content(self, env, tmp_path):
        original = make_image(tmp_path / "a" / "car.png", RED, vehicles=1)
        copy_elsewhere = tmp_path / "b" / "совсем_другое_имя.png"
        copy_elsewhere.parent.mkdir()
        shutil.copy(original, copy_elsewhere)

        await env.run(original)
        again = await env.run(copy_elsewhere)

        assert again.status == "skipped"
        assert await env.count("observations") == 1

    @pytest.mark.asyncio
    async def test_different_images_are_both_loaded(self, env, tmp_path):
        await env.run(make_image(tmp_path / "a.png", RED))
        await env.run(make_image(tmp_path / "b.png", GREEN))
        assert await env.count("observations") == 2
        assert await env.count("ingest_sources") == 2

    @pytest.mark.asyncio
    async def test_resumed_run_processes_only_new_files(self, env, tmp_path):
        folder = tmp_path / "in"
        make_image(folder / "1.png", RED)
        make_image(folder / "2.png", GREEN)
        for path in iter_image_files([folder]):
            await env.run(path)
        make_image(folder / "3.png", BLUE)  # появился новый файл

        statuses = [(p.name, (await env.run(p)).status) for p in iter_image_files([folder])]

        assert statuses == [("1.png", "skipped"), ("2.png", "skipped"), ("3.png", "added")]
        assert await env.count("observations") == 3

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing_anywhere(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)

        result = await env.run(frame, dry_run=True)

        assert (result.status, result.observations) == ("added", 2)   # "было бы добавлено"
        assert await env.count("observations") == 0
        assert await env.count("ingest_sources") == 0
        assert env.files("images") == [] and env.files("crops") == []

    @pytest.mark.asyncio
    async def test_dry_run_is_aware_of_already_loaded_files(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED)
        await env.run(frame)
        assert (await env.run(frame, dry_run=True)).status == "skipped"


@needs_database
class TestDuplicatesWithinOneRun:
    """Копия файла в той же папке: внутри одного запуска содержимое обрабатывается один раз."""

    @staticmethod
    def _folder_with_a_copy(tmp_path):
        original = make_image(tmp_path / "in" / "bus.png", RED, vehicles=2)
        shutil.copy(original, tmp_path / "in" / "копия_bus.png")
        return sorted((tmp_path / "in").iterdir())

    async def _run_all(self, env, files, **kwargs):
        seen, results = {}, []
        for path in files:
            results.append(await env.run(path, seen=seen, **kwargs))
        return results

    @pytest.mark.asyncio
    async def test_replace_does_not_replace_what_was_just_loaded(self, env, tmp_path):
        files = self._folder_with_a_copy(tmp_path)
        await self._run_all(env, files)                       # первичная загрузка

        results = await self._run_all(env, files, replace=True)

        assert [r.status for r in results] == ["added", "skipped"]    # копия - пропуск, а не "замена замены"
        assert sum(r.replaced for r in results) == 2                  # заменены ровно 2 старых наблюдения, не 4
        assert await env.count("observations") == 2

    @pytest.mark.asyncio
    async def test_dry_run_counts_a_copy_once_like_the_real_run_would(self, env, tmp_path):
        files = self._folder_with_a_copy(tmp_path)

        results = await self._run_all(env, files, dry_run=True)

        assert [(r.status, r.observations) for r in results] == [("added", 2), ("skipped", 2)]
        assert await env.count("observations") == 0       # и ничего не записано

    @pytest.mark.asyncio
    async def test_without_a_shared_dict_calls_are_independent(self, env, tmp_path):
        # Поведение по умолчанию: одиночный вызов ничего не "помнит" между вызовами,
        # единственная память - таблица ingest_sources.
        files = self._folder_with_a_copy(tmp_path)
        results = [await env.run(path, dry_run=True) for path in files]
        assert [r.status for r in results] == ["added", "added"]

    @pytest.mark.asyncio
    async def test_failed_file_is_not_remembered_so_its_copy_gets_a_chance(self, env, tmp_path, monkeypatch):
        files = self._folder_with_a_copy(tmp_path)
        original = env.search_service.embedder.embed_as_list
        state = {"fail": True}

        def flaky(pre):
            if state["fail"]:
                state["fail"] = False   # падает только первый раз
                raise RuntimeError("временный сбой")
            return original(pre)

        monkeypatch.setattr(env.search_service.embedder, "embed_as_list", flaky)

        results = await self._run_all(env, files)

        assert [r.status for r in results] == ["failed", "added"]
        assert await env.count("observations") == 2


@needs_database
class TestReplace:
    @pytest.mark.asyncio
    async def test_replace_swaps_observations_instead_of_duplicating_them(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)
        await env.run(frame)
        old_ids = await env.observation_ids()
        old_crops = set(env.files("crops"))

        result = await env.run(frame, replace=True)

        assert (result.status, result.observations, result.replaced) == ("added", 2, 2)
        new_ids = await env.observation_ids()
        assert len(new_ids) == 2 and set(new_ids).isdisjoint(old_ids)   # заменены, не добавлены поверх
        assert await env.count("ingest_sources") == 1
        assert len(env.files("images")) == 1 and len(env.files("crops")) == 2
        assert set(env.files("crops")).isdisjoint(old_crops)            # старые файлы удалены

    @pytest.mark.asyncio
    async def test_replace_with_another_mode_changes_the_observations(self, env, tmp_path):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=3)
        await env.run(frame, mode="all")
        assert await env.count("observations") == 3

        result = await env.run(frame, mode="whole", replace=True)

        assert (result.observations, result.replaced) == (1, 3)
        assert await env.count("observations") == 1
        async with env.session_factory() as session:
            (obs,), _ = await repository.list_observations(session, limit=5)
        frame_width = Image.open(frame).width          # 64 + число "машин" (см. make_image)
        assert (obs.bbox_x1, obs.bbox_x2) == (0, frame_width)   # теперь кадр целиком, а не рамки детектора

    @pytest.mark.asyncio
    async def test_replace_on_a_new_file_is_a_plain_load(self, env, tmp_path):
        result = await env.run(make_image(tmp_path / "x.png"), replace=True)
        assert (result.status, result.replaced) == ("added", 0)

    @pytest.mark.asyncio
    async def test_replace_leaves_other_files_untouched(self, env, tmp_path):
        keep = make_image(tmp_path / "keep.png", GREEN)
        target = make_image(tmp_path / "target.png", RED)
        await env.run(keep)
        await env.run(target)
        keep_id = (await env.observation_ids())[0]

        await env.run(target, replace=True)

        assert keep_id in await env.observation_ids()
        assert await env.count("observations") == 2

    @pytest.mark.asyncio
    async def test_replace_never_deletes_files_outside_the_storage(self, env, tmp_path):
        # Запись из demo_populate_db.py указывает на файл исходного датасета:
        # replace не должен его удалять.
        dataset_file = make_image(tmp_path / "VeRi" / "image_test" / "0001.png", BLUE, vehicles=0)
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=1)
        await env.run(frame)
        async with env.session_factory() as session:
            ids = (await repository.get_ingest_source(session, sha(frame))).observation_ids
            obs = await repository.get_observation(session, ids[0])
            obs.crop_path = str(dataset_file)      # подменяем путь на внешний файл
            obs.image_path = str(dataset_file)
            await session.commit()

        await env.run(frame, replace=True)

        assert dataset_file.exists()


@needs_database
class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_corrupt_file_fails_alone_and_does_not_stop_the_batch(self, env, tmp_path):
        bad = tmp_path / "in" / "broken.jpg"
        bad.parent.mkdir()
        bad.write_bytes("это совсем не JPEG".encode("utf-8"))
        good = make_image(tmp_path / "in" / "good.png", RED)

        bad_result = await env.run(bad)
        good_result = await env.run(good)

        assert bad_result.status == "failed" and "изображение" in bad_result.error
        assert good_result.status == "added"
        assert await env.count("observations") == 1

    @pytest.mark.asyncio
    async def test_unreadable_path_is_a_failure_not_a_crash(self, env, tmp_path):
        result = await env.run(tmp_path / "нет_такого_файла.png")
        assert result.status == "failed" and "прочитать" in result.error

    @pytest.mark.asyncio
    async def test_model_failure_mid_frame_leaves_nothing_behind(self, env, tmp_path, monkeypatch):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=3)
        original = env.search_service.embedder.embed_as_list
        calls = {"n": 0}

        def flaky(preprocessed):
            calls["n"] += 1
            if calls["n"] == 2:                      # модель падает на втором ТС из трёх
                raise RuntimeError("ONNX упал")
            return original(preprocessed)

        monkeypatch.setattr(env.search_service.embedder, "embed_as_list", flaky)

        result = await env.run(frame)

        assert result.status == "failed" and "ONNX упал" in result.error
        assert await env.count("observations") == 0       # не "половина кадра"
        assert await env.count("ingest_sources") == 0     # а значит, повторный запуск попробует снова
        assert env.files("images") == [] and env.files("crops") == []

    @pytest.mark.asyncio
    async def test_database_failure_mid_frame_rolls_back_rows_and_files(self, env, tmp_path, monkeypatch):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=3)
        original = env.search_service.add_observation
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("БД моргнула")
            return await original(*args, **kwargs)

        monkeypatch.setattr(env.search_service, "add_observation", flaky)

        result = await env.run(frame)

        assert result.status == "failed"
        assert await env.count("observations") == 0        # первое ТС откатилось вместе с остальными
        assert await env.count("ingest_sources") == 0
        assert env.files("images") == [] and env.files("crops") == []   # файлы, уже записанные на диск, удалены

    @pytest.mark.asyncio
    async def test_failed_file_can_be_loaded_by_the_next_run(self, env, tmp_path, monkeypatch):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)
        original = env.search_service.embedder.embed_as_list
        monkeypatch.setattr(env.search_service.embedder, "embed_as_list",
                            lambda p: (_ for _ in ()).throw(RuntimeError("сбой")))
        assert (await env.run(frame)).status == "failed"
        monkeypatch.setattr(env.search_service.embedder, "embed_as_list", original)

        retry = await env.run(frame)

        assert (retry.status, retry.observations) == ("added", 2)
        assert await env.count("observations") == 2

    @pytest.mark.asyncio
    async def test_failed_replace_keeps_the_old_data_intact(self, env, tmp_path, monkeypatch):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)
        await env.run(frame)
        ids_before = await env.observation_ids()
        images_before, crops_before = env.files("images"), env.files("crops")

        async def always_fails(*args, **kwargs):
            raise RuntimeError("БД упала при замене")

        monkeypatch.setattr(env.search_service, "add_observation", always_fails)
        result = await env.run(frame, replace=True)

        assert result.status == "failed"
        assert await env.observation_ids() == ids_before     # транзакция откатилась: старое на месте
        assert await env.count("ingest_sources") == 1
        assert env.files("images") == images_before and env.files("crops") == crops_before

    @pytest.mark.asyncio
    async def test_model_failure_during_replace_keeps_the_old_data_intact(self, env, tmp_path, monkeypatch):
        frame = make_image(tmp_path / "in" / "frame.png", RED, vehicles=2)
        await env.run(frame)
        ids_before = await env.observation_ids()
        crops_before = env.files("crops")

        monkeypatch.setattr(env.search_service.embedder, "embed_as_list",
                            lambda p: (_ for _ in ()).throw(RuntimeError("модель упала")))
        result = await env.run(frame, replace=True)

        assert result.status == "failed"
        assert await env.observation_ids() == ids_before
        assert env.files("crops") == crops_before


@needs_database
class TestLoadedObservationsAreUsable:
    @pytest.mark.asyncio
    async def test_ingested_images_are_found_by_search(self, env, tmp_path):
        # Цель всей функции: загруженное реально находится поиском.
        env.ingest.detector = None
        red = make_image(tmp_path / "in" / "red.png", RED, vehicles=0)
        for path, color in ((red, RED),
                            (make_image(tmp_path / "in" / "green.png", GREEN, vehicles=0), GREEN),
                            (make_image(tmp_path / "in" / "blue.png", BLUE, vehicles=0), BLUE)):
            await env.run(path, mode="whole")

        async with env.session_factory() as session:
            response = await env.search_service.search(
                session, crop=Image.open(red).convert("RGB"), query_bbox=(0, 0, 64, 64), top_k=3
            )
            red_ids = (await repository.get_ingest_source(session, sha(red))).observation_ids

        assert response.results[0].similarity == pytest.approx(1.0, abs=1e-3)
        assert response.results[0].observation.id == red_ids[0]    # нашёлся именно красный
        assert response.results[1].similarity < 0.99

    @pytest.mark.asyncio
    async def test_ingested_observations_appear_in_the_observation_list_with_files(self, env, tmp_path):
        await env.run(make_image(tmp_path / "in" / "frame.png", RED, vehicles=2))

        async with env.session_factory() as session:
            items, total = await repository.list_observations(session, limit=10)

        assert total == 2
        assert all(Path(o.image_path).is_file() and Path(o.crop_path).is_file() for o in items)
        # файлы лежат в постоянном хранилище, а не указывают на исходную папку
        assert all((tmp_path / "permanent") in Path(o.crop_path).parents for o in items)


@needs_database
class TestDeletingObservationCleansFiles:
    @pytest.mark.asyncio
    async def test_shared_frame_lives_until_its_last_observation_is_deleted(self, env, tmp_path):
        await env.run(make_image(tmp_path / "in" / "frame.png", RED, vehicles=2))
        first_id, second_id = await env.observation_ids()

        async def delete(observation_id):
            async with env.session_factory() as session:
                files = await env.search_service.delete_observation(session, observation_id)
            for path in files:
                env.storage.delete_permanent(path)
            return files

        files = await delete(first_id)
        assert len(files) == 1                                   # только кроп: кадр нужен второй записи
        assert len(env.files("images")) == 1 and len(env.files("crops")) == 1

        files = await delete(second_id)
        assert len(files) == 2                                   # теперь и кроп, и кадр
        assert env.files("images") == [] and env.files("crops") == []
        assert await env.count("observations") == 0