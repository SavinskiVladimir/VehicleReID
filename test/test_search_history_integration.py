"""
Сквозной тест журнала поиска и повторного поиска на НАСТОЯЩЕМ стеке:
приложение FastAPI целиком (lifespan, зависимости, маршруты), настоящий
PostgreSQL с pgvector, настоящая ONNX-модель (крошечный граф, собранный в
тесте) через настоящий onnxruntime, настоящее файловое хранилище.
Подменён только детектор (в тестах нет весов YOLO): он "находит" ТС во весь
кадр.

Что проверяется - то, что моками проверить нельзя:
  * поиск реально пишется в журнал вместе с кропом и вектором;
  * повторный поиск идёт по вектору из БД и НЕ запускает модели;
  * повторный поиск видит ТЕКУЩУЮ базу наблюдений;
  * журнал и база наблюдений - независимые сущности;
  * журнал ограничивается лимитом, файлы вычищаются вместе с записями;
  * в базе без таблицы журнала (старая БД) поиск продолжает работать.

Пропускается без TEST_DATABASE_URL или без пакета onnx.

Запуск: из корня проекта  ->  pytest tests/test_search_history_integration.py
"""

import asyncio
import io
import os
import types

from dotenv import load_dotenv

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
load_dotenv()

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import backend.app.main as main_module
from backend.app.config import get_settings
from backend.app.db.models import Base
from backend.app.dependencies import get_detector_service

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
ADMIN = {"X-Admin-Token": "integration-token"}

pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL не задан")

RED, GREEN, BLUE = (230, 20, 20), (20, 230, 20), (20, 20, 230)


def _build_toy_embedder(path, out_dim=512, size=64):
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper, save

    np.random.seed(0)
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["p"]),
         helper.make_node("Flatten", ["p"], ["f"], axis=1),
         helper.make_node("MatMul", ["f", "W"], ["embedding"])],
        "toy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, size, size])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, out_dim])],
        initializer=[numpy_helper.from_array(np.random.randn(3, out_dim).astype(np.float32), "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, str(path))


def _reset_database(*, drop_history_table=False):
    async def _run():
        engine = create_async_engine(TEST_DATABASE_URL)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            if drop_history_table:
                await conn.execute(text("DROP TABLE search_queries"))
        await engine.dispose()

    asyncio.run(_run())


class _WholeFrameDetector:
    """Вместо YOLO: единственное ТС занимает весь кадр."""

    def detect(self, image):
        return [types.SimpleNamespace(bbox=(0, 0, image.width, image.height), confidence=0.95)]

    def crop_with_padding(self, image, bbox):
        return image.crop(bbox)


def _jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def make_stack(tmp_path, monkeypatch):
    pytest.importorskip("onnx")

    def _make(*, history_max=500, drop_history_table=False):
        toy = tmp_path / "reid.onnx"
        _build_toy_embedder(toy)
        _reset_database(drop_history_table=drop_history_table)

        monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
        monkeypatch.setenv("REID_MODEL_PATH", str(toy))
        monkeypatch.setenv("REID_IMAGE_HEIGHT", "64")
        monkeypatch.setenv("REID_IMAGE_WIDTH", "64")
        monkeypatch.setenv("TEMP_UPLOAD_DIR", str(tmp_path / "temp"))
        monkeypatch.setenv("PERMANENT_STORAGE_DIR", str(tmp_path / "permanent"))
        monkeypatch.setenv("ADMIN_API_KEY", "integration-token")
        monkeypatch.setenv("SEARCH_HISTORY_MAX_ITEMS", str(history_max))
        get_settings.cache_clear()

        # Настоящими остаются ReID-модель, проверка размерности эмбеддинга, БД, хранилище.
        monkeypatch.setattr(main_module, "DetectorService", lambda **kwargs: object())
        monkeypatch.setattr(main_module, "check_model_files", lambda settings: None)

        app = main_module.create_app()
        app.dependency_overrides[get_detector_service] = lambda: _WholeFrameDetector()
        return app

    return _make


class _Api:
    """Тонкая обёртка над клиентом - чтобы тесты читались как сценарии пользователя."""

    def __init__(self, client):
        self.client = client

    def _upload(self, color):
        response = self.client.post("/api/detect", files={"file": ("x.jpg", _jpeg(color), "image/jpeg")})
        assert response.status_code == 200, response.text
        return response.json()["upload_id"]

    def save_observation(self, color):
        response = self.client.post(
            "/api/observations", json={"upload_id": self._upload(color), "detection_index": 0}, headers=ADMIN
        )
        assert response.status_code == 200, response.text
        return response.json()

    def search(self, color, top_k=10):
        return self.client.post(
            "/api/search", json={"upload_id": self._upload(color), "detection_index": 0, "top_k": top_k}
        )


def _files_in(directory):
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


class TestSearchIsRecordedInJournal:
    def test_search_creates_a_journal_entry_with_crop_and_summary(self, make_stack, tmp_path):
        with TestClient(make_stack()) as client:
            api = _Api(client)
            for color in (RED, GREEN, BLUE):
                api.save_observation(color)

            response = api.search(RED, top_k=5)
            assert response.status_code == 200
            body = response.json()
            query_id = body["query_id"]
            assert query_id is not None
            assert body["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-3)  # красный нашёл красный

            history = client.get("/api/search-history").json()
            assert history["total"] == 1
            item = history["items"][0]
            assert item["id"] == query_id
            assert item["result_count"] == 3
            assert item["top_k"] == 5
            assert item["top_similarity"] == pytest.approx(1.0, abs=1e-3)

            crop = client.get(f"/api/search-history/{query_id}/crop")
            assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
            assert len(_files_in(tmp_path / "permanent" / "queries")) == 1

    def test_journal_and_observation_base_do_not_mix(self, make_stack):
        with TestClient(make_stack()) as client:
            api = _Api(client)
            for color in (RED, GREEN, BLUE):
                api.save_observation(color)
            for _ in range(4):
                assert api.search(RED).status_code == 200

            assert client.get("/api/observations").json()["total"] == 3     # поиски базу не пополняют
            assert client.get("/api/search-history").json()["total"] == 4   # а журнал растёт


class TestRerunFromJournal:
    def test_rerun_needs_no_models_and_no_upload(self, make_stack):
        app = make_stack()
        with TestClient(app) as client:
            api = _Api(client)
            for color in (RED, GREEN, BLUE):
                api.save_observation(color)
            query_id = api.search(RED).json()["query_id"]

            # С этого момента любой запуск ReID-модели - ошибка теста.
            def forbidden(*args, **kwargs):
                raise AssertionError("повторный поиск не должен запускать ReID-модель")

            app.state.search_service.embedder.embed_as_list = forbidden

            response = client.post(f"/api/search-history/{query_id}/search")  # без файла и без токена

            assert response.status_code == 200
            body = response.json()
            assert body["query_id"] == query_id
            assert body["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-3)
            assert len(body["results"]) == 3

    def test_rerun_sees_the_current_database(self, make_stack):
        with TestClient(make_stack()) as client:
            api = _Api(client)
            api.save_observation(GREEN)
            query_id = api.search(RED).json()["query_id"]
            before = client.post(f"/api/search-history/{query_id}/search").json()
            assert before["results"][0]["similarity"] < 0.99   # красного в базе ещё нет

            api.save_observation(RED)  # в базе появилось то, что искали
            after = client.post(f"/api/search-history/{query_id}/search").json()

            assert after["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-3)
            assert len(after["results"]) == 2

    def test_rerun_is_not_recorded_as_a_new_search(self, make_stack):
        with TestClient(make_stack()) as client:
            api = _Api(client)
            api.save_observation(RED)
            query_id = api.search(RED).json()["query_id"]

            for _ in range(3):
                client.post(f"/api/search-history/{query_id}/search")

            assert client.get("/api/search-history").json()["total"] == 1

    def test_top_k_defaults_to_original_and_can_be_overridden(self, make_stack):
        with TestClient(make_stack()) as client:
            api = _Api(client)
            for color in (RED, GREEN, BLUE):
                api.save_observation(color)
            query_id = api.search(RED, top_k=2).json()["query_id"]

            assert len(client.post(f"/api/search-history/{query_id}/search").json()["results"]) == 2
            assert len(client.post(f"/api/search-history/{query_id}/search?top_k=3").json()["results"]) == 3

    def test_unknown_record(self, make_stack):
        import uuid

        with TestClient(make_stack()) as client:
            assert client.post(f"/api/search-history/{uuid.uuid4()}/search").status_code == 404


class TestJournalMaintenance:
    def test_journal_is_capped_and_old_crop_files_are_removed(self, make_stack, tmp_path):
        with TestClient(make_stack(history_max=2)) as client:
            api = _Api(client)
            api.save_observation(RED)
            ids = [api.search(RED).json()["query_id"] for _ in range(4)]

            history = client.get("/api/search-history").json()
            assert history["total"] == 2
            assert [item["id"] for item in history["items"]] == [ids[3], ids[2]]  # остались новые
            # и на диске не осталось кропов удалённых записей
            assert len(_files_in(tmp_path / "permanent" / "queries")) == 2

    def test_deleting_an_entry_removes_its_crop_and_leaves_the_base_alone(self, make_stack, tmp_path):
        with TestClient(make_stack()) as client:
            api = _Api(client)
            api.save_observation(RED)
            query_id = api.search(RED).json()["query_id"]

            assert client.delete(f"/api/search-history/{query_id}").status_code == 422        # без токена
            assert client.delete(f"/api/search-history/{query_id}", headers=ADMIN).status_code == 200

            assert client.get("/api/search-history").json()["total"] == 0
            assert _files_in(tmp_path / "permanent" / "queries") == []
            assert client.get("/api/observations").json()["total"] == 1
            # изображения самой базы наблюдений остались на месте
            assert len(_files_in(tmp_path / "permanent" / "crops")) == 1

    def test_clear_empties_the_journal_but_not_the_base(self, make_stack, tmp_path):
        with TestClient(make_stack()) as client:
            api = _Api(client)
            api.save_observation(RED)
            for _ in range(3):
                api.search(RED)

            response = client.delete("/api/search-history", headers=ADMIN)

            assert response.json() == {"deleted": 3}
            assert client.get("/api/search-history").json()["total"] == 0
            assert _files_in(tmp_path / "permanent" / "queries") == []
            assert client.get("/api/observations").json()["total"] == 1


class TestDatabaseCreatedBeforeJournalExisted:
    """Сценарий реального пользователя: БД создана старыми скриптами, таблицы search_queries нет."""

    def test_search_keeps_working_and_server_reports_the_problem(self, make_stack):
        with TestClient(make_stack(drop_history_table=True)) as client:
            api = _Api(client)
            api.save_observation(RED)

            response = api.search(RED)

            assert response.status_code == 200                       # поиск не сломан
            assert response.json()["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-3)
            assert response.json()["query_id"] is None               # истории нет, но об этом честно сказано

            health = client.get("/health").json()
            assert health["status"] == "ok"
            assert health["schema"]["search_history_table"] is False
            assert any("create_table" in w for w in health["warnings"])