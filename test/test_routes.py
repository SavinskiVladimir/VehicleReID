"""
Тесты для backend/app/api/routes.py.

Подход: lifespan должен успешно стартовать (поэтому DetectorService/
ReIDEmbedderService подменяются заглушками на уровне main.py, как в
test_main.py - без этого даже поднять приложение для теста не
получится). А вот ПОВЕДЕНИЕ конкретных эндпоинтов контролируется через
FastAPI-идиоматичный механизм app.dependency_overrides - подменяем
get_detector_service/get_search_service на фейки с нужным для теста
поведением, независимо от того, что реально создал lifespan.

FileStorage НЕ мокается - используется настоящая реализация поверх
pytest tmp_path, через переменные окружения TEMP_UPLOAD_DIR/
PERMANENT_STORAGE_DIR - дешёвые локальные файловые операции, нет
смысла их имитировать, полезнее проверить по-настоящему.

Запуск: из корня проекта  ->  pytest tests/test_routes.py
"""

import io
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
load_dotenv()

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.app.main as main_module
from backend.app.config import get_settings
from backend.app.dependencies import get_detector_service, get_search_service
from backend.app.schemas.vehicle import (
    BoundingBox,
    ObservationListResponse,
    ObservationOut,
    SearchHistoryItem,
    SearchHistoryResponse,
    SearchResponse,
    SearchResultItem,
)


ADMIN_TOKEN = "test-admin-token-123"


class FakeDetectorService:
    """Подменяет реальный ultralytics.YOLO при старте lifespan."""

    def __init__(self, **kwargs):
        pass


class FakeReIDEmbedderService:
    """Подменяет реальный onnxruntime.InferenceSession при старте lifespan."""

    def __init__(self, **kwargs):
        pass


@pytest.fixture(autouse=True)
def patch_ml_services_for_lifespan(monkeypatch):
    monkeypatch.setattr(main_module, "DetectorService", FakeDetectorService)
    monkeypatch.setattr(main_module, "ReIDEmbedderService", FakeReIDEmbedderService)
    # Файлов моделей в тестах нет, заглушки не считают эмбеддинги - проверки
    # старта отключены (они покрыты в test_startup_checks.py).
    monkeypatch.setattr(main_module, "check_model_files", lambda settings: None)
    monkeypatch.setattr(main_module, "check_embedding_dimension", lambda *args, **kwargs: 512)
    yield


@pytest.fixture(autouse=True)
def configure_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_TOKEN)
    monkeypatch.setenv("TEMP_UPLOAD_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PERMANENT_STORAGE_DIR", str(tmp_path / "permanent"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_jpeg_bytes(size=(60, 40), color=(120, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeVehicleDetection:
    def __init__(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence


class FakeDetector:
    """Настраиваемая заглушка для dependency_overrides[get_detector_service]."""

    def __init__(self, detections=None):
        self._detections = detections if detections is not None else []

    def detect(self, image):
        return self._detections

    def crop_with_padding(self, image, bbox):
        return image.crop(bbox) if bbox[2] > bbox[0] and bbox[3] > bbox[1] else image


class FakeSearchServiceRaises:
    """Имитирует недоступность подсистемы поиска/хранения (ТЗ п.1.1.2.4)."""

    def compute_embedding(self, crop):
        return [0.1] * 512  # эмбеддинг считается без БД - падает именно поиск

    async def search_by_embedding(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def search(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def add_observation(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def record_search(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def prune_search_history(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def list_search_history(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def get_search_crop_path(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def rerun_search(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def delete_search_query(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def clear_search_history(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def list_observations(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def delete_observation(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def get_observation(self, *args, **kwargs):
        raise RuntimeError("database down")


def _fake_observation_out(image_path="a.jpg") -> ObservationOut:
    return ObservationOut(
        id=uuid.uuid4(),
        image_path=image_path,
        crop_path=image_path.replace(".jpg", "_crop.jpg"),
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        created_at=datetime.now(timezone.utc),
    )


class FakeSearchServiceOk:
    def __init__(self):
        self.add_observation_called_with = None
        self.delete_observation_result = True
        self.delete_files = []                # файлы, ставшие ненужными после удаления
        self.observation_to_return = None  # для эндпоинтов отдачи изображений
        # --- журнал поиска ---
        self.record_search_calls = []
        self.record_search_error = None       # если задано - record_search бросает это исключение
        self.prune_calls = []
        self.paths_to_prune = []              # какие пути "удалены" при обрезке журнала
        self.history_items = []               # что вернёт list_search_history
        self.crop_path_to_return = None       # что вернёт get_search_crop_path
        self.rerun_response = None            # что вернёт rerun_search (None = нет записи)
        self.rerun_calls = []
        self.delete_query_result = None       # путь кропа удалённой записи или None (нет записи)
        self.cleared_paths = []

    def compute_embedding(self, crop):
        return [0.1] * 512

    async def search_by_embedding(self, session, embedding, query_bbox, top_k):
        obs = _fake_observation_out()
        return SearchResponse(
            query_bbox=BoundingBox(x1=query_bbox[0], y1=query_bbox[1], x2=query_bbox[2], y2=query_bbox[3]),
            results=[SearchResultItem(observation=obs, similarity=0.9)],
        )

    async def record_search(self, session, crop_path, bbox, embedding, top_k, response):
        if self.record_search_error:
            raise self.record_search_error
        self.record_search_calls.append({"crop_path": crop_path, "bbox": bbox, "top_k": top_k})
        return SearchHistoryItem(
            id=uuid.uuid4(), created_at=datetime.now(timezone.utc),
            bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
            top_k=top_k, result_count=len(response.results), top_similarity=0.9,
        )

    async def prune_search_history(self, session, keep):
        self.prune_calls.append(keep)
        return self.paths_to_prune

    async def list_search_history(self, session, limit, offset):
        return SearchHistoryResponse(items=self.history_items, total=len(self.history_items), limit=limit, offset=offset)

    async def get_search_crop_path(self, session, query_id):
        return self.crop_path_to_return

    async def rerun_search(self, session, query_id, top_k=None):
        self.rerun_calls.append({"query_id": query_id, "top_k": top_k})
        return self.rerun_response

    async def delete_search_query(self, session, query_id):
        return self.delete_query_result

    async def clear_search_history(self, session):
        return self.cleared_paths

    async def add_observation(self, session, image_path, crop_path, crop, bbox):
        self.add_observation_called_with = {"image_path": image_path, "crop_path": crop_path, "bbox": bbox}
        return _fake_observation_out(image_path)

    async def list_observations(self, session, limit, offset):
        return ObservationListResponse(items=[_fake_observation_out()], total=1, limit=limit, offset=offset)

    async def delete_observation(self, session, observation_id):
        return self.delete_files if self.delete_observation_result else None

    async def get_observation(self, session, observation_id):
        return self.observation_to_return


def _make_client(detector=None, search_service=None):
    app = main_module.create_app()
    if detector is not None:
        app.dependency_overrides[get_detector_service] = lambda: detector
    if search_service is not None:
        app.dependency_overrides[get_search_service] = lambda: search_service
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/detect
# ---------------------------------------------------------------------------

class TestDetectEndpoint:
    def test_success_returns_upload_id_and_detections(self):
        detector = FakeDetector(detections=[
            _FakeVehicleDetection((5, 5, 40, 30), 0.92),
            _FakeVehicleDetection((10, 10, 20, 20), 0.55),
        ])
        client = _make_client(detector=detector)

        with client:
            response = client.post(
                "/api/detect",
                files={"file": ("car.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert len(data["detections"]) == 2
        assert data["detections"][0]["confidence"] == pytest.approx(0.92)
        assert data["detections"][0]["bbox"] == {"x1": 5, "y1": 5, "x2": 40, "y2": 30}

    def test_no_detections_returns_empty_list(self):
        client = _make_client(detector=FakeDetector(detections=[]))

        with client:
            response = client.post(
                "/api/detect",
                files={"file": ("car.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )

        assert response.status_code == 200
        assert response.json()["detections"] == []

    def test_rejects_invalid_file_format(self):
        client = _make_client(detector=FakeDetector())

        with client:
            response = client.post(
                "/api/detect",
                files={"file": ("not_image.txt", b"hello world", "text/plain")},
            )

        assert response.status_code == 400

    def test_rejects_oversized_file(self, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "100")
        get_settings.cache_clear()
        client = _make_client(detector=FakeDetector())

        with client:
            response = client.post(
                "/api/detect",
                files={"file": ("car.jpg", _make_jpeg_bytes(size=(200, 200)), "image/jpeg")},
            )

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/search
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    def _detect_then_get_upload_id(self, client, detections):
        client.app.dependency_overrides[get_detector_service] = lambda: FakeDetector(detections=detections)
        response = client.post("/api/detect", files={"file": ("car.jpg", _make_jpeg_bytes(), "image/jpeg")})
        assert response.status_code == 200
        return response.json()["upload_id"]

    def test_success(self):
        detections = [_FakeVehicleDetection((0, 0, 40, 30), 0.9)]
        search_service = FakeSearchServiceOk()
        client = _make_client(search_service=search_service)

        with client:
            upload_id = self._detect_then_get_upload_id(client, detections)

            response = client.post(
                "/api/search",
                json={"upload_id": upload_id, "detection_index": 0, "top_k": 5},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["similarity"] == pytest.approx(0.9)

    def test_unknown_upload_id_returns_404(self):
        client = _make_client(detector=FakeDetector(), search_service=FakeSearchServiceOk())

        with client:
            response = client.post(
                "/api/search",
                json={"upload_id": str(uuid.uuid4()), "detection_index": 0, "top_k": 5},
            )

        assert response.status_code == 404

    def test_detection_index_out_of_range_returns_400(self):
        detections = [_FakeVehicleDetection((0, 0, 40, 30), 0.9)]
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            upload_id = self._detect_then_get_upload_id(client, detections)
            response = client.post(
                "/api/search",
                json={"upload_id": upload_id, "detection_index": 5, "top_k": 5},
            )

        assert response.status_code == 400

    def test_search_backend_unavailable_returns_503(self):
        detections = [_FakeVehicleDetection((0, 0, 40, 30), 0.9)]
        client = _make_client(search_service=FakeSearchServiceRaises())

        with client:
            upload_id = self._detect_then_get_upload_id(client, detections)
            response = client.post(
                "/api/search",
                json={"upload_id": upload_id, "detection_index": 0, "top_k": 5},
            )

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/observations (admin)
# ---------------------------------------------------------------------------

class TestAddObservationEndpoint:
    def _detect_then_get_upload_id(self, client, detections):
        client.app.dependency_overrides[get_detector_service] = lambda: FakeDetector(detections=detections)
        response = client.post("/api/detect", files={"file": ("car.jpg", _make_jpeg_bytes(), "image/jpeg")})
        return response.json()["upload_id"]

    def test_missing_admin_token_returns_422(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.post(
                "/api/observations",
                json={"upload_id": str(uuid.uuid4()), "detection_index": 0},
            )

        assert response.status_code == 422  # Header(...) обязателен на уровне FastAPI

    def test_wrong_admin_token_returns_403(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.post(
                "/api/observations",
                json={"upload_id": str(uuid.uuid4()), "detection_index": 0},
                headers={"X-Admin-Token": "wrong-token"},
            )

        assert response.status_code == 403

    def test_correct_admin_token_succeeds(self):
        detections = [_FakeVehicleDetection((0, 0, 40, 30), 0.9)]
        search_service = FakeSearchServiceOk()
        client = _make_client(search_service=search_service)

        with client:
            upload_id = self._detect_then_get_upload_id(client, detections)
            response = client.post(
                "/api/observations",
                json={"upload_id": upload_id, "detection_index": 0},
                headers={"X-Admin-Token": ADMIN_TOKEN},
            )

        assert response.status_code == 200
        assert search_service.add_observation_called_with is not None
        # Пути к файлам должны быть РЕАЛЬНЫМИ путями на диске (permanent-хранилище),
        # а не просто какими-то строками-заглушками.
        from pathlib import Path
        assert Path(search_service.add_observation_called_with["image_path"]).is_file()
        assert Path(search_service.add_observation_called_with["crop_path"]).is_file()

    def test_storage_unavailable_returns_503(self):
        detections = [_FakeVehicleDetection((0, 0, 40, 30), 0.9)]
        client = _make_client(search_service=FakeSearchServiceRaises())

        with client:
            upload_id = self._detect_then_get_upload_id(client, detections)
            response = client.post(
                "/api/observations",
                json={"upload_id": upload_id, "detection_index": 0},
                headers={"X-Admin-Token": ADMIN_TOKEN},
            )

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/observations (история - НЕ admin)
# ---------------------------------------------------------------------------

class TestObservationListEndpoint:
    def test_no_admin_token_required(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.get("/api/observations")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1

    def test_pagination_params_passed_through(self):
        search_service = FakeSearchServiceOk()
        client = _make_client(search_service=search_service)

        with client:
            response = client.get("/api/observations?limit=5&offset=10")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 10

    def test_invalid_limit_returns_422(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.get("/api/observations?limit=0")

        assert response.status_code == 422

    def test_backend_unavailable_returns_503(self):
        client = _make_client(search_service=FakeSearchServiceRaises())

        with client:
            response = client.get("/api/observations")

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# DELETE /api/observations/{id} (admin)
# ---------------------------------------------------------------------------

class TestDeleteObservationEndpoint:
    def test_missing_admin_token_returns_422(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.delete(f"/api/observations/{uuid.uuid4()}")

        assert response.status_code == 422

    def test_wrong_admin_token_returns_403(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.delete(
                f"/api/observations/{uuid.uuid4()}", headers={"X-Admin-Token": "wrong"}
            )

        assert response.status_code == 403

    def test_correct_token_existing_observation_succeeds(self):
        search_service = FakeSearchServiceOk()
        search_service.delete_observation_result = True
        client = _make_client(search_service=search_service)

        with client:
            response = client.delete(
                f"/api/observations/{uuid.uuid4()}", headers={"X-Admin-Token": ADMIN_TOKEN}
            )

        assert response.status_code == 200
        assert response.json()["deleted"] is True

    def test_correct_token_nonexistent_observation_returns_404(self):
        search_service = FakeSearchServiceOk()
        search_service.delete_observation_result = False
        client = _make_client(search_service=search_service)

        with client:
            response = client.delete(
                f"/api/observations/{uuid.uuid4()}", headers={"X-Admin-Token": ADMIN_TOKEN}
            )

        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.delete("/api/observations/not-a-uuid", headers={"X-Admin-Token": ADMIN_TOKEN})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/observations/{id}/crop и /image
# ---------------------------------------------------------------------------

class TestObservationImageEndpoints:
    """
    Пути к файлам берутся ИЗ БД, а не из жёстко заданной папки: записи,
    добавленные демо-скриптом (путь на исходный файл датасета), и
    записи, добавленные через API (permanent-хранилище), отдаются
    одинаково.
    """

    def _observation(self, image_path, crop_path):
        return ObservationOut(
            id=uuid.uuid4(), image_path=str(image_path), crop_path=str(crop_path),
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10), created_at=datetime.now(timezone.utc),
        )

    def _save_jpeg(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_make_jpeg_bytes())
        return path

    def test_crop_served_from_arbitrary_path_stored_in_db(self, tmp_path):
        # Файл лежит НЕ в permanent-хранилище, а где-то ещё (как у
        # записей из demo_populate_db.py) - путь всё равно из БД.
        crop = self._save_jpeg(tmp_path / "some_dataset" / "0371_c015_00065220_0.jpg")
        search_service = FakeSearchServiceOk()
        search_service.observation_to_return = self._observation(crop, crop)
        client = _make_client(search_service=search_service)

        with client:
            response = client.get(f"/api/observations/{uuid.uuid4()}/crop")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == crop.read_bytes()

    def test_full_image_endpoint(self, tmp_path):
        image = self._save_jpeg(tmp_path / "images" / "full.jpg")
        crop = self._save_jpeg(tmp_path / "crops" / "crop.jpg")
        search_service = FakeSearchServiceOk()
        search_service.observation_to_return = self._observation(image, crop)
        client = _make_client(search_service=search_service)

        with client:
            response = client.get(f"/api/observations/{uuid.uuid4()}/image")

        assert response.status_code == 200
        assert response.content == image.read_bytes()

    def test_unknown_observation_returns_404(self):
        search_service = FakeSearchServiceOk()
        search_service.observation_to_return = None
        client = _make_client(search_service=search_service)

        with client:
            response = client.get(f"/api/observations/{uuid.uuid4()}/crop")

        assert response.status_code == 404

    def test_file_missing_on_disk_returns_404(self, tmp_path):
        missing = tmp_path / "deleted.jpg"
        search_service = FakeSearchServiceOk()
        search_service.observation_to_return = self._observation(missing, missing)
        client = _make_client(search_service=search_service)

        with client:
            response = client.get(f"/api/observations/{uuid.uuid4()}/crop")

        assert response.status_code == 404

    def test_non_image_extension_is_never_served(self, tmp_path):
        secret = tmp_path / "secrets.txt"
        secret.write_text("password=123")
        search_service = FakeSearchServiceOk()
        search_service.observation_to_return = self._observation(secret, secret)
        client = _make_client(search_service=search_service)

        with client:
            response = client.get(f"/api/observations/{uuid.uuid4()}/crop")

        assert response.status_code == 404
        assert b"password" not in response.content

    def test_database_unavailable_returns_503(self):
        client = _make_client(search_service=FakeSearchServiceRaises())

        with client:
            response = client.get(f"/api/observations/{uuid.uuid4()}/crop")

        assert response.status_code == 503

    def test_invalid_uuid_returns_422(self):
        client = _make_client(search_service=FakeSearchServiceOk())

        with client:
            response = client.get("/api/observations/not-a-uuid/crop")

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Запись в журнал поиска при POST /api/search
# ---------------------------------------------------------------------------

def _upload_and_get_id(client):
    client.app.dependency_overrides[get_detector_service] = lambda: FakeDetector(
        detections=[_FakeVehicleDetection((0, 0, 40, 30), 0.9)]
    )
    response = client.post("/api/detect", files={"file": ("car.jpg", _make_jpeg_bytes(), "image/jpeg")})
    return response.json()["upload_id"]


def _history_item(top_k=10) -> SearchHistoryItem:
    return SearchHistoryItem(
        id=uuid.uuid4(), created_at=datetime.now(timezone.utc),
        bbox=BoundingBox(x1=1, y1=2, x2=30, y2=40), top_k=top_k, result_count=3, top_similarity=0.87,
    )


class TestSearchWritesToHistory:
    def test_successful_search_is_recorded_and_query_id_returned(self, tmp_path):
        search_service = FakeSearchServiceOk()
        client = _make_client(search_service=search_service)

        with client:
            upload_id = _upload_and_get_id(client)
            response = client.post("/api/search", json={"upload_id": upload_id, "detection_index": 0, "top_k": 7})

        assert response.status_code == 200
        assert response.json()["query_id"] is not None
        assert len(search_service.record_search_calls) == 1
        call = search_service.record_search_calls[0]
        assert call["top_k"] == 7
        assert call["bbox"] == (0, 0, 40, 30)
        # Кроп запроса РЕАЛЬНО сохранён в постоянное хранилище (под превью истории).
        from pathlib import Path
        assert Path(call["crop_path"]).is_file()
        assert (tmp_path / "permanent" / "queries") in Path(call["crop_path"]).parents

    def test_journal_is_pruned_to_configured_limit(self, monkeypatch):
        monkeypatch.setenv("SEARCH_HISTORY_MAX_ITEMS", "7")
        get_settings.cache_clear()
        search_service = FakeSearchServiceOk()
        client = _make_client(search_service=search_service)

        with client:
            upload_id = _upload_and_get_id(client)
            client.post("/api/search", json={"upload_id": upload_id, "detection_index": 0})

        assert search_service.prune_calls == [7]

    def test_files_of_pruned_records_are_deleted_from_disk(self, tmp_path):
        old_crop = tmp_path / "permanent" / "queries" / "old.jpg"
        old_crop.parent.mkdir(parents=True)
        old_crop.write_bytes(_make_jpeg_bytes())
        search_service = FakeSearchServiceOk()
        search_service.paths_to_prune = [str(old_crop)]
        client = _make_client(search_service=search_service)

        with client:
            upload_id = _upload_and_get_id(client)
            client.post("/api/search", json={"upload_id": upload_id, "detection_index": 0})

        assert not old_crop.exists()

    def test_pruning_never_deletes_files_outside_the_storage(self, tmp_path):
        # Путь в журнале может указывать куда угодно (запись из БД) - удалять
        # разрешено только внутри постоянного хранилища.
        foreign = tmp_path / "dataset" / "precious.jpg"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(_make_jpeg_bytes())
        search_service = FakeSearchServiceOk()
        search_service.paths_to_prune = [str(foreign)]
        client = _make_client(search_service=search_service)

        with client:
            upload_id = _upload_and_get_id(client)
            client.post("/api/search", json={"upload_id": upload_id, "detection_index": 0})

        assert foreign.exists()

    def test_search_still_succeeds_when_history_write_fails(self, tmp_path):
        # Журнал вторичен: пользователь получил результат поиска, сбой записи
        # истории не должен превращать его в ошибку.
        search_service = FakeSearchServiceOk()
        search_service.record_search_error = RuntimeError("таблицы search_queries нет")
        client = _make_client(search_service=search_service)

        with client:
            upload_id = _upload_and_get_id(client)
            response = client.post("/api/search", json={"upload_id": upload_id, "detection_index": 0})

        assert response.status_code == 200
        assert len(response.json()["results"]) == 1
        assert response.json()["query_id"] is None
        # и не осталось файла-сироты без записи в журнале
        queries_dir = tmp_path / "permanent" / "queries"
        assert not queries_dir.exists() or list(queries_dir.iterdir()) == []

    def test_failed_search_is_not_recorded(self):
        client = _make_client(search_service=FakeSearchServiceRaises())

        with client:
            upload_id = _upload_and_get_id(client)
            response = client.post("/api/search", json={"upload_id": upload_id, "detection_index": 0})

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/search-history, /{id}/crop, POST /{id}/search, DELETE
# ---------------------------------------------------------------------------

class TestSearchHistoryList:
    def test_open_to_everyone_and_paginated(self):
        search_service = FakeSearchServiceOk()
        search_service.history_items = [_history_item(), _history_item()]
        client = _make_client(search_service=search_service)

        with client:
            response = client.get("/api/search-history?limit=5&offset=10")  # без токена

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2 and data["limit"] == 5 and data["offset"] == 10
        assert data["items"][0]["top_similarity"] == pytest.approx(0.87)
        assert data["items"][0]["result_count"] == 3

    def test_response_exposes_no_server_file_paths_or_vectors(self):
        search_service = FakeSearchServiceOk()
        search_service.history_items = [_history_item()]
        client = _make_client(search_service=search_service)

        with client:
            text = client.get("/api/search-history").text

        assert "crop_path" not in text and "embedding" not in text

    def test_invalid_limit_returns_422(self):
        client = _make_client(search_service=FakeSearchServiceOk())
        with client:
            assert client.get("/api/search-history?limit=0").status_code == 422
            assert client.get("/api/search-history?limit=101").status_code == 422

    def test_backend_unavailable_returns_503(self):
        client = _make_client(search_service=FakeSearchServiceRaises())
        with client:
            assert client.get("/api/search-history").status_code == 503


class TestSearchHistoryCrop:
    def test_serves_crop_image(self, tmp_path):
        crop = tmp_path / "permanent" / "queries" / "q.jpg"
        crop.parent.mkdir(parents=True)
        crop.write_bytes(_make_jpeg_bytes())
        search_service = FakeSearchServiceOk()
        search_service.crop_path_to_return = str(crop)
        client = _make_client(search_service=search_service)

        with client:
            response = client.get(f"/api/search-history/{uuid.uuid4()}/crop")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == crop.read_bytes()

    def test_unknown_record_returns_404(self):
        client = _make_client(search_service=FakeSearchServiceOk())  # crop_path_to_return=None
        with client:
            assert client.get(f"/api/search-history/{uuid.uuid4()}/crop").status_code == 404

    def test_missing_file_returns_404(self, tmp_path):
        search_service = FakeSearchServiceOk()
        search_service.crop_path_to_return = str(tmp_path / "gone.jpg")
        client = _make_client(search_service=search_service)
        with client:
            assert client.get(f"/api/search-history/{uuid.uuid4()}/crop").status_code == 404

    def test_backend_unavailable_returns_503(self):
        client = _make_client(search_service=FakeSearchServiceRaises())
        with client:
            assert client.get(f"/api/search-history/{uuid.uuid4()}/crop").status_code == 503


class TestRerunSearch:
    def _response(self, query_id):
        return SearchResponse(
            query_bbox=BoundingBox(x1=1, y1=2, x2=30, y2=40),
            results=[SearchResultItem(observation=_fake_observation_out(), similarity=0.8)],
            query_id=query_id,
        )

    def test_rerun_returns_results_without_any_upload(self):
        query_id = uuid.uuid4()
        search_service = FakeSearchServiceOk()
        search_service.rerun_response = self._response(query_id)
        client = _make_client(search_service=search_service)

        with client:
            response = client.post(f"/api/search-history/{query_id}/search")  # ни файла, ни токена

        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == str(query_id)
        assert data["results"][0]["similarity"] == pytest.approx(0.8)
        assert search_service.rerun_calls == [{"query_id": query_id, "top_k": None}]

    def test_rerun_is_not_written_to_history_again(self):
        query_id = uuid.uuid4()
        search_service = FakeSearchServiceOk()
        search_service.rerun_response = self._response(query_id)
        client = _make_client(search_service=search_service)

        with client:
            client.post(f"/api/search-history/{query_id}/search")

        assert search_service.record_search_calls == []

    def test_top_k_override_is_passed_through(self):
        query_id = uuid.uuid4()
        search_service = FakeSearchServiceOk()
        search_service.rerun_response = self._response(query_id)
        client = _make_client(search_service=search_service)

        with client:
            client.post(f"/api/search-history/{query_id}/search?top_k=25")

        assert search_service.rerun_calls[0]["top_k"] == 25

    def test_unknown_record_returns_404(self):
        client = _make_client(search_service=FakeSearchServiceOk())  # rerun_response=None
        with client:
            assert client.post(f"/api/search-history/{uuid.uuid4()}/search").status_code == 404

    def test_invalid_top_k_returns_422(self):
        client = _make_client(search_service=FakeSearchServiceOk())
        with client:
            assert client.post(f"/api/search-history/{uuid.uuid4()}/search?top_k=0").status_code == 422

    def test_backend_unavailable_returns_503(self):
        client = _make_client(search_service=FakeSearchServiceRaises())
        with client:
            assert client.post(f"/api/search-history/{uuid.uuid4()}/search").status_code == 503


class TestDeleteSearchHistory:
    def test_delete_one_requires_admin_token(self):
        client = _make_client(search_service=FakeSearchServiceOk())
        with client:
            qid = uuid.uuid4()
            assert client.delete(f"/api/search-history/{qid}").status_code == 422
            assert client.delete(f"/api/search-history/{qid}", headers={"X-Admin-Token": "wrong"}).status_code == 403

    def test_delete_one_removes_record_and_its_crop_file(self, tmp_path):
        crop = tmp_path / "permanent" / "queries" / "q.jpg"
        crop.parent.mkdir(parents=True)
        crop.write_bytes(_make_jpeg_bytes())
        search_service = FakeSearchServiceOk()
        search_service.delete_query_result = str(crop)
        client = _make_client(search_service=search_service)

        with client:
            response = client.delete(f"/api/search-history/{uuid.uuid4()}", headers={"X-Admin-Token": ADMIN_TOKEN})

        assert response.status_code == 200
        assert not crop.exists()

    def test_delete_one_unknown_record_returns_404(self):
        client = _make_client(search_service=FakeSearchServiceOk())  # delete_query_result=None
        with client:
            response = client.delete(f"/api/search-history/{uuid.uuid4()}", headers={"X-Admin-Token": ADMIN_TOKEN})
        assert response.status_code == 404

    def test_delete_one_never_touches_files_outside_storage(self, tmp_path):
        foreign = tmp_path / "dataset" / "precious.jpg"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(_make_jpeg_bytes())
        search_service = FakeSearchServiceOk()
        search_service.delete_query_result = str(foreign)
        client = _make_client(search_service=search_service)

        with client:
            client.delete(f"/api/search-history/{uuid.uuid4()}", headers={"X-Admin-Token": ADMIN_TOKEN})

        assert foreign.exists()

    def test_clear_requires_admin_token(self):
        client = _make_client(search_service=FakeSearchServiceOk())
        with client:
            assert client.delete("/api/search-history").status_code == 422
            assert client.delete("/api/search-history", headers={"X-Admin-Token": "wrong"}).status_code == 403

    def test_clear_removes_all_crop_files_and_reports_count(self, tmp_path):
        crops = []
        for name in ("a.jpg", "b.jpg"):
            path = tmp_path / "permanent" / "queries" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_make_jpeg_bytes())
            crops.append(path)
        search_service = FakeSearchServiceOk()
        search_service.cleared_paths = [str(p) for p in crops]
        client = _make_client(search_service=search_service)

        with client:
            response = client.delete("/api/search-history", headers={"X-Admin-Token": ADMIN_TOKEN})

        assert response.status_code == 200
        assert response.json() == {"deleted": 2}
        assert not any(p.exists() for p in crops)

    def test_backend_unavailable_returns_503(self):
        client = _make_client(search_service=FakeSearchServiceRaises())
        with client:
            assert client.delete(f"/api/search-history/{uuid.uuid4()}", headers={"X-Admin-Token": ADMIN_TOKEN}).status_code == 503
            assert client.delete("/api/search-history", headers={"X-Admin-Token": ADMIN_TOKEN}).status_code == 503


class TestDeleteObservationRemovesFiles:
    @staticmethod
    def _files(tmp_path, *names):
        paths = []
        for name in names:
            path = tmp_path / "permanent" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_make_jpeg_bytes())
            paths.append(path)
        return paths

    def _delete(self, search_service):
        client = _make_client(search_service=search_service)
        with client:
            return client.delete(f"/api/observations/{uuid.uuid4()}", headers={"X-Admin-Token": ADMIN_TOKEN})

    def test_files_of_the_deleted_observation_are_removed_from_storage(self, tmp_path):
        crop, image = self._files(tmp_path, "crops/c.jpg", "images/i.jpg")
        search_service = FakeSearchServiceOk()
        search_service.delete_files = [str(crop), str(image)]

        assert self._delete(search_service).status_code == 200
        assert not crop.exists() and not image.exists()

    def test_files_outside_the_storage_are_never_removed(self, tmp_path):
        # запись демо-скрипта указывает на файл исходного датасета
        dataset_file = tmp_path / "VeRi" / "0001.jpg"
        dataset_file.parent.mkdir()
        dataset_file.write_bytes(_make_jpeg_bytes())
        search_service = FakeSearchServiceOk()
        search_service.delete_files = [str(dataset_file)]

        assert self._delete(search_service).status_code == 200
        assert dataset_file.exists()

    def test_unknown_observation_removes_nothing(self, tmp_path):
        (crop,) = self._files(tmp_path, "crops/c.jpg")
        search_service = FakeSearchServiceOk()
        search_service.delete_observation_result = False
        search_service.delete_files = [str(crop)]

        assert self._delete(search_service).status_code == 404
        assert crop.exists()