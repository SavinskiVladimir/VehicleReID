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
from backend.app.schemas.vehicle import BoundingBox, ObservationHistoryResponse, ObservationOut, SearchResponse, SearchResultItem


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

    async def search(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def add_observation(self, *args, **kwargs):
        raise RuntimeError("database down")

    async def get_history(self, *args, **kwargs):
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
        self.observation_to_return = None  # для эндпоинтов отдачи изображений

    async def search(self, session, crop, query_bbox, top_k):
        obs = _fake_observation_out()
        return SearchResponse(
            query_bbox=BoundingBox(x1=query_bbox[0], y1=query_bbox[1], x2=query_bbox[2], y2=query_bbox[3]),
            results=[SearchResultItem(observation=obs, similarity=0.9)],
        )

    async def add_observation(self, session, image_path, crop_path, crop, bbox):
        self.add_observation_called_with = {"image_path": image_path, "crop_path": crop_path, "bbox": bbox}
        return _fake_observation_out(image_path)

    async def get_history(self, session, limit, offset):
        return ObservationHistoryResponse(items=[_fake_observation_out()], total=1, limit=limit, offset=offset)

    async def delete_observation(self, session, observation_id):
        return self.delete_observation_result

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

class TestHistoryEndpoint:
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