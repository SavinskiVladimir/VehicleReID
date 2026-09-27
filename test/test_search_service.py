"""
Тесты для backend/app/services/search_service.py.

Все зависимости (ImageService, ReIDEmbedderService, repository) - моки:
логика самого SearchService (оркестрация вызовов и сборка Pydantic-ответа)
проверяется изолированно. Реальный сквозной прогон (с настоящим ONNX и
настоящим PostgreSQL+pgvector) проводился вручную при разработке.

Запуск: из корня проекта  ->  pytest tests/test_search_service.py
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from PIL import Image

import backend.app.services.search_service as search_service_module
from backend.app.services.search_service import SearchService


class FakeImageService:
    def preprocess(self, image):
        return "PREPROCESSED"


class FakeEmbedder:
    def embed_as_list(self, preprocessed):
        assert preprocessed == "PREPROCESSED"
        return [0.1] * 512


def _fake_observation(image_path="a.jpg"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        image_path=image_path,
        crop_path=image_path.replace(".jpg", "_crop.jpg"),
        bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def search_service():
    return SearchService(image_service=FakeImageService(), embedder=FakeEmbedder())


class TestComputeEmbedding:
    def test_delegates_to_image_service_and_embedder(self, search_service):
        embedding = search_service.compute_embedding(Image.new("RGB", (10, 10)))
        assert embedding == [0.1] * 512


class TestSearch:
    @pytest.mark.asyncio
    async def test_returns_ranked_results_with_query_bbox(self, search_service, monkeypatch):
        fake_results = [
            (_fake_observation("a.jpg"), 0.95),
            (_fake_observation("b.jpg"), 0.80),
        ]

        async def fake_search_similar(session, embedding, top_k):
            assert embedding == [0.1] * 512
            assert top_k == 5
            return fake_results

        monkeypatch.setattr(
            search_service_module.repository, "search_similar", fake_search_similar
        )

        response = await search_service.search(
            session=object(), crop=Image.new("RGB", (10, 10)),
            query_bbox=(1, 2, 30, 40), top_k=5,
        )

        assert response.query_bbox.x1 == 1
        assert response.query_bbox.y2 == 40
        assert response.top_k == 2
        assert response.results[0].observation.image_path == "a.jpg"
        assert response.results[0].similarity == 0.95
        assert response.results[1].similarity == 0.80

    @pytest.mark.asyncio
    async def test_empty_results(self, search_service, monkeypatch):
        async def fake_search_similar(session, embedding, top_k):
            return []

        monkeypatch.setattr(
            search_service_module.repository, "search_similar", fake_search_similar
        )

        response = await search_service.search(
            session=object(), crop=Image.new("RGB", (10, 10)),
            query_bbox=(0, 0, 10, 10), top_k=5,
        )
        assert response.top_k == 0
        assert response.results == []


class TestAddObservation:
    @pytest.mark.asyncio
    async def test_calls_repository_with_computed_embedding(self, search_service, monkeypatch):
        captured = {}

        async def fake_add_observation(session, image_path, crop_path, bbox, embedding):
            captured["image_path"] = image_path
            captured["crop_path"] = crop_path
            captured["bbox"] = bbox
            captured["embedding"] = embedding
            return _fake_observation(image_path)

        monkeypatch.setattr(
            search_service_module.repository, "add_observation", fake_add_observation
        )

        result = await search_service.add_observation(
            session=object(), image_path="frame.jpg", crop_path="crop.jpg",
            crop=Image.new("RGB", (10, 10)), bbox=(5, 5, 50, 60),
        )

        assert captured["image_path"] == "frame.jpg"
        assert captured["crop_path"] == "crop.jpg"
        assert captured["bbox"] == (5, 5, 50, 60)
        assert captured["embedding"] == [0.1] * 512
        assert result.image_path == "frame.jpg"


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_returns_paginated_response(self, search_service, monkeypatch):
        fake_items = [_fake_observation("a.jpg"), _fake_observation("b.jpg")]

        async def fake_list_observations(session, limit, offset):
            assert limit == 10
            assert offset == 5
            return fake_items, 42

        monkeypatch.setattr(
            search_service_module.repository, "list_observations", fake_list_observations
        )

        response = await search_service.get_history(session=object(), limit=10, offset=5)

        assert response.total == 42
        assert response.limit == 10
        assert response.offset == 5
        assert len(response.items) == 2


class TestDeleteObservation:
    @pytest.mark.asyncio
    async def test_delegates_to_repository(self, search_service, monkeypatch):
        async def fake_delete_observation(session, observation_id):
            assert observation_id == "some-id"
            return True

        monkeypatch.setattr(
            search_service_module.repository, "delete_observation", fake_delete_observation
        )

        result = await search_service.delete_observation(session=object(), observation_id="some-id")
        assert result is True
