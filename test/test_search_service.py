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


class TestListObservations:
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

        response = await search_service.list_observations(session=object(), limit=10, offset=5)

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


class TestGetObservation:
    @pytest.mark.asyncio
    async def test_returns_schema_for_existing_observation(self, search_service, monkeypatch):
        fake = _fake_observation("a.jpg")

        async def fake_get_observation(session, observation_id):
            assert observation_id == "some-id"
            return fake

        monkeypatch.setattr(
            search_service_module.repository, "get_observation", fake_get_observation
        )

        result = await search_service.get_observation(session=object(), observation_id="some-id")
        assert result.image_path == "a.jpg"
        assert result.bbox.x2 == 10

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, search_service, monkeypatch):
        async def fake_get_observation(session, observation_id):
            return None

        monkeypatch.setattr(
            search_service_module.repository, "get_observation", fake_get_observation
        )

        result = await search_service.get_observation(session=object(), observation_id="missing")
        assert result is None


# ---------------------------------------------------------------------------
# Журнал поиска и повторный поиск
# ---------------------------------------------------------------------------

def _fake_query(top_k=10, embedding=None):
    return SimpleNamespace(
        id=uuid.uuid4(), created_at=datetime.now(timezone.utc),
        crop_path="/store/queries/q.jpg", bbox={"x1": 1, "y1": 2, "x2": 30, "y2": 40},
        bbox_x1=1, bbox_y1=2, bbox_x2=30, bbox_y2=40,
        embedding=embedding if embedding is not None else [0.5] * 512,
        top_k=top_k, result_count=2, top_similarity=0.9,
    )


def _response_with(similarities):
    from backend.app.schemas.vehicle import SearchResponse, SearchResultItem, BoundingBox
    return SearchResponse(
        query_bbox=BoundingBox(x1=1, y1=2, x2=30, y2=40),
        results=[SearchResultItem(observation=_fake_observation(f"{i}.jpg"), similarity=s)
                 for i, s in enumerate(similarities)],
    )


class TestSearchByEmbedding:
    @pytest.mark.asyncio
    async def test_uses_given_vector_without_running_any_model(self, monkeypatch):
        # Сервис БЕЗ image_service/embedder: повторному поиску модели не нужны.
        service = SearchService(image_service=None, embedder=None)
        captured = {}

        async def fake_search_similar(session, embedding, top_k):
            captured["embedding"] = embedding
            captured["top_k"] = top_k
            return [(_fake_observation("a.jpg"), 0.9)]

        monkeypatch.setattr(search_service_module.repository, "search_similar", fake_search_similar)

        response = await service.search_by_embedding(
            session=object(), embedding=[0.3] * 512, query_bbox=(1, 2, 30, 40), top_k=7
        )

        assert captured == {"embedding": [0.3] * 512, "top_k": 7}
        assert response.query_bbox.x2 == 30
        assert response.results[0].similarity == 0.9


class TestSimilarityRange:
    @pytest.mark.asyncio
    async def test_out_of_range_similarity_does_not_break_the_search(self, monkeypatch):
        # Косинусное сходство лежит в [-1, 1]. Раньше отрицательное значение
        # (непохожий кандидат) нарушало ограничение схемы и весь поиск
        # возвращал 503; так же - выход за 1.0 из-за погрешности float32.
        service = SearchService(image_service=None, embedder=None)

        async def fake_search_similar(session, embedding, top_k):
            return [
                (_fake_observation("same.jpg"), 1.0000001),
                (_fake_observation("mid.jpg"), 0.42),
                (_fake_observation("opposite.jpg"), -0.337),
            ]

        monkeypatch.setattr(search_service_module.repository, "search_similar", fake_search_similar)

        response = await service.search_by_embedding(object(), [0.1] * 512, (0, 0, 1, 1), top_k=10)

        assert [item.similarity for item in response.results] == [1.0, 0.42, 0.0]


class TestRecordSearch:
    @pytest.mark.asyncio
    async def test_stores_outcome_summary(self, search_service, monkeypatch):
        captured = {}

        async def fake_add_search_query(session, **kwargs):
            captured.update(kwargs)
            return _fake_query()

        monkeypatch.setattr(search_service_module.repository, "add_search_query", fake_add_search_query)

        await search_service.record_search(
            session=object(), crop_path="/store/queries/q.jpg", bbox=(1, 2, 30, 40),
            embedding=[0.1] * 512, top_k=10, response=_response_with([0.7, 0.95, 0.4]),
        )

        assert captured["result_count"] == 3
        assert captured["top_similarity"] == 0.95  # лучшее, а не первое
        assert captured["top_k"] == 10
        assert captured["crop_path"] == "/store/queries/q.jpg"

    @pytest.mark.asyncio
    async def test_empty_result_has_no_top_similarity(self, search_service, monkeypatch):
        captured = {}

        async def fake_add_search_query(session, **kwargs):
            captured.update(kwargs)
            return _fake_query()

        monkeypatch.setattr(search_service_module.repository, "add_search_query", fake_add_search_query)
        await search_service.record_search(
            object(), "/x.jpg", (0, 0, 1, 1), [0.1] * 512, 10, _response_with([])
        )

        assert captured["result_count"] == 0
        assert captured["top_similarity"] is None


class TestRerunSearch:
    @pytest.mark.asyncio
    async def test_searches_by_stored_vector_and_bbox(self, monkeypatch):
        service = SearchService(image_service=None, embedder=None)
        stored = _fake_query(top_k=12, embedding=[0.25] * 512)
        captured = {}

        async def fake_get(session, query_id):
            return stored

        async def fake_search_similar(session, embedding, top_k):
            captured["embedding"] = embedding
            captured["top_k"] = top_k
            return [(_fake_observation("a.jpg"), 0.8)]

        monkeypatch.setattr(search_service_module.repository, "get_search_query", fake_get)
        monkeypatch.setattr(search_service_module.repository, "search_similar", fake_search_similar)

        response = await service.rerun_search(session=object(), query_id=stored.id)

        assert captured == {"embedding": [0.25] * 512, "top_k": 12}  # top_k - как в исходном поиске
        assert response.query_id == stored.id
        assert response.query_bbox.x1 == 1 and response.query_bbox.y2 == 40

    @pytest.mark.asyncio
    async def test_top_k_override(self, monkeypatch):
        service = SearchService(image_service=None, embedder=None)
        stored = _fake_query(top_k=10)
        captured = {}

        async def fake_get(session, query_id):
            return stored

        async def fake_search_similar(session, embedding, top_k):
            captured["top_k"] = top_k
            return []

        monkeypatch.setattr(search_service_module.repository, "get_search_query", fake_get)
        monkeypatch.setattr(search_service_module.repository, "search_similar", fake_search_similar)

        await service.rerun_search(object(), stored.id, top_k=30)
        assert captured["top_k"] == 30

    @pytest.mark.asyncio
    async def test_unknown_record_returns_none(self, monkeypatch):
        service = SearchService(image_service=None, embedder=None)

        async def fake_get(session, query_id):
            return None

        monkeypatch.setattr(search_service_module.repository, "get_search_query", fake_get)
        assert await service.rerun_search(object(), uuid.uuid4()) is None


class TestSearchHistoryListing:
    @pytest.mark.asyncio
    async def test_returns_paginated_schema_without_paths_or_vectors(self, search_service, monkeypatch):
        async def fake_list(session, limit, offset):
            return [_fake_query(), _fake_query()], 42

        monkeypatch.setattr(search_service_module.repository, "list_search_queries", fake_list)

        response = await search_service.list_search_history(object(), limit=2, offset=4)

        assert response.total == 42 and response.limit == 2 and response.offset == 4
        assert len(response.items) == 2
        dumped = response.model_dump_json()
        assert "crop_path" not in dumped and "embedding" not in dumped

    @pytest.mark.asyncio
    async def test_crop_path_lookup(self, search_service, monkeypatch):
        stored = _fake_query()

        async def fake_get(session, query_id):
            return stored if query_id == stored.id else None

        monkeypatch.setattr(search_service_module.repository, "get_search_query", fake_get)

        assert await search_service.get_search_crop_path(object(), stored.id) == "/store/queries/q.jpg"
        assert await search_service.get_search_crop_path(object(), uuid.uuid4()) is None