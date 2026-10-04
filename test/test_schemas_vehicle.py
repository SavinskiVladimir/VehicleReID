"""
Тесты для backend/app/schemas/vehicle.py.

Запуск: из корня проекта  ->  pytest tests/test_schemas_vehicle.py
"""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.schemas.vehicle import (
    BoundingBox,
    ObservationCreate,
    ObservationListResponse,
    ObservationOut,
    SearchHistoryItem,
    SearchHistoryResponse,
    SearchResponse,
    SearchResultItem,
)


class TestBoundingBox:
    def test_valid_bbox(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        assert bbox.as_tuple() == (10, 20, 100, 150)

    def test_x2_must_exceed_x1(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=100, y1=0, x2=50, y2=100)

    def test_y2_must_exceed_y1(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=0, y1=100, x2=50, y2=50)

    def test_equal_coordinates_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=10, x2=10, y2=50)


class TestObservationCreate:
    def _valid_bbox(self):
        return BoundingBox(x1=0, y1=0, x2=10, y2=10)

    def test_valid_embedding_dimension(self):
        obs = ObservationCreate(
            image_path="a.jpg", crop_path="a_crop.jpg",
            bbox=self._valid_bbox(), embedding=[0.1] * 512,
        )
        assert len(obs.embedding) == 512

    def test_wrong_embedding_dimension_rejected(self):
        with pytest.raises(ValidationError):
            ObservationCreate(
                image_path="a.jpg", crop_path="a_crop.jpg",
                bbox=self._valid_bbox(), embedding=[0.1] * 100,
            )

    def test_empty_embedding_rejected(self):
        with pytest.raises(ValidationError):
            ObservationCreate(
                image_path="a.jpg", crop_path="a_crop.jpg",
                bbox=self._valid_bbox(), embedding=[],
            )


class TestObservationOut:
    def test_builds_from_dict(self):
        obs = ObservationOut(
            id=uuid.uuid4(), image_path="a.jpg", crop_path="a_crop.jpg",
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            created_at=datetime.now(timezone.utc),
        )
        assert obs.bbox.x2 == 10

    def test_builds_from_object_with_attributes(self):
        # Имитация SQLAlchemy ORM-объекта: обычный объект с атрибутами,
        # а не dict - именно так search_service.py строит ObservationOut
        # из Observation через model_validate(obs, from_attributes=True).
        class FakeOrmObject:
            def __init__(self):
                self.id = uuid.uuid4()
                self.image_path = "a.jpg"
                self.crop_path = "a_crop.jpg"
                self.bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
                self.created_at = datetime.now(timezone.utc)

        obs = ObservationOut.model_validate(FakeOrmObject())
        assert obs.bbox.x2 == 10
        assert obs.image_path == "a.jpg"


class TestSearchResultItem:
    def test_similarity_bounds(self):
        obs = ObservationOut(
            id=uuid.uuid4(), image_path="a.jpg", crop_path="a_crop.jpg",
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            created_at=datetime.now(timezone.utc),
        )
        SearchResultItem(observation=obs, similarity=0.0)
        SearchResultItem(observation=obs, similarity=1.0)
        with pytest.raises(ValidationError):
            SearchResultItem(observation=obs, similarity=1.5)
        with pytest.raises(ValidationError):
            SearchResultItem(observation=obs, similarity=-0.1)


class TestSearchResponse:
    def test_top_k_matches_results_length(self):
        obs = ObservationOut(
            id=uuid.uuid4(), image_path="a.jpg", crop_path="a_crop.jpg",
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            created_at=datetime.now(timezone.utc),
        )
        response = SearchResponse(
            query_bbox=BoundingBox(x1=0, y1=0, x2=20, y2=20),
            results=[
                SearchResultItem(observation=obs, similarity=0.9),
                SearchResultItem(observation=obs, similarity=0.7),
            ],
        )
        assert response.top_k == 2


class TestObservationListResponse:
    def test_pagination_fields(self):
        response = ObservationListResponse(items=[], total=42, limit=10, offset=20)
        assert response.total == 42
        assert response.limit == 10
        assert response.offset == 20


class TestSearchHistorySchemas:
    def test_item_builds_from_orm_like_object_and_hides_internals(self):
        class FakeQuery:
            id = uuid.uuid4()
            created_at = datetime.now(timezone.utc)
            bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
            top_k = 10
            result_count = 4
            top_similarity = 0.91
            crop_path = "/server/secret/path.jpg"      # не должно попасть в схему
            embedding = [0.1] * 512                    # и это тоже

        item = SearchHistoryItem.model_validate(FakeQuery())
        dumped = item.model_dump()

        assert item.top_similarity == 0.91
        assert "crop_path" not in dumped and "embedding" not in dumped

    def test_top_similarity_is_optional(self):
        item = SearchHistoryItem(
            id=uuid.uuid4(), created_at=datetime.now(timezone.utc),
            bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1), top_k=10, result_count=0,
        )
        assert item.top_similarity is None

    def test_page(self):
        page = SearchHistoryResponse(items=[], total=5, limit=2, offset=2)
        assert page.total == 5

    def test_search_response_query_id_is_optional(self):
        response = SearchResponse(query_bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1), results=[])
        assert response.query_id is None
        qid = uuid.uuid4()
        assert SearchResponse(query_bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1), results=[], query_id=qid).query_id == qid