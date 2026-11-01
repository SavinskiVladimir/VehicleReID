"""
Pydantic-схемы для API слоя.

Реализует контракт данных для функций 6-9 из Таблицы 1.3 ТЗ:
    6. Поиск ближайших соседей -> ранжированный список идентификаторов
       с метрикой сходства.
    7. Формирование и выдача результата -> N кандидатов с превью и
       значением метрики сходства пользователю.
    8. Добавление нового наблюдения в базу.
    9. Хранение и просмотр истории: журнал выполненных поисков с
       возможностью повторного поиска (отдельно от БАЗЫ НАБЛЮДЕНИЙ).

Эти схемы не зависят от конкретной СУБД/векторного индекса (pgvector) -
они описывают только внешний контракт API, согласно ТЗ п.1.3.2.1:
"взаимодействие с подсистемой хранения и поиска должно быть реализовано
через отдельный слой доступа к данным, инкапсулирующий детали
конкретной реализации векторного поиска".
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Координаты ограничивающего прямоугольника ТС на исходном изображении."""

    x1: int
    y1: int
    x2: int
    y2: int

    @field_validator("x2")
    @classmethod
    def _x2_after_x1(cls, v: int, info) -> int:
        x1 = info.data.get("x1")
        if x1 is not None and v <= x1:
            raise ValueError("x2 должен быть больше x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y2_after_y1(cls, v: int, info) -> int:
        y1 = info.data.get("y1")
        if y1 is not None and v <= y1:
            raise ValueError("y2 должен быть больше y1")
        return v

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class DetectionResult(BaseModel):
    """Одно обнаруженное ТС (функция 2 ТЗ) - результат POST /api/detect."""

    index: int = Field(..., description="Номер детекции в ответе - используется в SearchRequest.detection_index")
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    """Ответ POST /api/detect: upload_id для последующего /api/search + список найденных ТС."""

    upload_id: str = Field(..., description="Идентификатор временно сохранённого изображения")
    detections: List[DetectionResult]


class SearchRequest(BaseModel):
    """Тело запроса POST /api/search - ссылается на изображение из /api/detect по upload_id."""

    upload_id: str
    detection_index: int = Field(..., description="Индекс выбранной детекции из ответа /api/detect")
    top_k: int = Field(default=10, ge=1, le=100)


class AddObservationRequest(BaseModel):
    """Тело запроса POST /api/observations (функция 8 ТЗ, admin-доступ)."""

    upload_id: str
    detection_index: int


class ObservationCreate(BaseModel):
    """
    Данные, необходимые для добавления нового наблюдения в базу
    (функция 8 ТЗ). Само изображение передаётся отдельно как файл
    в multipart-запросе - здесь только метаданные, полученные уже
    ПОСЛЕ детекции и формирования эмбеддинга.
    """

    image_path: str = Field(..., description="Путь/ссылка на сохранённое исходное изображение")
    crop_path: str = Field(..., description="Путь/ссылка на кадрированное изображение ТС")
    bbox: BoundingBox
    embedding: List[float] = Field(..., description="Нормализованное векторное представление ТС")

    @field_validator("embedding")
    @classmethod
    def _embedding_correct_dim(cls, v: List[float]) -> List[float]:
        # Единообразие формата векторного описания для всех компонентов
        # системы (требование ТЗ п.1.3.2.3) - фиксируем размерность здесь,
        # а не полагаемся на то, что вызывающий код передаст правильную.
        expected_dim = 512
        if len(v) != expected_dim:
            raise ValueError(
                f"Эмбеддинг должен иметь размерность {expected_dim}, получено {len(v)}"
            )
        return v


class ObservationOut(BaseModel):
    """Одно наблюдение из БАЗЫ НАБЛЮДЕНИЙ (используется в её списке и в результатах поиска)."""

    id: uuid.UUID
    image_path: str
    crop_path: str
    bbox: BoundingBox
    created_at: datetime

    model_config = {"from_attributes": True}  # позволяет строить из ORM-объекта напрямую


class SearchResultItem(BaseModel):
    """
    Один кандидат в результатах поиска (функция 7 ТЗ): наблюдение +
    значение метрики сходства с query.
    """

    observation: ObservationOut
    similarity: float = Field(..., ge=0.0, le=1.0, description="Косинусное сходство, 1.0 = идентично")


class SearchResponse(BaseModel):
    """Ответ на поисковый запрос: ранжированный список из N кандидатов."""

    query_bbox: BoundingBox
    results: List[SearchResultItem]
    # id записи в журнале поиска (None, если запись не удалось сделать -
    # поиск при этом всё равно выполнен: журнал вторичен по отношению к нему).
    query_id: Optional[uuid.UUID] = None

    @property
    def top_k(self) -> int:
        return len(self.results)


class ObservationListResponse(BaseModel):
    """Список записей БАЗЫ НАБЛЮДЕНИЙ с пагинацией."""

    items: List[ObservationOut]
    total: int
    limit: int
    offset: int


class SearchHistoryItem(BaseModel):
    """Одна запись ЖУРНАЛА ПОИСКА (функция 9 ТЗ): что искали, когда и с каким итогом."""

    id: uuid.UUID
    created_at: datetime
    bbox: BoundingBox
    top_k: int
    result_count: int
    top_similarity: Optional[float] = Field(
        default=None, description="Лучшее сходство среди найденных на момент поиска (None, если ничего не найдено)"
    )

    model_config = {"from_attributes": True}


class SearchHistoryResponse(BaseModel):
    """Страница журнала поиска."""

    items: List[SearchHistoryItem]
    total: int
    limit: int
    offset: int

    
class FeedbackSet(BaseModel):
    query_id: uuid.UUID
    candidate_id: uuid.UUID
    label: Literal["match", "no_match"]


class FeedbackOut(BaseModel):
    query_id: Optional[uuid.UUID] = None
    candidate_id: uuid.UUID
    label: str
    operator_id: str
    created_at: datetime
    updated_at: datetime


class FeedbackListResponse(BaseModel):
    items: List[FeedbackOut]