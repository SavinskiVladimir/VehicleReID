"""
Сервис поиска ближайших соседей по признаковому сходству (функция 6 из
Таблицы 1.3 ТЗ), добавления наблюдений в БАЗУ НАБЛЮДЕНИЙ (функция 8) и
ведения ЖУРНАЛА ПОИСКА с повторным поиском (функция 9).

Две сущности намеренно разделены (см. db/models.py): база наблюдений -
то, В ЧЁМ ищем; журнал поиска - то, ЧТО искали.

Оркестрирует полную цепочку: кроп ТС -> предобработка (ImageService) ->
эмбеддинг (ReIDEmbedderService) -> запрос к БД (repository.py) ->
Pydantic-схема ответа (functions 6-9). Согласно ТЗ (п.1.3.2.1), routes.py
должен зависеть только от этого сервиса, а не от repository.py или
ML-сервисов напрямую - SearchService и есть та единая точка входа,
инкапсулирующая весь пайплайн для API-слоя.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Sequence, Tuple

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import repository
from ..schemas.vehicle import (
    BoundingBox,
    ObservationListResponse,
    ObservationOut,
    SearchHistoryItem,
    SearchHistoryResponse,
    SearchResponse,
    SearchResultItem,
)
from .image_service import ImageService
from .reid_embedder import ReIDEmbedderService


def _clamp_similarity(value: float) -> float:
    """
    similarity = 1 - косинусное расстояние лежит в [-1, 1]: у противоположных
    по направлению векторов оно отрицательное (и теоретически возможен выход
    за 1.0 из-за погрешности float32). API и интерфейс работают с долей
    [0, 1] ("процент сходства"), а схема ответа это проверяет - без
    ограничения один неудачный кандидат превращал весь поиск в ошибку 503.
    Отрицательное сходство для пользователя равносильно "совсем не похоже", то
    есть 0.
    """
    return min(1.0, max(0.0, float(value)))


class SearchService:
    """
    Args:
        image_service: сервис предобработки кропа (функция 4 ТЗ).
        embedder: сервис формирования эмбеддинга (функция 5 ТЗ).
    """

    def __init__(self, image_service: ImageService, embedder: ReIDEmbedderService):
        self.image_service = image_service
        self.embedder = embedder

    def compute_embedding(self, crop: Image.Image) -> list:
        """
        Прогоняет кроп ТС через предобработку и ReID-модель (функции 4-5
        ТЗ). Вынесено в отдельный метод, чтобы можно было переиспользовать
        и в search(), и в add_observation() - в обоих случаях нужен один
        и тот же эмбеддинг, посчитанный одинаковым образом.
        """
        preprocessed = self.image_service.preprocess(crop)
        return self.embedder.embed_as_list(preprocessed)

    async def search_by_embedding(
        self,
        session: AsyncSession,
        embedding: Sequence[float],
        query_bbox: Tuple[int, int, int, int],
        top_k: int = 10,
    ) -> SearchResponse:
        """
        Поиск по УЖЕ готовому вектору (функции 6-7 ТЗ). Отдельный метод, потому
        что он нужен и основному поиску (после compute_embedding), и
        повторному поиску из журнала - там вектор берётся из БД и модели
        не запускаются вообще.
        """
        raw_results = await repository.search_similar(session, embedding, top_k=top_k)

        results = [
            SearchResultItem(
                observation=ObservationOut.model_validate(obs),
                similarity=_clamp_similarity(similarity),
            )
            for obs, similarity in raw_results
        ]

        return SearchResponse(
            query_bbox=BoundingBox(
                x1=query_bbox[0], y1=query_bbox[1], x2=query_bbox[2], y2=query_bbox[3]
            ),
            results=results,
        )

    async def search(
        self,
        session: AsyncSession,
        crop: Image.Image,
        query_bbox: Tuple[int, int, int, int],
        top_k: int = 10,
    ) -> SearchResponse:
        """
        Полный сценарий поиска (функции 4-7 ТЗ): кроп -> эмбеддинг -> поиск в БД
        -> ранжированный ответ с превью и метрикой сходства.

        Args:
            crop: уже кадрированное изображение ТС (результат
                DetectorService.crop_with_padding - сама детекция
                находится вне ответственности этого сервиса).
            query_bbox: координаты бокса на ИСХОДНОМ изображении query
                (для включения в ответ, чтобы клиент знал, какой именно
                объект искали, при нескольких ТС на кадре).
            top_k: сколько кандидатов вернуть (N в терминах функции 7 ТЗ).
        """
        embedding = self.compute_embedding(crop)
        return await self.search_by_embedding(session, embedding, query_bbox, top_k)

    async def add_observation(
        self,
        session: AsyncSession,
        image_path: str,
        crop_path: str,
        crop: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> ObservationOut:
        """
        Добавление нового наблюдения в базу (функция 8 ТЗ) - используется
        как в основном сценарии поиска (после показа результатов
        пользователю решено сохранить query как новое наблюдение), так
        и в режиме пакетного пополнения базы без поиска (ТЗ п.1.3.2.1,
        "дополнительный режим - пакетное добавление изображений в базу
        наблюдений без выполнения поиска").
        """
        embedding = self.compute_embedding(crop)
        observation = await repository.add_observation(
            session, image_path=image_path, crop_path=crop_path,
            bbox=bbox, embedding=embedding,
        )
        return ObservationOut.model_validate(observation)

    async def get_observation(self, session: AsyncSession, observation_id) -> "ObservationOut | None":
        """Одно наблюдение по id (None, если не найдено) - нужно для отдачи его изображений."""
        observation = await repository.get_observation(session, observation_id)
        if observation is None:
            return None
        return ObservationOut.model_validate(observation)

    async def list_observations(
        self, session: AsyncSession, limit: int = 50, offset: int = 0
    ) -> ObservationListResponse:
        """Список записей БАЗЫ НАБЛЮДЕНИЙ с пагинацией (не журнал поиска)."""
        items, total = await repository.list_observations(session, limit=limit, offset=offset)
        return ObservationListResponse(
            items=[ObservationOut.model_validate(obs) for obs in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def delete_observation(self, session: AsyncSession, observation_id) -> bool:
        """Удаление наблюдения (требование ТЗ п.1.3.2.3)."""
        return await repository.delete_observation(session, observation_id)

    # ------------------------------------------------------------------
    # Журнал поиска (функция 9 ТЗ) - ОТДЕЛЬНО от базы наблюдений
    # ------------------------------------------------------------------

    async def record_search(
        self,
        session: AsyncSession,
        crop_path: str,
        bbox: Tuple[int, int, int, int],
        embedding: Sequence[float],
        top_k: int,
        response: SearchResponse,
    ) -> SearchHistoryItem:
        """Записывает выполненный поиск в журнал вместе с его итогом."""
        similarities = [item.similarity for item in response.results]
        query = await repository.add_search_query(
            session,
            crop_path=crop_path,
            bbox=bbox,
            embedding=embedding,
            top_k=top_k,
            result_count=len(similarities),
            top_similarity=max(similarities) if similarities else None,
        )
        return SearchHistoryItem.model_validate(query)

    async def list_search_history(
        self, session: AsyncSession, limit: int = 20, offset: int = 0
    ) -> SearchHistoryResponse:
        items, total = await repository.list_search_queries(session, limit=limit, offset=offset)
        return SearchHistoryResponse(
            items=[SearchHistoryItem.model_validate(q) for q in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_search_crop_path(self, session: AsyncSession, query_id: uuid.UUID) -> Optional[str]:
        """Путь к кропу запроса (для отдачи превью) либо None, если записи нет."""
        query = await repository.get_search_query(session, query_id)
        return query.crop_path if query is not None else None

    async def rerun_search(
        self, session: AsyncSession, query_id: uuid.UUID, top_k: Optional[int] = None
    ) -> Optional[SearchResponse]:
        """
        Повторный поиск из журнала (функция 9 ТЗ: "с возможностью повторного
        поиска"): по СОХРАНЁННОМУ вектору, с ТЕКУЩИМ содержимым базы наблюдений.
        Ни исходное изображение, ни модели не нужны. Результат может
        отличаться от исходного - в базе могли появиться или пропасть записи.
        В журнал повтор НЕ записывается (это тот же запрос, не новый).

        Returns:
            None, если записи с таким id нет в журнале.
        """
        query = await repository.get_search_query(session, query_id)
        if query is None:
            return None

        response = await self.search_by_embedding(
            session,
            embedding=query.embedding,
            query_bbox=(query.bbox_x1, query.bbox_y1, query.bbox_x2, query.bbox_y2),
            top_k=top_k if top_k is not None else query.top_k,
        )
        return response.model_copy(update={"query_id": query.id})

    async def delete_search_query(self, session: AsyncSession, query_id: uuid.UUID) -> Optional[str]:
        """Удаляет запись журнала; возвращает путь её кропа (для удаления файла) или None."""
        return await repository.delete_search_query(session, query_id)

    async def clear_search_history(self, session: AsyncSession) -> List[str]:
        """Очищает журнал целиком; возвращает пути кропов (для удаления файлов)."""
        return await repository.clear_search_queries(session)

    async def prune_search_history(self, session: AsyncSession, keep: int) -> List[str]:
        """Оставляет `keep` последних записей; возвращает пути кропов удалённых."""
        return await repository.prune_search_queries(session, keep)