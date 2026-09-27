"""
Сервис поиска ближайших соседей по признаковому сходству (функция 6 из
Таблицы 1.3 ТЗ), а также вспомогательные операции добавления наблюдения
и просмотра истории (функции 8, 9).

Оркестрирует полную цепочку: кроп ТС -> предобработка (ImageService) ->
эмбеддинг (ReIDEmbedderService) -> запрос к БД (repository.py) ->
Pydantic-схема ответа (functions 6-9). Согласно ТЗ (п.1.3.2.1), routes.py
должен зависеть только от этого сервиса, а не от repository.py или
ML-сервисов напрямую - SearchService и есть та единая точка входа,
инкапсулирующая весь пайплайн для API-слоя.
"""

from __future__ import annotations

from typing import Tuple

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import repository
from ..schemas.vehicle import (
    BoundingBox,
    ObservationHistoryResponse,
    ObservationOut,
    SearchResponse,
    SearchResultItem,
)
from .image_service import ImageService
from .reid_embedder import ReIDEmbedderService


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

    async def search(
        self,
        session: AsyncSession,
        crop: Image.Image,
        query_bbox: Tuple[int, int, int, int],
        top_k: int = 10,
    ) -> SearchResponse:
        """
        Полный сценарий поиска (функции 4-7 ТЗ): кроп -> эмбеддинг ->
        поиск в БД -> ранжированный ответ с превью и метрикой сходства.

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
        raw_results = await repository.search_similar(session, embedding, top_k=top_k)

        results = [
            SearchResultItem(
                observation=ObservationOut.model_validate(obs),
                similarity=similarity,
            )
            for obs, similarity in raw_results
        ]

        return SearchResponse(
            query_bbox=BoundingBox(
                x1=query_bbox[0], y1=query_bbox[1], x2=query_bbox[2], y2=query_bbox[3]
            ),
            results=results,
        )

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

    async def get_history(
        self, session: AsyncSession, limit: int = 50, offset: int = 0
    ) -> ObservationHistoryResponse:
        """Просмотр истории наблюдений (функция 9 ТЗ), с пагинацией."""
        items, total = await repository.list_observations(session, limit=limit, offset=offset)
        return ObservationHistoryResponse(
            items=[ObservationOut.model_validate(obs) for obs in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def delete_observation(self, session: AsyncSession, observation_id) -> bool:
        """Удаление наблюдения (требование ТЗ п.1.3.2.3)."""
        return await repository.delete_observation(session, observation_id)