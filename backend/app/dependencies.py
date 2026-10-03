"""
Dependency injection функции для api/routes.py.

Все сервисы (детектор, поиск) и фабрика сессий БД создаются ОДИН РАЗ
при старте приложения (см. main.py: lifespan) и хранятся в
request.app.state - функции здесь только достают их оттуда для
конкретного запроса, не создают заново. Для БД это означает ОДНУ
async-сессию на запрос (стандартная практика FastAPI+SQLAlchemy),
закрывающуюся автоматически по завершении обработки запроса.

Вынесено в отдельный модуль (а не определено прямо в main.py), чтобы
routes.py мог импортировать эти функции, не импортируя сам main.py -
иначе возник бы циклический импорт (main.py в будущем будет
подключать router из routes.py через app.include_router).
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from .services.detector import DetectorService
from .services.search_service import SearchService
from .storage import FileStorage


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Одна сессия БД на запрос, закрывается автоматически после обработки."""
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


def get_detector_service(request: Request) -> DetectorService:
    return request.app.state.detector_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_file_storage(request: Request) -> FileStorage:
    return request.app.state.file_storage