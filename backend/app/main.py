"""
Точка входа FastAPI-приложения.

Согласно ТЗ (п.1.3.2.1), HTTP-API-слой (этот файл + api/routes.py,
который появится следующим модулем) зависит только от сервисного слоя
(services/*.py) и слоя доступа к данным (db/repository.py), не от
деталей реализации моделей/СУБД напрямую.

ML-модели (детектор, ReID-экстрактор) и движок БД создаются ОДИН РАЗ
при старте приложения через lifespan - не на каждый запрос, иначе
задержка инференса складывалась бы ещё и со временем загрузки весов
ONNX-моделей, многократно превышая бюджет ТЗ на запрос (700 мс CPU /
300 мс GPU, п.1.3.2.2).

Запуск (из корня проекта):
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from .config import get_settings
from .db.models import EMBEDDING_DIM
from .services.detector import DetectorService
from .services.image_service import ImageService
from .services.reid_embedder import ReIDEmbedderService
from .services.search_service import SearchService
from .startup_checks import (
    admin_key_warnings,
    check_embedding_dimension,
    check_model_files,
    inspect_database,
)
from .storage import FileStorage


logger = logging.getLogger(__name__)


def _configure_app_logging() -> None:
    """
    uvicorn настраивает только СВОИ логгеры, поэтому сообщения приложения
    (стартовые проверки, ошибки в routes.py) без настройки выводятся
    запасным обработчиком Python: без уровня и только от WARNING и выше -
    "БД готова" (INFO) не видно вовсе. Подключаем обработчик в формате
    uvicorn ("INFO:     ...") к логгеру пакета приложения.

    Родительский логгер берём из __name__ ("app" при запуске из backend/,
    "backend.app" из корня проекта) - покрывает все модули пакета. Если у
    корневого логгера уже есть обработчики (pytest, чужая конфигурация) -
    ничего не добавляем, чтобы не дублировать вывод.
    """
    package_name = __name__.rpartition(".")[0]
    if not package_name or logging.getLogger().handlers:
        return
    package_logger = logging.getLogger(package_name)
    if package_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Неисправимые без человека проблемы (нет файла модели) - падаем сразу
    # с понятным сообщением, ДО создания каких-либо ресурсов.
    check_model_files(settings)

    # --- Слой хранения ---
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # --- ML-сервисы - загружаются ОДИН РАЗ здесь, переиспользуются на
    # все последующие запросы через dependencies.py ---
    detector_service = DetectorService(
        model_path=settings.detector_model_path,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        device=settings.device,
        padding_fraction=settings.padding_fraction,
    )
    image_service = ImageService(image_size=settings.reid_image_size)
    embedder_service = ReIDEmbedderService(
        model_path=settings.reid_model_path, device=settings.device
    )
    search_service = SearchService(image_service=image_service, embedder=embedder_service)

    # Размерность эмбеддинга модели должна совпасть со схемой БД -
    # иначе это вскрылось бы только на первом поиске как 503.
    embedding_dim = check_embedding_dimension(embedder_service, image_service, EMBEDDING_DIM)

    file_storage = FileStorage(
        temp_dir=settings.temp_upload_dir,
        permanent_dir=settings.permanent_storage_dir,
        max_upload_size_bytes=settings.max_upload_size_bytes,
        temp_ttl_seconds=settings.temp_upload_ttl_seconds,
    )

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.detector_service = detector_service
    app.state.search_service = search_service
    app.state.file_storage = file_storage
    app.state.embedding_dim = embedding_dim

    # Состояние БД НЕ валит приложение (ТЗ п.1.1.2.4: при недоступности
    # хранилища - сообщить пользователю, а не прекратить работу): только
    # лог с причиной и подсказкой. Актуальное состояние - в /health.
    db_status = await inspect_database(engine)
    if db_status.error:
        logger.warning("БД недоступна при старте: %s", db_status.error)
    for problem in db_status.problems():
        logger.warning("Проблема с БД: %s", problem)
    for warning in db_status.warnings():
        logger.warning("%s", warning)
    for warning in admin_key_warnings(settings.admin_api_key):
        logger.warning("%s", warning)
    if db_status.ready and not db_status.warnings():
        logger.info("БД готова: расширение vector, таблица observations и HNSW-индекс на месте.")

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    _configure_app_logging()
    settings = get_settings()

    app = FastAPI(
        title="Vehicle Re-ID API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ВАЖНО: это НЕ terminates TLS само по себе - HTTPSRedirectMiddleware
    # только перенаправляет http:// запросы на https://, предполагая,
    # что реальное шифрование уже обеспечено ПЕРЕД приложением (reverse-
    # proxy/балансировщик) либо самим uvicorn, запущенным с
    # --ssl-keyfile/--ssl-certfile. Выключено по умолчанию
    # (settings.enforce_https=False) - иначе локальная разработка без
    # сертификата была бы сломана перенаправлением в никуда.
    if settings.enforce_https:
        app.add_middleware(HTTPSRedirectMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_hsts_header(request: Request, call_next):
        response = await call_next(request)
        if settings.enforce_https:
            # Strict-Transport-Security - говорит браузеру ВСЕГДА
            # использовать HTTPS для этого origin в будущем, даже если
            # пользователь наберёт адрес без https:// вручную.
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

    @app.get("/health", tags=["service"])
    async def health_check(request: Request) -> dict:
        """
        Готовность сервиса. status="ok" только если БД доступна И схема
        готова (расширение vector + таблица observations). Отсутствие
        HNSW-индекса не делает сервис нерабочим (поиск идёт полным
        перебором) - оно попадает в warnings, статус остаётся "ok".

        Не роняет приложение при недоступной БД (ТЗ п.1.1.2.4) - сообщает
        "degraded". Технические детали ошибки подключения наружу не
        отдаются (только в лог при старте): /health публичный.
        schema.* равны null, если БД недоступна - про схему тогда
        ничего не известно.
        """
        db_status = await inspect_database(request.app.state.engine)

        return {
            "status": "ok" if db_status.ready else "degraded",
            "database": "ok" if db_status.reachable else "unavailable",
            "schema": {
                "vector_extension": db_status.vector_extension,
                "table": db_status.table_exists,
                "hnsw_index": db_status.hnsw_index,
            },
            "problems": db_status.problems(),
            "warnings": db_status.warnings(),
            "detector_loaded": request.app.state.detector_service is not None,
            "search_service_loaded": request.app.state.search_service is not None,
            "embedding_dim": request.app.state.embedding_dim,
        }

    from .api.routes import router as api_router
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()