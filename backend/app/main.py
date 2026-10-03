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
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from .config import get_settings
from .services.detector import DetectorService
from .services.image_service import ImageService
from .services.reid_embedder import ReIDEmbedderService
from .services.search_service import SearchService
from .storage import FileStorage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

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

    yield

    await engine.dispose()


def create_app() -> FastAPI:
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
        Проверка готовности сервиса: отвечает "ok" только если и БД
        доступна, и ML-сервисы успешно загружены при старте. Не роняет
        приложение при недоступности БД (ТЗ п.1.3.2.4: требование к
        надёжности) - просто сообщает "degraded", чтобы вызывающая
        сторона (мониторинг/балансировщик) могла на это отреагировать.
        """
        database_ok = True
        try:
            async with request.app.state.session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception:
            database_ok = False

        return {
            "status": "ok" if database_ok else "degraded",
            "database": "ok" if database_ok else "unavailable",
            "detector_loaded": request.app.state.detector_service is not None,
            "search_service_loaded": request.app.state.search_service is not None,
        }

    from .api.routes import router as api_router
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()