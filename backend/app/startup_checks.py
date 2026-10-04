"""
Проверки согласованности компонентов системы при старте приложения.

Зачем: система собрана из независимо подготовленных частей (ONNX-модели,
экспортированные в Colab/локально; схема БД; настройки в .env), и
рассогласование между ними иначе проявляется поздно и непонятно - как
503 на первом же поиске. Здесь оно ловится при запуске, с указанием,
что именно не так и как это исправить.

Два класса проблем обрабатываются по-разному:

  * НЕИСПРАВИМЫЕ без вмешательства человека (нет файла модели, размерность
    эмбеддинга не совпадает со схемой БД) - приложение не стартует,
    поднимается StartupCheckError. Работать с такой конфигурацией
    бессмысленно, а запуск "как будто всё нормально" только откладывает
    ошибку до первого запроса пользователя.

  * СОСТОЯНИЕ БД (недоступна, нет таблицы/расширения/индекса) - приложение
    НЕ падает: ТЗ п.1.1.2.4 требует при недоступности подсистемы хранения
    информировать пользователя о временной невозможности запроса, а не
    прекращать работу. Проблемы описываются в логе и в /health.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .config import DEFAULT_ADMIN_API_KEY

logger = logging.getLogger(__name__)


class StartupCheckError(RuntimeError):
    """Конфигурация системы несогласованна - запуск невозможен."""


# ---------------------------------------------------------------------------
# Файлы моделей
# ---------------------------------------------------------------------------

def check_model_files(settings) -> None:
    """Проверяет, что файлы моделей детектора и ReID существуют."""
    missing: List[str] = []

    if not Path(settings.detector_model_path).is_file():
        missing.append(
            f"  - детектор: {settings.detector_model_path}\n"
            f"    получить: python ml/detection/export_detector.py "
            f"--checkpoint <best.pt> --output ml/detection/exported\n"
            f"    либо указать другой путь в .env: DETECTOR_MODEL_PATH"
        )

    if not Path(settings.reid_model_path).is_file():
        missing.append(
            f"  - ReID-экстрактор: {settings.reid_model_path}\n"
            f"    получить: python ml/reid/export_reid.py "
            f"--checkpoint <best_checkpoint.pt> --output ml/reid/exported\n"
            f"    либо указать другой путь в .env: REID_MODEL_PATH"
        )

    if missing:
        raise StartupCheckError(
            "Не найдены файлы моделей, без которых система не может работать:\n"
            + "\n".join(missing)
        )


# ---------------------------------------------------------------------------
# Размерность эмбеддинга
# ---------------------------------------------------------------------------

def check_embedding_dimension(embedder, image_service, expected_dim: int) -> int:
    """
    Прогоняет через ReID-модель тестовое изображение и проверяет, что
    размерность эмбеддинга совпадает с колонкой VECTOR(expected_dim) в БД
    (backend/app/db/models.py: EMBEDDING_DIM, ТЗ п.1.1.2.3: "формат
    представления векторного описания должен быть единообразным для
    всех компонентов системы").

    Без этой проверки модель, экспортированная с другим --embedding-dim,
    обнаруживалась бы только на первом сохранении/поиске - как ошибка
    pgvector о несовпадении размерностей.

    Returns:
        Фактическую размерность эмбеддинга.
    """
    height, width = image_service.image_size
    probe = Image.new("RGB", (width, height), (128, 128, 128))

    try:
        embedding = embedder.embed_as_list(image_service.preprocess(probe))
    except Exception as exc:
        raise StartupCheckError(
            f"ReID-модель не смогла обработать тестовое изображение {height}x{width}.\n"
            f"Вероятная причина: размер входа в настройках (REID_IMAGE_HEIGHT/"
            f"REID_IMAGE_WIDTH) не совпадает с --imgsz, с которым модель "
            f"экспортировалась скриптом export_reid.py.\n"
            f"Исходная ошибка: {exc}"
        ) from exc

    actual_dim = len(embedding)
    if actual_dim != expected_dim:
        raise StartupCheckError(
            f"Размерность эмбеддинга ReID-модели ({actual_dim}) не совпадает "
            f"с размерностью колонки в БД ({expected_dim}).\n"
            f"Переэкспортируйте модель с подходящим --embedding-dim либо "
            f"пересоздайте схему БД под новую размерность "
            f"(EMBEDDING_DIM в backend/app/db/models.py и таблица observations)."
        )

    return actual_dim


# ---------------------------------------------------------------------------
# Состояние БД
# ---------------------------------------------------------------------------

@dataclass
class DatabaseStatus:
    """
    Состояние подсистемы хранения. Поля схемы равны None, если БД
    недоступна - тогда про схему ничего неизвестно (это не то же
    самое, что "расширения нет").
    """

    reachable: bool
    vector_extension: Optional[bool] = None
    table_exists: Optional[bool] = None
    hnsw_index: Optional[bool] = None
    history_table: Optional[bool] = None  # таблица журнала поиска search_queries
    error: Optional[str] = field(default=None, repr=False)  # только для лога, наружу не отдаётся

    @property
    def ready(self) -> bool:
        """БД готова обслуживать запросы: доступна, есть расширение и таблица."""
        return bool(self.reachable and self.vector_extension and self.table_exists)

    def problems(self) -> List[str]:
        """Что мешает работе, с подсказкой, как исправить. Без технических деталей подключения."""
        if not self.reachable:
            return ["База данных недоступна. Проверьте, что она запущена (docker compose up -d db) "
                    "и что DATABASE_URL в .env указывает на неё."]
        result = []
        if not self.vector_extension:
            result.append("В базе не включено расширение pgvector "
                          "(CREATE EXTENSION vector; при docker compose это делает db/init.sql "
                          "ТОЛЬКО при первой инициализации тома).")
        if not self.table_exists:
            result.append("Таблица observations не создана: выполните python db/create_table.py")
        return result

    def warnings(self) -> List[str]:
        """Не блокирует работу поиска, но что-то работает хуже или не работает."""
        result = []
        if self.ready and not self.hnsw_index:
            result.append("Нет HNSW-индекса на observations.embedding: поиск работает полным перебором "
                          "и не уложится в требование ТЗ (30 мс на 100 000 записей). "
                          "Выполните python db/create_indexes.py")
        # is False, а не "not": None означает "неизвестно" (БД недоступна), это не отсутствие.
        if self.ready and self.history_table is False:
            result.append("Нет таблицы журнала поиска search_queries: поиск работает, но история "
                          "поиска не сохраняется. Выполните python db/create_table.py "
                          "(существующие данные не затрагиваются)")
        return result


async def inspect_database(engine: AsyncEngine, timeout_seconds: float = 5.0) -> DatabaseStatus:
    """
    Проверяет доступность БД и готовность схемы. Никогда не бросает
    исключений: любая проблема отражается в возвращаемом статусе
    (недоступная БД не должна ни ронять приложение при старте, ни
    подвешивать /health - отсюда таймаут).
    """

    async def _inspect() -> DatabaseStatus:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            extension = (await conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )).scalar()
            table = (await conn.execute(
                text("SELECT to_regclass('public.observations') IS NOT NULL")
            )).scalar()
            history = (await conn.execute(
                text("SELECT to_regclass('public.search_queries') IS NOT NULL")
            )).scalar()
            hnsw = (await conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_indexes "
                     "WHERE tablename = 'observations' AND indexdef ILIKE '%USING hnsw%')")
            )).scalar()
            return DatabaseStatus(
                reachable=True,
                vector_extension=bool(extension),
                table_exists=bool(table),
                hnsw_index=bool(hnsw),
                history_table=bool(history),
            )

    try:
        return await asyncio.wait_for(_inspect(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return DatabaseStatus(reachable=False, error=f"таймаут {timeout_seconds} с")
    except Exception as exc:
        return DatabaseStatus(reachable=False, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Токен администратора
# ---------------------------------------------------------------------------

def admin_key_warnings(admin_api_key: str) -> List[str]:
    """
    Предупреждения о небезопасном или неработоспособном ADMIN_API_KEY.
    Не блокируют запуск: система работает и без админ-функций.
    """
    warnings = []
    if admin_api_key == DEFAULT_ADMIN_API_KEY:
        warnings.append(
            "ADMIN_API_KEY не задан: используется значение по умолчанию, известное "
            "каждому, кто видел код. Придумайте свой токен и впишите его в .env."
        )
    if not admin_api_key.isascii():
        warnings.append(
            "ADMIN_API_KEY содержит не-ASCII символы (например, кириллицу): браузер не "
            "сможет передать такой токен в HTTP-заголовке, админ-действия будут "
            "недоступны. Используйте латиницу, цифры и знаки."
        )
    return warnings