"""
HTTP-эндпоинты API. Реализует функции 1-9 из Таблицы 1.3 ТЗ.

Поток запроса организован в ДВА шага (аналогично --preview-only/
--select в scripts/demo_search.py):

    1. POST /api/detect  - загрузка изображения (функция 1), детекция
       всех ТС на кадре (функция 2). Изображение сохраняется во
       временное хранилище, клиенту возвращается upload_id - дальше не
       нужно повторно передавать файл целиком.
    2. POST /api/search   - по upload_id + индексу ВЫБРАННОЙ детекции
       (требование ТЗ: "при наличии нескольких ТС система должна
       предоставлять пользователю возможность выбора") - кадрирование
       (функция 3), предобработка (функция 4), эмбеддинг (функция 5),
       поиск (функция 6), ранжированный ответ (функция 7).

Детекция в /api/search запускается ПОВТОРНО (а не переиспользует
результат /api/detect) - сознательное упрощение: ONNX-инференс в
eval-режиме детерминирован (нет dropout/рандома), поэтому результат
гарантированно совпадёт с тем, что видел клиент на шаге /detect, а
хранить список детекций отдельным sidecar-файлом ради экономии одного
короткого прогона детектора (~10 мс по ранее измеренным бенчмаркам)
не оправдано дополнительной сложностью.

Требования к надёжности (ТЗ п.1.1.2.4) реализованы единообразно: ошибки
ML-обработки и недоступность БД возвращают понятный JSON с кодом
ошибки, а не падение процесса или сырой traceback.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import require_admin
from ..dependencies import (
    get_app_settings,
    get_db_session,
    get_detector_service,
    get_file_storage,
    get_search_service,
)
from ..schemas.vehicle import (
    AddObservationRequest,
    BoundingBox,
    DetectionResult,
    DetectResponse,
    ObservationListResponse,
    ObservationOut,
    SearchHistoryResponse,
    SearchRequest,
    SearchResponse,
)
from ..services.detector import DetectorService
from ..services.search_service import SearchService
from ..storage import FileStorage, UploadNotFoundError, UploadValidationError

router = APIRouter(tags=["vehicles"])
logger = logging.getLogger(__name__)


_SERVABLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_file_response(path_str: str) -> FileResponse:
    """
    Отдаёт файл изображения по пути, сохранённому в БД. Путь берётся из
    БД, а не от клиента, но на всякий случай отдаём ТОЛЬКО файлы с
    расширениями изображений - чтобы запись с произвольным путём в БД
    не превращала этот эндпоинт в чтение любых файлов сервера.
    """
    path = Path(path_str)
    if path.suffix.lower() not in _SERVABLE_EXTENSIONS or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Файл изображения не найден на сервере.",
        )
    return FileResponse(path)


def _bbox_to_schema(bbox: tuple) -> BoundingBox:
    x1, y1, x2, y2 = bbox
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _get_selected_detection(detector: DetectorService, image, detection_index: int):
    """
    Общая логика для /search и /observations: повторная детекция +
    валидация индекса выбранной машины. Поднимает HTTPException с
    понятным сообщением при некорректном индексе, а не падает с
    IndexError.
    """
    detections = detector.detect(image)
    if not detections:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Детектор не нашёл ни одного ТС на изображении.",
        )
    if not (0 <= detection_index < len(detections)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"detection_index={detection_index} вне диапазона "
                   f"(найдено {len(detections)} ТС, допустимо 0..{len(detections) - 1}).",
        )
    return detections[detection_index]


@router.post("/detect", response_model=DetectResponse)
async def detect_vehicles(
    file: UploadFile = File(...),
    detector: DetectorService = Depends(get_detector_service),
    storage: FileStorage = Depends(get_file_storage),
) -> DetectResponse:
    """Функции 1-2 ТЗ: приём изображения + детекция ТС."""
    try:
        image_bytes = await storage.validate_and_read(file)
    except UploadValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    upload_id = storage.save_temp(image_bytes)

    try:
        image = storage.load_temp(upload_id)
        detections = detector.detect(image)
    except Exception:
        # Полный traceback - в лог сервера (для диагностики), пользователю
        # - только общее сообщение (ТЗ п.1.1.2.4: "корректное сообщение об
        # ошибке без прекращения работы серверной части").
        logger.exception("Ошибка на этапе детекции (upload_id=%s)", upload_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось обработать изображение на этапе детекции.",
        )

    return DetectResponse(
        upload_id=upload_id,
        detections=[
            DetectionResult(index=i, bbox=_bbox_to_schema(d.bbox), confidence=d.confidence)
            for i, d in enumerate(detections)
        ],
    )


async def _record_search_best_effort(
    search_service, storage, session, settings, crop, bbox, embedding, top_k, response
):
    """
    Записывает выполненный поиск в журнал. ЖУРНАЛ ВТОРИЧЕН: пользователь уже
    получил результат поиска, и сбой записи в журнал (нет таблицы, БД
    моргнула, нет места на диске) не должен превращать удачный поиск в ошибку.
    Сбой логируется, клиент получает ответ без query_id.

    Returns:
        id записи журнала либо None, если записать не удалось.
    """
    saved_path = None
    try:
        _, saved_path = storage.save_permanent(crop, subdir="queries")
        item = await search_service.record_search(
            session, saved_path, bbox, embedding, top_k, response
        )
    except Exception:
        logger.exception("Не удалось записать поиск в журнал (сам поиск при этом выполнен)")
        if saved_path:
            storage.delete_permanent(saved_path)  # не оставлять файл без записи
        try:
            await session.rollback()
        except Exception:
            pass
        return None

    # Ограничение размера журнала. Сбой здесь не отменяет уже сделанную запись.
    try:
        for old_path in await search_service.prune_search_history(
            session, settings.search_history_max_items
        ):
            storage.delete_permanent(old_path)
    except Exception:
        logger.exception("Не удалось обрезать журнал поиска до лимита")

    return item.id


@router.post("/search", response_model=SearchResponse)
async def search_vehicle(
    body: SearchRequest,
    detector: DetectorService = Depends(get_detector_service),
    search_service: SearchService = Depends(get_search_service),
    storage: FileStorage = Depends(get_file_storage),
    settings=Depends(get_app_settings),
    session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    """
    Функции 3-7 ТЗ: кроп выбранного ТС -> эмбеддинг -> поиск -> ответ. Выполненный
    поиск записывается в журнал (функция 9) - автоматически и без влияния на
    сам результат.
    """
    try:
        image = storage.load_temp(body.upload_id)
    except UploadNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    chosen = _get_selected_detection(detector, image, body.detection_index)
    crop = detector.crop_with_padding(image, chosen.bbox)

    try:
        embedding = search_service.compute_embedding(crop)
        response = await search_service.search_by_embedding(
            session, embedding, chosen.bbox, body.top_k
        )
    except HTTPException:
        raise
    except Exception:
        # ТЗ п.1.1.2.4: "при недоступности подсистемы хранения и поиска
        # система должна информировать пользователя о временной
        # невозможности выполнения запроса" - 503, а не 500/крах.
        logger.exception("Ошибка на этапе поиска (upload_id=%s)", body.upload_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис поиска временно недоступен. Попробуйте позже.",
        )

    query_id = await _record_search_best_effort(
        search_service, storage, session, settings, crop,
        chosen.bbox, embedding, body.top_k, response,
    )
    return response.model_copy(update={"query_id": query_id})


@router.post(
    "/observations",
    response_model=ObservationOut,
    dependencies=[Depends(require_admin)],
)
async def add_observation(
    body: AddObservationRequest,
    detector: DetectorService = Depends(get_detector_service),
    search_service: SearchService = Depends(get_search_service),
    storage: FileStorage = Depends(get_file_storage),
    session: AsyncSession = Depends(get_db_session),
) -> ObservationOut:
    """
    Функция 8 ТЗ: добавление нового наблюдения в базу.

    Административная функция (ТЗ п.1.1.2.4, требования по
    безопасности) - требует заголовок X-Admin-Token, см. auth.py.
    """
    try:
        image = storage.load_temp(body.upload_id)
    except UploadNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    chosen = _get_selected_detection(detector, image, body.detection_index)
    crop = detector.crop_with_padding(image, chosen.bbox)

    _, image_path = storage.save_permanent(image, subdir="images")
    _, crop_path = storage.save_permanent(crop, subdir="crops")

    try:
        result = await search_service.add_observation(
            session,
            image_path=image_path,
            crop_path=crop_path,
            crop=crop,
            bbox=chosen.bbox,
        )
    except Exception:
        logger.exception("Ошибка при сохранении наблюдения (upload_id=%s)", body.upload_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Не удалось сохранить наблюдение - хранилище временно недоступно.",
        )

    storage.delete_temp(body.upload_id)
    return result


@router.get("/observations", response_model=ObservationListResponse)
async def list_observations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> ObservationListResponse:
    """Список записей БАЗЫ НАБЛЮДЕНИЙ (то, в чём ищем) с пагинацией. Журнал поиска - /search-history."""
    try:
        return await search_service.list_observations(session, limit=limit, offset=offset)
    except Exception:
        logger.exception("Ошибка при получении списка наблюдений")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="База наблюдений временно недоступна.",
        )


@router.delete("/observations/{observation_id}", dependencies=[Depends(require_admin)])
async def delete_observation(
    observation_id: uuid.UUID,
    search_service: SearchService = Depends(get_search_service),
    storage: FileStorage = Depends(get_file_storage),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Удаление наблюдения (ТЗ п.1.1.2.3: "система хранения должна
    обеспечивать возможность... удаления записей"). Вместе с записью удаляются её файлы
    в хранилище (кроп и, если на него больше никто не ссылается, полный кадр). Файлы вне
    хранилища (записи демо-скрипта указывают на файлы датасета) не затрагиваются.
    Административная функция - требует X-Admin-Token.
    """
    try:
        unused_files = await search_service.delete_observation(session, observation_id)
    except Exception:
        logger.exception("Ошибка при удалении наблюдения (id=%s)", observation_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис хранения временно недоступен.",
        )

    if unused_files is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Наблюдение {observation_id} не найдено.",
        )

    for path in unused_files:
        storage.delete_permanent(path)
    return {"deleted": True, "id": str(observation_id)}


async def _load_observation_or_404(search_service, session, observation_id):
    try:
        observation = await search_service.get_observation(session, observation_id)
    except Exception:
        logger.exception("Ошибка при чтении наблюдения (id=%s)", observation_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис хранения временно недоступен.",
        )
    if observation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Наблюдение {observation_id} не найдено.",
        )
    return observation


@router.get("/observations/{observation_id}/crop")
async def get_observation_crop(
    observation_id: uuid.UUID,
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> FileResponse:
    """Кроп ТС наблюдения (для превью в результатах поиска и в списке базы)."""
    observation = await _load_observation_or_404(search_service, session, observation_id)
    return _image_file_response(observation.crop_path)


@router.get("/observations/{observation_id}/image")
async def get_observation_image(
    observation_id: uuid.UUID,
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> FileResponse:
    """Исходный кадр наблюдения целиком."""
    observation = await _load_observation_or_404(search_service, session, observation_id)
    return _image_file_response(observation.image_path)


# ---------------------------------------------------------------------------
# Журнал поиска (функция 9 ТЗ) - ОТДЕЛЬНО от базы наблюдений (/observations)
# ---------------------------------------------------------------------------

@router.get("/search-history", response_model=SearchHistoryResponse)
async def list_search_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> SearchHistoryResponse:
    """Журнал выполненных поисков, сначала новые, с пагинацией."""
    try:
        return await search_service.list_search_history(session, limit=limit, offset=offset)
    except Exception:
        logger.exception("Ошибка при получении журнала поиска")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="История поиска временно недоступна.",
        )


@router.get("/search-history/{query_id}/crop")
async def get_search_query_crop(
    query_id: uuid.UUID,
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> FileResponse:
    """Кроп, по которому выполнялся поиск (превью записи журнала)."""
    try:
        crop_path = await search_service.get_search_crop_path(session, query_id)
    except Exception:
        logger.exception("Ошибка при чтении записи журнала (id=%s)", query_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="История поиска временно недоступна.",
        )
    if crop_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Запись журнала {query_id} не найдена.",
        )
    return _image_file_response(crop_path)


@router.post("/search-history/{query_id}/search", response_model=SearchResponse)
async def rerun_search(
    query_id: uuid.UUID,
    top_k: Optional[int] = Query(default=None, ge=1, le=100),
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    """
    Повторный поиск из журнала (функция 9 ТЗ): по сохранённому вектору, с ТЕКУЩИМ
    содержимым базы наблюдений. Исходное изображение и модели не нужны. В журнал
    повтор не записывается. top_k по умолчанию - как в исходном поиске.
    """
    try:
        response = await search_service.rerun_search(session, query_id, top_k=top_k)
    except Exception:
        logger.exception("Ошибка при повторном поиске (id=%s)", query_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис поиска временно недоступен. Попробуйте позже.",
        )
    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Запись журнала {query_id} не найдена.",
        )
    return response


@router.delete("/search-history/{query_id}", dependencies=[Depends(require_admin)])
async def delete_search_query(
    query_id: uuid.UUID,
    search_service: SearchService = Depends(get_search_service),
    storage: FileStorage = Depends(get_file_storage),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Удаление одной записи журнала (вместе с файлом кропа). Только администратор."""
    try:
        crop_path = await search_service.delete_search_query(session, query_id)
    except Exception:
        logger.exception("Ошибка при удалении записи журнала (id=%s)", query_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис хранения временно недоступен.",
        )
    if crop_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Запись журнала {query_id} не найдена.",
        )
    storage.delete_permanent(crop_path)
    return {"deleted": True, "id": str(query_id)}


@router.delete("/search-history", dependencies=[Depends(require_admin)])
async def clear_search_history(
    search_service: SearchService = Depends(get_search_service),
    storage: FileStorage = Depends(get_file_storage),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Полная очистка журнала поиска. Только администратор: у системы нет
    учётных записей пользователей, журнал общий - очищать чужие запросы
    любому посетителю нельзя.
    """
    try:
        crop_paths = await search_service.clear_search_history(session)
    except Exception:
        logger.exception("Ошибка при очистке журнала поиска")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис хранения временно недоступен.",
        )
    for path in crop_paths:
        storage.delete_permanent(path)
    return {"deleted": len(crop_paths)}