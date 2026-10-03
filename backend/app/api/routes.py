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

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import require_admin
from ..dependencies import get_db_session, get_detector_service, get_file_storage, get_search_service
from ..schemas.vehicle import (
    AddObservationRequest,
    BoundingBox,
    DetectionResult,
    DetectResponse,
    ObservationHistoryResponse,
    ObservationOut,
    SearchRequest,
    SearchResponse,
)
from ..services.detector import DetectorService
from ..services.search_service import SearchService
from ..storage import FileStorage, UploadNotFoundError, UploadValidationError

router = APIRouter(tags=["vehicles"])
logger = logging.getLogger(__name__)


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


@router.post("/search", response_model=SearchResponse)
async def search_vehicle(
    body: SearchRequest,
    detector: DetectorService = Depends(get_detector_service),
    search_service: SearchService = Depends(get_search_service),
    storage: FileStorage = Depends(get_file_storage),
    session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    """Функции 3-7 ТЗ: кроп выбранного ТС -> эмбеддинг -> поиск -> ответ."""
    try:
        image = storage.load_temp(body.upload_id)
    except UploadNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    chosen = _get_selected_detection(detector, image, body.detection_index)
    crop = detector.crop_with_padding(image, chosen.bbox)

    try:
        response = await search_service.search(
            session, crop=crop, query_bbox=chosen.bbox, top_k=body.top_k
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

    return response


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


@router.get("/observations", response_model=ObservationHistoryResponse)
async def get_observation_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> ObservationHistoryResponse:
    """Функция 9 ТЗ: история ранее загруженных наблюдений, с пагинацией."""
    try:
        return await search_service.get_history(session, limit=limit, offset=offset)
    except Exception:
        logger.exception("Ошибка при получении истории наблюдений")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="История наблюдений временно недоступна.",
        )


@router.delete("/observations/{observation_id}", dependencies=[Depends(require_admin)])
async def delete_observation(
    observation_id: uuid.UUID,
    search_service: SearchService = Depends(get_search_service),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Удаление наблюдения (ТЗ п.1.1.2.3: "система хранения должна
    обеспечивать возможность... удаления записей"). Административная
    функция - требует X-Admin-Token.
    """
    try:
        deleted = await search_service.delete_observation(session, observation_id)
    except Exception:
        logger.exception("Ошибка при удалении наблюдения (id=%s)", observation_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис хранения временно недоступен.",
        )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Наблюдение {observation_id} не найдено.",
        )

    return {"deleted": True, "id": str(observation_id)}