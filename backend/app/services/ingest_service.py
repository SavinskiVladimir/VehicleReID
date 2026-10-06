"""
Пакетное пополнение базы наблюдений (ТЗ п.1.1.2.1: "дополнительный режим -
пакетное добавление изображений в базу наблюдений без выполнения поиска").

Проходит ТОТ ЖЕ путь, что и добавление через API (функция 8 ТЗ): детекция ->
кроп с отступом -> эмбеддинг -> постоянное хранилище -> БД. Отличие одно -
нет пользователя, выбирающего ТС на кадре, поэтому выбор задаётся режимом:

    all    каждое найденное ТС становится наблюдением (кадры камер);
    best   только самое уверенное ТС кадра;
    whole  без детектора: кадр целиком считается изображением ТС (для готовых
           кропов, например VeRi-776). Детектор в этом режиме не нужен вообще.

Гарантии:

  * Идемпотентность. Файл опознаётся по SHA-256 содержимого (таблица
    ingest_sources): повторный запуск, прерванный и продолженный запуск,
    переименованная или скопированная картинка не создают дубликатов.
    Режим replace заменяет наблюдения файла при загрузке с другими настройками.

  * Атомарность на уровне файла. Все наблюдения одного файла и отметка о его
    загрузке попадают в БД ОДНОЙ транзакцией. Все эмбеддинги считаются ДО
    записи чего-либо, а при любом сбое транзакция откатывается и сохранённые
    файлы удаляются: ни записей без файлов, ни файлов без записей.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import repository
from ..storage import FileStorage
from .search_service import SearchService

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODES = ("all", "best", "whole")

BBox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Поиск файлов
# ---------------------------------------------------------------------------

def iter_image_files(paths: Iterable[str | Path], recursive: bool = True) -> List[Path]:
    """
    Файлы изображений из списка путей (файлы и/или папки), отсортированные и без
    повторов - порядок обработки воспроизводим от запуска к запуску.

    Скрытые файлы (имя начинается с точки, например "._img.jpg" от macOS) не
    берутся: это не изображения, хоть и с подходящим расширением.

    Raises:
        FileNotFoundError: путь не существует (опечатка в пути не должна
            превращаться в тихую "загрузку нуля файлов").
    """
    found = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Путь не найден: {path}")
        candidates = [path] if path.is_file() else (path.rglob("*") if recursive else path.glob("*"))
        for candidate in candidates:
            if (
                candidate.is_file()
                and not candidate.name.startswith(".")
                and candidate.suffix.lower() in IMAGE_EXTENSIONS
            ):
                found[candidate.resolve()] = candidate
    return [found[key] for key in sorted(found)]


# ---------------------------------------------------------------------------
# Результаты
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    """Итог обработки одного файла."""

    status: str  # "added" | "no_vehicles" | "skipped" | "failed"
    observations: int = 0  # создано наблюдений (для skipped - сколько уже было)
    replaced: int = 0      # удалено старых наблюдений (режим replace)
    error: Optional[str] = None
    source_hash: str = ""


@dataclass
class IngestStats:
    """Накопленная статистика запуска."""

    added_files: int = 0
    skipped: int = 0
    no_vehicles: int = 0
    failed: int = 0
    observations: int = 0
    replaced: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return self.added_files + self.skipped + self.no_vehicles + self.failed

    def record(self, path: str | Path, result: IngestResult) -> None:
        if result.status == "added":
            self.added_files += 1
            self.observations += result.observations
        elif result.status == "skipped":
            self.skipped += 1
        elif result.status == "no_vehicles":
            self.no_vehicles += 1
        else:
            self.failed += 1
            self.errors.append((str(path), result.error or "неизвестная ошибка"))
        self.replaced += result.replaced


# ---------------------------------------------------------------------------
# Сервис
# ---------------------------------------------------------------------------

class IngestService:
    """
    Args:
        search_service: эмбеддинг и запись наблюдения (тот же сервис, что в API).
        storage: постоянное хранилище файлов.
        detector: DetectorService; может быть None, если используется только
            режим whole.
    """

    def __init__(self, search_service: SearchService, storage: FileStorage, detector=None):
        self.search_service = search_service
        self.storage = storage
        self.detector = detector

    async def ingest_file(
        self,
        session: AsyncSession,
        path: str | Path,
        *,
        mode: str = "all",
        replace: bool = False,
        dry_run: bool = False,
        seen: Optional[Dict[str, int]] = None,
    ) -> IngestResult:
        """
        Загружает один файл. Никогда не бросает исключений из-за самого файла
        или сбоя модели/БД: любая проблема возвращается как status="failed"
        с описанием, чтобы один плохой файл не останавливал загрузку тысяч.

        Args:
            seen: словарь {хеш содержимого: число наблюдений} ОДНОГО запуска -
                вызывающий код передаёт один и тот же словарь на все файлы
                запуска. Копия уже обработанного в этом запуске файла
                пропускается. Без этого: в режиме replace копия заменяла бы
                только что созданное (лишняя работа и завышенная статистика
                замен), а в dry-run копия считалась бы дважды, хотя реальная
                загрузка добавит её один раз.
        """
        if mode not in MODES:
            raise ValueError(f"Неизвестный режим {mode!r}, допустимо: {', '.join(MODES)}")
        if mode != "whole" and self.detector is None:
            raise ValueError(f"Режиму {mode!r} нужен детектор (режим whole работает без него)")

        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            return IngestResult("failed", error=f"не удалось прочитать файл: {exc}")
        source_hash = hashlib.sha256(data).hexdigest()

        if seen is not None and source_hash in seen:
            return IngestResult("skipped", observations=seen[source_hash], source_hash=source_hash)

        result = await self._ingest_unseen(
            session, path, data, source_hash, mode=mode, replace=replace, dry_run=dry_run
        )
        if seen is not None and result.status != "failed":
            seen[source_hash] = result.observations
        return result

    async def _ingest_unseen(
        self, session: AsyncSession, path: Path, data: bytes, source_hash: str,
        *, mode: str, replace: bool, dry_run: bool,
    ) -> IngestResult:
        try:
            existing = await repository.get_ingest_source(session, source_hash)
            if existing is not None and not replace:
                return IngestResult(
                    "skipped", observations=len(existing.observation_ids), source_hash=source_hash
                )

            try:
                image = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:
                return IngestResult("failed", error=f"не удалось открыть как изображение: {exc}",
                                    source_hash=source_hash)

            regions = self._find_regions(image, mode)
            replaced = len(existing.observation_ids) if existing is not None else 0

            if dry_run:
                return IngestResult("added" if regions else "no_vehicles",
                                    observations=len(regions), replaced=replaced,
                                    source_hash=source_hash)

            # ВСЕ эмбеддинги - до записи чего-либо: сбой модели на втором ТС
            # кадра не должен оставить в базе половину кадра.
            embeddings = [self.search_service.compute_embedding(crop) for _, crop in regions]

            return await self._store(
                session, path, source_hash, image, regions, embeddings, mode, existing
            )
        except Exception as exc:
            await _safe_rollback(session)
            logger.exception("Ошибка при загрузке %s", path)
            return IngestResult("failed", error=f"{type(exc).__name__}: {exc}", source_hash=source_hash)

    # ------------------------------------------------------------------

    def _find_regions(self, image: Image.Image, mode: str) -> List[Tuple[BBox, Image.Image]]:
        """Что именно станет наблюдениями: список (рамка, кроп)."""
        if mode == "whole":
            return [((0, 0, image.width, image.height), image)]

        detections = self.detector.detect(image)
        if mode == "best":
            detections = detections[:1]
        return [(d.bbox, self.detector.crop_with_padding(image, d.bbox)) for d in detections]

    async def _store(
        self,
        session: AsyncSession,
        path: Path,
        source_hash: str,
        image: Image.Image,
        regions: Sequence[Tuple[BBox, Image.Image]],
        embeddings: Sequence[Sequence[float]],
        mode: str,
        existing,
    ) -> IngestResult:
        saved_files: List[str] = []
        old_paths: List[Tuple[str, str]] = []
        new_ids = []
        try:
            # Полный кадр сохраняется ОДИН раз на файл: у нескольких ТС одного
            # кадра общий image_path (в отличие от кропов - их по одному на ТС).
            image_path = None
            if regions:
                _, image_path = self.storage.save_permanent(image, subdir="images")
                saved_files.append(image_path)
            crop_paths = []
            for _, crop in regions:
                _, crop_path = self.storage.save_permanent(crop, subdir="crops")
                saved_files.append(crop_path)
                crop_paths.append(crop_path)

            if existing is not None:  # режим replace: старое - прочь, в той же транзакции
                old_paths = await repository.delete_observations_by_ids(
                    session, existing.observation_ids
                )
                await repository.delete_ingest_source(session, source_hash)

            for (bbox, crop), embedding, crop_path in zip(regions, embeddings, crop_paths):
                observation = await self.search_service.add_observation(
                    session, image_path=image_path, crop_path=crop_path,
                    crop=crop, bbox=bbox, embedding=embedding, commit=False,
                )
                new_ids.append(observation.id)

            await repository.add_ingest_source(session, source_hash, str(path), mode, new_ids)
            await session.commit()
        except Exception:
            await _safe_rollback(session)
            for file_path in saved_files:  # ни файлов без записей, ни записей без файлов
                self.storage.delete_permanent(file_path)
            raise

        # Файлы старых наблюдений удаляются только ПОСЛЕ коммита: при откате
        # записи остались бы в базе уже без файлов.
        for old_path in {p for pair in old_paths for p in pair}:
            self.storage.delete_permanent(old_path)

        return IngestResult(
            "added" if new_ids else "no_vehicles",
            observations=len(new_ids),
            replaced=len(old_paths),
            source_hash=source_hash,
        )


async def _safe_rollback(session: AsyncSession) -> None:
    try:
        await session.rollback()
    except Exception:  # откат на уже сломанном соединении - не должен прятать исходную ошибку
        pass