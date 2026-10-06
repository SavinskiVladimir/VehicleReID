"""
Пакетное пополнение базы наблюдений изображениями из папки (ТЗ п.1.1.2.1:
"дополнительный режим - пакетное добавление изображений в базу наблюдений без
выполнения поиска").

Проходит тот же путь, что и добавление через интерфейс: детекция -> кроп ->
эмбеддинг -> постоянное хранилище -> БД (см. backend/app/services/ingest_service.py).
Настройки (БД, модели, хранилище) берутся из .env, как и у сервера.

Режимы (--mode):
    all    (по умолчанию) каждое найденное на кадре ТС - отдельное наблюдение
    best   только самое уверенное ТС кадра
    whole  без детектора: кадр целиком = изображение ТС. Для готовых кропов
           (VeRi-776, CityFlow-кропы). Детектор в этом режиме не нужен.

Повторный запуск безопасен: файл опознаётся по SHA-256 содержимого, уже
загруженное пропускается (дубликаты не создаются, в том числе для переименованных
копий). Прерванную загрузку можно просто запустить снова. Чтобы загрузить заново
с другими настройками - --replace (старые наблюдения файла заменяются).

Перед первым запуском нужна таблица учёта загрузок:  python db/create_table.py

Примеры (из корня проекта):

    # проверить, что будет загружено, ничего не записывая
    python scripts/ingest_images.py data/frames --dry-run

    # загрузить кадры камер
    python scripts/ingest_images.py data/frames

    # загрузить готовые кропы машин (без детектора)
    python scripts/ingest_images.py data/VeRi/image_test --mode whole --limit 500

    # перезагрузить те же файлы с другим режимом
    python scripts/ingest_images.py data/frames --mode best --replace

Код возврата: 0 - всё хорошо; 1 - не удалось начать (модели, БД, пути);
2 - часть файлов не загружена (список в конце); 130 - прервано пользователем.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.config import get_settings
from backend.app.services.image_service import ImageService
from backend.app.services.ingest_service import MODES, IngestService, IngestStats, iter_image_files
from backend.app.services.reid_embedder import ReIDEmbedderService
from backend.app.services.search_service import SearchService
from backend.app.startup_checks import StartupCheckError, check_embedding_dimension, inspect_database
from backend.app.storage import FileStorage
from backend.app.db.models import EMBEDDING_DIM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Пакетное пополнение базы наблюдений изображениями из папки",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Подробности и примеры - в документации в начале файла scripts/ingest_images.py",
    )
    parser.add_argument("paths", nargs="+", help="Папки и/или файлы изображений (jpg, jpeg, png, webp)")
    parser.add_argument("--mode", choices=MODES, default="all",
                        help="Что считать наблюдением на кадре: all - каждое ТС, best - самое "
                             "уверенное, whole - кадр целиком без детектора (по умолчанию all)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Порог уверенности детектора (по умолчанию - DETECTOR_CONFIDENCE_THRESHOLD из .env)")
    parser.add_argument("--no-recursive", action="store_true", help="Не заходить во вложенные папки")
    parser.add_argument("--limit", type=int, default=None, help="Обработать не более N файлов (для пробного запуска)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только посчитать, что будет загружено: в БД и на диск ничего не пишется")
    parser.add_argument("--replace", action="store_true",
                        help="Файлы, уже загруженные ранее, загрузить заново: их старые наблюдения заменяются")
    parser.add_argument("-v", "--verbose", action="store_true", help="Печатать результат по каждому файлу")
    parser.add_argument("--progress-every", type=int, default=50, metavar="N",
                        help="Как часто печатать строку прогресса, файлов (по умолчанию 50)")
    return parser


def _fail(message: str) -> int:
    print(f"ОШИБКА: {message}", file=sys.stderr)
    return 1


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} с"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {seconds:02d} с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"


async def run(args: argparse.Namespace, detector=None) -> int:
    """
    detector - готовый объект детектора; нужен тестам, чтобы не загружать
    реальные веса YOLO. В обычном запуске создаётся из настроек.
    """
    if args.limit is not None and args.limit < 1:
        return _fail("--limit должен быть не меньше 1")

    settings = get_settings()

    # --- 1. Окружение: всё, что можно проверить ДО начала долгой работы ---
    if not Path(settings.reid_model_path).is_file():
        return _fail(f"не найдена ReID-модель: {settings.reid_model_path}\n"
                     f"Получить: python ml/reid/export_reid.py (см. README, шаг 5)")
    if args.mode != "whole" and detector is None and not Path(settings.detector_model_path).is_file():
        return _fail(f"не найдена модель детектора: {settings.detector_model_path}\n"
                     f"Режим {args.mode!r} требует детектор. Для готовых кропов можно --mode whole.")

    try:
        files = iter_image_files(args.paths, recursive=not args.no_recursive)
    except FileNotFoundError as exc:
        return _fail(str(exc))
    if not files:
        return _fail("в указанных путях не найдено изображений (jpg, jpeg, png, webp)")
    total_found = len(files)
    if args.limit is not None:
        files = files[: args.limit]

    image_service = ImageService(image_size=settings.reid_image_size)
    embedder = ReIDEmbedderService(model_path=settings.reid_model_path, device=settings.device)
    try:
        check_embedding_dimension(embedder, image_service, EMBEDDING_DIM)
    except StartupCheckError as exc:
        return _fail(str(exc))

    if args.mode != "whole" and detector is None:
        from backend.app.services.detector import DetectorService  # тяжёлый импорт - только когда нужен

        detector = DetectorService(
            model_path=settings.detector_model_path,
            confidence_threshold=(
                args.confidence if args.confidence is not None else settings.detector_confidence_threshold
            ),
            iou_threshold=settings.detector_iou_threshold,
            device=settings.device,
            padding_fraction=settings.padding_fraction,
        )

    engine = create_async_engine(settings.database_url, echo=False)
    try:
        status = await inspect_database(engine)
        if not status.ready:
            return _fail("база данных не готова к работе:\n  " + "\n  ".join(status.problems()))
        async with engine.connect() as conn:
            has_sources = (await conn.execute(
                text("SELECT to_regclass('public.ingest_sources') IS NOT NULL")
            )).scalar()
        if not has_sources:
            return _fail("нет таблицы учёта загрузок ingest_sources. Выполните: python db/create_table.py\n"
                         "(существующие данные не затрагиваются)")

        storage = FileStorage(
            temp_dir=settings.temp_upload_dir,
            permanent_dir=settings.permanent_storage_dir,
            max_upload_size_bytes=settings.max_upload_size_bytes,
            temp_ttl_seconds=settings.temp_upload_ttl_seconds,
        )
        search_service = SearchService(image_service=image_service, embedder=embedder)
        ingest = IngestService(search_service=search_service, storage=storage, detector=detector)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        return await _ingest_all(args, files, total_found, ingest, session_factory)
    finally:
        await engine.dispose()


async def _ingest_all(args, files, total_found, ingest, session_factory) -> int:
    header = f"Найдено изображений: {total_found}"
    if len(files) < total_found:
        header += f", обрабатывается: {len(files)} (--limit)"
    print(header)
    print(f"Режим: {args.mode}" + ("  |  ПРОВЕРКА (dry-run): ничего не записывается" if args.dry_run else "")
          + ("  |  --replace: загруженное ранее заменяется" if args.replace else ""))
    print()

    stats = IngestStats()
    seen = {}  # содержимое, уже обработанное В ЭТОМ запуске (копии файлов не обрабатываются дважды)
    started = time.monotonic()
    interrupted = False

    try:
        for index, path in enumerate(files, start=1):
            async with session_factory() as session:
                result = await ingest.ingest_file(
                    session, path, mode=args.mode, replace=args.replace, dry_run=args.dry_run, seen=seen
                )
            stats.record(path, result)

            if result.status == "failed":
                print(f"ОШИБКА  {path}: {result.error}", file=sys.stderr)
            elif args.verbose:
                added_label = "добавилось бы" if args.dry_run else "добавлено"   # в dry-run ничего не добавлено
                label = {"added": added_label, "no_vehicles": "нет ТС", "skipped": "пропущено"}[result.status]
                extra = f" ({result.observations} шт.)" if result.status in ("added", "skipped") else ""
                print(f"{label:<14}{path}{extra}")

            if args.progress_every and index % args.progress_every == 0 and index < len(files):
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed > 0 else 0
                remaining = (len(files) - index) / rate if rate > 0 else 0
                print(f"[{index:>{len(str(len(files)))}}/{len(files)}] добавлено: {stats.added_files}, "
                      f"пропущено: {stats.skipped}, без ТС: {stats.no_vehicles}, ошибок: {stats.failed} "
                      f"| {rate:.1f} изобр./с, осталось ~{_format_duration(remaining)}")
    except (asyncio.CancelledError, KeyboardInterrupt):
        # Каждый файл загружается атомарно, так что прерывание безопасно:
        # продолжить можно простым повторным запуском.
        interrupted = True

    elapsed = time.monotonic() - started
    _print_summary(args, stats, len(files), elapsed, interrupted)

    if interrupted:
        return 130
    return 2 if stats.failed else 0


def _print_summary(args, stats: IngestStats, planned: int, elapsed: float, interrupted: bool) -> None:
    print()
    print("Прервано пользователем, безопасно: можно запустить снова." if interrupted
          else f"Готово за {_format_duration(elapsed)}"
               + (f" ({stats.processed / elapsed:.1f} изобр./с)" if elapsed > 0 and stats.processed else "")
               + ".")
    if args.dry_run:
        print("РЕЖИМ ПРОВЕРКИ: в базу и на диск ничего не записано.")
    verb = "было бы добавлено" if args.dry_run else "добавлено"
    print(f"  Обработано файлов:       {stats.processed} из {planned}")
    print(f"  С новыми наблюдениями:   {stats.added_files:<6} ({verb} наблюдений: {stats.observations})")
    print(f"  Пропущено (уже были):    {stats.skipped}")
    print(f"  Без найденных ТС:        {stats.no_vehicles}")
    if stats.replaced:
        print(f"  Заменено наблюдений:     {stats.replaced}")
    print(f"  Ошибок:                  {stats.failed}")
    if stats.errors:
        print()
        print("Не загружены (остальные загружены, повторный запуск попробует эти снова):")
        for path, error in stats.errors[:20]:
            print(f"  {path}: {error}")
        if len(stats.errors) > 20:
            print(f"  ... и ещё {len(stats.errors) - 20}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130


if __name__ == "__main__":
    sys.exit(main())