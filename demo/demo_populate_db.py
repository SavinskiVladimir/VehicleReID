"""
ДЕМОНСТРАЦИОННЫЙ скрипт. Для штатного пополнения базы используйте
scripts/ingest_images.py (детектор, постоянное хранилище, откат при сбоях;
остановка и продолжение там тоже поддерживаются). Этот скрипт проще: пишет в БД
пути НА ИСХОДНЫЕ ФАЙЛЫ датасета, без копирования в хранилище и без учёта в
таблице ingest_sources. Остановку и продолжение он поддерживает, уже
загруженные изображения определяются по пути файла (observation_exists).

Наполняет базу данных "галереей" для визуальной демонстрации поиска:
берёт N изображений из data/VeRi/image_test (gallery-сплит VeRi-776,
уже скачанный локально при обучении ReID-модели), считает для каждого
эмбеддинг реальной обученной моделью и сохраняет в Postgres.

Изображения VeRi-776 УЖЕ являются кропами одного ТС (в отличие от
сырых кадров UA-DETRAC) - детектор здесь не нужен, предобработка и
эмбеддинг применяются к изображению напрямую, как к уже готовому кропу.

Пример запуска (из корня проекта):

    python scripts/demo_populate_db.py \
        --veri-root data/VeRi \
        --reid-model ml/reid/exported/reid_embedder.onnx \
        --num-images 300
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import random
import sys

from dotenv import load_dotenv
from PIL import Image
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db import repository
from backend.app.db.models import Base
from backend.app.services.image_service import ImageService
from backend.app.services.reid_embedder import ReIDEmbedderService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Наполнение демо-базы изображениями VeRi-776")

    parser.add_argument("--veri-root", type=str, default="data/VeRi",
                         help="Корень VeRi-776 (используется image_test/ - gallery-сплит)")
    parser.add_argument("--reid-model", type=str, required=True,
                         help="Путь к reid_embedder.onnx (ml/reid/export_reid.py)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--num-images", type=int, default=300,
                         help="Сколько изображений из gallery добавить (для скорости - "
                              "не весь gallery сразу, его там тысячи)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-existing", action="store_true",
                         help="Удалить все текущие записи из базы перед заполнением "
                              "(иначе новые изображения добавляются поверх существующих)")

    parser.add_argument("--split", type=str, default="image_test",
                         choices=["image_test", "image_train", "image_query"],
                         help="Какую часть VeRi-776 загружать: image_test (gallery), image_train "
                              "(обучающая, около 37 тыс.), image_query (1678)")
    parser.add_argument("--store-prefix", type=str, default=None,
                         help="Префикс путей, которые записываются в БД, вместо абсолютного пути на "
                              "хосте. Для Docker: /app/data/veri - изображения будут отображаться в "
                              "контейнере без правки путей в БД")

    return parser.parse_args()


def stored_path(path: str, args: argparse.Namespace) -> str:
    """Путь, под которым изображение записывается в БД (и по которому определяется, что оно уже загружено)."""
    if args.store_prefix:
        return f"{args.store_prefix.rstrip('/')}/{args.split}/{os.path.basename(path)}"
    return os.path.abspath(path)


async def main() -> None:
    args = parse_args()
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL не найден ни в окружении, ни в .env. "
            "Проверьте, что .env лежит в корне проекта и содержит DATABASE_URL."
        )

    gallery_dir = os.path.join(args.veri_root, args.split)
    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(f"Не найдена папка {gallery_dir}")

    all_images = sorted(glob.glob(os.path.join(gallery_dir, "*.jpg")))
    if not all_images:
        raise FileNotFoundError(f"В {gallery_dir} не найдено .jpg файлов")

    random.seed(args.seed)
    random.shuffle(all_images)
    selected = all_images[: args.num_images]
    print(f"Найдено {len(all_images)} изображений в gallery, добавляю {len(selected)}.")

    image_service = ImageService(image_size=tuple(args.image_size))
    embedder = ReIDEmbedderService(model_path=args.reid_model, device=args.device)

    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        if args.clear_existing:
            existing, total = await repository.list_observations(session, limit=1)
            print(f"Удаляю {total} существующих записей...")
            # Простая пагинация удаления - забираем и удаляем пачками,
            # пока база не опустеет.
            while True:
                items, total = await repository.list_observations(session, limit=100, offset=0)
                if not items:
                    break
                for item in items:
                    await repository.delete_observation(session, item.id)
            print("База очищена.")

        added = 0

        # Первый проход: определяем, какие изображения уже есть в БД.
        print("Проверяю существующие изображения...")

        new_images = []

        for path in selected:
            image_path = stored_path(path, args)

            if await repository.observation_exists(session, image_path):
                continue

            new_images.append(path)

        skipped = len(selected) - len(new_images)

        print(f"Пропускаю {skipped} уже существующих изображений.")
        print(f"Начинаю загрузку {len(new_images)} новых изображений...")

        # Второй проход: обрабатываем только новые изображения.
        for path in new_images:
            image = Image.open(path).convert("RGB")
            preprocessed = image_service.preprocess(image)
            embedding = embedder.embed_as_list(preprocessed)

            image_path = stored_path(path, args)

            await repository.add_observation(
                session,
                image_path=image_path,
                crop_path=image_path,
                bbox=(0, 0, image.width, image.height),
                embedding=embedding,
            )

            added += 1

            if added % 50 == 0:
                print(f"  Добавлено {added}/{len(new_images)}...")

    await engine.dispose()
    print()
    print(f"Готово. В базу добавлено {added} изображений.")
    print(f"Пропущено уже существующих: {skipped}.")


if __name__ == "__main__":
    asyncio.run(main())