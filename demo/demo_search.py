"""
Визуальная демонстрация полного пайплайна: детекция ТС на кадре ->
кадрирование с отступом -> формирование эмбеддинга -> поиск похожих
изображений в базе -> наглядная картинка с результатом.

Поддерживает два режима входного изображения:

    --detect (по умолчанию): входное изображение - "сырой" кадр
        (может содержать несколько ТС на фоне сцены, как кадры
        UA-DETRAC). По умолчанию для КАЖДОГО найденного ТС отдельно
        выполняется поиск - см. --select ниже, если нужно искать только
        одно конкретное ТС из нескольких найденных.

    --no-detect: входное изображение УЖЕ является кропом одного ТС
        (например, query-изображение из VeRi-776) - шаг детекции
        пропускается, эмбеддинг считается по всему изображению целиком.

Если на кадре несколько ТС, а искать нужно только одно конкретное -
двухэтапный сценарий:

    1. Сначала --preview-only - посмотреть, под какими номерами
       детектор нашёл машины, БЕЗ подключения к базе и без поиска:

           python scripts/demo_search.py --image photo.jpg \
               --detector-model ml/detection/exported/vehicle_detector.onnx \
               --preview-only --output outputs/preview.png

       Откройте outputs/preview.png, посмотрите на номера рамок (#0,
       #1, #2...), выберите нужную машину.

    2. Затем --select N с номером нужной рамки - поиск выполнится
       ТОЛЬКО для неё:

           python scripts/demo_search.py --image photo.jpg \
               --detector-model ml/detection/exported/vehicle_detector.onnx \
               --reid-model ml/reid/exported/reid_embedder.onnx \
               --select 2 --top-k 5 --output outputs/demo_result.png

Пример запуска (из корня проекта):

    python scripts/demo_search.py \
        --image path/to/photo.jpg \
        --detector-model ml/detection/exported/vehicle_detector.onnx \
        --reid-model ml/reid/exported/reid_embedder.onnx \
        --top-k 5 \
        --output outputs/demo_result.png

    # Если входное изображение уже кроп (например, из VeRi-776 query):
    python scripts/demo_search.py \
        --image data/VeRi/image_query/0002_c002_00030600_0.jpg \
        --no-detect \
        --reid-model ml/reid/exported/reid_embedder.onnx \
        --top-k 5 \
        --output outputs/demo_result.png
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db import repository
from backend.app.services.detector import DetectorService, VehicleDetection
from backend.app.services.image_service import ImageService
from backend.app.services.reid_embedder import ReIDEmbedderService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуальная демонстрация поиска ТС")

    parser.add_argument("--image", type=str, required=True,
                         help="Путь к входному изображению")
    parser.add_argument("--detect", dest="detect", action="store_true", default=True,
                         help="Запускать детектор на входном изображении (по умолчанию)")
    parser.add_argument("--no-detect", dest="detect", action="store_false",
                         help="Считать входное изображение уже готовым кропом одного ТС "
                              "(например, query из VeRi-776) - шаг детекции пропускается")

    parser.add_argument("--detector-model", type=str, default=None,
                         help="Путь к vehicle_detector.onnx/.pt (нужен, если --detect)")
    parser.add_argument("--reid-model", type=str, default=None,
                         help="Путь к reid_embedder.onnx (не нужен при --preview-only)")

    parser.add_argument("--preview-only", action="store_true",
                         help="Только прогнать детектор и сохранить картинку с "
                              "пронумерованными рамками, БЕЗ поиска по базе (ReID-модель "
                              "и подключение к БД не требуются). Используйте, чтобы "
                              "сначала посмотреть номера найденных ТС, а затем выбрать "
                              "нужный через --select.")
    parser.add_argument("--select", type=int, default=None,
                         help="Искать только ОДНО конкретное ТС из найденных детектором, "
                              "по его номеру (см. --preview-only). Без этого флага "
                              "поиск выполняется для ВСЕХ найденных ТС сразу.")

    parser.add_argument("--confidence", type=float, default=0.4,
                         help="Порог confidence для детектора")
    parser.add_argument("--padding", type=float, default=0.075,
                         help="Доля отступа при кадрировании (5-10% по архитектуре проекта)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                         help="Размер входа ReID-модели (должен совпадать с обучением)")
    parser.add_argument("--top-k", type=int, default=5,
                         help="Сколько кандидатов показывать на одно найденное ТС")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--output", type=str, default="outputs/demo_result.png",
                         help="Куда сохранить итоговую картинку")

    args = parser.parse_args()
    if args.detect and not args.detector_model:
        parser.error("--detector-model обязателен, если используется режим --detect "
                      "(по умолчанию). Для --no-detect он не нужен.")
    if not args.preview_only and not args.reid_model:
        parser.error("--reid-model обязателен, если не используется --preview-only.")
    if args.select is not None and not args.detect:
        parser.error("--select имеет смысл только вместе с детекцией (--detect, "
                      "по умолчанию включён) - при --no-detect и так ровно одно ТС.")
    return args


def draw_boxes(image: Image.Image, detections: list, display_indices: list = None) -> Image.Image:
    """
    Рисует bounding box'ы на копии изображения, с номером каждой детекции.

    display_indices: какие номера показывать для каждого элемента
    detections (по умолчанию 0..N-1). Нужен, когда detections уже
    отфильтрован до одного элемента через --select N - без этого
    параметра единственный оставшийся бокс подписался бы как "#0",
    даже если пользователь выбирал "#1" на preview, что сбивает с толку.
    """
    if display_indices is None:
        display_indices = list(range(len(detections)))

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for idx, det in zip(display_indices, detections):
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=max(2, image.width // 300))
        label = f"#{idx} ({det.confidence:.2f})"
        # Смещаем подпись по вертикали на idx*16px - если несколько боксов
        # расположены близко друг к другу (например, несколько
        # перекрывающихся детекций от недостаточно уверенной модели),
        # подписи не накладываются друг на друга и остаются читаемыми.
        label_y = max(0, y1 - 18 - idx * 16)
        draw.rectangle([x1, label_y, x1 + 10 * len(label), label_y + 18], fill="red")
        draw.text((x1 + 2, label_y + 1), label, fill="white")
    return annotated


async def run_search(session, crops, image_service, embedder, top_k):
    """Для каждого кропа считает эмбеддинг и ищет top_k кандидатов в базе."""
    all_results = []
    for crop in crops:
        preprocessed = image_service.preprocess(crop)
        embedding = embedder.embed_as_list(preprocessed)
        results = await repository.search_similar(session, embedding, top_k=top_k)
        all_results.append(results)
    return all_results


def build_figure(original, detections, crops, search_results, output_path, no_detect_mode, display_indices=None):
    """
    Собирает итоговую картинку:
        - верхняя строка: исходное изображение (с рамками, если был детект)
        - по одной строке на каждое найденное ТС: [кроп] + [top-K кандидатов]
    """
    if display_indices is None:
        display_indices = list(range(len(crops)))

    num_rows = 1 + len(crops)
    num_cols = max(1 + max((len(r) for r in search_results), default=0), 2)

    fig = plt.figure(figsize=(3.2 * num_cols, 3.2 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols)

    # --- верхняя строка: исходное изображение целиком ---
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(original)
    title = "Входное изображение" if no_detect_mode else f"Входное изображение ({len(detections)} ТС найдено)"
    ax_top.set_title(title, fontsize=11)
    ax_top.axis("off")

    # --- по одной строке на каждое ТС ---
    for row_idx, (crop, results, display_idx) in enumerate(zip(crops, search_results, display_indices), start=1):
        ax_crop = fig.add_subplot(gs[row_idx, 0])
        ax_crop.imshow(crop)
        label = "Query (кроп)" if no_detect_mode else f"Кроп ТС #{display_idx}"
        ax_crop.set_title(label, fontsize=10)
        ax_crop.axis("off")

        for col_idx, (observation, similarity) in enumerate(results, start=1):
            ax_cand = fig.add_subplot(gs[row_idx, col_idx])
            try:
                candidate_img = Image.open(observation.image_path).convert("RGB")
                ax_cand.imshow(candidate_img)
            except FileNotFoundError:
                ax_cand.text(0.5, 0.5, "файл\nне найден", ha="center", va="center")
            ax_cand.set_title(f"#{col_idx} sim={similarity:.3f}", fontsize=9)
            ax_cand.axis("off")

        if not results:
            ax_empty = fig.add_subplot(gs[row_idx, 1])
            ax_empty.text(0.5, 0.5, "база пуста -\nнечего найти", ha="center", va="center")
            ax_empty.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


async def main() -> None:
    args = parse_args()
    load_dotenv()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Не найдено входное изображение: {args.image}")

    original = Image.open(args.image).convert("RGB")

    if args.detect:
        detector = DetectorService(
            model_path=args.detector_model,
            confidence_threshold=args.confidence,
            padding_fraction=args.padding,
            device=args.device,
        )
        detections = detector.detect(original)
        if not detections:
            print("Детектор не нашёл ни одного ТС на изображении. Попробуйте другое фото "
                  "или понизьте --confidence.")
            return
        print(f"Найдено {len(detections)} ТС на изображении.")

        display_indices = list(range(len(detections)))

        if args.preview_only:
            # Поиск не нужен - ReID-модель и БД здесь вообще не
            # задействуются, только отрисовка пронумерованных рамок.
            annotated = draw_boxes(original, detections, display_indices)
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            annotated.save(args.output)
            print(f"\nPreview сохранён: {args.output}")
            print("Посмотрите на номера рамок (#0, #1, ...) и запустите снова с "
                  "--select N и --reid-model, чтобы искать конкретное ТС.")
            return

        if args.select is not None:
            if not (0 <= args.select < len(detections)):
                print(
                    f"Ошибка: --select {args.select} вне диапазона. Детектор нашёл "
                    f"{len(detections)} ТС, допустимые номера: 0..{len(detections) - 1}. "
                    f"Запустите с --preview-only, чтобы посмотреть, какой номер у нужной машины."
                )
                return
            detections = [detections[args.select]]
            display_indices = [args.select]  # сохраняем исходный номер для подписей
            print(f"Ищу только ТС #{args.select} (остальные найденные ТС игнорируются).")

        crops = [detector.crop_with_padding(original, det.bbox) for det in detections]
    else:
        # Режим --no-detect: имитируем "одну детекцию" на весь кадр, чтобы
        # дальнейшая логика (визуализация, поиск) не требовала отдельной ветки.
        detections = [VehicleDetection(bbox=(0, 0, original.width, original.height), confidence=1.0)]
        crops = [original]
        display_indices = [0]
        print("Режим --no-detect: входное изображение считается готовым кропом.")

    # DATABASE_URL и ReID-модель нужны только отсюда и дальше - при
    # --preview-only выше уже произошёл return, эта точка недостижима
    # без реального намерения выполнить поиск.
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL не найден ни в окружении, ни в .env. "
            "Проверьте, что .env лежит в корне проекта и содержит DATABASE_URL."
        )

    image_service = ImageService(image_size=tuple(args.image_size))
    embedder = ReIDEmbedderService(model_path=args.reid_model, device=args.device)

    engine = create_async_engine(database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        search_results = await run_search(session, crops, image_service, embedder, args.top_k)

    await engine.dispose()

    for display_idx, results in zip(display_indices, search_results):
        print(f"\nТС #{display_idx}: найдено {len(results)} кандидатов")
        for rank, (observation, similarity) in enumerate(results, start=1):
            print(f"  #{rank}  similarity={similarity:.4f}  {observation.image_path}")

    annotated_original = draw_boxes(original, detections, display_indices) if args.detect else original
    build_figure(annotated_original, detections, crops, search_results, args.output,
                 no_detect_mode=not args.detect, display_indices=display_indices)

    print(f"\nИтоговая картинка сохранена: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())