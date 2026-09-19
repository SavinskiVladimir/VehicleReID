"""
Подготовка датасета UA-DETRAC для дообучения YOLO11 на задаче детекции
единого класса "vehicle" (согласно архитектуре системы - детектор не
различает car/bus/van/truck, а находит любое транспортное средство).

Почему UA-DETRAC, а не COCO: UA-DETRAC - это именно кадры со стационарных
камер городского видеонаблюдения (Пекин/Тяньцзинь), что напрямую
соответствует домену эксплуатации системы согласно ТЗ ("камеры
видеонаблюдения, установленные в городской среде"). YOLO11 уже
предобучен на COCO (там есть car/truck/bus), поэтому дообучение именно
на CCTV-специфичных кадрах даёт реальную пользу, а не просто повторяет
то, что модель уже знает.

Ожидаемая структура исходных файлов UA-DETRAC (нативный релиз датасета,
https://detrac-db.rit.albany.edu/download):

    UA-DETRAC/
        Insight-MVT_Annotation_Train/
            MVI_20011/
                img00001.jpg
                img00002.jpg
                ...
            MVI_20012/
                ...
        DETRAC-Train-Annotations-XML/
            MVI_20011.xml
            MVI_20012.xml
            ...

Формат XML-аннотации (подтверждённая структура нативного релиза):

    <sequence name="MVI_20011">
      <frame num="1">
        <target_list>
          <target id="1">
            <box left="..." top="..." width="..." height="..."/>
            <attribute vehicle_type="car" .../>
          </target>
        </target_list>
      </frame>
      ...
    </sequence>

Скрипт:
    1. Парсит все XML-файлы аннотаций.
    2. Конвертирует box (left, top, width, height) в формат YOLO
       (class x_center y_center width height, все значения нормированы
       на размер кадра в диапазоне [0,1]).
    3. Все типы vehicle_type (car, bus, van, others) сливаются в ОДИН
       класс "vehicle" (индекс 0), согласно архитектуре системы.
    4. Опционально отбрасывает сильно перекрытые/обрезанные боксы
       (occlusion_ratio / truncation_ratio выше порога) - такие объекты
       обычно вредят обучению детектора, а не помогают.
    5. Разбивает ПОСЛЕДОВАТЕЛЬНОСТИ (не отдельные кадры!) на train/val,
       чтобы избежать утечки данных - соседние кадры одной
       последовательности почти идентичны, и если раскидать их по
       train/val случайно, оценка качества будет необъективно завышена.
    6. Копирует/симлинкает изображения в стандартную YOLO-структуру
       (images/train, images/val, labels/train, labels/val) и пишет
       data.yaml для передачи в ultralytics.

Пример запуска:

    python ml/detection/prepare_dataset.py \
        --detrac-root /path/to/UA-DETRAC \
        --output data/detection_yolo \
        --val-fraction 0.15
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple

import yaml


VEHICLE_CLASS_ID = 0  # единственный класс - "vehicle"


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвертация UA-DETRAC в формат YOLO (единый класс vehicle)"
    )
    parser.add_argument("--detrac-root", type=str, required=True,
                         help="Корень UA-DETRAC (содержит Insight-MVT_Annotation_Train/ и DETRAC-Train-Annotations-XML/)")
    parser.add_argument("--images-subdir", type=str, default="Insight-MVT_Annotation_Train")
    parser.add_argument("--annotations-subdir", type=str, default="DETRAC-Train-Annotations-XML")
    parser.add_argument("--output", type=str, required=True,
                         help="Куда записать датасет в формате YOLO")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Доля ПОСЛЕДОВАТЕЛЬНОСТЕЙ (не кадров), уходящих в val")
    parser.add_argument("--max-sequences", type=int, default=None,
                         help="Ограничить число обрабатываемых последовательностей "
                              "(случайный выбор с учётом --seed). Полезно, когда "
                              "полный датасет слишком велик для доступного места/времени "
                              "(например, при обучении в Google Colab) - можно взять, "
                              "скажем, 15 последовательностей из 100 вместо всех. "
                              "По умолчанию используются ВСЕ найденные последовательности.")
    parser.add_argument("--max-occlusion-ratio", type=float, default=1.0,
                         help="Отбрасывать боксы с occlusion_ratio выше порога (1.0 = не фильтровать)")
    parser.add_argument("--max-truncation-ratio", type=float, default=1.0,
                         help="Отбрасывать боксы с truncation_ratio выше порога (1.0 = не фильтровать)")
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="symlink",
                         help="copy - дублировать файлы изображений; symlink - создавать символьные "
                              "ссылки (быстрее и не расходует лишнее место на диске, но требует "
                              "прав на symlink - на Windows без прав администратора может не сработать)")
    parser.add_argument("--frame-stride", type=int, default=1,
                         help="Брать только каждый N-й кадр последовательности (по номеру "
                              "кадра). UA-DETRAC снят с 25 fps, и соседние кадры почти "
                              "дублируют друг друга - для обучения детектора это в основном "
                              "избыточность, а не полезное разнообразие. --frame-stride 5 "
                              "снижает объём датасета впятеро почти без потери качества "
                              "обучения, что особенно полезно при ограниченном месте/времени "
                              "(например, в Google Colab). По умолчанию 1 = берутся все кадры.")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _parse_sequence_xml(
    xml_path: str,
    max_occlusion_ratio: float,
    max_truncation_ratio: float,
) -> dict:
    """
    Парсит один XML-файл аннотаций UA-DETRAC (одна последовательность).

    Returns:
        {frame_num: [YoloBox без нормировки на размер кадра (в пикселях,
         но уже в формате left/top/width/height -> x_center/y_center/w/h)]}
        Нормировка на размер изображения делается отдельно, т.к. XML не
        всегда содержит явный размер кадра - берём его из самого файла
        изображения при записи.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames = {}
    for frame_tag in root.findall("frame"):
        frame_num = int(frame_tag.attrib["num"])
        boxes_px = []  # (left, top, width, height) в пикселях

        target_list = frame_tag.find("target_list")
        if target_list is None:
            frames[frame_num] = boxes_px
            continue

        for target in target_list.findall("target"):
            box_tag = target.find("box")
            if box_tag is None:
                continue

            attribute_tag = target.find("attribute")
            occlusion_ratio = 0.0
            truncation_ratio = 0.0
            if attribute_tag is not None:
                occlusion_ratio = float(attribute_tag.attrib.get("occlusion_ratio", 0.0) or 0.0)
                truncation_ratio = float(attribute_tag.attrib.get("truncation_ratio", 0.0) or 0.0)

            if occlusion_ratio > max_occlusion_ratio:
                continue
            if truncation_ratio > max_truncation_ratio:
                continue

            left = float(box_tag.attrib["left"])
            top = float(box_tag.attrib["top"])
            width = float(box_tag.attrib["width"])
            height = float(box_tag.attrib["height"])
            boxes_px.append((left, top, width, height))

        frames[frame_num] = boxes_px

    return frames


def _boxes_to_yolo_lines(boxes_px: List[Tuple[float, float, float, float]], img_w: int, img_h: int) -> List[str]:
    lines = []
    for left, top, width, height in boxes_px:
        x_center = (left + width / 2.0) / img_w
        y_center = (top + height / 2.0) / img_h
        w_norm = width / img_w
        h_norm = height / img_h

        # Отбрасываем полностью вырожденные/вышедшие за кадр боксы.
        if w_norm <= 0 or h_norm <= 0:
            continue
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        w_norm = min(w_norm, 1.0)
        h_norm = min(h_norm, 1.0)

        lines.append(f"{VEHICLE_CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return lines


def _link_or_copy(src: str, dst: str, mode: str) -> None:
    if os.path.exists(dst):
        return
    if mode == "symlink":
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError:
            pass  # на некоторых системах (Windows без прав) symlink не создать - копируем
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    images_root = os.path.join(args.detrac_root, args.images_subdir)
    annotations_root = os.path.join(args.detrac_root, args.annotations_subdir)

    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Не найдена папка изображений: {images_root}")
    if not os.path.isdir(annotations_root):
        raise FileNotFoundError(f"Не найдена папка аннотаций: {annotations_root}")

    xml_files = sorted(f for f in os.listdir(annotations_root) if f.endswith(".xml"))
    if not xml_files:
        raise FileNotFoundError(f"В {annotations_root} не найдено .xml файлов")

    sequence_names = [os.path.splitext(f)[0] for f in xml_files]
    random.shuffle(sequence_names)

    if args.max_sequences is not None and args.max_sequences < len(sequence_names):
        print(
            f"Ограничиваю число последовательностей: {len(sequence_names)} -> "
            f"{args.max_sequences} (случайный выбор, seed={args.seed})"
        )
        sequence_names = sequence_names[: args.max_sequences]

    num_val = max(1, int(round(len(sequence_names) * args.val_fraction)))
    val_sequences = set(sequence_names[:num_val])
    train_sequences = set(sequence_names[num_val:])

    print(f"Всего последовательностей: {len(sequence_names)}")
    print(f"  train: {len(train_sequences)}, val: {len(val_sequences)}")
    print(
        "  (разбиение по ПОСЛЕДОВАТЕЛЬНОСТЯМ целиком, а не по отдельным "
        "кадрам - иначе соседние кадры одной сцены попадут и в train, и "
        "в val, завышая оценку качества)"
    )

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output, "labels", split), exist_ok=True)

    # Отложенный импорт, чтобы скрипт можно было запустить даже без Pillow
    # установленного глобально (проверка происходит только здесь).
    from PIL import Image

    stats = {"train": {"images": 0, "boxes": 0}, "val": {"images": 0, "boxes": 0}}

    for seq_name in sequence_names:
        split = "val" if seq_name in val_sequences else "train"

        xml_path = os.path.join(annotations_root, f"{seq_name}.xml")
        seq_image_dir = os.path.join(images_root, seq_name)
        if not os.path.isdir(seq_image_dir):
            print(f"  ПРЕДУПРЕЖДЕНИЕ: нет папки изображений для {seq_name}, пропускаю")
            continue

        frames = _parse_sequence_xml(xml_path, args.max_occlusion_ratio, args.max_truncation_ratio)

        for frame_num, boxes_px in frames.items():
            if (frame_num - 1) % args.frame_stride != 0:
                continue

            img_filename = f"img{frame_num:05d}.jpg"
            src_img_path = os.path.join(seq_image_dir, img_filename)
            if not os.path.isfile(src_img_path):
                continue

            with Image.open(src_img_path) as img:
                img_w, img_h = img.size

            yolo_lines = _boxes_to_yolo_lines(boxes_px, img_w, img_h)

            # Уникальное имя, т.к. img00001.jpg повторяется в каждой
            # последовательности - добавляем префикс seq_name.
            unique_name = f"{seq_name}_{img_filename}"
            dst_img_path = os.path.join(args.output, "images", split, unique_name)
            dst_label_path = os.path.join(
                args.output, "labels", split, os.path.splitext(unique_name)[0] + ".txt"
            )

            _link_or_copy(src_img_path, dst_img_path, args.link_mode)
            with open(dst_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n")

            stats[split]["images"] += 1
            stats[split]["boxes"] += len(yolo_lines)

    data_yaml = {
        "path": os.path.abspath(args.output),
        "train": "images/train",
        "val": "images/val",
        "names": {VEHICLE_CLASS_ID: "vehicle"},
    }
    data_yaml_path = os.path.join(args.output, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print()
    print("Готово.")
    for split in ("train", "val"):
        print(f"  {split}: {stats[split]['images']} изображений, {stats[split]['boxes']} боксов")
    print(f"data.yaml записан в: {data_yaml_path}")


if __name__ == "__main__":
    main()
