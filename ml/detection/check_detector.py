"""
Диагностика детектора ТС на ваших изображениях: почему на кропах (VeRi-776) он
находит фрагменты машины или ничего, и насколько рамки соответствуют разметке.

Два режима (выбираются наличием --gt-labels):

  КРОПЫ (по умолчанию). В папке готовые кропы одной машины, например
  data/VeRi/image_test: машина занимает почти весь кадр. Для каждого
  изображения смотрим, какую часть кадра закрывает лучшая рамка детектора:
      верно       лучшая рамка закрывает >= --good-coverage кадра (0.6)
      фрагмент    рамки есть, но лучшая закрывает меньше (часть машины)
      не найдено  ни одной рамки выше порога
  Проверяются возможные "рычаги", каждый из которых мог бы объяснить провал:
      порог достоверности   (--app-conf, --conf)
      размер входа imgsz    (--imgsz; только для .pt, у ONNX он зафиксирован)
      масштаб ТС в кадре    (--canvas: кроп кладётся в центр серого холста в
                             k раз больше, машина становится "мелкой", как
                             в обучающих кадрах UA-DETRAC)
  Итог - вывод "что именно помогает" (или что ничего не помогает и нужно
  до-обучение).

  КАДРЫ С РАЗМЕТКОЙ (--gt-labels). В папке кадры, рядом (или в --gt-labels)
  txt-разметка в формате YOLO (class cx cy w h, нормированная), например
  валидационная часть датасета UA-DETRAC после prepare_dataset.py. Считаются
  точность, полнота и главное - СИСТЕМАТИЧЕСКОЕ отличие размера рамок от
  разметки (медиана отношения ширины, высоты и площади). Если рамки в среднем
  меньше разметки, скрипт подсказывает, какого PADDING_FRACTION хватило бы,
  чтобы это компенсировать.

Что получается на выходе (папка --out):
    summary.json            все числа и вывод
    sheet_<вариант>.jpg     контактные листы с рамками: зелёная (green) - достоверность
                            не ниже порога приложения, оранжевая (orange) - ниже,
                            синяя (blue) - разметка (только в режиме с разметкой).
                            Подписи на листах латиницей: встроенный шрифт не умеет кириллицу

Скрипт вызывает YOLO тем же способом, что backend/app/services/detector.py
(PIL-изображение, те же conf/iou), поэтому результат по умолчанию показывает
то, что реально происходит в приложении.

Примеры (из корня проекта):

    # кропы VeRi: что мешает детектору?
    python ml/detection/check_detector.py --images data/VeRi/image_test --limit 60

    # то же на ONNX, который реально используется приложением (без перебора imgsz)
    python ml/detection/check_detector.py --images data/VeRi/image_test \\
        --model ml/detection/exported/vehicle_detector.onnx

    # размер рамок против разметки на кадрах UA-DETRAC
    python ml/detection/check_detector.py --images data/ua_detrac_yolo/images/val \\
        --gt-labels data/ua_detrac_yolo/labels/val --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

Box = Tuple[float, float, float, float]            # x1, y1, x2, y2
Detection = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence
# predictor(изображение, минимальная достоверность, imgsz или None) -> список рамок
Predictor = Callable[[Image.Image, float, Optional[int]], List[Detection]]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CANVAS_GRAY = (114, 114, 114)  # тот же серый, что у letterbox в YOLO


# ===========================================================================
# Геометрия
# ===========================================================================

def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(box: Sequence[float], rect: Sequence[float]) -> float:
    """Какая доля прямоугольника rect (например, всего кадра) закрыта рамкой box."""
    area = box_area(rect)
    return intersection_area(box, rect) / area if area > 0 else 0.0


def place_on_canvas(image: Image.Image, factor: int) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Кладёт изображение в центр серого холста в `factor` раз больше. Возвращает
    (холст, смещение). При factor <= 1 - само изображение. Машина на холсте
    выглядит мельче, чем на исходном кропе, - примерно как на обучающих кадрах.
    """
    if factor <= 1:
        return image, (0, 0)
    width, height = image.size
    canvas = Image.new("RGB", (width * factor, height * factor), CANVAS_GRAY)
    offset = ((width * factor - width) // 2, (height * factor - height) // 2)
    canvas.paste(image, offset)
    return canvas, offset


def shift_detection(det: Detection, dx: float, dy: float) -> Detection:
    return (det[0] - dx, det[1] - dy, det[2] - dx, det[3] - dy, det[4])


def above_threshold(dets: Sequence[Detection], conf: float) -> List[Detection]:
    """Рамки с достоверностью не ниже conf, сначала самые уверенные."""
    return sorted((d for d in dets if d[4] >= conf), key=lambda d: -d[4])


# ===========================================================================
# Варианты запуска детектора
# ===========================================================================

@dataclass(frozen=True)
class InferenceConfig:
    """Один способ запустить детектор: размер входа и масштаб кадра (холст)."""

    imgsz: Optional[int] = None  # None - как в приложении (по умолчанию модели)
    canvas: int = 1              # 1 - без холста

    @property
    def is_default(self) -> bool:
        return self.imgsz is None and self.canvas == 1

    @property
    def title(self) -> str:
        if self.is_default:
            return "как в приложении (imgsz по умолчанию, без холста)"
        parts = []
        if self.imgsz is not None:
            parts.append(f"imgsz={self.imgsz}")
        if self.canvas > 1:
            parts.append(f"кроп на холсте x{self.canvas}")
        return ", ".join(parts)

    @property
    def slug(self) -> str:
        if self.is_default:
            return "default"
        parts = []
        if self.imgsz is not None:
            parts.append(f"imgsz{self.imgsz}")
        if self.canvas > 1:
            parts.append(f"canvas{self.canvas}")
        return "_".join(parts)


def build_inference_configs(
    imgsz_values: Sequence[int], canvas_values: Sequence[int], allow_imgsz: bool, gt_mode: bool
) -> List[InferenceConfig]:
    """
    Первым всегда идёт вариант "как в приложении". Холст осмыслен только для кропов
    (в режиме с разметкой координаты разметки к холсту не относятся); imgsz
    перебирается только у .pt - у ONNX размер входа зафиксирован при экспорте.
    """
    configs = [InferenceConfig()]
    if allow_imgsz:
        configs += [InferenceConfig(imgsz=v) for v in imgsz_values]
    if not gt_mode:
        configs += [InferenceConfig(canvas=v) for v in canvas_values if v > 1]
    return configs


def detect_all(
    images: Sequence[Image.Image], predictor: Predictor, config: InferenceConfig, min_conf: float
) -> List[List[Detection]]:
    """
    Запускает детектор на всех изображениях ОДИН раз с самым низким порогом;
    остальные пороги получаются фильтрацией (greedy NMS сверху вниз по
    достоверности: рамки выше порога не зависят от того, были ли ниже него другие).
    Рамки возвращаются в координатах ИСХОДНОГО изображения (смещение холста снято).
    """
    result = []
    for image in images:
        canvas, (dx, dy) = place_on_canvas(image, config.canvas)
        dets = predictor(canvas, min_conf, config.imgsz)
        result.append([shift_detection(d, dx, dy) for d in dets])
    return result


# ===========================================================================
# Метрики: кропы
# ===========================================================================

@dataclass
class CropStats:
    images: int
    good: float       # доля кадров: лучшая рамка закрывает >= порога кадра
    fragment: float   # рамки есть, но лучшая закрывает меньше порога
    missed: float     # ни одной рамки
    mean_boxes: float
    median_coverage: Optional[float]  # медиана покрытия лучшей рамкой (среди кадров с рамками)


def compute_crop_stats(
    per_image_dets: Sequence[Sequence[Detection]],
    sizes: Sequence[Tuple[int, int]],
    conf: float,
    good_coverage: float = 0.6,
) -> CropStats:
    good = fragment = missed = 0
    box_counts, best_coverages = [], []
    for dets, (width, height) in zip(per_image_dets, sizes):
        kept = above_threshold(dets, conf)
        box_counts.append(len(kept))
        if not kept:
            missed += 1
            continue
        best = coverage(kept[0], (0, 0, width, height))  # kept[0] - самая достоверная, её и берёт приложение
        best_coverages.append(best)
        if best >= good_coverage:
            good += 1
        else:
            fragment += 1
    n = len(sizes)
    return CropStats(
        images=n,
        good=good / n,
        fragment=fragment / n,
        missed=missed / n,
        mean_boxes=sum(box_counts) / n,
        median_coverage=statistics.median(best_coverages) if best_coverages else None,
    )


# ===========================================================================
# Метрики: кадры с разметкой
# ===========================================================================

def load_yolo_labels(path: Path, width: int, height: int) -> List[Box]:
    """
    Разметка YOLO (class cx cy w h, нормированная) -> рамки в пикселях.
    Нет файла - нет объектов (так принято в YOLO-датасетах). Битые строки пропускаются.
    """
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append(((cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def match_detections(
    dets: Sequence[Detection], gts: Sequence[Box], iou_threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """
    Жадное сопоставление: достоверные рамки первыми, каждая берёт лучшую свободную
    разметку с IoU не ниже порога. Возвращает (индекс рамки, индекс разметки, IoU).
    """
    matches, used = [], set()
    order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
    for i in order:
        best_j, best_iou = None, iou_threshold
        for j, gt in enumerate(gts):
            if j in used:
                continue
            value = iou(dets[i][:4], gt)
            if value >= best_iou:
                best_j, best_iou = j, value
        if best_j is not None:
            used.add(best_j)
            matches.append((i, best_j, best_iou))
    return matches


@dataclass
class GtStats:
    images: int
    gt_total: int
    pred_total: int
    matched: int
    precision: Optional[float]
    recall: Optional[float]
    mean_iou: Optional[float]
    width_ratio: Optional[float]   # медиана ширины рамки / ширины разметки
    height_ratio: Optional[float]
    area_ratio: Optional[float]
    false_positives_per_image: float


def compute_gt_stats(
    per_image_dets: Sequence[Sequence[Detection]],
    per_image_gts: Sequence[Sequence[Box]],
    conf: float,
    iou_threshold: float = 0.5,
) -> GtStats:
    gt_total = pred_total = matched = 0
    ious, w_ratios, h_ratios, a_ratios = [], [], [], []
    for dets, gts in zip(per_image_dets, per_image_gts):
        kept = above_threshold(dets, conf)
        gt_total += len(gts)
        pred_total += len(kept)
        for i, j, value in match_detections(kept, gts, iou_threshold):
            matched += 1
            ious.append(value)
            pred, gt = kept[i], gts[j]
            gw, gh = gt[2] - gt[0], gt[3] - gt[1]
            if gw > 0 and gh > 0:
                w_ratios.append((pred[2] - pred[0]) / gw)
                h_ratios.append((pred[3] - pred[1]) / gh)
                a_ratios.append(box_area(pred) / box_area(gt))

    def median_or_none(values):
        return statistics.median(values) if values else None

    n = len(per_image_dets)
    return GtStats(
        images=n, gt_total=gt_total, pred_total=pred_total, matched=matched,
        precision=matched / pred_total if pred_total else None,
        recall=matched / gt_total if gt_total else None,
        mean_iou=sum(ious) / len(ious) if ious else None,
        width_ratio=median_or_none(w_ratios),
        height_ratio=median_or_none(h_ratios),
        area_ratio=median_or_none(a_ratios),
        false_positives_per_image=(pred_total - matched) / n if n else 0.0,
    )


def compensating_padding(width_ratio: float, height_ratio: float) -> float:
    """
    Отступ на сторону (доля размера рамки), который возвращает рамку к размеру
    разметки. Рамка w, разметка w/r: (1 + 2p) * w = w / r  =>  p = (1/r - 1) / 2.
    Ровно так отступ применяет DetectorService.crop_with_padding (на каждую сторону).
    """
    ratio = (width_ratio + height_ratio) / 2
    return 0.0 if ratio <= 0 or ratio >= 1 else (1 / ratio - 1) / 2


# ===========================================================================
# Результаты и выводы
# ===========================================================================

@dataclass
class ScenarioResult:
    config: InferenceConfig
    conf: float
    crop: Optional[CropStats] = None
    gt: Optional[GtStats] = None


def crops_verdict(results: Sequence[ScenarioResult], app_conf: float) -> str:
    """Что именно мешает детектору на кропах и что помогает."""
    baseline = next(r for r in results if r.config.is_default and r.conf == app_conf)
    base = baseline.crop
    others = [r for r in results if r is not baseline]
    best = max(others, key=lambda r: (r.crop.good, r.config.is_default), default=None)

    lines = []
    if base.good >= 0.8:
        lines.append(f"На этих изображениях детектор работает как нужно: верно в {base.good:.0%} случаев "
                     f"при пороге приложения {app_conf}.")
    elif best is not None and best.crop.good >= base.good + 0.2:
        levers = []
        if best.config.imgsz is not None:
            levers.append(f"меньший размер входа (imgsz={best.config.imgsz})")
        if best.config.canvas > 1:
            levers.append(f"меньший масштаб машины в кадре (кроп на холсте x{best.config.canvas})")
        if best.conf < app_conf:
            levers.append(f"более низкий порог достоверности ({best.conf})")
        lines.append(
            f"Как в приложении верно только {base.good:.0%} кадров "
            f"(фрагмент {base.fragment:.0%}, не найдено {base.missed:.0%}). "
            f"Лучше всего помогает: {', '.join(levers)} - верно {best.crop.good:.0%}."
        )
        if best.config.canvas > 1 or best.config.imgsz is not None:
            lines.append("Это проблема МАСШТАБА: детектор обучался на кадрах, где машины мелкие, и не справляется "
                         "с машиной во весь кадр. Варианты: запасной вариант \"искать по всему изображению\", "
                         "или до-обучение детектора с близкими планами и более сильной аугментацией масштаба.")
        elif best.conf < app_conf:
            lines.append("Это проблема ПОРОГА: уверенность детектора на таких кадрах ниже порога приложения. "
                         "Можно понизить DETECTOR_CONFIDENCE_THRESHOLD, проверив число ложных срабатываний "
                         "на обычных кадрах.")
    else:
        lines.append(
            f"Верно только {base.good:.0%} кадров (фрагмент {base.fragment:.0%}, не найдено {base.missed:.0%}), "
            "и ни порог, ни размер входа, ни масштаб не помогают заметно. "
            "Это разрыв между обучающими данными и этими изображениями: нужно до-обучение детектора."
        )
    if base.fragment >= 0.3:
        lines.append(f"Заметная доля кадров ({base.fragment:.0%}) даёт рамку на ФРАГМЕНТ машины: "
                     f"медиана покрытия кадра лучшей рамкой {base.median_coverage:.0%}.")
    return "\n".join(lines)


def gt_verdict(results: Sequence[ScenarioResult], app_conf: float) -> str:
    """Насколько детектор соответствует разметке и систематически ли занижен размер рамок."""
    baseline = next(r for r in results if r.config.is_default and r.conf == app_conf)
    g = baseline.gt
    if g.matched == 0:
        return "Ни одна рамка не сопоставилась с разметкой (IoU 0.5): проверьте, что папки кадров и разметки соответствуют друг другу."
    lines = []
    recall = g.recall or 0.0
    precision = g.precision or 0.0
    lines.append(f"При пороге приложения {app_conf}: полнота {recall:.0%}, точность {precision:.0%}, "
                 f"средний IoU совпавших {g.mean_iou:.2f}.")
    if recall < 0.7:
        lines.append(f"Пропущено {1 - recall:.0%} размеченных машин: смотрите, не слишком ли высок порог (колонки с меньшими порогами).")
    area = g.area_ratio
    if area < 0.9:
        padding = compensating_padding(g.width_ratio, g.height_ratio)
        lines.append(f"Рамки систематически МЕНЬШЕ разметки: медиана ширины {g.width_ratio:.2f}, высоты {g.height_ratio:.2f}, "
                     f"площади {area:.2f}. Чтобы вернуть размер разметки, хватило бы PADDING_FRACTION примерно {padding:.2f} "
                     f"(сейчас по умолчанию 0.075). Оценивать эффект нужно по качеству поиска, а не по совпадению с разметкой "
                     f"UA-DETRAC: для ReID важен не размер рамки, а то, какую часть машины видит модель.")
    elif area > 1.1:
        lines.append(f"Рамки систематически БОЛЬШЕ разметки (медиана площади {area:.2f}).")
    else:
        lines.append(f"Размер рамок соответствует разметке (медиана площади {area:.2f}, в пределах 10%). "
                     "Если на глаз рамки кажутся малыми, дело в стиле разметки датасета, а не в сдвиге модели: "
                     "её нужно компенсировать отступом при кадрировании.")
    return "\n".join(lines)


# ===========================================================================
# Отчёт и контактные листы
# ===========================================================================

def format_report(
    results: Sequence[ScenarioResult], verdict: str, gt_mode: bool, app_conf: float, good_coverage: float
) -> str:
    out = []
    configs = []
    for r in results:
        if r.config not in configs:
            configs.append(r.config)
    for config in configs:
        out.append(f"[{config.title}]")
        for r in (x for x in results if x.config == config):
            marker = "  <- порог приложения" if r.conf == app_conf else ""
            if gt_mode:
                g = r.gt
                fmt = lambda v, p=False: "-" if v is None else (f"{v:.0%}" if p else f"{v:.2f}")
                out.append(f"  conf {r.conf:<5} полнота {fmt(g.recall, True):>5}  точность {fmt(g.precision, True):>5}  "
                           f"IoU {fmt(g.mean_iou):>5}  размер рамки/разметки: ширина {fmt(g.width_ratio)} высота {fmt(g.height_ratio)} "
                           f"площадь {fmt(g.area_ratio)}{marker}")
            else:
                c = r.crop
                out.append(f"  conf {r.conf:<5} верно {c.good:>4.0%}  фрагмент {c.fragment:>4.0%}  не найдено {c.missed:>4.0%}  "
                           f"рамок на кадр {c.mean_boxes:.1f}{marker}")
        out.append("")
    out.append("ВЫВОД")
    out.append(verdict)
    if not gt_mode:
        out.append(f"\n(\"верно\" = лучшая по достоверности рамка закрывает не менее {good_coverage:.0%} кадра; "
                   "кроп содержит одну машину во весь кадр)")
    return "\n".join(out)


def draw_tile(
    image: Image.Image, config: InferenceConfig, dets: Sequence[Detection], app_conf: float,
    gts: Sequence[Box] = (), tile_height: int = 180,
) -> Image.Image:
    """Плитка для листа: то, что видел детектор (с холстом - холст), и рамки на нём."""
    shown, (dx, dy) = place_on_canvas(image, config.canvas)
    scale = tile_height / shown.height
    tile = shown.resize((max(1, round(shown.width * scale)), tile_height))
    draw = ImageDraw.Draw(tile)

    def to_tile(box):
        return ((box[0] + dx) * scale, (box[1] + dy) * scale, (box[2] + dx) * scale, (box[3] + dy) * scale)

    for gt in gts:
        draw.rectangle(to_tile(gt), outline=(40, 80, 230), width=2)
    for det in dets:
        color = (30, 170, 60) if det[4] >= app_conf else (240, 150, 20)
        x1, y1, x2, y2 = to_tile(det)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1 + 3, max(0, y1 + 1)), f"{det[4]:.2f}", fill=color)
    return tile


def sheet_title(config: InferenceConfig, app_conf: float, gt_mode: bool) -> str:
    """
    Подпись на листе - ТОЛЬКО ASCII. Встроенный шрифт PIL не содержит кириллицы
    (русский текст на картинке превращается в квадраты), а системные шрифты у
    всех разные. Русские пояснения остаются в консоли и в summary.json, а лист
    опознаётся по имени варианта (slug) и цветовой легенде.
    """
    title = f"{config.slug} | green: conf >= {app_conf} | orange: lower"
    return title + (" | blue: ground truth" if gt_mode else "")


def make_contact_sheet(tiles: Sequence[Image.Image], title: str, columns: int = 6, gap: int = 6) -> Image.Image:
    rows = [tiles[i:i + columns] for i in range(0, len(tiles), columns)]
    title_h = 22
    row_heights = [max(t.height for t in row) for row in rows]
    row_widths = [sum(t.width for t in row) + gap * (len(row) - 1) for row in rows]
    sheet = Image.new("RGB", (max(row_widths), title_h + sum(row_heights) + gap * len(rows)), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 4), title, fill=(16, 33, 63))
    y = title_h
    for row, height in zip(rows, row_heights):
        x = 0
        for tile in row:
            sheet.paste(tile, (x, y))
            x += tile.width + gap
        y += height + gap
    return sheet


# ===========================================================================
# Запуск
# ===========================================================================

def make_ultralytics_predictor(model_path: str, device: Optional[str], iou_threshold: float) -> Predictor:
    """
    Обёртка над YOLO, повторяющая вызов в backend/app/services/detector.py:
    PIL-изображение, conf и iou; imgsz передаётся только когда задан (иначе
    размер по умолчанию, как в приложении).
    """
    from ultralytics import YOLO  # тяжёлый импорт - только когда реально нужен детектор

    model = YOLO(model_path)

    def predict(image: Image.Image, conf: float, imgsz: Optional[int]) -> List[Detection]:
        kwargs = {"imgsz": imgsz} if imgsz is not None else {}
        results = model.predict(source=image, conf=conf, iou=iou_threshold, device=device, verbose=False, **kwargs)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for xyxy, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                detections.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(score)))
        return detections

    return predict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Диагностика детектора ТС на кропах или кадрах с разметкой",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Подробности и примеры - в документации в начале файла ml/detection/check_detector.py",
    )
    parser.add_argument("--images", required=True, help="Папка с изображениями")
    parser.add_argument("--model", default="ml/detection/runs/vehicle_detector/weights/best.pt",
                        help="Модель детектора (.pt или .onnx). Перебор imgsz возможен только для .pt")
    parser.add_argument("--gt-labels", default=None,
                        help="Папка с разметкой YOLO (<имя кадра>.txt) - включает режим сравнения с разметкой")
    parser.add_argument("--limit", type=int, default=60, help="Сколько изображений взять (случайно, по умолчанию 60)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--app-conf", type=float, default=0.4,
                        help="Порог достоверности приложения (DETECTOR_CONFIDENCE_THRESHOLD, по умолчанию 0.4)")
    parser.add_argument("--conf", type=float, nargs="*", default=[0.25, 0.1],
                        help="Дополнительные пороги для сравнения (по умолчанию 0.25 0.1)")
    parser.add_argument("--imgsz", type=int, nargs="*", default=[480, 320, 256],
                        help="Размеры входа для перебора (только для .pt)")
    parser.add_argument("--canvas", type=int, nargs="*", default=[2, 3],
                        help="Во сколько раз холст больше кропа (режим кропов)")
    parser.add_argument("--iou", type=float, default=0.5, help="Порог IoU для NMS (как у приложения)")
    parser.add_argument("--good-coverage", type=float, default=0.6,
                        help="Какую долю кадра должна закрывать рамка, чтобы считаться верной (режим кропов)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="runs/check_detector", help="Куда сохранить summary.json и контактные листы")
    parser.add_argument("--sheet-count", type=int, default=24, help="Сколько изображений на каждом листе")
    parser.add_argument("--no-sheets", action="store_true", help="Не рисовать контактные листы")
    return parser


def _fail(message: str) -> int:
    print(f"ОШИБКА: {message}", file=sys.stderr)
    return 1


def run(args: argparse.Namespace, predictor: Optional[Predictor] = None) -> int:
    """predictor - готовая функция детектора; нужна тестам, обычный запуск создаёт её из --model."""
    images_dir = Path(args.images)
    if not images_dir.is_dir():
        return _fail(f"папка с изображениями не найдена: {images_dir}")
    files = sorted(p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))
    if not files:
        return _fail(f"в {images_dir} нет изображений (jpg, jpeg, png, webp)")
    if args.limit < 1:
        return _fail("--limit должен быть не меньше 1")
    random.Random(args.seed).shuffle(files)
    files = files[: args.limit]

    gt_mode = args.gt_labels is not None
    gt_dir = Path(args.gt_labels) if gt_mode else None
    if gt_mode and not gt_dir.is_dir():
        return _fail(f"папка с разметкой не найдена: {gt_dir}")

    model_is_pt = Path(args.model).suffix.lower() == ".pt"
    if predictor is None:
        if not Path(args.model).is_file():
            return _fail(f"модель не найдена: {args.model}")
        predictor = make_ultralytics_predictor(args.model, args.device, args.iou)

    notes = []
    if not model_is_pt and args.imgsz:
        notes.append("Модель в формате ONNX: размер входа зафиксирован при экспорте, перебор imgsz пропущен "
                     "(для него нужен .pt).")
    configs = build_inference_configs(args.imgsz, args.canvas, allow_imgsz=model_is_pt, gt_mode=gt_mode)
    confs = sorted({args.app_conf, *args.conf}, reverse=True)

    images = [Image.open(p).convert("RGB") for p in files]
    sizes = [im.size for im in images]
    gts = [load_yolo_labels(gt_dir / f"{p.stem}.txt", *im.size) for p, im in zip(files, images)] if gt_mode else None

    print(f"Изображений: {len(images)} из {images_dir}" + (f", разметка: {gt_dir}" if gt_mode else ", режим кропов"))
    print(f"Модель: {args.model}; пороги: {', '.join(str(c) for c in confs)}; вариантов запуска: {len(configs)}")
    for note in notes:
        print(note)
    print()

    results: List[ScenarioResult] = []
    raw: Dict[str, List[List[Detection]]] = {}
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.title} ...", flush=True)
        per_image = detect_all(images, predictor, config, min_conf=min(confs))
        raw[config.slug] = per_image
        for conf in confs:
            if gt_mode:
                results.append(ScenarioResult(config, conf, gt=compute_gt_stats(per_image, gts, conf)))
            else:
                results.append(ScenarioResult(config, conf,
                                              crop=compute_crop_stats(per_image, sizes, conf, args.good_coverage)))

    verdict = gt_verdict(results, args.app_conf) if gt_mode else crops_verdict(results, args.app_conf)
    print()
    print(format_report(results, verdict, gt_mode, args.app_conf, args.good_coverage))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "gt" if gt_mode else "crops",
        "images": len(images), "images_dir": str(images_dir), "model": args.model,
        "app_conf": args.app_conf, "good_coverage": args.good_coverage, "notes": notes,
        "scenarios": [
            {"config": r.config.title, "slug": r.config.slug, "conf": r.conf,
             "stats": asdict(r.gt if gt_mode else r.crop)} for r in results
        ],
        "verdict": verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sheets:
        count = min(args.sheet_count, len(images))
        for config in configs:
            tiles = [draw_tile(images[i], config, raw[config.slug][i], args.app_conf,
                               gts=gts[i] if gt_mode else ()) for i in range(count)]
            sheet = make_contact_sheet(tiles, title=sheet_title(config, args.app_conf, gt_mode))
            sheet.save(out_dir / f"sheet_{config.slug}.jpg", quality=90)
    print(f"\nРезультаты сохранены в {out_dir}: summary.json" + ("" if args.no_sheets else " и контактные листы sheet_*.jpg"))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())