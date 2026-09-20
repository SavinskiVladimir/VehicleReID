"""
Сервис детекции транспортных средств.

Реализует функции 2 и 3 из Таблицы 1.3 ТЗ:
    2. Детекция ТС на изображении -> координаты bounding box + confidence.
    3. Выделение области изображения с учётом отступа -> кроп ТС.

Согласно ТЗ (п. 1.3.2.1), backend взаимодействует с ML-модулем через
отдельный сервисный интерфейс, обеспечивающий независимую замену
ML-модуля без изменения остальной части системы - этот класс и есть
такая точка входа для детекции. search_service.py / routes.py должны
зависеть только от публичного интерфейса DetectorService, а не от
деталей ultralytics/YOLO.

Модель может быть в любом формате, который понимает ultralytics.YOLO:
.pt (обучающий чекпоинт) или .onnx (экспортированный, см.
ml/detection/export_detector.py) - выбор делается путём подсуванием
соответствующего пути, код инференса не меняется. Согласно ТЗ
(п. 1.3.2.3), сервис должен уметь работать как на GPU, так и на CPU -
это обеспечивается параметром device и автоопределением по умолчанию.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
from ultralytics import YOLO


@dataclass
class VehicleDetection:
    """Один обнаруженный объект класса vehicle."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) в пикселях исходного изображения
    confidence: float


class DetectorService:
    """
    Сервис детекции ТС на изображении.

    Пример использования (в search_service.py или routes.py):

        detector = DetectorService(model_path="ml/detection/exported/vehicle_detector.onnx")
        detections = detector.detect(image)
        if not detections:
            # см. Требования к надёжности (ТЗ п.1.3.2.4): корректное
            # сообщение об ошибке, без падения серверной части
            raise VehicleNotFoundError(...)
        crop = detector.crop_with_padding(image, detections[0].bbox)

    Требования к быстродействию (ТЗ п.1.3.2.2): время детекции входит в
    общий бюджет обработки запроса (2000 мс CPU / 500 мс GPU) - при
    развёртывании стоит проверить реальную latency конкретной модели
    (yolo11s против yolo11m) на целевом оборудовании и при необходимости
    выбрать более лёгкий вариант.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = None,
        padding_fraction: float = 0.075,
    ):
        """
        Args:
            model_path: путь к .pt или .onnx файлу модели детектора.
            confidence_threshold: минимальная достоверность детекции,
                ниже которой объект отбрасывается (ТЗ функция 2 требует
                "оценку достоверности детекции" - порог именно здесь).
            iou_threshold: порог IoU для Non-Maximum Suppression
                (подавление дублирующихся боксов одного и того же ТС).
            device: 'cuda'/'cpu'/None (автовыбор ultralytics: GPU, если
                доступен, иначе CPU - соответствует требованию ТЗ о
                работе системы в обоих режимах).
            padding_fraction: доля отступа от каждой стороны bounding
                box при кадрировании (0.05-0.10 согласно архитектуре
                проекта: "Bounding box + padding 5-10%"). Отступ нужен,
                т.к. YOLO иногда слегка "обрезает" боксы по краю
                кузова, и без запаса ReID-модель может недополучить
                значимые детали (фары, бампер).
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.padding_fraction = padding_fraction

    def detect(self, image: Image.Image) -> List[VehicleDetection]:
        """
        Находит все транспортные средства на изображении.

        Возвращает список VehicleDetection, отсортированный по убыванию
        confidence. Пустой список, если ничего не найдено (ТЗ п.1.3.2.4:
        это штатная ситуация, а не ошибка - обработка отсутствия
        результата остаётся на стороне вызывающего кода/бизнес-логики).

        Согласно ТЗ ("при наличии на изображении нескольких ТС система
        должна предоставлять пользователю возможность выбора") -
        намеренно возвращаются ВСЕ найденные объекты выше порога
        достоверности, а не только один "лучший", чтобы пользователь
        мог выбрать нужный на клиенте.
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                detections.append(
                    VehicleDetection(
                        bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                        confidence=confidence,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def crop_with_padding(
        self, image: Image.Image, bbox: Tuple[int, int, int, int]
    ) -> Image.Image:
        """
        Кадрирует изображение по bounding box с отступом padding_fraction
        от каждой стороны, ограничивая результат границами исходного
        изображения (отступ не выходит за пределы кадра).
        """
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = int(round(box_w * self.padding_fraction))
        pad_y = int(round(box_h * self.padding_fraction))

        img_w, img_h = image.size
        padded_x1 = max(0, x1 - pad_x)
        padded_y1 = max(0, y1 - pad_y)
        padded_x2 = min(img_w, x2 + pad_x)
        padded_y2 = min(img_h, y2 + pad_y)

        return image.crop((padded_x1, padded_y1, padded_x2, padded_y2))

    def detect_and_crop(
        self, image: Image.Image
    ) -> List[Tuple[VehicleDetection, Image.Image]]:
        """
        Удобный метод, объединяющий функции 2 и 3 из ТЗ за один вызов:
        детекция + кадрирование с отступом для каждого найденного ТС.
        Возвращает список пар (детекция, кроп), в порядке убывания
        confidence - соответствует последовательности "Кроп ТС" на
        архитектурной схеме проекта.
        """
        detections = self.detect(image)
        return [(det, self.crop_with_padding(image, det.bbox)) for det in detections]
