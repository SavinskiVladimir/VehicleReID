"""
Тесты для DetectorService.

Реальные веса YOLO не нужны и не загружаются: ultralytics.YOLO
подменяется лёгкой заглушкой (FakeYOLO), возвращающей заранее заданные
"результаты детекции" в том же формате, что и настоящая библиотека
(result.boxes -> итерируемые объекты с .xyxy/.conf). Это позволяет
проверить именно ЛОГИКУ DetectorService (парсинг боксов, сортировку по
confidence, паддинг при кадрировании) изолированно от самой модели.

Запуск: из папки backend/app/services/  ->  pytest test_detector.py
"""

import types

import pytest
import torch
from PIL import Image

import detector as detector_module
from detector import DetectorService, VehicleDetection


# ---------------------------------------------------------------------------
# Заглушка вместо ultralytics.YOLO
# ---------------------------------------------------------------------------

class FakeBoxes:
    """Имитирует result.boxes: итерируемый список объектов с .xyxy/.conf."""

    def __init__(self, boxes_xyxy, confs):
        self._boxes = []
        for xyxy, conf in zip(boxes_xyxy, confs):
            box = types.SimpleNamespace()
            box.xyxy = torch.tensor([xyxy], dtype=torch.float32)
            box.conf = torch.tensor([conf], dtype=torch.float32)
            self._boxes.append(box)

    def __iter__(self):
        return iter(self._boxes)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes  # None или FakeBoxes


class FakeYOLO:
    """Заглушка вместо ultralytics.YOLO - не загружает никаких весов."""

    next_results = []
    last_predict_kwargs = None

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, **kwargs):
        FakeYOLO.last_predict_kwargs = kwargs
        return FakeYOLO.next_results


@pytest.fixture(autouse=True)
def patch_yolo(monkeypatch):
    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    FakeYOLO.next_results = []
    FakeYOLO.last_predict_kwargs = None
    yield


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestDetectorServiceInit:
    def test_stores_config(self):
        service = DetectorService(
            "fake_path.onnx",
            confidence_threshold=0.5,
            iou_threshold=0.6,
            device="cpu",
            padding_fraction=0.1,
        )
        assert service.confidence_threshold == 0.5
        assert service.iou_threshold == 0.6
        assert service.device == "cpu"
        assert service.padding_fraction == 0.1
        assert isinstance(service.model, FakeYOLO)

    def test_default_thresholds(self):
        service = DetectorService("fake.onnx")
        assert service.confidence_threshold == 0.4
        assert service.iou_threshold == 0.5
        assert service.padding_fraction == 0.075


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------

class TestDetect:
    def test_no_detections_returns_empty_list(self):
        FakeYOLO.next_results = [FakeResult(boxes=None)]
        service = DetectorService("fake.onnx")
        detections = service.detect(Image.new("RGB", (100, 100)))
        assert detections == []

    def test_parses_boxes_into_vehicle_detection(self):
        FakeYOLO.next_results = [
            FakeResult(boxes=FakeBoxes(
                boxes_xyxy=[[10, 20, 50, 60], [0, 0, 30, 30]],
                confs=[0.9, 0.5],
            ))
        ]
        service = DetectorService("fake.onnx")
        detections = service.detect(Image.new("RGB", (100, 100)))

        assert len(detections) == 2
        assert all(isinstance(d, VehicleDetection) for d in detections)

    def test_sorted_by_confidence_descending(self):
        FakeYOLO.next_results = [
            FakeResult(boxes=FakeBoxes(
                boxes_xyxy=[[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]],
                confs=[0.2, 0.9, 0.5],
            ))
        ]
        service = DetectorService("fake.onnx")
        detections = service.detect(Image.new("RGB", (100, 100)))

        confidences = [d.confidence for d in detections]
        assert confidences == sorted(confidences, reverse=True)

    def test_bbox_coordinates_are_rounded_ints(self):
        FakeYOLO.next_results = [
            FakeResult(boxes=FakeBoxes(boxes_xyxy=[[10.4, 20.6, 50.5, 60.49]], confs=[0.9]))
        ]
        service = DetectorService("fake.onnx")
        detections = service.detect(Image.new("RGB", (100, 100)))

        assert detections[0].bbox == (10, 21, 50, 60)  # round-half-to-even/стандартное round()

    def test_passes_thresholds_and_device_to_predict(self):
        FakeYOLO.next_results = [FakeResult(boxes=None)]
        service = DetectorService(
            "fake.onnx", confidence_threshold=0.7, iou_threshold=0.4, device="cuda"
        )
        service.detect(Image.new("RGB", (100, 100)))

        kwargs = FakeYOLO.last_predict_kwargs
        assert kwargs["conf"] == 0.7
        assert kwargs["iou"] == 0.4
        assert kwargs["device"] == "cuda"

    def test_collects_boxes_from_multiple_results(self):
        FakeYOLO.next_results = [
            FakeResult(boxes=FakeBoxes([[0, 0, 10, 10]], [0.9])),
            FakeResult(boxes=FakeBoxes([[0, 0, 20, 20]], [0.8])),
        ]
        service = DetectorService("fake.onnx")
        detections = service.detect(Image.new("RGB", (100, 100)))
        assert len(detections) == 2


# ---------------------------------------------------------------------------
# crop_with_padding()
# ---------------------------------------------------------------------------

class TestCropWithPadding:
    def test_no_padding_matches_bbox_exactly(self):
        service = DetectorService("fake.onnx", padding_fraction=0.0)
        image = Image.new("RGB", (200, 200))
        crop = service.crop_with_padding(image, (50, 50, 100, 120))
        assert crop.size == (50, 70)

    def test_padding_expands_crop_symmetrically(self):
        service = DetectorService("fake.onnx", padding_fraction=0.1)
        image = Image.new("RGB", (200, 200))
        # box_w=50, box_h=100 -> pad_x=5, pad_y=10 -> итог (60, 120)
        crop = service.crop_with_padding(image, (50, 50, 100, 150))
        assert crop.size == (60, 120)

    def test_padding_clamped_to_image_bounds(self):
        service = DetectorService("fake.onnx", padding_fraction=0.5)
        image = Image.new("RGB", (100, 100))
        crop = service.crop_with_padding(image, (0, 0, 20, 20))
        assert crop.size[0] <= 100
        assert crop.size[1] <= 100

    def test_bbox_at_image_edge_does_not_crash(self):
        service = DetectorService("fake.onnx", padding_fraction=0.075)
        image = Image.new("RGB", (50, 50))
        crop = service.crop_with_padding(image, (40, 40, 50, 50))
        assert crop.size[0] > 0
        assert crop.size[1] > 0

    def test_crop_is_at_least_as_large_as_bbox(self):
        service = DetectorService("fake.onnx", padding_fraction=0.1)
        image = Image.new("RGB", (300, 300))
        bbox = (50, 50, 150, 200)
        crop = service.crop_with_padding(image, bbox)
        box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        assert crop.size[0] >= box_w
        assert crop.size[1] >= box_h


# ---------------------------------------------------------------------------
# detect_and_crop()
# ---------------------------------------------------------------------------

class TestDetectAndCrop:
    def test_combines_detect_and_crop_for_each_detection(self):
        FakeYOLO.next_results = [
            FakeResult(boxes=FakeBoxes([[10, 10, 50, 50], [60, 60, 90, 90]], [0.9, 0.7]))
        ]
        service = DetectorService("fake.onnx")
        pairs = service.detect_and_crop(Image.new("RGB", (100, 100)))

        assert len(pairs) == 2
        for det, crop in pairs:
            assert isinstance(det, VehicleDetection)
            assert isinstance(crop, Image.Image)
            assert crop.size[0] > 0 and crop.size[1] > 0

    def test_empty_list_when_nothing_detected(self):
        FakeYOLO.next_results = [FakeResult(boxes=None)]
        service = DetectorService("fake.onnx")
        pairs = service.detect_and_crop(Image.new("RGB", (100, 100)))
        assert pairs == []

    def test_order_matches_detect_confidence_order(self):
        FakeYOLO.next_results = [
            FakeResult(boxes=FakeBoxes(
                boxes_xyxy=[[0, 0, 10, 10], [20, 20, 40, 40]],
                confs=[0.3, 0.95],
            ))
        ]
        service = DetectorService("fake.onnx")
        pairs = service.detect_and_crop(Image.new("RGB", (100, 100)))
        confidences = [det.confidence for det, _ in pairs]
        assert confidences == sorted(confidences, reverse=True)
