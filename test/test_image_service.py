"""
Тесты для backend/app/services/image_service.py.

Запуск: из корня проекта  ->  pytest tests/test_image_service.py
"""

import numpy as np
import pytest
from PIL import Image

from backend.app.services.image_service import ImageService


class TestPreprocess:
    def test_output_shape(self):
        service = ImageService(image_size=(224, 224))
        img = Image.new("RGB", (100, 80), (128, 64, 32))
        result = service.preprocess(img)
        assert result.shape == (1, 3, 224, 224)

    def test_output_dtype_is_float32(self):
        service = ImageService(image_size=(64, 64))
        img = Image.new("RGB", (50, 50), (10, 20, 30))
        result = service.preprocess(img)
        assert result.dtype == np.float32

    def test_converts_non_rgb_mode(self):
        service = ImageService(image_size=(32, 32))
        img = Image.new("L", (40, 40), 128)  # grayscale
        result = service.preprocess(img)
        assert result.shape == (1, 3, 32, 32)  # 3 канала, несмотря на grayscale вход

    def test_different_sizes_produce_configured_output_shape(self):
        service = ImageService(image_size=(128, 96))  # H=128, W=96
        img = Image.new("RGB", (300, 200), (0, 0, 0))
        result = service.preprocess(img)
        assert result.shape == (1, 3, 128, 96)

    def test_matches_torchvision_reference_normalization(self):
        # Кросс-проверка с эталонной реализацией через torchvision,
        # использованной при обучении (ml/reid/train.py).
        torch = pytest.importorskip("torch")
        from torchvision import transforms

        np.random.seed(1)
        img_array = (np.random.rand(150, 100, 3) * 255).astype("uint8")
        img = Image.fromarray(img_array)

        service = ImageService(image_size=(224, 224))
        ours = service.preprocess(img)

        ref_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        reference = ref_transform(img).unsqueeze(0).numpy()

        assert np.max(np.abs(ours - reference)) < 1e-4


class TestPreprocessBatch:
    def test_batch_shape(self):
        service = ImageService(image_size=(64, 64))
        images = [
            Image.new("RGB", (50, 50), (255, 0, 0)),
            Image.new("RGB", (50, 50), (0, 255, 0)),
            Image.new("RGB", (50, 50), (0, 0, 255)),
        ]
        result = service.preprocess_batch(images)
        assert result.shape == (3, 3, 64, 64)

    def test_batch_matches_individual_preprocessing(self):
        service = ImageService(image_size=(48, 48))
        img1 = Image.new("RGB", (60, 60), (100, 100, 100))
        img2 = Image.new("RGB", (60, 60), (200, 50, 10))

        batch_result = service.preprocess_batch([img1, img2])
        individual_1 = service.preprocess(img1)
        individual_2 = service.preprocess(img2)

        assert np.allclose(batch_result[0:1], individual_1)
        assert np.allclose(batch_result[1:2], individual_2)

    def test_empty_batch_returns_correctly_shaped_empty_array(self):
        service = ImageService(image_size=(32, 32))
        result = service.preprocess_batch([])
        assert result.shape == (0, 3, 32, 32)
