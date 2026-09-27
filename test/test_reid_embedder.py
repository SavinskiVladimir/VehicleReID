"""
Тесты для backend/app/services/reid_embedder.py.

onnxruntime.InferenceSession подменяется лёгкой заглушкой - реальная
ONNX-модель не нужна для проверки логики самого сервиса (распаковка
батча/одиночного результата, embed_as_list). Полноценная сверка с
настоящим экспортированным .onnx проводилась вручную при разработке
(см. журнал разработки) - здесь фиксируется контракт класса.

Запуск: из корня проекта  ->  pytest tests/test_reid_embedder.py
"""

import numpy as np
import pytest

import backend.app.services.reid_embedder as reid_embedder_module
from backend.app.services.reid_embedder import ReIDEmbedderService


class FakeInferenceSession:
    """Заглушка вместо onnxruntime.InferenceSession."""

    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.providers = providers

    def get_inputs(self):
        return [type("Node", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("Node", (), {"name": "embedding"})()]

    def run(self, output_names, input_feed):
        batch = input_feed["input"]
        n = batch.shape[0]
        # Возвращаем детерминированный, но различающийся по батчу
        # "эмбеддинг" фиксированной размерности - достаточно для
        # проверки распаковки формы, не для проверки качества модели.
        fake_embedding = np.tile(np.arange(8, dtype=np.float32), (n, 1)) + np.arange(n).reshape(-1, 1)
        return [fake_embedding]


@pytest.fixture(autouse=True)
def patch_onnxruntime(monkeypatch):
    monkeypatch.setattr(reid_embedder_module.ort, "InferenceSession", FakeInferenceSession)
    yield


class TestReIDEmbedderServiceInit:
    def test_default_device_uses_cpu_provider(self):
        service = ReIDEmbedderService("fake.onnx")
        assert service.session.providers == ["CPUExecutionProvider"]

    def test_cuda_device_includes_cuda_provider_with_cpu_fallback(self):
        service = ReIDEmbedderService("fake.onnx", device="cuda")
        assert service.session.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


class TestEmbed:
    def test_single_image_returns_1d_array(self):
        service = ReIDEmbedderService("fake.onnx")
        batch = np.zeros((1, 3, 64, 64), dtype=np.float32)
        result = service.embed(batch)
        assert result.ndim == 1
        assert result.shape == (8,)

    def test_batch_returns_2d_array(self):
        service = ReIDEmbedderService("fake.onnx")
        batch = np.zeros((4, 3, 64, 64), dtype=np.float32)
        result = service.embed(batch)
        assert result.shape == (4, 8)

    def test_different_batch_items_get_different_embeddings(self):
        service = ReIDEmbedderService("fake.onnx")
        batch = np.zeros((3, 3, 64, 64), dtype=np.float32)
        result = service.embed(batch)
        assert not np.allclose(result[0], result[1])
        assert not np.allclose(result[1], result[2])


class TestEmbedAsList:
    def test_returns_plain_python_list(self):
        service = ReIDEmbedderService("fake.onnx")
        batch = np.zeros((1, 3, 64, 64), dtype=np.float32)
        result = service.embed_as_list(batch)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        assert len(result) == 8

    def test_raises_for_batch_input(self):
        service = ReIDEmbedderService("fake.onnx")
        batch = np.zeros((2, 3, 64, 64), dtype=np.float32)  # batch > 1
        with pytest.raises(ValueError):
            service.embed_as_list(batch)
