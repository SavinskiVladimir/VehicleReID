"""
Тесты для models/efficientnetv2_reid.py.

Не скачивают ImageNet-веса (pretrained=False) - тесты про архитектуру и
формы тензоров, а не про качество претрейна.

Запуск: из папки ml/reid/  ->  pytest test_efficientnetv2_reid.py
"""

import pytest
import torch

from models.efficientnetv2_reid import VehicleReIDNet


@pytest.fixture(scope="module")
def small_model():
    # embedding_dim маленький и pretrained=False - тест лёгкий и быстрый.
    return VehicleReIDNet(num_classes=10, embedding_dim=16, pretrained=False)


class TestVehicleReIDNetTrainMode:
    def test_output_shapes(self, small_model):
        small_model.train()
        batch = torch.randn(4, 3, 64, 64)

        embedding, logits, normalized_embedding = small_model(batch)

        assert embedding.shape == (4, 16)
        assert logits.shape == (4, 10)
        assert normalized_embedding.shape == (4, 16)

    def test_normalized_embedding_has_unit_norm(self, small_model):
        small_model.train()
        batch = torch.randn(4, 3, 64, 64)
        _, _, normalized_embedding = small_model(batch)

        norms = normalized_embedding.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_logits_are_not_none_in_train_mode(self, small_model):
        small_model.train()
        batch = torch.randn(2, 3, 64, 64)
        _, logits, _ = small_model(batch)
        assert logits is not None


class TestVehicleReIDNetEvalMode:
    def test_logits_are_none_in_eval_mode(self, small_model):
        small_model.eval()
        batch = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            _, logits, _ = small_model(batch)
        assert logits is None

    def test_embedding_still_produced_in_eval_mode(self, small_model):
        small_model.eval()
        batch = torch.randn(3, 3, 64, 64)
        with torch.no_grad():
            embedding, _, normalized_embedding = small_model(batch)
        assert embedding.shape == (3, 16)
        assert normalized_embedding.shape == (3, 16)

    def test_eval_mode_is_deterministic(self, small_model):
        # В eval-режиме dropout/batchnorm-статистики фиксированы ->
        # одинаковый вход должен давать одинаковый выход.
        small_model.eval()
        batch = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            _, _, emb1 = small_model(batch)
            _, _, emb2 = small_model(batch)
        assert torch.allclose(emb1, emb2)


class TestVehicleReIDNetConfig:
    def test_different_embedding_dim(self):
        model = VehicleReIDNet(num_classes=5, embedding_dim=32, pretrained=False)
        model.eval()
        batch = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            embedding, _, _ = model(batch)
        assert embedding.shape == (2, 32)

    def test_extract_features_returns_flat_vector(self, small_model):
        batch = torch.randn(2, 3, 64, 64)
        feat = small_model.extract_features(batch)
        assert feat.dim() == 2
        assert feat.shape[0] == 2
