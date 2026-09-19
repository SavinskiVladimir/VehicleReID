"""
Тесты для losses/triplet.py.

Запуск: из папки ml/reid/  ->  pytest test_triplet.py
"""

import pytest
import torch

from losses.triplet import BatchHardTripletLoss, _pairwise_euclidean_dist


class TestPairwiseEuclideanDist:
    def test_known_distances(self):
        # Три точки на прямой: 0, 3, 7 -> расстояния 3, 7, 4
        embeddings = torch.tensor([[0.0], [3.0], [7.0]])
        dist = _pairwise_euclidean_dist(embeddings)

        assert dist.shape == (3, 3)
        assert torch.allclose(dist[0, 1], torch.tensor(3.0), atol=1e-5)
        assert torch.allclose(dist[0, 2], torch.tensor(7.0), atol=1e-5)
        assert torch.allclose(dist[1, 2], torch.tensor(4.0), atol=1e-5)

    def test_diagonal_is_zero(self):
        embeddings = torch.randn(5, 8)
        dist = _pairwise_euclidean_dist(embeddings)
        # atol чуть шире стандартного: ||a||^2+||a||^2-2*a.a должно быть
        # ровно 0 математически, но из-за catastrophic cancellation в
        # float32 иногда остаётся крошечный (~1e-3 после sqrt) остаток -
        # это особенность вычислений с плавающей точкой, а не баг функции.
        assert torch.allclose(torch.diag(dist), torch.zeros(5), atol=1e-2)

    def test_symmetric(self):
        embeddings = torch.randn(6, 4)
        dist = _pairwise_euclidean_dist(embeddings)
        assert torch.allclose(dist, dist.t(), atol=1e-5)

    def test_non_negative(self):
        embeddings = torch.randn(10, 16)
        dist = _pairwise_euclidean_dist(embeddings)
        assert (dist >= 0).all()


class TestBatchHardTripletLoss:
    def test_zero_loss_when_perfectly_separated_beyond_margin(self):
        # 2 идентичности, кластеры далеко друг от друга и плотные внутри
        # себя -> hardest_negative >> hardest_positive + margin -> loss=0
        margin = 0.3
        embeddings = torch.tensor([
            [0.0, 0.0], [0.01, 0.0], [0.0, 0.01],   # identity 0, кластер у (0,0)
            [10.0, 10.0], [10.01, 10.0], [10.0, 10.01],  # identity 1, далеко
        ])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        loss_fn = BatchHardTripletLoss(margin=margin)
        loss = loss_fn(embeddings, labels)

        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_loss_when_clusters_overlap(self):
        # Идентичности перемешаны в пространстве -> hardest negative
        # ближе, чем hardest positive -> loss > 0
        embeddings = torch.tensor([
            [0.0, 0.0], [1.0, 0.0],   # identity 0 - далеко друг от друга
            [0.05, 0.0], [0.95, 0.0],  # identity 1 - переплетены с identity 0
        ])
        labels = torch.tensor([0, 0, 1, 1])

        loss_fn = BatchHardTripletLoss(margin=0.3)
        loss = loss_fn(embeddings, labels)

        assert loss.item() > 0.0

    def test_raises_if_identity_has_no_positive_pair(self):
        # Каждая identity встречается только один раз -> нет positive
        embeddings = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])

        loss_fn = BatchHardTripletLoss(margin=0.3)
        with pytest.raises(ValueError):
            loss_fn(embeddings, labels)

    def test_soft_margin_variant_runs_without_margin(self):
        embeddings = torch.randn(6, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        loss_fn = BatchHardTripletLoss(margin=None)  # soft-margin: log(1+exp(...))
        loss = loss_fn(embeddings, labels)

        assert torch.isfinite(loss)
        assert loss.item() > 0.0  # log(1+exp(x)) > 0 всегда

    def test_loss_is_differentiable(self):
        embeddings = torch.randn(6, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        loss_fn = BatchHardTripletLoss(margin=0.3)
        loss = loss_fn(embeddings, labels)
        loss.backward()

        assert embeddings.grad is not None
        assert torch.isfinite(embeddings.grad).all()
