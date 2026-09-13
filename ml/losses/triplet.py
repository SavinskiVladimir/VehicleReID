"""
Batch-Hard Triplet Loss (Hermans et al., "In Defense of the Triplet
Loss for Person Re-Identification", 2017).

Идея: вместо того чтобы заранее формировать конкретные тройки
(anchor, positive, negative), для каждого anchor'а в батче на лету
выбирается САМЫЙ ТРУДНЫЙ позитив (наиболее удалённый образец той же
идентичности) и САМЫЙ ТРУДНЫЙ негатив (наиболее близкий образец другой
идентичности). Это возможно только если батч построен по схеме P x K
(P идентичностей, K изображений каждая) — именно для этого нужен
PKSampler из ml/samplers/pk_sampler.py.

Loss для одного anchor'а i:
    L_i = max(0, d(a_i, hardest_positive_i) - d(a_i, hardest_negative_i) + margin)

Итоговый loss — среднее по всем anchor'ам в батче.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _pairwise_euclidean_dist(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Матрица попарных евклидовых расстояний между эмбеддингами батча.

    Args:
        embeddings: тензор [N, D].

    Returns:
        dist: тензор [N, N], dist[i, j] = ||e_i - e_j||_2.

    Реализовано через тождество:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
    вместо прямого попарного вычитания — это заметно быстрее на GPU
    и не требует O(N^2 * D) промежуточной памяти.
    """
    # sq_norms[i] = ||e_i||^2
    sq_norms = (embeddings ** 2).sum(dim=1, keepdim=True)  # [N, 1]

    # dist_sq[i, j] = ||e_i||^2 + ||e_j||^2 - 2 * e_i . e_j
    dist_sq = sq_norms + sq_norms.t() - 2.0 * torch.mm(embeddings, embeddings.t())

    # Из-за ошибок округления могут появиться небольшие отрицательные
    # значения на диагонали (расстояние точки до самой себя) —
    # обрезаем снизу нулём перед взятием корня.
    dist_sq = dist_sq.clamp(min=0.0)

    # Для численной стабильности градиента sqrt в нуле (d/dx sqrt(x) -> inf
    # при x->0) добавляем небольшой эпсилон только там, где значение
    # действительно нулевое.
    eps_mask = (dist_sq == 0.0).float()
    dist = torch.sqrt(dist_sq + eps_mask * 1e-12)

    return dist


class BatchHardTripletLoss(nn.Module):
    """
    Args:
        margin: отступ alpha в формуле triplet loss. Если None,
            используется "soft margin" вариант: log(1 + exp(d_ap - d_an))
            вместо max(0, d_ap - d_an + margin) — это устраняет
            необходимость подбирать margin вручную (см. Hermans et al.,
            раздел про soft-margin), но может сходиться медленнее на
            старте обучения.
    """

    def __init__(self, margin: Optional[float] = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [N, D] — эмбеддинги батча (рекомендуется подавать
                L2-нормализованные эмбеддинги, т.е. normalized_embedding
                из VehicleReIDNet.forward).
            labels: [N] — идентичности (pid) для каждого эмбеддинга.
                Батч должен быть сформирован PKSampler'ом, чтобы у
                каждой идентичности было >= 2 образцов (иначе для неё
                не найдётся positive).

        Returns:
            Скалярный тензор — усреднённый по батчу loss.
        """
        dist_mat = _pairwise_euclidean_dist(embeddings)  # [N, N]
        n = embeddings.size(0)

        # is_pos[i, j] = True, если j - позитив для i (тот же pid, i != j)
        # is_neg[i, j] = True, если j - негатив для i (другой pid)
        labels = labels.view(-1, 1)
        same_id = labels.eq(labels.t())  # [N, N], True если labels совпадают

        # Убираем диагональ (i == j) из позитивов - сам с собой не сравниваем.
        identity_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        is_pos = same_id & (~identity_mask)
        is_neg = ~same_id

        # Для hardest positive нам нужен МАКСИМУМ расстояния среди
        # позитивов -> заменяем расстояния до непозитивов на -inf,
        # чтобы они не попали в max.
        dist_ap = dist_mat.masked_fill(~is_pos, float("-inf")).max(dim=1).values

        # Для hardest negative нужен МИНИМУМ расстояния среди
        # негативов -> заменяем расстояния до непозитивов на +inf.
        dist_an = dist_mat.masked_fill(~is_neg, float("inf")).min(dim=1).values

        if torch.isinf(dist_ap).any():
            raise ValueError(
                "У некоторых идентичностей в батче нет позитивной пары "
                "(только 1 изображение на pid). Проверьте, что PKSampler "
                "настроен с num_instances >= 2."
            )

        if self.margin is not None:
            # Классическая формула (уравнение 11 в обзорной статье):
            # max(0, d_ap - d_an + margin)
            loss = torch.clamp(dist_ap - dist_an + self.margin, min=0.0)
        else:
            # Soft-margin вариант: log(1 + exp(d_ap - d_an))
            loss = torch.log1p(torch.exp(dist_ap - dist_an))

        return loss.mean()


if __name__ == "__main__":
    # Быстрая проверка на синтетическом батче P=4 идентичности x K=4.
    torch.manual_seed(0)
    P, K, D = 4, 4, 512
    embeddings = torch.randn(P * K, D)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # L2-норм.
    labels = torch.arange(P).repeat_interleave(K)

    criterion = BatchHardTripletLoss(margin=0.3)
    loss = criterion(embeddings, labels)
    print("Batch-hard triplet loss на случайном батче:", loss.item())