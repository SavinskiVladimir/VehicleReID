"""
Модель для Vehicle Re-Identification: EfficientNetV2-S backbone +
embedding head + classification head.

Архитектура соответствует пайплайну проекта:

    Crop -> EfficientNetV2-S -> GAP -> Embedding head (512D)
                                          |
                                          +--> Classification head (CE)
                                          |
                                          +--> L2-norm --> Triplet loss

Используется приём "BNNeck" (широко применяется в person/vehicle ReID,
см., например, Luo et al. "Bag of Tricks and A Strong Baseline for
Deep Person Re-identification", 2019):

    - Triplet loss считается на эмбеддинге ДО BatchNorm1d.
    - Classification (ID) loss считается на эмбеддинге ПОСЛЕ BatchNorm1d.

Причина: BatchNorm выравнивает масштаб признаков, что помогает
softmax-классификации, но одновременно "сплющивает" метрическое
пространство, что вредит triplet loss. Разделение этих двух путей
после общего BatchNorm обычно даёт заметный прирост качества по
сравнению с использованием одного и того же вектора для обеих задач.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class VehicleReIDNet(nn.Module):
    """
    Args:
        num_classes: количество уникальных идентичностей в train-сплите
            (dataset.num_train_pids из VeRi776). Нужно для classification
            head.
        embedding_dim: размерность итогового эмбеддинга (по умолчанию 512,
            согласно архитектуре проекта).
        pretrained: использовать ли ImageNet-претрейн для backbone.
        dropout_p: вероятность dropout в embedding head.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)

        # У efficientnet_v2_s из torchvision структура:
        #   backbone.features   -> свёрточные блоки (то, что нам нужно)
        #   backbone.avgpool    -> глобальный average pooling (GAP)
        #   backbone.classifier -> исходная ImageNet-голова (нам не нужна)
        #
        # Забираем только feature extractor + GAP, отбрасываем classifier.
        self.backbone_features = backbone.features
        self.gap = backbone.avgpool  # nn.AdaptiveAvgPool2d(1)

        # Число каналов на выходе последнего свёрточного блока
        # efficientnet_v2_s (перед классификатором) — 1280.
        backbone_out_dim = backbone.classifier[1].in_features

        # --- Embedding head ---
        # Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear(embedding_dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim // 2),
            nn.BatchNorm1d(backbone_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(backbone_out_dim // 2, embedding_dim),
        )

        # BNNeck: отдельный BatchNorm1d после эмбеддинга, вход для
        # classification head. Bias отключён — стандартная практика
        # для BNNeck (см. Luo et al., 2019), т.к. смещение съедается
        # последующим Linear-слоем классификатора.
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)

        # --- Classification head ---
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-инициализация для новых (не претрейн) слоёв."""
        for m in self.embedding_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + GAP -> плоский вектор признаков (до embedding head)."""
        x = self.backbone_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            embedding: "сырой" эмбеддинг (до BatchNorm), используется
                для triplet loss после L2-нормализации. Также именно
                этот вектор (нормализованный) следует сохранять в
                pgvector/FAISS для инференса.
            logits: выход classification head (для CrossEntropyLoss),
                считается только в режиме обучения (self.training),
                иначе None — на инференсе классификатор не нужен.
            normalized_embedding: L2-нормализованный embedding —
                именно его нужно использовать для cosine similarity
                search и для triplet loss.
        """
        feat = self.extract_features(x)
        embedding = self.embedding_head(feat)

        normalized_embedding = F.normalize(embedding, p=2, dim=1)

        logits = None
        if self.training:
            bn_embedding = self.bottleneck(embedding)
            logits = self.classifier(bn_embedding)

        return embedding, logits, normalized_embedding


if __name__ == "__main__":
    # Быстрая проверка форм тензоров на случайном батче.
    model = VehicleReIDNet(num_classes=576, pretrained=False)
    model.train()
    dummy = torch.randn(8, 3, 224, 224)
    embedding, logits, norm_embedding = model(dummy)
    print("embedding:", embedding.shape)          # [8, 512]
    print("logits:", logits.shape)                # [8, 576]
    print("normalized_embedding:", norm_embedding.shape)  # [8, 512]

    model.eval()
    with torch.no_grad():
        _, logits_eval, norm_embedding_eval = model(dummy)
    print("logits in eval mode:", logits_eval)     # None