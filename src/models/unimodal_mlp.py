from __future__ import annotations

import torch
from torch import nn


class FeatureMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_labels: int = 7,
        dropout: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }
