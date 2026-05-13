"""Multi-task head on the lightweight backbone (morphology + auxiliary science tasks)."""

from __future__ import annotations

import torch
import torch.nn as nn

from galaxy_morphology.models.cnn import LightweightGalaxyCNN


class LightweightMultiTaskGalaxyCNN(LightweightGalaxyCNN):
    """Morphology logits plus merger / bar / asymmetry heads on shared embeddings.

    Auxiliary targets are optional at training time (per-sample mask). ``asymmetry`` is a
    scalar in ``[0, 1]`` (e.g. human or catalog asymmetry score); the head uses sigmoid.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout)
        d = 128
        self.merger_head = nn.Linear(d, 1)
        self.bar_head = nn.Linear(d, 1)
        self.asym_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.forward_features(x)
        return {
            "morph": self.fc2(z),
            "merger": self.merger_head(z).squeeze(-1),
            "bar": self.bar_head(z).squeeze(-1),
            "asymmetry": torch.sigmoid(self.asym_head(z)).squeeze(-1),
        }
