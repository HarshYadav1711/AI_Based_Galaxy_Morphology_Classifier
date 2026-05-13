"""Wrap multi-task models so Grad-CAM sees a single morphology logit tensor."""

from __future__ import annotations

import torch.nn as nn

from galaxy_morphology.utils.model_outputs import morph_logits


class MorphologyOnlyWrapper(nn.Module):
    """Thin wrapper: ``forward`` returns morphology logits for CAM / softmax."""

    def __init__(self, multitask_model: nn.Module) -> None:
        super().__init__()
        self.backbone = multitask_model

    def forward(self, x):
        return morph_logits(self.backbone(x))
