"""Test-time augmentation (rotation averaging) for morphology logits — local, no deps."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from galaxy_morphology.utils.model_outputs import morph_logits


def predict_with_rotation_tta(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    angles: tuple[int, ...] = (0, 90, 180, 270),
) -> torch.Tensor:
    """Average softmax probabilities over rotations (CPU/GPU safe)."""
    model.eval()
    probs_stack: list[torch.Tensor] = []
    with torch.no_grad():
        for deg in angles:
            xt = x.to(device) if deg == 0 else torch.rot90(x.to(device), k=deg // 90, dims=[2, 3])
            logits = morph_logits(model(xt))
            probs_stack.append(F.softmax(logits, dim=1))
    return torch.stack(probs_stack, dim=0).mean(dim=0)
