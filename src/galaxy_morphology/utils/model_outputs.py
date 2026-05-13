"""Normalize model outputs (single-task tensor vs multi-task dict)."""

from __future__ import annotations

import torch


def morph_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Return morphology logits ``(N, C)`` whether the model is single- or multi-task."""
    if isinstance(output, dict):
        return output["morph"]
    return output
