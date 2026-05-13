"""Model forward pass smoke test."""

from __future__ import annotations

import torch

from galaxy_morphology.models.cnn import get_model


def test_lightweight_forward() -> None:
    m = get_model("lightweight", num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 3)


def test_efficient_forward() -> None:
    m = get_model("efficient", num_classes=3)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    assert y.shape == (1, 3)
