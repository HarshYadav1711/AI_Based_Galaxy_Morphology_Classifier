"""Model registry smoke tests (no pretrained weight download)."""

from __future__ import annotations

import torch

from galaxy_morphology.models.registry import build_model, list_model_names


def test_list_models_includes_transfer_backbones() -> None:
    names = list_model_names()
    for n in ("lightweight", "efficient", "efficientnet_b0", "convnext_tiny", "resnet50"):
        assert n in names


def test_build_each_model_forward_cpu() -> None:
    x = torch.randn(1, 3, 224, 224)
    for name in list_model_names():
        m = build_model(name, num_classes=3, pretrained=False)
        y = m(x)
        assert y.shape == (1, 3)
