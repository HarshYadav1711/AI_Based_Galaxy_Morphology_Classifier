"""Centralized model factory (registry) for all supported architectures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from galaxy_morphology.models.backbones import (
    build_convnext_tiny,
    build_efficientnet_b0,
    build_resnet50,
)
from galaxy_morphology.models.cnn import EfficientGalaxyNet, LightweightGalaxyCNN
from galaxy_morphology.models.multitask_cnn import LightweightMultiTaskGalaxyCNN

# Registry: name -> builder(num_classes, pretrained, **kwargs)
_ModelBuilder = Callable[..., nn.Module]


def _build_lightweight(num_classes: int, pretrained: bool = False, **_: Any) -> nn.Module:
    _ = pretrained
    return LightweightGalaxyCNN(num_classes=num_classes)


def _build_lightweight_multitask(num_classes: int, pretrained: bool = False, **_: Any) -> nn.Module:
    _ = pretrained
    return LightweightMultiTaskGalaxyCNN(num_classes=num_classes)


def _build_efficient_galaxy(num_classes: int, pretrained: bool = False, **_: Any) -> nn.Module:
    _ = pretrained
    return EfficientGalaxyNet(num_classes=num_classes)


def _build_efficientnet_b0(num_classes: int, pretrained: bool = True, **_: Any) -> nn.Module:
    return build_efficientnet_b0(num_classes, pretrained=pretrained)


def _build_convnext_tiny(num_classes: int, pretrained: bool = True, **_: Any) -> nn.Module:
    return build_convnext_tiny(num_classes, pretrained=pretrained)


def _build_resnet50(num_classes: int, pretrained: bool = True, **_: Any) -> nn.Module:
    return build_resnet50(num_classes, pretrained=pretrained)


MODEL_REGISTRY: dict[str, _ModelBuilder] = {
    "lightweight": _build_lightweight,
    "lightweight_multitask": _build_lightweight_multitask,
    "efficient": _build_efficient_galaxy,
    "efficientnet_b0": _build_efficientnet_b0,
    "convnext_tiny": _build_convnext_tiny,
    "resnet50": _build_resnet50,
}


def list_model_names() -> list[str]:
    """Return sorted supported ``model.name`` strings."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(
    name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Construct a model by registry name.

    Args:
        name: One of :func:`list_model_names`.
        num_classes: Number of logits / classes.
        pretrained: For torchvision backbones, load ImageNet weights when True.
            Custom CNNs ignore this flag.

    Returns:
        Trainable module moved by caller to device.
    """
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(list_model_names())}")
    return MODEL_REGISTRY[key](num_classes, pretrained=pretrained, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
