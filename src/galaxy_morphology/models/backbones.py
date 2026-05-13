"""ImageNet-pretrained torchvision classifiers with modern ``weights`` APIs."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet50,
)


def build_efficientnet_b0(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with replaced classifier head."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_convnext_tiny(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """ConvNeXt Tiny with replaced linear head."""
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """ResNet50 with replaced ``fc`` layer."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
