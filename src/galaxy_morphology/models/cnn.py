"""Lightweight CNN architectures for galaxy morphology classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightGalaxyCNN(nn.Module):
    """A lightweight CNN for galaxy morphology classification.

    Uses convolutional blocks, batch normalization, and global average pooling.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class EfficientGalaxyNet(nn.Module):
    """MobileNet-inspired depthwise separable CNN."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__()

        def depthwise_separable_conv(
            in_channels: int, out_channels: int, stride: int = 1
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            depthwise_separable_conv(32, 64, stride=1),
            depthwise_separable_conv(64, 128, stride=2),
            depthwise_separable_conv(128, 128, stride=1),
            depthwise_separable_conv(128, 256, stride=2),
            depthwise_separable_conv(256, 256, stride=1),
            depthwise_separable_conv(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model(
    model_name: str = "lightweight", num_classes: int = 3, pretrained: bool = False
) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_name: ``lightweight`` or ``efficient``.
        num_classes: Number of output logits.
        pretrained: Reserved for future use (custom models are not pretrained here).

    Returns:
        A trainable ``nn.Module``.
    """
    _ = pretrained  # API compatibility; no pretrained weights for custom heads.
    if model_name == "lightweight":
        return LightweightGalaxyCNN(num_classes=num_classes)
    if model_name == "efficient":
        return EfficientGalaxyNet(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
