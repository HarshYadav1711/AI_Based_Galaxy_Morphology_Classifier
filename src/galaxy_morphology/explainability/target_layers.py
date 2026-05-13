"""Select convolutional layers for Grad-CAM per architecture."""

from __future__ import annotations

import torch.nn as nn


def gradcam_target_layers(model: nn.Module, model_name: str) -> list[nn.Module]:
    """Return one or more modules suitable for ``pytorch_grad_cam.GradCAM``."""
    key = (model_name or "lightweight").lower().strip()

    if key in ("lightweight", "lightweight_multitask"):
        return [model.conv4]

    if key == "efficient":
        # Last spatial block before global pooling
        return [model.features[-2]]

    if key == "resnet50":
        return [model.layer4[-1]]

    if key in ("efficientnet_b0", "convnext_tiny"):
        return [model.features[-1]]

    # Fallback: last Conv2d in the module tree
    last_conv: nn.Conv2d | None = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError(f"No Conv2d found for Grad-CAM (model_name={model_name}).")
    return [last_conv]
