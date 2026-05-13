"""Grad-CAM overlays using the maintained ``grad-cam`` (``pytorch_grad_cam``) package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from galaxy_morphology.explainability.target_layers import gradcam_target_layers

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_to_rgb_image(chw: torch.Tensor) -> np.ndarray:
    """Denormalize ImageNet-normalized CHW tensor to HWC float RGB in [0, 1]."""
    x = chw.detach().cpu().float().numpy()
    for c in range(3):
        x[c] = x[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    x = np.clip(x.transpose(1, 2, 0), 0.0, 1.0)
    return x.astype(np.float32)


def explain_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    model_name: str,
    target_class_idx: int,
    *,
    out_dir: Path,
    stem: str,
) -> dict[str, Any]:
    """Run Grad-CAM, save overlay + side-by-side figure, return paths and metadata.

    ``input_tensor`` is a single image batch ``(1, 3, H, W)`` on ``device``.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    targets = [ClassifierOutputTarget(target_class_idx)]
    layers = gradcam_target_layers(model, model_name)
    cam = GradCAM(model=model, target_layers=layers)

    rgb = tensor_to_rgb_image(input_tensor[0])
    x = input_tensor.detach().clone().to(device)
    x.requires_grad_(True)
    try:
        with torch.enable_grad():
            grayscale = cam(input_tensor=x, targets=targets)
    finally:
        if hasattr(cam, "release"):
            cam.release()

    cam_map = (
        np.asarray(grayscale[0]) if isinstance(grayscale, list) else np.asarray(grayscale)
    )
    if cam_map.ndim == 3:
        cam_map = cam_map[0]
    cam_map = np.squeeze(cam_map)
    overlay = show_cam_on_image(rgb, cam_map, use_rgb=True)

    overlay_path = out_dir / f"{stem}_gradcam_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    compare_path = out_dir / f"{stem}_gradcam_compare.png"
    fig.savefig(compare_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved Grad-CAM: %s, %s", overlay_path, compare_path)
    return {
        "overlay_path": str(overlay_path),
        "compare_path": str(compare_path),
        "target_class_idx": target_class_idx,
    }
