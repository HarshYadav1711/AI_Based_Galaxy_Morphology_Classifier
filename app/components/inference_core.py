"""Image tensors and explainability (no Streamlit imports)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from galaxy_morphology.explainability.pipeline import run_explainability_for_image
from galaxy_morphology.inference.predictor import predict


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tf(image.convert("RGB")).unsqueeze(0)


def tensor_from_uploaded_bytes(data: bytes, image_size: int) -> torch.Tensor:
    with Image.open(io.BytesIO(data)) as im:
        return pil_to_tensor(im, image_size)


def predict_from_tensor(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    class_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    return predict(model, tensor, device, class_names)


def run_full_explain(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    class_names: list[str],
    model_name: str,
    out_dir: Path,
    stem: str,
    mc_dropout_samples: int,
) -> dict[str, Any]:
    return run_explainability_for_image(
        model,
        tensor,
        device,
        class_names,
        model_name,
        stem=stem,
        out_dir=out_dir,
        mc_dropout_samples=mc_dropout_samples,
    )
