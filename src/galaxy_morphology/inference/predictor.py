"""Inference: load checkpoint, preprocess, predict."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from galaxy_morphology.models.cnn import get_model
from galaxy_morphology.utils.torch_io import load_checkpoint

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "lightweight",
) -> tuple[torch.nn.Module, list[str]]:
    """Load weights and class names from a training checkpoint."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    class_names = list(ckpt.get("class_names", ["spiral", "elliptical", "irregular"]))
    num_classes = len(class_names)
    model = get_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names


def preprocess_image(image_path: str, image_size: int = 224) -> tuple[torch.Tensor, Image.Image]:
    """Load and tensor-normalize a single RGB image."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    class_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    """Return predicted label, confidence, and per-class probabilities."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    predicted_class = class_names[int(predicted.item())]
    confidence_score = float(confidence.item())
    all_probs = probabilities[0].cpu().numpy()
    class_probs = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}
    return predicted_class, confidence_score, class_probs


def predict_batch(
    model: torch.nn.Module,
    image_paths: list[str],
    device: torch.device,
    class_names: list[str],
    image_size: int = 224,
) -> list[dict[str, Any]]:
    """Predict on many image paths; failed reads include an ``error`` key."""
    results: list[dict[str, Any]] = []
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            predicted_class = class_names[int(predicted.item())]
            confidence_score = float(confidence.item())
            all_probs = probabilities[0].cpu().numpy()
            class_probs = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}
            results.append(
                {
                    "image_path": path,
                    "predicted_class": predicted_class,
                    "confidence": confidence_score,
                    "probabilities": class_probs,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed on %s: %s", path, exc)
            results.append({"image_path": path, "error": str(exc)})
    return results
