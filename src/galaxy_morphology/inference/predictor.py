"""Inference: load checkpoint, preprocess, predict."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from galaxy_morphology.models.registry import build_model
from galaxy_morphology.utils.torch_io import load_checkpoint

logger = logging.getLogger(__name__)


class _ImagePathDataset(Dataset):
    """Single-channel path index for batched file inference."""

    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = paths
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        p = self.paths[idx]
        with Image.open(p) as im:
            tensor = self.tf(im.convert("RGB"))
        return tensor, p


def load_model(
    checkpoint_path: str,
    device: torch.device,
    model_name: str | None = None,
) -> tuple[torch.nn.Module, list[str]]:
    """Load weights and class names from a training checkpoint."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    class_names = list(ckpt.get("class_names", ["spiral", "elliptical", "irregular"]))
    num_classes = len(class_names)
    resolved_name = model_name or ckpt.get("model_name", "lightweight")
    # Weights are loaded from checkpoint; avoid re-downloading ImageNet backbones.
    model = build_model(str(resolved_name), num_classes, pretrained=False)
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
    """Predict on many image paths (one image per forward); failed reads include ``error``."""
    return predict_paths_batched(
        model,
        image_paths,
        device,
        class_names,
        image_size=image_size,
        batch_size=1,
    )


def predict_paths_batched(
    model: torch.nn.Module,
    image_paths: list[str],
    device: torch.device,
    class_names: list[str],
    *,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    """Batched inference over valid image paths (skips unreadable files with errors)."""
    results: list[dict[str, Any]] = []
    valid: list[str] = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im2:
                im2.load()
            valid.append(p)
        except Exception as exc:  # noqa: BLE001
            results.append({"image_path": p, "error": str(exc)})

    if not valid:
        return results

    ds = _ImagePathDataset(valid, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_paths in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            for i in range(batch_x.size(0)):
                pc = class_names[int(pred[i].item())]
                prs = {class_names[j]: float(probs[i, j].item()) for j in range(len(class_names))}
                results.append(
                    {
                        "image_path": batch_paths[i],
                        "predicted_class": pc,
                        "confidence": float(conf[i].item()),
                        "probabilities": prs,
                    }
                )
    return results


def benchmark_inference(
    model: torch.nn.Module,
    image_paths: list[str],
    device: torch.device,
    _class_names: list[str] | None = None,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    warmup_batches: int = 1,
) -> dict[str, float]:
    """Return images/sec on ``image_paths`` using batched inference."""
    valid = [p for p in image_paths if p]  # caller filters existence
    if not valid:
        return {"images_per_sec": 0.0, "num_images": 0.0}
    ds = _ImagePathDataset(valid, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    n = 0
    with torch.no_grad():
        for _ in range(warmup_batches):
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch_x, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            _ = model(batch_x)
            n += batch_x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    return {"images_per_sec": float(n / elapsed), "num_images": float(n)}
