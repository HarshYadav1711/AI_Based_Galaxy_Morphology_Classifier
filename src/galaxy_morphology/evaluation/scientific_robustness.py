"""Robustness-style checks (rotations, noise, resolution) on a validation loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from galaxy_morphology.utils.model_outputs import morph_logits

logger = logging.getLogger(__name__)


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


@torch.no_grad()
def evaluate_baseline(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_multitask_batch: bool,
) -> float:
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="baseline", leave=False):
        if use_multitask_batch:
            images, labels, _, _ = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        logits = morph_logits(model(images))
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_rotation_avg(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_multitask_batch: bool,
    angles: tuple[int, ...] = (0, 90, 180, 270),
) -> float:
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="rotation_avg", leave=False):
        if use_multitask_batch:
            images, labels, _, _ = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        acc_probs = None
        for deg in angles:
            xt = images if deg == 0 else torch.rot90(images, k=deg // 90, dims=[2, 3])
            lg = morph_logits(model(xt))
            pr = F.softmax(lg, dim=1)
            acc_probs = pr if acc_probs is None else acc_probs + pr
        acc_probs = acc_probs / len(angles)  # type: ignore[operator]
        pred = acc_probs.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_gaussian_noise(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_multitask_batch: bool,
    std: float = 0.08,
) -> float:
    correct = 0
    total = 0
    for batch in tqdm(loader, desc=f"noise_{std}", leave=False):
        if use_multitask_batch:
            images, labels, _, _ = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        noise = torch.randn_like(images) * std
        logits = morph_logits(model(images + noise))
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_low_resolution(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_multitask_batch: bool,
    small_side: int = 96,
    full_side: int = 224,
) -> float:
    correct = 0
    total = 0
    for batch in tqdm(loader, desc=f"lowres_{small_side}", leave=False):
        if use_multitask_batch:
            images, labels, _, _ = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        x = F.interpolate(
            images,
            size=(small_side, small_side),
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(x, size=(full_side, full_side), mode="bilinear", align_corners=False)
        logits = morph_logits(model(x))
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


def run_scientific_suite(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    use_multitask_batch: bool,
) -> dict[str, Any]:
    """Run a small suite; returns JSON-serializable metrics."""
    model.eval()
    out: dict[str, Any] = {}
    out["accuracy_baseline"] = evaluate_baseline(
        model, val_loader, device, use_multitask_batch=use_multitask_batch
    )
    out["accuracy_rotation_avg_90"] = evaluate_rotation_avg(
        model, val_loader, device, use_multitask_batch=use_multitask_batch, angles=(0, 90, 180, 270)
    )
    out["accuracy_noise_std_0.05"] = evaluate_gaussian_noise(
        model, val_loader, device, use_multitask_batch=use_multitask_batch, std=0.05
    )
    out["accuracy_noise_std_0.12"] = evaluate_gaussian_noise(
        model, val_loader, device, use_multitask_batch=use_multitask_batch, std=0.12
    )
    out["accuracy_lowres_96"] = evaluate_low_resolution(
        model, val_loader, device, use_multitask_batch=use_multitask_batch, small_side=96
    )
    logger.info("Scientific suite: %s", out)
    return out


def write_scientific_report_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
