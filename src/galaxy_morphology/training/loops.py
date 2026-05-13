"""Training and validation step functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.amp import GradScaler, autocast
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: GradScaler | None,
    use_amp: bool,
    max_grad_norm: float,
) -> tuple[float, float]:
    """Run one training epoch with optional mixed precision and gradient clipping."""
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    amp_enabled = use_amp and device.type == "cuda"

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            with autocast("cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += float(loss.detach().item())
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

    n = max(len(train_loader), 1)
    epoch_loss = running_loss / n
    epoch_acc = float(accuracy_score(all_labels, all_preds))
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    *,
    use_amp: bool,
) -> tuple[float, float, dict, np.ndarray]:
    """Validate and return loss, accuracy, sklearn report dict, confusion matrix."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    amp_enabled = use_amp and device.type == "cuda"

    for images, labels in tqdm(val_loader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if amp_enabled:
            with autocast("cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += float(loss.item())
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = max(len(val_loader), 1)
    epoch_loss = running_loss / n
    epoch_acc = float(accuracy_score(all_labels, all_preds))
    num_classes = len(class_names)
    all_class_indices = list(range(num_classes))
    report = classification_report(
        all_labels,
        all_preds,
        labels=all_class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=all_class_indices)
    return epoch_loss, epoch_acc, report, cm
