"""Training and validation step functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from galaxy_morphology.evaluation.classification_metrics import compute_extended_metrics
from galaxy_morphology.training.mixup import mixup_criterion, mixup_data

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
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> tuple[float, float]:
    """Run one training epoch with optional AMP, gradient clipping, and mixup."""
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

        if use_mixup and mixup_alpha > 0:
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device=device)
            if amp_enabled and scaler is not None:
                with autocast("cuda", enabled=True):
                    outputs = model(mixed_x)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(mixed_x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y_a.detach().cpu().numpy().tolist())
        elif amp_enabled and scaler is not None:
            with autocast("cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        running_loss += float(loss.detach().item())
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
) -> tuple[float, float, dict, np.ndarray, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Validate; return metrics plus arrays for ROC/PR plots."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
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
        probs = F.softmax(outputs.float(), dim=1)
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    n = max(len(val_loader), 1)
    epoch_loss = running_loss / n
    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_preds, dtype=int)
    y_score = np.vstack(all_probs) if all_probs else np.zeros((0, len(class_names)))
    epoch_acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    num_classes = len(class_names)
    all_class_indices = list(range(num_classes))
    report = classification_report(
        y_true,
        y_pred,
        labels=all_class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_class_indices)
    extended = compute_extended_metrics(y_true, y_pred, y_score, class_names)
    return epoch_loss, epoch_acc, report, cm, extended, y_true, y_pred, y_score
