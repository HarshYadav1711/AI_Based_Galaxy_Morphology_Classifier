"""Training / validation for multi-task morphology + optional auxiliary heads."""

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
from galaxy_morphology.utils.model_outputs import morph_logits

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


def multitask_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    aux: torch.Tensor,
    mask: torch.Tensor,
    *,
    ce: nn.Module,
    w_merger: float,
    w_bar: float,
    w_asym: float,
) -> torch.Tensor:
    """Combined loss; auxiliary BCE/MSE terms are masked (missing labels excluded)."""
    loss = ce(out["morph"], labels)
    if mask[:, 0].sum() > 0:
        bce = F.binary_cross_entropy_with_logits(out["merger"], aux[:, 0], reduction="none")
        loss = loss + w_merger * (bce * mask[:, 0]).sum() / (mask[:, 0].sum() + 1e-6)
    if mask[:, 1].sum() > 0:
        bce_b = F.binary_cross_entropy_with_logits(out["bar"], aux[:, 1], reduction="none")
        loss = loss + w_bar * (bce_b * mask[:, 1]).sum() / (mask[:, 1].sum() + 1e-6)
    if mask[:, 2].sum() > 0:
        mse = F.mse_loss(out["asymmetry"], aux[:, 2], reduction="none")
        loss = loss + w_asym * (mse * mask[:, 2]).sum() / (mask[:, 2].sum() + 1e-6)
    return loss


def train_epoch_multitask(
    model: nn.Module,
    train_loader: DataLoader,
    ce: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: GradScaler | None,
    use_amp: bool,
    max_grad_norm: float,
    w_merger: float,
    w_bar: float,
    w_asym: float,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    amp_enabled = use_amp and device.type == "cuda"

    for images, labels, aux, mask in tqdm(train_loader, desc="Training (multi-task)"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            with autocast("cuda", enabled=True):
                out = model(images)
                loss = multitask_loss(
                    out, labels, aux, mask, ce=ce, w_merger=w_merger, w_bar=w_bar, w_asym=w_asym
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = multitask_loss(
                out, labels, aux, mask, ce=ce, w_merger=w_merger, w_bar=w_bar, w_asym=w_asym
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        logits = morph_logits(out)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        running_loss += float(loss.detach().item())

    n = max(len(train_loader), 1)
    return running_loss / n, float(accuracy_score(all_labels, all_preds))


@torch.no_grad()
def validate_multitask(
    model: nn.Module,
    val_loader: DataLoader,
    ce: nn.Module,
    device: torch.device,
    class_names: list[str],
    *,
    use_amp: bool,
    w_merger: float,
    w_bar: float,
    w_asym: float,
) -> tuple[float, float, dict, np.ndarray, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    amp_enabled = use_amp and device.type == "cuda"

    for images, labels, aux, mask in tqdm(val_loader, desc="Validating (multi-task)"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if amp_enabled:
            with autocast("cuda", enabled=True):
                out = model(images)
                loss = multitask_loss(
                    out, labels, aux, mask, ce=ce, w_merger=w_merger, w_bar=w_bar, w_asym=w_asym
                )
        else:
            out = model(images)
            loss = multitask_loss(
                out, labels, aux, mask, ce=ce, w_merger=w_merger, w_bar=w_bar, w_asym=w_asym
            )
        running_loss += float(loss.item())
        logits = morph_logits(out)
        probs = F.softmax(logits.float(), dim=1)
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
