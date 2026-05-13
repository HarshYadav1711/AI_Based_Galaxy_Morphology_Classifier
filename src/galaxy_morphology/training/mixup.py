"""Optional mixup augmentation for training batches."""

from __future__ import annotations

import torch


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed inputs, pair of labels, and lambda for soft targets.

    If ``alpha <= 0``, returns original batch and ``lam=1.0`` (no mixing).
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    if lam == 1.0:
        return x, y, y, lam

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: torch.nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Combine two supervised losses with ``lam``."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
