"""Centralized random seed and PyTorch reproducibility helpers."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducible runs.

    When ``deterministic_cudnn`` is True and CUDA is available, enables
    deterministic algorithms where supported (may reduce performance).

    Args:
        seed: Integer seed applied across libraries.
        deterministic_cudnn: If True, prefer deterministic cuDNN behavior.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Prefer deterministic algorithms when available (PyTorch 1.8+)
        with contextlib.suppress(TypeError, AttributeError):
            torch.use_deterministic_algorithms(True, warn_only=True)
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
