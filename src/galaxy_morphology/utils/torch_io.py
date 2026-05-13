"""PyTorch checkpoint I/O with version-tolerant ``torch.load``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, map_location: Any) -> dict[str, Any]:
    """Load a training checkpoint dict (state dicts + metadata).

    Uses ``weights_only=False`` when supported (PyTorch 2+) for pickled metadata.
    """
    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(p, map_location=map_location)
