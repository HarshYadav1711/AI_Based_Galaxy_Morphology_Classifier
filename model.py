"""Shim for legacy ``import model`` (package lives under ``src/galaxy_morphology``)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from galaxy_morphology.models.cnn import (  # noqa: F401
    EfficientGalaxyNet,
    LightweightGalaxyCNN,
    count_parameters,
    get_model,
)

__all__ = [
    "EfficientGalaxyNet",
    "LightweightGalaxyCNN",
    "count_parameters",
    "get_model",
]
