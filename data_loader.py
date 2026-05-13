"""Shim for legacy ``import data_loader``."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from galaxy_morphology.data.dataset import GalaxyDataset  # noqa: F401
from galaxy_morphology.data.downloader import SDSSDataDownloader  # noqa: F401
from galaxy_morphology.data.loaders import create_sample_dataset, load_dataset  # noqa: F401

__all__ = ["GalaxyDataset", "SDSSDataDownloader", "create_sample_dataset", "load_dataset"]
