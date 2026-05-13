"""Dataset returning morphology label plus optional auxiliary targets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GalaxyMultiTaskDataset(Dataset):
    """``__getitem__`` returns ``(image, morph_label, aux_tensor, aux_mask)``."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        aux_targets: np.ndarray,
        aux_mask: np.ndarray,
        transform: Callable | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.aux_targets = aux_targets.astype(np.float32)
        self.aux_mask = aux_mask.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        aux = torch.from_numpy(self.aux_targets[idx])
        mask = torch.from_numpy(self.aux_mask[idx])
        return image, label, aux, mask
