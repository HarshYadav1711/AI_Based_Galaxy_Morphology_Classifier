"""PyTorch ``Dataset`` for on-disk galaxy images."""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image
from torch.utils.data import Dataset


class GalaxyDataset(Dataset):
    """Loads RGB galaxy images and integer labels from paths."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Callable | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
