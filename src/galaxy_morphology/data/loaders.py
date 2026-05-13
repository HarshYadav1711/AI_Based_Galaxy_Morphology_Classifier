"""Build DataLoaders from a folder-per-class layout."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from galaxy_morphology.data.dataset import GalaxyDataset

logger = logging.getLogger(__name__)


def create_sample_dataset(
    output_dir: str = "data/galaxies", num_samples_per_class: int = 100
) -> None:
    """Create empty class subdirectories for spiral / elliptical / irregular.

    Args:
        output_dir: Root folder for class subfolders.
        num_samples_per_class: Kept for API compatibility; does not create images.
    """
    _ = num_samples_per_class
    os.makedirs(output_dir, exist_ok=True)
    for class_name in ["spiral", "elliptical", "irregular"]:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    logger.info("Created dataset structure in %s", output_dir)
    logger.info("Populate with Galaxy Zoo or SDSS data; see README for dataset layout.")


def load_dataset(
    data_dir: str = "data/galaxies",
    train_split: float = 0.8,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str], list[int], list[str], list[str]]:
    """Load train/validation loaders from ``data_dir/<class>/*.jpg|png``.

    Args:
        data_dir: Root with subfolders per class name.
        train_split: Fraction of samples for training.
        image_size: Square resize side length.
        batch_size: Batch size for both loaders.
        num_workers: ``DataLoader`` workers (0 is robust on Windows).
        seed: Random seed for splits when stratification fails.

    Returns:
        Tuple of ``(train_loader, val_loader, class_names, train_labels, train_paths, val_paths)``.

    Raises:
        ValueError: If no images are found under ``data_dir``.
    """
    class_names = ["spiral", "elliptical", "irregular"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_paths: list[str] = []
    labels: list[int] = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning("Class directory missing, creating: %s", class_dir)
            os.makedirs(class_dir, exist_ok=True)
            continue
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_to_idx[class_name])

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Add images to class subfolders.")

    labels_array = np.array(labels)
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    logger.info("Dataset statistics:")
    for label_idx, count in zip(unique_labels, counts, strict=False):
        logger.info("  %s: %d images", class_names[int(label_idx)], int(count))

    if len(unique_labels) < len(class_names):
        missing = [class_names[i] for i in range(len(class_names)) if i not in unique_labels]
        logger.warning("Missing classes in dataset: %s", missing)

    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths,
            labels,
            test_size=1 - train_split,
            stratify=labels,
            random_state=seed,
        )
        logger.info("Split: %d train, %d validation", len(train_paths), len(val_paths))
    except ValueError as exc:
        logger.warning("Stratified split failed (%s); using random split.", exc)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(image_paths))
        image_paths_shuffled = [image_paths[i] for i in indices]
        labels_shuffled = [labels[i] for i in indices]
        split_idx = int(len(image_paths_shuffled) * train_split)
        train_paths = image_paths_shuffled[:split_idx]
        train_labels = labels_shuffled[:split_idx]
        val_paths = image_paths_shuffled[split_idx:]
        val_labels = labels_shuffled[split_idx:]
        val_unique = set(val_labels)
        if len(val_unique) < len(class_names):
            missing_in_val = [
                class_names[i] for i in range(len(class_names)) if i not in val_unique
            ]
            logger.warning("Classes missing in validation: %s", missing_in_val)

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_dataset = GalaxyDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = GalaxyDataset(val_paths, val_labels, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, class_names, train_labels, train_paths, val_paths
