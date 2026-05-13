"""Usage examples (adds ``src`` to path for uninstalled runs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from galaxy_morphology.data.loaders import create_sample_dataset, load_dataset
from galaxy_morphology.models.cnn import count_parameters, get_model


def example_model_creation() -> None:
    print("=" * 60)
    print("Example 1: Model Creation")
    print("=" * 60)
    model_light = get_model("lightweight", num_classes=3)
    print(f"\nLightweight Model:\n  Parameters: {count_parameters(model_light):,}")
    model_eff = get_model("efficient", num_classes=3)
    print(f"\nEfficient Model:\n  Parameters: {count_parameters(model_eff):,}")
    x = torch.randn(1, 3, 224, 224)
    output = model_light(x)
    print(f"\nInput shape: {x.shape}\nOutput shape: {output.shape}")


def example_dataset_setup() -> None:
    print("\n" + "=" * 60)
    print("Example 2: Dataset Setup")
    print("=" * 60)
    data_dir = "data/galaxies"
    create_sample_dataset(data_dir)
    print(f"\nDataset structure at: {data_dir}")


def example_training_command() -> None:
    print("\n" + "=" * 60)
    print("Example 3: Training Command")
    print("=" * 60)
    print("\n  python scripts/train.py --config configs/train.yaml")
    print("\n  galaxy-train --config configs/train.yaml --epochs 10")


def example_inference_command() -> None:
    print("\n" + "=" * 60)
    print("Example 4: Inference Command")
    print("=" * 60)
    print("\n  python scripts/inference.py --checkpoint checkpoints/best_model.pth --image path/to.jpg")
    print("\n  galaxy-infer --checkpoint checkpoints/best_model.pth --image path/to.jpg")


def example_data_loading() -> None:
    print("\n" + "=" * 60)
    print("Example 5: Data Loading")
    print("=" * 60)
    data_dir = "data/galaxies"
    if not os.path.exists(data_dir):
        print(f"\n{data_dir} missing; run download_sample_data first.")
        return
    try:
        train_loader, val_loader, class_names, _train_labels = load_dataset(
            data_dir=data_dir,
            image_size=224,
            batch_size=32,
        )
        print(f"\nLoaded classes={class_names}, train_batches={len(train_loader)}, val_batches={len(val_loader)}")
        if len(train_loader) > 0:
            images, labels = next(iter(train_loader))
            print(f"Sample batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
    except ValueError as e:
        print(f"\n{e}")


if __name__ == "__main__":
    example_model_creation()
    example_dataset_setup()
    example_training_command()
    example_inference_command()
    example_data_loading()
    print("\nDone. See README.md for full documentation.")
