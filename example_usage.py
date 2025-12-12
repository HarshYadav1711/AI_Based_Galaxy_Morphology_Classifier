"""
Example usage script demonstrating how to use the galaxy classifier.
This script shows basic usage patterns.
"""

import os
import torch
from model import get_model, count_parameters
from data_loader import create_sample_dataset, load_dataset


def example_model_creation():
    """Example: Create and inspect models."""
    print("=" * 60)
    print("Example 1: Model Creation")
    print("=" * 60)
    
    # Create lightweight model
    model_light = get_model('lightweight', num_classes=3)
    print(f"\nLightweight Model:")
    print(f"  Parameters: {count_parameters(model_light):,}")
    
    # Create efficient model
    model_eff = get_model('efficient', num_classes=3)
    print(f"\nEfficient Model:")
    print(f"  Parameters: {count_parameters(model_eff):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model_light(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output classes: {output.shape[1]}")


def example_dataset_setup():
    """Example: Set up dataset structure."""
    print("\n" + "=" * 60)
    print("Example 2: Dataset Setup")
    print("=" * 60)
    
    # Create sample dataset structure
    data_dir = 'data/galaxies'
    create_sample_dataset(data_dir)
    
    print(f"\nDataset structure created at: {data_dir}")
    print("\nTo add your own data:")
    print("1. Download Galaxy Zoo dataset from https://data.galaxyzoo.org/")
    print("2. Organize images into:")
    print(f"   {data_dir}/spiral/")
    print(f"   {data_dir}/elliptical/")
    print(f"   {data_dir}/irregular/")


def example_training_command():
    """Example: Show training command."""
    print("\n" + "=" * 60)
    print("Example 3: Training Command")
    print("=" * 60)
    
    print("\nTo train the model, run:")
    print("\n  python train.py --data_dir data/galaxies --epochs 50")
    print("\nOr with more options:")
    print("\n  python train.py \\")
    print("    --data_dir data/galaxies \\")
    print("    --model lightweight \\")
    print("    --epochs 50 \\")
    print("    --batch_size 32 \\")
    print("    --lr 0.001 \\")
    print("    --save_dir checkpoints")


def example_inference_command():
    """Example: Show inference command."""
    print("\n" + "=" * 60)
    print("Example 4: Inference Command")
    print("=" * 60)
    
    print("\nTo predict on a single image:")
    print("\n  python inference.py \\")
    print("    --checkpoint checkpoints/best_model.pth \\")
    print("    --image path/to/galaxy.jpg")
    
    print("\nTo predict on multiple images:")
    print("\n  python inference.py \\")
    print("    --checkpoint checkpoints/best_model.pth \\")
    print("    --image_dir test_galaxies/ \\")
    print("    --output predictions.csv")


def example_data_loading():
    """Example: Load dataset (if data exists)."""
    print("\n" + "=" * 60)
    print("Example 5: Data Loading")
    print("=" * 60)
    
    data_dir = 'data/galaxies'
    
    if not os.path.exists(data_dir):
        print(f"\nDataset directory {data_dir} does not exist.")
        print("Run example_dataset_setup() first or create it manually.")
        return
    
    try:
        train_loader, val_loader, class_names = load_dataset(
            data_dir=data_dir,
            image_size=224,
            batch_size=32
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"  Classes: {class_names}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Show a sample batch
        if len(train_loader) > 0:
            images, labels = next(iter(train_loader))
            print(f"\nSample batch:")
            print(f"  Image shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Label values: {labels[:5].tolist()}")
    
    except ValueError as e:
        print(f"\nError: {e}")
        print("Add some images to the dataset directories first.")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Galaxy Morphology Classifier - Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_model_creation()
    example_dataset_setup()
    example_training_command()
    example_inference_command()
    
    # Try to load dataset if it exists
    example_data_loading()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nFor more information, see README.md")

