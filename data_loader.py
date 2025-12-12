"""
Data loader for galaxy morphology classification.
Downloads and preprocesses galaxy images from SDSS/Galaxy Zoo datasets.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import requests
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class GalaxyDataset(Dataset):
    """Dataset class for galaxy images."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of integer labels (0: spiral, 1: elliptical, 2: irregular)
            transform: Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SDSSDataDownloader:
    """Downloads galaxy images from SDSS using astropy."""
    
    def __init__(self, output_dir='data/galaxies', image_size=224):
        self.output_dir = output_dir
        self.image_size = image_size
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each class
        for class_name in ['spiral', 'elliptical', 'irregular']:
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    def download_sdss_image(self, ra, dec, objid, scale=0.396, width=224, height=224):
        """
        Download a cutout image from SDSS.
        
        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            objid: SDSS object ID
            scale: Arcseconds per pixel
            width: Image width in pixels
            height: Image height in pixels
        """
        try:
            from astroquery.sdss import SDSS
            
            # Query SDSS for image
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            result = SDSS.get_images(coords, band='g', data_release=16)
            
            if len(result) > 0:
                hdu = result[0]
                data = hdu[0].data
                
                # Normalize and convert to PIL Image
                data = np.nan_to_num(data)
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                data = (data * 255).astype(np.uint8)
                
                # Resize if needed
                img = Image.fromarray(data).convert('RGB')
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                
                return img
        except Exception as e:
            print(f"Error downloading SDSS image: {e}")
            return None
    
    def download_from_url(self, url, save_path):
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
                img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                img.save(save_path)
                return True
        except Exception as e:
            print(f"Error downloading from URL: {e}")
            return False


def create_sample_dataset(output_dir='data/galaxies', num_samples_per_class=100):
    """
    Create a sample dataset structure.
    For a real implementation, you would download actual SDSS/Galaxy Zoo data.
    This function creates the directory structure and provides a template.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in ['spiral', 'elliptical', 'irregular']:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    print(f"Created dataset structure in {output_dir}")
    print("To populate with real data:")
    print("1. Download Galaxy Zoo dataset from https://data.galaxyzoo.org/")
    print("2. Or use SDSS API to download images based on coordinates")
    print("3. Organize images into spiral/, elliptical/, irregular/ subdirectories")


def load_dataset(data_dir='data/galaxies', train_split=0.8, image_size=224, batch_size=32):
    """
    Load dataset from directory structure.
    
    Expected structure:
    data/galaxies/
        spiral/
            img1.jpg
            img2.jpg
            ...
        elliptical/
            img1.jpg
            ...
        irregular/
            img1.jpg
            ...
    """
    class_names = ['spiral', 'elliptical', 'irregular']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    # Collect all images
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist. Creating empty directory.")
            os.makedirs(class_dir, exist_ok=True)
            continue
            
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_to_idx[class_name])
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Please add images to subdirectories.")
    
    # Count samples per class
    labels_array = np.array(labels)
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    print(f"\nDataset statistics:")
    for label_idx, count in zip(unique_labels, counts):
        print(f"  {class_names[label_idx]}: {count} images")
    
    # Check for class imbalance
    if len(unique_labels) < len(class_names):
        missing_classes = [class_names[i] for i in range(len(class_names)) if i not in unique_labels]
        print(f"\nWarning: Missing classes in dataset: {missing_classes}")
        print("Please ensure all three classes (spiral, elliptical, irregular) have images.")
    
    # Use stratified split to ensure all classes are in both train and val
    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=1-train_split, 
            stratify=labels,  # Ensures balanced split
            random_state=42
        )
        print(f"\nSplit: {len(train_paths)} train, {len(val_paths)} validation")
    except ValueError as e:
        # If stratification fails (e.g., a class has too few samples), use random split
        print(f"\nWarning: Stratified split failed ({e}). Using random split instead.")
        print("Consider adding more samples to underrepresented classes.")
        indices = np.random.permutation(len(image_paths))
        image_paths_shuffled = [image_paths[i] for i in indices]
        labels_shuffled = [labels[i] for i in indices]
        
        split_idx = int(len(image_paths_shuffled) * train_split)
        train_paths = image_paths_shuffled[:split_idx]
        train_labels = labels_shuffled[:split_idx]
        val_paths = image_paths_shuffled[split_idx:]
        val_labels = labels_shuffled[split_idx:]
        
        # Verify all classes are in validation set
        val_unique = set(val_labels)
        if len(val_unique) < len(class_names):
            missing_in_val = [class_names[i] for i in range(len(class_names)) if i not in val_unique]
            print(f"Warning: Classes missing in validation set: {missing_in_val}")
            print("This may cause errors in classification report. Add more samples to these classes.")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = GalaxyDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = GalaxyDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, class_names


if __name__ == '__main__':
    # Create sample dataset structure
    create_sample_dataset()
    print("\nDataset structure created. Add your galaxy images to the subdirectories.")

