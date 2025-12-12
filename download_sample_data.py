"""
Script to download sample galaxy images for testing.
This script provides example code for downloading SDSS images.
For production use, download Galaxy Zoo dataset from https://data.galaxyzoo.org/
"""

import os
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import time


def download_sdss_cutout(ra, dec, scale=0.396, width=224, height=224, band='g'):
    """
    Download a cutout image from SDSS using their image cutout service.
    
    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        scale: Arcseconds per pixel
        width: Image width in pixels
        height: Image height in pixels
        band: Filter band (u, g, r, i, z)
    
    Returns:
        PIL Image or None if download fails
    """
    base_url = "https://skyserver.sdss.org/dr16/SkyServerWS/SkyServerWS.asmx/getJpeg"
    
    params = {
        'ra': ra,
        'dec': dec,
        'scale': scale,
        'width': width,
        'height': height,
        'opt': f'G'  # Grid overlay
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
    except Exception as e:
        print(f"Error downloading SDSS image: {e}")
        return None
    
    return None


def download_sample_galaxies(output_dir='data/galaxies', num_per_class=10):
    """
    Download sample galaxy images from SDSS.
    Note: These are example coordinates. For real training, use Galaxy Zoo dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Example galaxy coordinates (RA, Dec) from SDSS
    # These are just examples - for real data, use Galaxy Zoo classifications
    sample_coords = {
        'spiral': [
            (146.714, 0.395),  # Example spiral galaxy
            (150.123, 1.234),
            (145.567, 0.789),
        ],
        'elliptical': [
            (200.123, 0.456),  # Example elliptical galaxy
            (201.234, 0.567),
            (199.890, 0.345),
        ],
        'irregular': [
            (180.456, 0.234),  # Example irregular galaxy
            (181.567, 0.345),
            (179.234, 0.123),
        ]
    }
    
    print("Downloading sample galaxy images from SDSS...")
    print("Note: This is for demonstration. For real training, download Galaxy Zoo dataset.")
    print()
    
    for class_name, coords_list in sample_coords.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Downloading {class_name} galaxies...")
        for i, (ra, dec) in enumerate(coords_list[:num_per_class]):
            img = download_sdss_cutout(ra, dec)
            if img:
                save_path = os.path.join(class_dir, f'{class_name}_{i+1}.jpg')
                img.save(save_path)
                print(f"  Saved: {save_path}")
            else:
                print(f"  Failed to download image at RA={ra}, Dec={dec}")
            
            time.sleep(0.5)  # Be nice to the server
    
    print("\nDownload complete!")
    print("\nFor real training data:")
    print("1. Visit https://data.galaxyzoo.org/ to download Galaxy Zoo dataset")
    print("2. Or use the SDSS API with proper galaxy classifications")
    print("3. Organize images into spiral/, elliptical/, irregular/ subdirectories")


def create_dummy_dataset(output_dir='data/galaxies', num_per_class=5):
    """
    Create dummy dataset structure with placeholder images.
    Useful for testing the code structure without downloading real data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in ['spiral', 'elliptical', 'irregular']:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Creating dummy images for {class_name}...")
        for i in range(num_per_class):
            # Create a simple colored image as placeholder
            img = Image.new('RGB', (224, 224), color=(50 + i*10, 100 + i*5, 150 + i*3))
            save_path = os.path.join(class_dir, f'{class_name}_{i+1}.png')
            img.save(save_path)
            print(f"  Created: {save_path}")
    
    print(f"\nDummy dataset created in {output_dir}")
    print("Replace these with real galaxy images for actual training.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download sample galaxy images')
    parser.add_argument('--mode', type=str, default='dummy',
                       choices=['dummy', 'sdss'],
                       help='Download mode: dummy (create placeholder) or sdss (download from SDSS)')
    parser.add_argument('--output_dir', type=str, default='data/galaxies',
                       help='Output directory')
    parser.add_argument('--num_per_class', type=int, default=5,
                       help='Number of images per class')
    
    args = parser.parse_args()
    
    if args.mode == 'dummy':
        create_dummy_dataset(args.output_dir, args.num_per_class)
    else:
        download_sample_galaxies(args.output_dir, args.num_per_class)

