# AI-Based Galaxy Morphology Classifier

A lightweight deep learning model for classifying galaxy morphologies into three categories: **Spiral**, **Elliptical**, and **Irregular**. This project uses PyTorch and is designed to work with public datasets from SDSS (Sloan Digital Sky Survey) and Galaxy Zoo.

## 🚀 Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Create test data (optional, for quick testing):**
```bash
python download_sample_data.py --mode dummy --num_per_class 5
```

**3. Train the model:**
```bash
python train.py --data_dir data/galaxies
```

**4. Make predictions:**
```bash
python inference.py --checkpoint checkpoints/best_model.pth --image your_image.jpg
```

📖 **For detailed step-by-step instructions, see [QUICKSTART.md](QUICKSTART.md)**

## Features

- **Lightweight CNN Architecture**: Two model options - a custom lightweight CNN and an efficient MobileNet-inspired architecture
- **Data Augmentation**: Built-in augmentation for better generalization
- **Easy to Use**: Simple command-line interface for training and inference
- **Citizen Science Support**: Designed to assist in galaxy classification workflows

## Project Structure

```
.
├── data_loader.py          # Dataset loading and preprocessing
├── model.py                # CNN model architectures
├── train.py                # Training script
├── inference.py            # Inference script for predictions
├── download_sample_data.py # Script to download sample data
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Install astroquery for SDSS data access**:
```bash
pip install astroquery
```

## Dataset Setup

### Option 1: Galaxy Zoo Dataset (Recommended)

1. Visit [Galaxy Zoo Data Portal](https://data.galaxyzoo.org/)
2. Download the galaxy images and classifications
3. Organize images into the following structure:
```
data/galaxies/
    spiral/
        img1.jpg
        img2.jpg
        ...
    elliptical/
        img1.jpg
        img2.jpg
        ...
    irregular/
        img1.jpg
        img2.jpg
        ...
```

### Option 2: SDSS API

You can use the SDSS API to download images programmatically. See `download_sample_data.py` for example code.

### Option 3: Create Dummy Dataset (For Testing)

To test the code structure without real data:
```bash
python download_sample_data.py --mode dummy --num_per_class 5
```

## Usage

### Training

Train the model with default settings:
```bash
python train.py --data_dir data/galaxies
```

With custom parameters:
```bash
python train.py \
    --data_dir data/galaxies \
    --model lightweight \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --image_size 224 \
    --save_dir checkpoints
```

**Parameters:**
- `--data_dir`: Directory containing galaxy images (default: `data/galaxies`)
- `--model`: Model architecture - `lightweight` or `efficient` (default: `lightweight`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--image_size`: Input image size (default: 224)
- `--save_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--resume`: Path to checkpoint to resume training from

### Inference

**Single Image Prediction:**
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/galaxy_image.jpg
```

**Batch Prediction:**
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image_dir path/to/galaxy_images/ \
    --output predictions.csv
```

**Parameters:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--image`: Path to single image for prediction
- `--image_dir`: Directory containing images for batch prediction
- `--model`: Model architecture used (default: `lightweight`)
- `--output`: CSV file to save batch predictions

## Model Architectures

### LightweightGalaxyCNN
A custom CNN with 4 convolutional blocks, batch normalization, and global average pooling. Designed for good accuracy with moderate computational requirements.

### EfficientGalaxyNet
A MobileNet-inspired architecture using depthwise separable convolutions. More efficient with fewer parameters, suitable for resource-constrained environments.

## Training Outputs

After training, you'll find:
- `checkpoints/best_model.pth`: Best model checkpoint
- `checkpoints/training_history.png`: Training curves
- `checkpoints/confusion_matrix.png`: Confusion matrix

## Example Workflow

1. **Prepare your dataset**:
```bash
# Create directory structure
mkdir -p data/galaxies/{spiral,elliptical,irregular}

# Add your galaxy images to respective folders
# Or use download_sample_data.py for testing
```

2. **Train the model**:
```bash
python train.py --data_dir data/galaxies --epochs 50
```

3. **Make predictions**:
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image_dir test_galaxies/ \
    --output results.csv
```

## Data Sources

- **Galaxy Zoo**: [https://data.galaxyzoo.org/](https://data.galaxyzoo.org/)
- **SDSS**: [https://www.sdss.org/](https://www.sdss.org/)
- **Zoobot**: Pre-trained galaxy classification models - [https://github.com/mwalmsley/zoobot](https://github.com/mwalmsley/zoobot)

## References

- Galaxy Zoo: Citizen science project for galaxy classification
- SDSS: Sloan Digital Sky Survey - comprehensive imaging and spectroscopic survey
- Zoobot: Open-source galaxy morphology classification tools

## Performance Tips

1. **Data Quality**: Ensure good quality, properly labeled images
2. **Data Augmentation**: The training script includes augmentation - adjust if needed
3. **Model Selection**: Use `efficient` model for faster inference, `lightweight` for better accuracy
4. **Batch Size**: Adjust based on your GPU memory
5. **Learning Rate**: The script uses learning rate scheduling - monitor training curves

## Troubleshooting

**No images found error:**
- Ensure images are in the correct subdirectories (spiral/, elliptical/, irregular/)
- Check image file extensions (.jpg, .png, .jpeg)

**CUDA out of memory:**
- Reduce batch size: `--batch_size 16`
- Use smaller image size: `--image_size 128`
- Use the `efficient` model which has fewer parameters

**Poor accuracy:**
- Ensure you have enough training data (recommended: 100+ images per class)
- Check data quality and labeling
- Try training for more epochs
- Adjust learning rate

## License

This project is provided as-is for educational and research purposes.

## Contributing

This is a lightweight implementation designed for citizen science workflows. Feel free to adapt and improve for your specific use case!

## Citation

If you use this code in your research, please consider citing:
- Galaxy Zoo dataset
- SDSS data
- Relevant papers on galaxy morphology classification

