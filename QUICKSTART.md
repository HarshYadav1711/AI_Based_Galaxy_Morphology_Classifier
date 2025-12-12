# Quick Start Guide

## Step-by-Step Instructions to Run the Project

### Step 1: Install Dependencies

Open a terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

**Note for Windows users:** If you get permission errors, try:
```bash
pip install --user -r requirements.txt
```

### Step 2: Set Up Test Data (Quick Test)

For a quick test without real galaxy images, create dummy data:

```bash
python download_sample_data.py --mode dummy --num_per_class 5
```

This creates placeholder images in `data/galaxies/` with the correct folder structure.

### Step 3: Train the Model

Run the training script:

```bash
python train.py --data_dir data/galaxies
```

**For a quick test with fewer epochs:**
```bash
python train.py --data_dir data/galaxies --epochs 10 --batch_size 16
```

**What happens:**
- The script loads images from `data/galaxies/`
- Splits them into train/validation sets
- Trains the model
- Saves the best model to `checkpoints/best_model.pth`
- Creates training plots in the `checkpoints/` folder

### Step 4: Make Predictions

Once training is complete, test the model:

**Single image:**
```bash
python inference.py --checkpoint checkpoints/best_model.pth --image data/galaxies/spiral/spiral_1.png
```

**Batch prediction:**
```bash
python inference.py --checkpoint checkpoints/best_model.pth --image_dir data/galaxies/spiral/ --output predictions.csv
```

---

## Using Real Galaxy Data

### Option A: Manual Setup

1. Create the folder structure:
```bash
mkdir -p data/galaxies/spiral
mkdir -p data/galaxies/elliptical
mkdir -p data/galaxies/irregular
```

2. Download galaxy images from:
   - [Galaxy Zoo](https://data.galaxyzoo.org/)
   - [SDSS](https://www.sdss.org/)

3. Place images in the correct folders:
   - Spiral galaxies → `data/galaxies/spiral/`
   - Elliptical galaxies → `data/galaxies/elliptical/`
   - Irregular galaxies → `data/galaxies/irregular/`

4. Run training:
```bash
python train.py --data_dir data/galaxies --epochs 50
```

### Option B: Use SDSS API (Advanced)

The `download_sample_data.py` script includes example code for downloading from SDSS. You'll need to modify it with actual galaxy coordinates and classifications.

---

## Common Commands

### Training Commands

**Basic training:**
```bash
python train.py --data_dir data/galaxies
```

**Custom training:**
```bash
python train.py --data_dir data/galaxies --epochs 100 --batch_size 32 --lr 0.0001
```

**Resume training:**
```bash
python train.py --data_dir data/galaxies --resume checkpoints/checkpoint_epoch_50.pth
```

### Inference Commands

**Single image:**
```bash
python inference.py --checkpoint checkpoints/best_model.pth --image your_image.jpg
```

**Batch prediction:**
```bash
python inference.py --checkpoint checkpoints/best_model.pth --image_dir test_images/ --output results.csv
```

---

## Troubleshooting

### "No images found" Error

**Problem:** The script can't find images in the data directory.

**Solution:**
1. Check that `data/galaxies/` exists
2. Verify subdirectories: `spiral/`, `elliptical/`, `irregular/`
3. Ensure images are in common formats: `.jpg`, `.png`, `.jpeg`
4. Run the dummy data script: `python download_sample_data.py --mode dummy`

### CUDA/GPU Issues

**Problem:** Out of memory or CUDA errors.

**Solution:**
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use smaller images: `--image_size 128`
- Use CPU (if GPU fails): The script automatically uses CPU if CUDA is unavailable

### Import Errors

**Problem:** `ModuleNotFoundError` when running scripts.

**Solution:**
```bash
pip install -r requirements.txt
```

If specific packages fail, install individually:
```bash
pip install torch torchvision
pip install numpy pillow matplotlib scikit-learn
```

---

## Expected Output

### During Training

You should see:
```
Using device: cuda  (or cpu)
Loading dataset...
Classes: ['spiral', 'elliptical', 'irregular']
Train batches: X, Val batches: Y
Model: lightweight
Parameters: XXX,XXX

Starting training...
Epoch 1/50
Training: 100%|████████| X/X [XX:XX<00:00, loss=0.XXXX]
Validating: 100%|████████| X/X [XX:XX<00:00]
Train Loss: X.XXXX, Train Acc: 0.XXXX
Val Loss: X.XXXX, Val Acc: 0.XXXX
New best model saved! Val Acc: 0.XXXX
```

### After Training

Check the `checkpoints/` folder:
- `best_model.pth` - The trained model
- `training_history.png` - Training curves
- `confusion_matrix.png` - Classification results

### During Inference

You should see:
```
Using device: cuda
Loading model from checkpoints/best_model.pth...
Model loaded. Classes: ['spiral', 'elliptical', 'irregular']

Predicting on: your_image.jpg

Prediction: spiral
Confidence: 0.9234

Class Probabilities:
  spiral: 0.9234
  elliptical: 0.0543
  irregular: 0.0223
```

---

## Next Steps

1. **Improve accuracy:** Add more training data (100+ images per class recommended)
2. **Experiment:** Try different models (`--model efficient`) or hyperparameters
3. **Deploy:** Use the trained model in your citizen science workflow
4. **Extend:** Modify the code to add more galaxy classes or features

For more details, see `README.md`.

