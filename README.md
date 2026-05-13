# AI-Based Galaxy Morphology Classifier

A **local-first**, reproducible PyTorch project for classifying galaxy images into **spiral**, **elliptical**, and **irregular** morphologies. The codebase is structured for research workflows (clear modules, YAML configs, exported metrics) while staying small enough to run on a laptop with no cloud services or API keys.

## Architecture overview

| Layer | Role |
|--------|------|
| **configs/** | Single source of truth for hyperparameters (`train.yaml`, `inference.yaml`). |
| **src/galaxy_morphology/data/** | `Dataset`, directory-based loaders, optional SDSS helpers, sample data script. |
| **src/galaxy_morphology/models/** | `LightweightGalaxyCNN` and `EfficientGalaxyNet` (depthwise separable). |
| **src/galaxy_morphology/training/** | Training loop with AMP (CUDA), gradient clipping, early stopping, checkpointing, CLI. |
| **src/galaxy_morphology/inference/** | Checkpoint load (`map_location`), preprocessing, single/batch prediction, CLI. |
| **src/galaxy_morphology/evaluation/** | Writes `metrics.json`, `classification_report.json`, `training_history.csv`. |
| **src/galaxy_morphology/visualization/** | Training curves and confusion matrix PNGs. |
| **src/galaxy_morphology/utils/** | YAML config merge, structured logging (tqdm-safe), seeds, checkpoint I/O. |
| **tests/** | pytest smoke tests for model, data, inference, and config loading. |

Training flow: **YAML + CLI overrides** → **deterministic seed / cuDNN flags** → **DataLoaders** → **forward + loss** (optional **AMP** + **grad clip**) → **validation** → **scheduler / early stopping** → **periodic and best checkpoints** → **metrics + plots**.

## Setup

**Python 3.10+** recommended.

```bash
# Virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Runtime dependencies
pip install -r requirements/base.txt

# Editable install (adds console commands + stable imports)
pip install -e .

# Optional: developer tools
pip install -r requirements/dev.txt
pre-commit install
```

Console entry points (after `pip install -e .`):

- `galaxy-train` — training CLI  
- `galaxy-infer` — inference CLI  
- `galaxy-sample-data` — dummy or SDSS sample images  

Without installation, use `python scripts/train.py` (scripts prepend `src/` to `sys.path`).

## Quick start

```bash
python scripts/download_sample_data.py --mode dummy --num-per-class 5
python scripts/train.py --config configs/train.yaml --epochs 3
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image data/galaxies/spiral/spiral_1.png
```

## Training

1. Put images under `data/galaxies/<class>/` with classes `spiral`, `elliptical`, `irregular` (see `configs/train.yaml` → `data.dir`).
2. Edit hyperparameters in **`configs/train.yaml`** (epochs, LR, AMP, early stopping, checkpoint cadence, etc.).
3. Run:

```bash
galaxy-train --config configs/train.yaml
# or
python train.py --config configs/train.yaml
```

**CLI overrides** (optional) merge into the YAML tree, for example:

```bash
python scripts/train.py --config configs/train.yaml --data-dir data/galaxies --epochs 20 --batch-size 16 --lr 0.0005 --model efficient
```

**Resume** from a checkpoint:

```bash
python scripts/train.py --config configs/train.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

Training uses:

- **Mixed precision** (`torch.amp`) when CUDA is available (toggle `training.amp` in YAML).
- **Gradient clipping** (`training.gradient_clip_norm`).
- **Early stopping** on `val_loss` or `val_acc` (`training.early_stopping`); set `patience: 0` to disable.
- **Periodic checkpoints** that always serialize the **current** full state (fixes a bug where an old dict could be saved).
- **`torch.load(..., map_location=device)`** (via `galaxy_morphology.utils.torch_io.load_checkpoint`) for resume and final evaluation.

## Inference

Edit **`configs/inference.yaml`** or pass flags:

```bash
galaxy-infer --checkpoint checkpoints/best_model.pth --image path/to/galaxy.jpg
galaxy-infer --checkpoint checkpoints/best_model.pth --image-dir path/to/folder/ --output predictions.csv
```

The checkpoint’s `class_names` are restored so labels stay consistent with training.

## Metrics and artifacts

After training you typically get:

| Artifact | Location |
|----------|-----------|
| Best weights | `checkpoints/best_model.pth` (and periodic `checkpoint_epoch_*.pth`) |
| Training / validation curves | `checkpoints/training_history.png` |
| Confusion matrix | `checkpoints/confusion_matrix.png` |
| Run-level metrics | `outputs/metrics.json` |
| Per-class sklearn report | `outputs/classification_report.json` |
| Per-epoch CSV | `outputs/training_history.csv` |

## Reproducibility

- Global seed: `seed` in `configs/train.yaml`.
- `galaxy_morphology.utils.seed.set_seed` sets Python, NumPy, and PyTorch seeds; optional **deterministic cuDNN** (`reproducibility.deterministic_cudnn`).
- Stratified split uses `random_state=seed`; `DataLoader` shuffling uses a fixed `torch.Generator` when possible.

Some GPU kernels remain non-deterministic even with these flags; for strict bitwise reproducibility prefer CPU or consult PyTorch deterministic ops notes.

## Project structure

```text
configs/                 # YAML configs
src/galaxy_morphology/   # Installable Python package
  data/ models/ training/ inference/ evaluation/ visualization/ utils/
scripts/                 # Runnable wrappers without prior pip install
tests/                   # pytest
notebooks/               # Optional experiments (.gitkeep)
outputs/                 # metrics.json, CSV, etc. (.gitkeep)
checkpoints/             # saved weights and plots (.gitkeep)
requirements/            # base.txt + dev.txt
pyproject.toml           # packaging, black/ruff/pytest settings
setup.cfg                # setuptools metadata
README.md
LICENSE                  # MIT
```

## Development

```bash
pip install -r requirements/dev.txt
pytest
ruff check src tests scripts
black src tests scripts
```

## Data sources (external, public)

- [Galaxy Zoo](https://data.galaxyzoo.org/)  
- [SDSS](https://www.sdss.org/)  

## License

MIT — see [LICENSE](LICENSE).
