# AI-Based Galaxy Morphology Classifier

A **local-first**, reproducible PyTorch project for classifying galaxy images into **spiral**, **elliptical**, and **irregular** morphologies. The codebase is structured for research workflows (clear modules, YAML configs, exported metrics) while staying small enough to run on a laptop with no cloud services or API keys.

For a minimal command sequence, see **[QUICKSTART.md](QUICKSTART.md)**.

## Architecture overview

| Layer | Role |
|--------|------|
| **configs/** | Single source of truth for hyperparameters (`train.yaml`, `inference.yaml`). |
| **src/galaxy_morphology/data/** | `Dataset`, loaders, **dataset quality** JSON, optional SDSS helpers, sample data script. |
| **src/galaxy_morphology/models/** | Custom CNNs plus **torchvision** transfer models (EfficientNet-B0, ConvNeXt Tiny, ResNet50) via **registry** (`build_model` / `list_model_names`). |
| **src/galaxy_morphology/training/** | AMP, grad clip, early stopping, **mixup**, **weighted CE**, **label smoothing**, **cosine or plateau** LR, local **experiment** dirs (JSONL + CSV). |
| **src/galaxy_morphology/inference/** | Checkpoint load, batched inference, optional **ONNX** export, **throughput** benchmark. |
| **src/galaxy_morphology/evaluation/** | Metrics, **macro/weighted F1**, balanced accuracy, ROC-AUC (OvR), **benchmark** → CSV + Markdown table. |
| **src/galaxy_morphology/visualization/** | Training curves, confusion matrix, **ROC**, **PR**, **class distribution**. |
| **src/galaxy_morphology/utils/** | YAML merge, structured logging, seeds, checkpoint I/O. |
| **tests/** | pytest smoke tests for model, data, inference, registry, and config loading. |

Training flow: **YAML + CLI overrides** → **optional dataset quality JSON** → **deterministic seed / cuDNN** → **DataLoaders** → **forward + loss** (optional **AMP**, **grad clip**, **mixup**, **class weights**, **label smoothing**) → **validation + extended metrics** → **scheduler / early stopping** → **checkpoints** (store `model_name`, `scheduler_kind`, `pretrained`) → **ROC/PR + experiment JSONL**.

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
- `galaxy-benchmark` — compare models (CSV + Markdown table)  

Without installation, use `python scripts/train.py` (scripts prepend `src/` to `sys.path`).

## Quick start

```bash
python scripts/download_sample_data.py --mode dummy --num-per-class 5
python scripts/train.py --config configs/train.yaml --epochs 3
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image data/galaxies/spiral/spiral_1.png
python scripts/benchmark_models.py --data-dir data/galaxies --epochs 1 --out-dir outputs/benchmarks
```

## Model registry and transfer learning

Models are built with **`galaxy_morphology.models.registry.build_model`** (same as **`get_model`** in `model.py` shims).

| `model.name` | Notes |
|--------------|--------|
| `lightweight` | Small custom CNN. |
| `efficient` | MobileNet-style custom CNN. |
| `efficientnet_b0` | torchvision EfficientNet-B0 + new head (`EfficientNet_B0_Weights`). |
| `convnext_tiny` | torchvision ConvNeXt Tiny (`ConvNeXt_Tiny_Weights`). |
| `resnet50` | torchvision ResNet50 (`ResNet50_Weights`). |

Use **`model.pretrained: true`** in `configs/train.yaml` for ImageNet initialization of torchvision backbones. Checkpoints store **`model_name`**, **`scheduler_kind`**, and **`pretrained`** for reproducible resume and inference.

**Tradeoffs:** custom CNNs are fastest for smoke tests and tiny data; ResNet50 / ConvNeXt Tiny are heavier than EfficientNet-B0 at 224² but often give better features when you have enough labels.

## Benchmarking

Fair-ish comparison on the **same stratified split** (`seed`):

```bash
python scripts/benchmark_models.py --data-dir data/galaxies --epochs 1 --out-dir outputs/benchmarks
galaxy-benchmark --models lightweight,efficientnet_b0 --epochs 1 --no-pretrained
```

Writes **`benchmark_results.csv`** and **`benchmark_table.md`** (parameters, val accuracy, macro F1, val-loader throughput, peak GPU memory on CUDA).

## Local experiment tracking

In `configs/train.yaml`:

```yaml
experiment:
  enabled: true
  root: experiments
  slug: ablation_name
```

Creates **`experiments/<UTC>_<slug>/`** with `config_copy.yaml`, `config_resolved.json`, per-epoch **`metrics.jsonl`**, `outputs/`, `checkpoints/`, and `figures/` (ROC, PR, confusion matrix, class distribution). No cloud tracking.

## Dataset quality

With `data.quality.enabled: true` (default), training writes **`outputs/dataset_statistics.json`**: per-class counts, files that fail `PIL` open/verify, **MD5 duplicate** path groups, and simple **imbalance** warnings.

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
python scripts/train.py --config configs/train.yaml --data-dir data/galaxies --epochs 20 --batch-size 16 --lr 0.0005 --model efficientnet_b0
```

**Resume** from a checkpoint:

```bash
python scripts/train.py --config configs/train.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

Training uses:

- **Mixed precision** (`torch.amp`) when CUDA is available (`training.amp`).
- **Gradient clipping** (`training.gradient_clip_norm`).
- **AdamW** optimizer; **ReduceLROnPlateau** or **cosine** schedule (`scheduler.type`: `plateau` or `cosine`).
- **Class-weighted** cross-entropy (`training.loss.weighted`), **label smoothing** (`training.loss.label_smoothing`).
- **Mixup** (`training.mixup.enabled`, `training.mixup.alpha`).
- **Early stopping** (`training.early_stopping`); `patience: 0` disables.
- **Periodic checkpoints** with full optimizer/scheduler/scaler state.
- **`torch.load(..., map_location=device)`** via `load_checkpoint` for resume and final eval.

## Inference

Edit **`configs/inference.yaml`** or pass flags:

```bash
galaxy-infer --checkpoint checkpoints/best_model.pth --image path/to/galaxy.jpg
galaxy-infer --checkpoint checkpoints/best_model.pth --image-dir path/to/folder/ --batch-size 32 --output predictions.csv --benchmark
galaxy-infer --checkpoint checkpoints/best_model.pth --onnx-export model.onnx
```

Directory mode uses a **DataLoader** for batched GPU inference. **`--benchmark`** logs images/sec after predictions. **`--onnx-export`** writes a dynamic-batch ONNX graph (opset 17; uses the non-Dynamo exporter when available).

The checkpoint’s `class_names` (and optional `model_name`) define outputs.

## Metrics and artifacts

After training you typically get:

| Artifact | Location |
|----------|-----------|
| Best weights | `checkpoints/best_model.pth` (and periodic `checkpoint_epoch_*.pth`) |
| Training / validation curves | `outputs/training_history.png`, or `experiments/.../figures/` when experiment mode is on |
| Confusion matrix | `outputs/confusion_matrix.png` (or experiment `figures/`) |
| ROC / PR curves | `outputs/roc_curves.png`, `outputs/pr_curves.png` (or experiment `figures/`) |
| Class distribution | `outputs/class_distribution_train.png` (or experiment `figures/`) |
| Run-level metrics | `outputs/metrics.json` (includes **extended** block: macro/weighted F1, balanced acc, ROC-AUC OvR) |
| Extended metrics JSON | `outputs/extended_metrics.json` |
| Per-class sklearn report | `outputs/classification_report.json` |
| ROC/PR raw points | `outputs/roc_curve_data.json`, `outputs/pr_curve_data.json` |
| Per-epoch CSV | `outputs/training_history.csv` |
| Dataset quality | `outputs/dataset_statistics.json` |

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
outputs/                 # metrics, plots (no experiment), CSV (.gitkeep)
checkpoints/             # saved weights (.gitkeep)
experiments/             # local experiment dirs when enabled (.gitkeep)
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
