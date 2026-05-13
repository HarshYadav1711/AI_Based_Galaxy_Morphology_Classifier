# Quick Start Guide

Short path from zero to a trained classifier. For architecture options, benchmarks, and experiment folders, see [README.md](README.md).

## 1. Install

From the project root:

```bash
pip install -r requirements/base.txt
pip install -e .
```

(Optional dev tools: `pip install -r requirements/dev.txt`)

## 2. Dummy data (offline)

```bash
python scripts/download_sample_data.py --mode dummy --num-per-class 5
```

This creates `data/galaxies/{spiral,elliptical,irregular}/` with placeholder PNGs.

## 3. Train (YAML + CLI overrides)

Default config: `configs/train.yaml`.

```bash
python scripts/train.py --config configs/train.yaml --epochs 10
# or, after `pip install -e .`:
galaxy-train --config configs/train.yaml --epochs 10
```

Common overrides:

```bash
python scripts/train.py --config configs/train.yaml --data-dir data/galaxies --epochs 20 --batch-size 16 --lr 0.0005 --model lightweight
python scripts/train.py --config configs/train.yaml --model efficientnet_b0 --epochs 5
```

Resume:

```bash
python scripts/train.py --config configs/train.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

**What you get**

- Weights: `checkpoints/best_model.pth` (unless `experiment.enabled` moves artifacts under `experiments/…/checkpoints/`).
- Metrics and tables: `outputs/` (or `experiments/…/outputs/`) — e.g. `metrics.json`, `training_history.csv`, `dataset_statistics.json`, ROC/PR JSON, `extended_metrics.json`.
- Figures: same `outputs/` or `experiments/…/figures/` — training curves, confusion matrix, ROC, PR, class distribution.

Logs use structured logging (timestamped lines) instead of plain `print`.

## 4. Inference

Single image:

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image data/galaxies/spiral/spiral_1.png
galaxy-infer --checkpoint checkpoints/best_model.pth --image path/to/galaxy.jpg
```

Directory (batched):

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image-dir data/galaxies/spiral --batch-size 16 --output predictions.csv
```

Optional throughput log:

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image-dir data/galaxies/spiral --batch-size 16 --benchmark
```

Optional ONNX export:

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth --onnx-export model.onnx
```

Use `--model <name>` only if you must override the architecture stored in the checkpoint; normally the checkpoint’s `model_name` is used.

## 5. Quick model comparison (optional)

```bash
python scripts/benchmark_models.py --data-dir data/galaxies --epochs 1 --no-pretrained --out-dir outputs/benchmarks
```

Writes `benchmark_results.csv` and `benchmark_table.md`.

---

## Real data

1. Create `data/galaxies/spiral`, `elliptical`, `irregular`.
2. Add labeled `.jpg` / `.png` images (see [Galaxy Zoo](https://data.galaxyzoo.org/) or [SDSS](https://www.sdss.org/)).
3. Tune `configs/train.yaml` (epochs, `model.name`, `scheduler.type`, `training.loss`, `training.mixup`, etc.) and run `galaxy-train` or `python scripts/train.py`.

---

## Troubleshooting

**No images found**

- Paths must be `data/galaxies/<class_name>/*.png` (or `.jpg`).
- Run the dummy script in step 2.

**CUDA out of memory**

- Lower `--batch-size` (training) or `data.batch_size` in YAML.
- Lower `data.image_size` in YAML.
- Prefer `efficientnet_b0` or `lightweight` over larger backbones.

**Wrong model at inference**

- Train and infer with the same `model.name`, or rely on the checkpoint’s saved `model_name` and omit `--model`.

**Import errors**

- Run `pip install -e .` from the repo root so `galaxy_morphology` is on the path, or use `python scripts/train.py` (scripts add `src/` automatically).

---

## Next steps

- Read [README.md](README.md) for the model registry, experiment tracking, and metric artifacts.
- For code examples: `python example_usage.py`
