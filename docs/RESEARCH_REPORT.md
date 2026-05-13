# Research report: AI-assisted galaxy morphology analysis

This document summarizes the **methodology**, **local experiments**, and **limitations** of this repository as an **astronomy-focused, local-first analysis system** (not merely a three-way classifier). Quantitative numbers should be filled from your own runs (`outputs/`, `outputs/scientific_eval/`, `outputs/dataset_study/`).

---

## 1. Methodology

### 1.1 Morphology and auxiliary tasks

- **Primary task:** three-class morphology (spiral / elliptical / irregular) with standard supervised training and optional class reweighting, label smoothing, and mixup.
- **Optional multi-task head** (`lightweight_multitask`): shared convolutional trunk with additional predictions:
  - **Merger likelihood** (binary logit + BCE when labels exist),
  - **Bar presence** (binary logit + BCE when labels exist),
  - **Asymmetry proxy** (scalar in [0, 1] with masked MSE when labels exist).
- **Label source:** CSV manifest keyed by paths relative to `data/<dataset_root>/` (see `configs/train.yaml` → `multitask.manifest_csv`). Missing cells yield **masked** auxiliary losses so training remains stable without fabricated labels.

### 1.2 Rotation robustness and TTA

- **Training:** configurable `RandomRotation(±rotation_degrees)` on the train split (`data.augmentation.rotation_degrees` in YAML).
- **Test-time augmentation (TTA):** `galaxy_morphology.inference.tta.predict_with_rotation_tta` averages softmax probabilities over 0°/90°/180°/270° rotations (no extra dependencies).

### 1.3 Active learning (human in the loop)

- **Queue:** JSONL records (e.g. low-confidence or `needs_human_review` samples) under `outputs/active_learning/review_queue.jsonl`.
- **Export:** `galaxy-al-export` writes a CSV with empty `merger`, `bar`, `asymmetry`, and `reviewer_notes` columns for manual completion.
- **Merge:** `galaxy_morphology.active_learning.merge_manifest.merge_review_to_manifest` produces a training-ready manifest from completed rows.

### 1.4 Scientific robustness evaluation

- **CLI:** `galaxy-scientific-eval` reports validation accuracy under:
  - baseline,
  - rotation-averaged predictions,
  - additive Gaussian noise in **normalized** image space,
  - synthetic low resolution (downsample then upsample).

### 1.5 Dataset characterization

- **CLI:** `galaxy-dataset-study` writes `outputs/dataset_study/report.md` with class counts, a Laplacian-variance sharpness proxy, and a small **augmentation impact** comparison (weak vs strong rotation policy on a synthetic patch).

---

## 2. Experiments (templates)

| Experiment | Command / artifact | What to record |
|------------|------------------|----------------|
| Baseline training | `galaxy-train --config configs/train.yaml` | Best `val_acc`, macro-F1, confusion matrix |
| Multi-task (optional manifest) | Set `model.name: lightweight_multitask` + `multitask.manifest_csv` | Auxiliary mask coverage, training stability |
| Robustness suite | `galaxy-scientific-eval --checkpoint checkpoints/best_model.pth` | `outputs/scientific_eval/robustness.json` |
| Dataset study | `galaxy-dataset-study` | `outputs/dataset_study/report.md` |
| Trust / calibration | `galaxy-trust-report` | `outputs/visualizations/evaluation_report.md` |

---

## 3. Findings (fill after your runs)

- **Rotation / TTA:** document whether rotation averaging recovers accuracy vs. baseline on your validation split.
- **Noise / low-res:** note accuracy drop magnitudes; relate to sensor noise and postage-stamp pixel scales relevant to your survey.
- **Class balance:** cite counts from the dataset study; link to any weighted loss or sampling choices.

---

## 4. Limitations

- **Auxiliary labels** are not inferred automatically from RGB alone; merger/bar/asymmetry require human or external catalog input for supervised auxiliary heads.
- **Multi-task architecture** is implemented for the **lightweight** CNN to keep the stack small and reproducible on CPU laptops.
- **Robustness metrics** are **in-distribution** stress tests; they do not replace domain shift benchmarks across telescopes or filters.
- **Active learning** exports are **local files**; workflow discipline (versioning manifests, reviewer provenance) is left to the research team.

---

## 5. Future work

- Extend multi-task heads to torchvision backbones behind a shared feature API.
- Incorporate redshift / band metadata when available (tabular fusion).
- Conformal or temperature scaling for **distribution-free** uncertainty beyond MC dropout.
- Larger rotation TTA policies and consistency regularization during training.

---

## References (informal)

- Galaxy Zoo / citizen-science morphologies for label semantics.
- SDSS / Rubin-era imaging for resolution and noise characteristics relevant to robustness interpretation.
