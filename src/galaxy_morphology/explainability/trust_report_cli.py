"""CLI: validation trust report (calibration, failures, Grad-CAM, markdown)."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from galaxy_morphology.data.loaders import load_dataset
from galaxy_morphology.explainability.calibration import (
    expected_calibration_error,
    plot_reliability_diagram,
)
from galaxy_morphology.explainability.failure_analysis import (
    plot_failure_montage,
    rank_failure_cases,
    records_from_arrays,
)
from galaxy_morphology.explainability.gradcam import explain_gradcam
from galaxy_morphology.explainability.mc_dropout import mc_dropout_stats
from galaxy_morphology.explainability.plots import plot_confusion_matrix_png
from galaxy_morphology.explainability.report import write_evaluation_report_md
from galaxy_morphology.inference.predictor import load_model, preprocess_image
from galaxy_morphology.training.loops import validate
from galaxy_morphology.utils.logging_utils import setup_logging
from galaxy_morphology.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _safe_stem(path: str) -> str:
    base = Path(path).stem
    return re.sub(r"[^\w\-]+", "_", base, flags=re.UNICODE)[:80] or "sample"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="galaxy-trust-report",
        description=(
            "Evaluate a checkpoint on the validation split: metrics, ECE, reliability plot, "
            "confusion matrix, failure montages, example Grad-CAMs, and a markdown report."
        ),
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Trained .pth checkpoint.")
    p.add_argument("--data-dir", type=str, default="data/galaxies", help="Dataset root.")
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/visualizations",
        help="Output folder for figures and evaluation_report.md.",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Validation batch size.")
    p.add_argument("--image-size", type=int, default=224, help="Resize side length.")
    p.add_argument("--seed", type=int, default=42, help="Reproducible split seed.")
    p.add_argument(
        "--mc-dropout-samples",
        type=int,
        default=15,
        help="Forward passes per image for MC-dropout subset stats.",
    )
    p.add_argument(
        "--mc-subset",
        type=int,
        default=48,
        help="Number of validation images for MC-dropout aggregate (cap for speed).",
    )
    p.add_argument(
        "--gradcam-examples",
        type=int,
        default=4,
        help="Number of validation images to explain with Grad-CAM.",
    )
    args = p.parse_args(argv)
    setup_logging("INFO")
    set_seed(args.seed, deterministic_cudnn=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "confusion").mkdir(parents=True, exist_ok=True)
    (out / "failure").mkdir(parents=True, exist_ok=True)
    (out / "gradcam").mkdir(parents=True, exist_ok=True)

    model, class_names, model_name = load_model(args.checkpoint, device)
    logger.info("Model=%s classes=%s device=%s", model_name, class_names, device)

    _train_loader, val_loader, _names, _tl, _tp, val_paths = load_dataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    if _names != class_names:
        logger.warning(
            "Class order in data dir may differ from checkpoint; using checkpoint names."
        )

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, report, cm, _ext, y_true, y_pred, y_score = validate(
        model, val_loader, criterion, device, class_names, use_amp=False
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(len(class_names))))

    confidences = y_score.max(axis=1)
    ece = expected_calibration_error(y_true, y_score, n_bins=15)
    rel_path = out / "calibration" / "reliability_diagram.png"
    plot_reliability_diagram(y_true, y_score, rel_path, n_bins=10, title="Reliability (validation)")

    cm_path = out / "confusion" / "confusion_matrix.png"
    plot_confusion_matrix_png(cm, class_names, cm_path, title="Validation confusion matrix")

    paths_ordered: list[str] = []
    pi = 0
    for batch_images, _ in val_loader:
        b = batch_images.size(0)
        paths_ordered.extend(val_paths[pi : pi + b])
        pi += b

    records = records_from_arrays(paths_ordered, y_true, y_pred, confidences)
    ranked = rank_failure_cases(records)
    plot_failure_montage(
        ranked["confident_wrong"],
        class_names,
        out / "failure" / "most_confident_wrong.png",
        title="Most confident wrong predictions",
    )
    plot_failure_montage(
        ranked["least_confident_correct"],
        class_names,
        out / "failure" / "least_confident_correct.png",
        title="Least confident correct predictions",
    )
    plot_failure_montage(
        ranked["low_confidence_samples"],
        class_names,
        out / "failure" / "low_confidence_samples.png",
        title="Low-confidence samples (sorted by confidence)",
    )

    n_mc = min(args.mc_subset, len(paths_ordered))
    mc_mean_confs: list[float] = []
    mc_unc: list[float] = []
    mc_review_frac: list[float] = []
    for i in range(n_mc):
        path = paths_ordered[i]
        tensor, _ = preprocess_image(path, args.image_size)
        tensor = tensor.to(device)
        st = mc_dropout_stats(
            model,
            tensor,
            num_samples=args.mc_dropout_samples,
            device=device,
        )
        mc_mean_confs.append(float(st["mean_confidence"]))
        mc_unc.append(float(st["uncertainty_score"]))
        mc_review_frac.append(1.0 if st["needs_human_review"] else 0.0)

    gc_rel: list[str] = []
    n_g = min(args.gradcam_examples, len(paths_ordered))
    for i in range(n_g):
        path = paths_ordered[i]
        tensor, _ = preprocess_image(path, args.image_size)
        tensor = tensor.to(device)
        stem = f"val_{i}_{_safe_stem(path)}"
        pred_i = int(y_pred[i])
        explain_gradcam(
            model,
            tensor,
            device,
            model_name,
            pred_i,
            out_dir=out / "gradcam",
            stem=stem,
        )
        gc_rel.append(f"gradcam/{stem}_gradcam_compare.png")

    per_class_lines = []
    for k, v in report.items():
        if isinstance(v, dict) and "precision" in v:
            p = v.get("precision", 0)
            r = v.get("recall", 0)
            f1 = v.get("f1-score", 0)
            per_class_lines.append(f"{k}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    metrics_lines = [
        f"- **Validation loss:** {val_loss:.4f}",
        f"- **Validation accuracy:** {val_acc:.4f}",
        f"- **Macro F1:** {macro_f1:.4f}",
        "",
        "### Per-class (sklearn report)",
        "```",
        *per_class_lines,
        "```",
        "",
        f"### MC dropout (first {n_mc} validation images, {args.mc_dropout_samples} samples each)",
        f"- **Mean of mean confidence:** {float(np.mean(mc_mean_confs)):.4f}",
        f"- **Mean uncertainty score:** {float(np.mean(mc_unc)):.4f}",
        f"- **Fraction flagged needs_human_review:** {float(np.mean(mc_review_frac)):.4f}",
    ]

    report_md = out / "evaluation_report.md"
    write_evaluation_report_md(
        report_md,
        title="Galaxy morphology — trust & explainability report",
        metrics_lines=metrics_lines,
        ece=ece,
        rel_diagram_rel="calibration/reliability_diagram.png",
        cm_rel="confusion/confusion_matrix.png",
        gradcam_examples_rel=gc_rel,
        failure_montages_rel=[
            "failure/most_confident_wrong.png",
            "failure/least_confident_correct.png",
            "failure/low_confidence_samples.png",
        ],
    )
    logger.info("Wrote report: %s", report_md.resolve())


if __name__ == "__main__":
    main()
