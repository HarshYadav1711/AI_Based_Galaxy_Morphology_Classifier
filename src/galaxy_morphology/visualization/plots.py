"""Plotting utilities for training curves and confusion matrix."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

logger = logging.getLogger(__name__)


def plot_training_history(history: dict[str, list[float]], save_path: str | Path) -> None:
    """Save train/val loss and accuracy curves to a PNG file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close(fig)
    logger.info("Saved training history plot: %s", p)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: Sequence[str], save_path: str | Path
) -> None:
    """Save confusion matrix heatmap to PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", p)


def plot_class_distribution(
    class_counts: dict[str, int],
    save_path: str | Path,
) -> None:
    """Bar chart of image counts per class."""
    names = list(class_counts.keys())
    vals = [class_counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, vals, color="steelblue")
    ax.set_ylabel("Count")
    ax.set_title("Training set class distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close(fig)
    logger.info("Saved class distribution: %s", p)


def plot_roc_curves(
    roc_data: dict[str, object],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """One-vs-rest ROC curves (uses JSON structure from :func:`roc_curve_data`)."""
    curves = roc_data.get("curves", {})
    if not curves:
        logger.warning("No ROC data to plot; skipping.")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in class_names:
        if name not in curves:
            continue
        c = curves[name]
        ax.plot(c["fpr"], c["tpr"], label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close(fig)
    logger.info("Saved ROC curves: %s", p)


def plot_pr_curves(
    pr_data: dict[str, object],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """Per-class precision–recall curves."""
    curves = pr_data.get("curves", {})
    if not curves:
        logger.warning("No PR data to plot; skipping.")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in class_names:
        if name not in curves:
            continue
        c = curves[name]
        ax.plot(c["recall"], c["precision"], label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall (one-vs-rest)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close(fig)
    logger.info("Saved PR curves: %s", p)
