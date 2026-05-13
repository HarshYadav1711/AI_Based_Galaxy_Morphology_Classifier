"""Calibration metrics (ECE) and reliability diagrams."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Expected calibration error using top-class confidence (common multiclass summary)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        elif i == 0:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        prop = float(mask.sum()) / max(n, 1)
        if prop < 1e-12:
            continue
        ece += abs(float(correct[mask].mean()) - float(conf[mask].mean())) * prop
    return float(ece)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram (top-class confidence)",
) -> None:
    """Save a reliability diagram (mean confidence vs accuracy per bin)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        elif i == 0:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        xs.append(float(conf[mask].mean()))
        ys.append(float(acc[mask].mean()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    if xs:
        ax.plot(xs, ys, "o-", label="Model", linewidth=2)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved reliability diagram: %s", p)
