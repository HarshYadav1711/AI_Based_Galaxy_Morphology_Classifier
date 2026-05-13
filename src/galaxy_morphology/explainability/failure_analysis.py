"""Failure-mode visualization helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SampleRecord(TypedDict, total=False):
    path: str
    y_true: int
    y_pred: int
    confidence: float
    correct: bool


def records_from_arrays(
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for i, p in enumerate(paths):
        out.append(
            {
                "path": p,
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "confidence": float(confidences[i]),
                "correct": bool(y_true[i] == y_pred[i]),
            }
        )
    return out


def plot_failure_montage(
    records: list[SampleRecord],
    class_names: list[str],
    save_path: str | Path,
    *,
    title: str,
    max_images: int = 6,
) -> None:
    """Show up to ``max_images`` thumbnails with path basename and confidence."""
    if not records:
        return
    take = records[:max_images]
    n = len(take)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, rec in zip(axes, take, strict=True):
        try:
            im = Image.open(rec["path"]).convert("RGB")
            ax.imshow(im)
        except Exception:  # noqa: BLE001
            ax.text(0.5, 0.5, "?", ha="center")
        ax.axis("off")
        pred = class_names[rec["y_pred"]]
        true = class_names[rec["y_true"]]
        ax.set_title(
            f"{Path(rec['path']).name}\nconf={rec['confidence']:.2f}\n{true}→{pred}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved montage: %s", p)


def rank_failure_cases(records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    """Confident wrong, least-confident correct, and lowest-confidence samples."""
    wrong = [r for r in records if not r["correct"]]
    right = [r for r in records if r["correct"]]

    confident_wrong = sorted(wrong, key=lambda r: r["confidence"], reverse=True)[:20]
    least_confident_right = sorted(right, key=lambda r: r["confidence"])[:20]
    uncertain = sorted(records, key=lambda r: r["confidence"])[:20]

    return {
        "confident_wrong": confident_wrong,
        "least_confident_correct": least_confident_right,
        "low_confidence_samples": uncertain,
    }
