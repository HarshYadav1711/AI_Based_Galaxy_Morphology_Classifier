"""Extended classification metrics (F1, balanced accuracy, ROC-AUC, etc.)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute macro / weighted F1, balanced accuracy, and ROC-AUC (OvR) when valid.

    Args:
        y_true: Integer labels ``(n,)``.
        y_pred: Integer predictions ``(n,)``.
        y_score: Softmax probabilities ``(n, n_classes)``.
        class_names: Label names in class index order.

    Returns:
        JSON-serializable dict including per-class AP where applicable.
    """
    n_classes = len(class_names)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    out: dict[str, Any] = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "roc_auc_ovr": None,
        "per_class_average_precision": {},
    }

    present = np.unique(y_true)
    if len(present) >= 2 and y_score.shape[1] >= 2:
        try:
            out["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            )
        except ValueError as exc:
            logger.debug("ROC-AUC skipped: %s", exc)

    y_bin = np.zeros((len(y_true), n_classes), dtype=int)
    for i, t in enumerate(y_true):
        if 0 <= int(t) < n_classes:
            y_bin[i, int(t)] = 1

    for c in range(n_classes):
        if np.any(y_bin[:, c]):
            try:
                ap = average_precision_score(y_bin[:, c], y_score[:, c])
                out["per_class_average_precision"][class_names[c]] = float(ap)
            except ValueError:
                pass

    return out


def roc_curve_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """FPR/TPR per class (one-vs-rest) for plotting."""
    n_classes = len(class_names)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    curves: dict[str, dict[str, list[float]]] = {}
    y_bin = np.zeros((len(y_true), n_classes), dtype=int)
    for i, t in enumerate(y_true):
        if 0 <= int(t) < n_classes:
            y_bin[i, int(t)] = 1
    for c in range(n_classes):
        if np.sum(y_bin[:, c]) == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, c])
            curves[class_names[c]] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        except ValueError:
            continue
    return {"curves": curves}


def precision_recall_curve_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Precision/recall arrays per class for plotting."""
    n_classes = len(class_names)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    curves: dict[str, dict[str, list[float]]] = {}
    y_bin = np.zeros((len(y_true), n_classes), dtype=int)
    for i, t in enumerate(y_true):
        if 0 <= int(t) < n_classes:
            y_bin[i, int(t)] = 1
    for c in range(n_classes):
        if np.sum(y_bin[:, c]) == 0:
            continue
        try:
            prec, rec, _ = precision_recall_curve(y_bin[:, c], y_score[:, c])
            curves[class_names[c]] = {"precision": prec.tolist(), "recall": rec.tolist()}
        except ValueError:
            continue
    return {"curves": curves}
