"""Tests for extended sklearn-based metrics."""

from __future__ import annotations

import numpy as np

from galaxy_morphology.evaluation.classification_metrics import compute_extended_metrics


def test_extended_metrics_basic() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    y_score = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.1, 0.7],
            [0.2, 0.7, 0.1],
        ]
    )
    names = ["a", "b", "c"]
    m = compute_extended_metrics(y_true, y_pred, y_score, names)
    assert "macro_f1" in m and "weighted_f1" in m
    assert m["balanced_accuracy"] >= 0.0
