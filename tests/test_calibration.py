"""Tests for calibration helpers."""

from __future__ import annotations

import numpy as np

from galaxy_morphology.explainability.calibration import expected_calibration_error


def test_ece_confident_and_correct() -> None:
    """Always correct with high confidence on predicted class -> low ECE."""
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_prob = np.tile(np.array([0.99, 0.005, 0.005], dtype=np.float64), (n, 1))
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05


def test_ece_miscalibrated_high_conf_wrong() -> None:
    """Model always predicts class 0 with 0.99 confidence but labels random -> high ECE."""
    n = 500
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, size=n)
    y_prob = np.full((n, 3), 0.005, dtype=np.float64)
    y_prob[:, 0] = 0.99
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece > 0.25
