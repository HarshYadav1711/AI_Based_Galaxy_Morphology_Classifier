"""Explainability: Grad-CAM, MC dropout uncertainty, calibration, failure analysis."""

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
from galaxy_morphology.explainability.pipeline import run_explainability_for_image
from galaxy_morphology.explainability.target_layers import gradcam_target_layers

__all__ = [
    "expected_calibration_error",
    "explain_gradcam",
    "gradcam_target_layers",
    "mc_dropout_stats",
    "plot_failure_montage",
    "plot_reliability_diagram",
    "rank_failure_cases",
    "records_from_arrays",
    "run_explainability_for_image",
]
