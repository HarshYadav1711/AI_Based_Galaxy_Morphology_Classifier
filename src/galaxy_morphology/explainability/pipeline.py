"""Per-image explainability: prediction summary, MC dropout, Grad-CAM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from galaxy_morphology.explainability.gradcam import explain_gradcam
from galaxy_morphology.explainability.mc_dropout import mc_dropout_stats
from galaxy_morphology.inference.predictor import predict

logger = logging.getLogger(__name__)


def top_k_class_probs(class_probs: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:k]


def run_explainability_for_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    class_names: list[str],
    model_name: str,
    *,
    stem: str,
    out_dir: Path,
    mc_dropout_samples: int = 20,
    gradcam_target_class: int | None = None,
) -> dict[str, Any]:
    """Run prediction, top-3, MC-dropout stats, and Grad-CAM; return structured result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_label, confidence, probs = predict(model, image_tensor, device, class_names)
    top3 = top_k_class_probs(probs, 3)
    pred_idx = next(i for i, n in enumerate(class_names) if n == pred_label)
    target_class = gradcam_target_class if gradcam_target_class is not None else pred_idx

    mc = mc_dropout_stats(
        model,
        image_tensor,
        num_samples=mc_dropout_samples,
        device=device,
    )

    gc = explain_gradcam(
        model,
        image_tensor,
        device,
        model_name,
        target_class,
        out_dir=out_dir,
        stem=stem,
    )

    result: dict[str, Any] = {
        "predicted_class": pred_label,
        "confidence": confidence,
        "top3": [{"class": n, "probability": float(p)} for n, p in top3],
        "mc_dropout": mc,
        "gradcam": gc,
    }
    logger.info(
        "Explain %s: pred=%s conf=%.3f review=%s",
        stem,
        pred_label,
        confidence,
        mc.get("needs_human_review"),
    )
    return result
