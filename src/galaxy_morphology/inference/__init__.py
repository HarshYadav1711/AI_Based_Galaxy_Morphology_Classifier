"""Inference package."""

from galaxy_morphology.inference.cli import build_parser, main
from galaxy_morphology.inference.predictor import (
    load_model,
    predict,
    predict_batch,
    preprocess_image,
)

__all__ = [
    "build_parser",
    "load_model",
    "main",
    "predict",
    "predict_batch",
    "preprocess_image",
]
