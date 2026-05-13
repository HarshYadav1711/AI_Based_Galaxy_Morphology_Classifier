"""Inference package."""

from galaxy_morphology.inference.cli import build_parser, main
from galaxy_morphology.inference.onnx_export import export_onnx
from galaxy_morphology.inference.predictor import (
    benchmark_inference,
    load_model,
    predict,
    predict_batch,
    predict_paths_batched,
    preprocess_image,
)

__all__ = [
    "benchmark_inference",
    "build_parser",
    "export_onnx",
    "load_model",
    "main",
    "predict",
    "predict_batch",
    "predict_paths_batched",
    "preprocess_image",
]
