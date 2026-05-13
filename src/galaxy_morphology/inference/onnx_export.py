"""Optional ONNX export for deployment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    input_size: int = 224,
    batch_size: int = 1,
    opset_version: int = 17,
) -> None:
    """Export ``model`` in eval mode to ONNX (CPU example batch)."""
    model.eval()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(batch_size, 3, input_size, input_size, device="cpu")
    model_cpu = model.cpu()
    export_kw: dict[str, Any] = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "dynamic_axes": {"images": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset_version,
    }
    try:
        torch.onnx.export(
            model_cpu,
            dummy,
            str(p),
            dynamo=False,
            **export_kw,
        )
    except TypeError:
        torch.onnx.export(model_cpu, dummy, str(p), **export_kw)
    logger.info("Exported ONNX model to %s", p.resolve())
