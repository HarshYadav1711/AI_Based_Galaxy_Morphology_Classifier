"""Model benchmarking utilities (local, no cloud)."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from galaxy_morphology.data.loaders import load_dataset
from galaxy_morphology.models.registry import build_model, count_parameters
from galaxy_morphology.training.loops import train_epoch, validate
from galaxy_morphology.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _reset_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_mem_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def benchmark_model(
    model_name: str,
    *,
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    train_epochs: int = 1,
    pretrained: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train briefly, validate, and measure params / speed / GPU memory.

    Uses the same stratified split as training (``seed``) for comparable metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed, deterministic_cudnn=False)
    train_loader, val_loader, class_names, _train_labels, _train_paths, _val_paths = load_dataset(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    num_classes = len(class_names)

    model = build_model(model_name, num_classes, pretrained=pretrained)
    model = model.to(device)
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    _reset_cuda_memory()
    for _ in range(train_epochs):
        train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=None,
            use_amp=False,
            max_grad_norm=1.0,
            use_mixup=False,
            mixup_alpha=0.0,
        )

    val_loss, val_acc, _, _, ext, _yt, _yp, _ys = validate(
        model, val_loader, criterion, device, class_names, use_amp=False
    )
    macro_f1 = float(ext.get("macro_f1", 0.0))

    model.eval()
    n_img = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            _ = model(images)
            n_img += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    ips = n_img / elapsed

    peak_mb = _peak_mem_mb()

    return {
        "model": model_name,
        "num_parameters": n_params,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "macro_f1": macro_f1,
        "inference_images_per_sec": float(ips),
        "peak_gpu_memory_mb": peak_mb,
        "train_epochs_ran": train_epochs,
    }


def write_benchmark_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in keys}
            w.writerow(row)
    logger.info("Wrote benchmark CSV: %s", p)


def write_benchmark_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Render a simple GitHub-flavored markdown table."""
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 1e6 else f"{v:.2e}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote benchmark markdown: %s", p)
