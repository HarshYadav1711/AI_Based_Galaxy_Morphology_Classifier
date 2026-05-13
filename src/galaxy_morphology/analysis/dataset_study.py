"""Dataset characterization: class balance, simple sharpness proxy, augmentation statistics."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


def _collect_paths(data_dir: Path) -> tuple[list[str], list[str]]:
    class_names = ["spiral", "elliptical", "irregular"]
    paths: list[str] = []
    labels: list[str] = []
    for c in class_names:
        d = data_dir / c
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                paths.append(str(f.resolve()))
                labels.append(c)
    return paths, labels


def _laplacian_variance(gray: np.ndarray) -> float:
    """Simple sharpness / focus proxy (higher often means sharper edges)."""
    g = gray.astype(np.float32)
    lap = (
        -4 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    return float(np.var(lap))


def augmentation_batch_stats(
    image_size: int = 224,
    *,
    rotation_weak: int = 15,
    rotation_strong: int = 180,
) -> dict[str, float]:
    """Compare mean/std of augmented tensors (no model) — mild vs strong rotation policy."""
    base = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    t_weak = transforms.Compose(
        base + [transforms.RandomRotation(rotation_weak), transforms.RandomHorizontalFlip()]
    )
    t_strong = transforms.Compose(
        base + [transforms.RandomRotation(rotation_strong), transforms.RandomHorizontalFlip()]
    )
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    im = Image.fromarray(x.astype(np.uint8))
    w = torch.stack([t_weak(im) for _ in range(32)])
    s = torch.stack([t_strong(im) for _ in range(32)])
    return {
        "weak_batch_mean": float(w.mean().item()),
        "weak_batch_std": float(w.std().item()),
        "strong_batch_mean": float(s.mean().item()),
        "strong_batch_std": float(s.std().item()),
    }


def write_dataset_study_report(
    data_dir: str | Path,
    out_md: str | Path,
    *,
    max_images_per_class: int = 80,
    rotation_weak: int = 15,
    rotation_strong: int = 180,
) -> None:
    """Write ``report.md`` with class counts, sharpness proxy summary, augmentation stats."""
    data_dir = Path(data_dir)
    out_md = Path(out_md)
    paths, labels = _collect_paths(data_dir)
    counts = Counter(labels)
    lines = [
        "# Dataset analysis (local)",
        "",
        "## Class distribution",
        "",
        "| Class | Count |",
        "|-------|-------|",
    ]
    for c in ["spiral", "elliptical", "irregular"]:
        lines.append(f"| {c} | {counts.get(c, 0)} |")
    lines.extend(["", "## Image sharpness proxy (Laplacian variance)", ""])

    sharp: list[float] = []
    by_class: dict[str, list[float]] = {c: [] for c in ["spiral", "elliptical", "irregular"]}
    for p, lab in zip(paths, labels, strict=False):
        if len(sharp) >= max_images_per_class * 3:
            break
        try:
            with Image.open(p) as im:
                g = np.asarray(im.convert("L"), dtype=np.uint8)
            v = _laplacian_variance(g)
            sharp.append(v)
            by_class[lab].append(v)
        except OSError:
            continue

    if sharp:
        lines.append(f"- **Mean Laplacian variance:** {float(np.mean(sharp)):.2f}")
        lines.append(f"- **Std:** {float(np.std(sharp)):.2f}")
        lines.append("")
        for c, vals in by_class.items():
            if vals:
                lines.append(f"- **{c} mean:** {float(np.mean(vals)):.2f}")
        lines.append("")
    else:
        lines.append("_No images sampled._", "")

    aug = augmentation_batch_stats(rotation_weak=rotation_weak, rotation_strong=rotation_strong)
    lines.extend(
        [
            "## Augmentation impact (tensor statistics, single random patch)",
            "",
            f"- Weak rotation (±{rotation_weak}°) batch std: **{aug['weak_batch_std']:.4f}**",
            f"- Strong rotation (±{rotation_strong}°) batch std: **{aug['strong_batch_std']:.4f}**",
            "",
            "Stronger rotation increases diversity (higher spread across stochastic samples).",
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote dataset study: %s", out_md)
