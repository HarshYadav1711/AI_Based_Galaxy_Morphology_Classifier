"""Dataset integrity: corrupted files, duplicates, imbalance, exportable stats."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def _file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def analyze_dataset(
    data_dir: str,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Scan class folders for readable images, duplicates (MD5), and imbalance.

    Returns:
        Structured report suitable for ``json`` export.
    """
    if class_names is None:
        class_names = ["spiral", "elliptical", "irregular"]

    report: dict[str, Any] = {
        "data_dir": str(Path(data_dir).resolve()),
        "per_class_counts": {},
        "corrupted_files": [],
        "duplicate_groups": [],
        "imbalance_warnings": [],
        "total_valid_images": 0,
    }

    hash_to_paths: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()

    for cls in class_names:
        cdir = Path(data_dir) / cls
        if not cdir.is_dir():
            report["imbalance_warnings"].append(f"Missing directory for class '{cls}': {cdir}")
            report["per_class_counts"][cls] = 0
            continue
        n = 0
        for name in os.listdir(cdir):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            fp = str(cdir / name)
            try:
                with Image.open(fp) as im:
                    im.verify()
            except Exception as exc:  # noqa: BLE001
                report["corrupted_files"].append({"path": fp, "error": str(exc)})
                continue
            try:
                with Image.open(fp) as im2:
                    im2.load()
            except Exception as exc:  # noqa: BLE001
                report["corrupted_files"].append({"path": fp, "error": str(exc)})
                continue
            digest = _file_md5(fp)
            hash_to_paths.setdefault(digest, []).append(fp)
            counts[cls] += 1
            n += 1
        report["per_class_counts"][cls] = n

    for paths in hash_to_paths.values():
        if len(paths) > 1:
            report["duplicate_groups"].append(paths)

    total = sum(counts.values())
    report["total_valid_images"] = total
    if total == 0:
        report["imbalance_warnings"].append("No valid images found.")
        return report

    max_c = max(counts.values()) if counts else 0
    min_c = min(v for v in counts.values() if v > 0) if any(counts.values()) else 0
    if min_c > 0 and max_c / min_c > 3.0:
        report["imbalance_warnings"].append(
            f"Class count ratio max/min ≈ {max_c / min_c:.2f} (>3); "
            "consider class weights or resampling."
        )
    if min_c < 5:
        report["imbalance_warnings"].append(
            "At least one class has fewer than 5 images; metrics and splits may be unstable."
        )

    return report


def save_dataset_statistics(path: str | Path, report: dict[str, Any]) -> None:
    """Write dataset analysis JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote dataset statistics: %s", p)
