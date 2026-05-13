"""Export queued samples to a CSV humans can fill for retraining / manifest merge."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_review_csv(
    records: list[dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Write a CSV with prediction metadata plus **empty** annotation columns.

    After review, fill ``merger``, ``bar`` (0/1), and ``asymmetry`` ([0,1]) then merge into
    ``data/multitask_manifest.csv`` using the same ``path`` column (relative to dataset root).
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "predicted_class",
        "confidence",
        "uncertainty_score",
        "needs_human_review",
        "merger",
        "bar",
        "asymmetry",
        "reviewer_notes",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: "" for k in fieldnames}
            row["path"] = r.get("path") or r.get("image_path") or ""
            row["predicted_class"] = r.get("predicted_class", "")
            row["confidence"] = r.get("confidence", "")
            row["uncertainty_score"] = r.get("uncertainty_score", "")
            row["needs_human_review"] = r.get("needs_human_review", "")
            w.writerow(row)
    logger.info("Wrote review CSV (%d rows): %s", len(records), p)
