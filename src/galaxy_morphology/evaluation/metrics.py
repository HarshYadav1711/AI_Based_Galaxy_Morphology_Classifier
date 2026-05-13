"""Export metrics and reports to disk."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """Convert nested dicts/lists with numpy scalars to JSON-serializable types."""
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    return obj


def save_metrics_json(path: str | Path, metrics: dict[str, Any]) -> None:
    """Write ``metrics`` to ``path`` as formatted JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    logger.info("Wrote metrics: %s", p)


def save_classification_report_json(path: str | Path, report: dict[str, Any]) -> None:
    """Write sklearn-style classification report dict to JSON."""
    save_metrics_json(path, report)


def append_training_history_csv(
    path: str | Path,
    row: dict[str, Any],
    fieldnames: list[str] | None = None,
) -> None:
    """Append one training epoch row to CSV (writes header on first row)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(row.keys())
    write_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def write_training_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Overwrite CSV with full training history."""
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("Wrote training history CSV: %s", p)
