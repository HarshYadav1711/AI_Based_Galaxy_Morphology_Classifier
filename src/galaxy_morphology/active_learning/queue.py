"""Append / read low-confidence samples for human review (JSONL queue)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUEUE = Path("outputs/active_learning/review_queue.jsonl")


def append_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Append one JSON object per line (easy to stream and version-control locally)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Appended %d records to %s", len(records), p)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_low_confidence(
    records: list[dict[str, Any]],
    *,
    max_confidence: float = 0.65,
    require_review_flag: bool | None = None,
) -> list[dict[str, Any]]:
    """Keep items with ``confidence`` <= threshold (active-learning style pool)."""
    out: list[dict[str, Any]] = []
    for r in records:
        if "confidence" not in r:
            continue
        if float(r["confidence"]) > max_confidence:
            continue
        flag = bool(r.get("needs_human_review"))
        if require_review_flag is not None and flag != require_review_flag:
            continue
        out.append(r)
    return out
