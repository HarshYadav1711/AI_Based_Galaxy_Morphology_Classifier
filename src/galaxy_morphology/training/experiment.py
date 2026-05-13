"""Lightweight local experiment directories (JSONL + CSV, no cloud)."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_experiment_dir(
    root: str | Path,
    *,
    slug: str,
    config_path: Path | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> Path:
    """Create ``<root>/<timestamp>_<slug>/`` and optionally copy config."""
    root = Path(root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp = root / f"{ts}_{slug}"
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "checkpoints").mkdir(exist_ok=True)
    (exp / "outputs").mkdir(exist_ok=True)
    (exp / "figures").mkdir(exist_ok=True)

    if config_path is not None and config_path.is_file():
        shutil.copy2(config_path, exp / "config_copy.yaml")
    if resolved_config is not None:
        with (exp / "config_resolved.json").open("w", encoding="utf-8") as f:
            json.dump(resolved_config, f, indent=2, default=str)

    logger.info("Experiment directory: %s", exp.resolve())
    return exp


def append_metrics_jsonl(exp_dir: Path, row: dict[str, Any]) -> None:
    """Append one JSON object per line to ``metrics.jsonl``."""
    path = exp_dir / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
