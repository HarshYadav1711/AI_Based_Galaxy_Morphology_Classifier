"""Merge a human-reviewed CSV into a multi-task manifest for training."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _blank(x: object) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and str(x).strip() == "":
        return True
    try:
        if pd.isna(x):
            return True
    except (TypeError, ValueError):
        pass
    return False


def merge_review_to_manifest(
    review_csv: str | Path,
    out_manifest: str | Path,
    *,
    only_completed_rows: bool = True,
) -> int:
    """Copy rows where auxiliary columns are filled into ``out_manifest`` (overwrite/create).

    Expected columns include ``path``, ``merger``, ``bar``, ``asymmetry``.
    Returns number of rows written.
    """
    review_csv = Path(review_csv)
    out_manifest = Path(out_manifest)
    if not review_csv.is_file():
        raise FileNotFoundError(review_csv)
    df = pd.read_csv(review_csv)
    cols = {c.lower().strip(): c for c in df.columns}
    pc = cols.get("path")
    if not pc:
        raise ValueError("review CSV needs a path column")

    rows: list[dict[str, float | str]] = []
    for _, r in df.iterrows():
        path = str(r[pc]).strip()
        if not path:
            continue
        mc = cols.get("merger", "merger")
        bc = cols.get("bar", "bar")
        ac = cols.get("asymmetry", "asymmetry")
        merger = r.get(mc, np.nan)
        bar = r.get(bc, np.nan)
        asym = r.get(ac, np.nan)
        if only_completed_rows and (_blank(merger) or _blank(bar) or _blank(asym)):
            continue
        rows.append(
            {
                "path": path.replace("\\", "/"),
                "merger": float(merger),
                "bar": float(bar),
                "asymmetry": float(asym),
            }
        )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    logger.info("Wrote manifest %s (%d rows)", out_manifest, len(rows))
    return len(rows)
