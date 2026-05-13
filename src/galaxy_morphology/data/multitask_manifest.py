"""Load optional multi-task labels (merger, bar, asymmetry) from a CSV manifest."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_multitask_manifest(
    csv_path: str | Path,
    data_dir: str | Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``path,merger,bar,asymmetry`` CSV (header required).

    The ``path`` column is relative to ``data_dir`` (e.g. ``spiral/foo.jpg``).
    Missing cells use mask 0 for that target. ``merger`` / ``bar`` are 0/1;
    ``asymmetry`` is in ``[0, 1]``.

    Returns:
        Map **absolute normalized path** -> ``(values (3,), mask (3,))``.
    """
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)
    if not csv_path.is_file():
        logger.warning("Multi-task manifest not found: %s", csv_path)
        return {}

    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    path_col = cols.get("path") or cols.get("relpath") or cols.get("relative_path")
    if not path_col:
        raise ValueError("Manifest needs a path column named path, relpath, or relative_path.")

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in df.iterrows():
        rel = str(row[path_col]).strip().replace("\\", "/")
        if not rel or rel.lower() == "nan":
            continue
        abs_path = str((data_dir / rel).resolve())
        key = os.path.normpath(abs_path)

        vals = np.zeros(3, dtype=np.float32)
        mask = np.zeros(3, dtype=np.float32)
        for i, name in enumerate(("merger", "bar", "asymmetry")):
            c = cols.get(name)
            if not c or pd.isna(row.get(c)):
                continue
            v = float(row[c])
            vals[i] = v
            mask[i] = 1.0

        out[key] = (vals, mask)

    logger.info("Loaded multi-task manifest: %d rows", len(out))
    return out


def multitask_targets_for_paths(
    paths: list[str],
    manifest: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``(aux, mask)`` per path (shape ``(N, 3)`` each)."""
    aux = np.zeros((len(paths), 3), dtype=np.float32)
    msk = np.zeros((len(paths), 3), dtype=np.float32)
    for i, p in enumerate(paths):
        key = os.path.normpath(str(Path(p).resolve()))
        hit = manifest.get(key)
        if hit is not None:
            aux[i], msk[i] = hit[0], hit[1]
    return aux, msk
