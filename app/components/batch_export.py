"""ZIP / multi-file batch inference helpers."""

from __future__ import annotations

import contextlib
import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images_from_zip(data: bytes) -> list[tuple[str, bytes]]:
    """Return (virtual_path, file_bytes) for each image inside the ZIP."""
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or Path(name).name.startswith("."):
                continue
            suf = Path(name).suffix.lower()
            if suf not in _IMAGE_EXTS:
                continue
            with zf.open(name) as f:
                out.append((name, f.read()))
    return out


def write_uploaded_images_to_temp(
    files: list[tuple[str, bytes]],
) -> tuple[tempfile.TemporaryDirectory[str], list[str]]:
    """Write (name, bytes) pairs to a temp tree; return (tmpdir_ctx, absolute_paths)."""
    tmp = tempfile.TemporaryDirectory(prefix="galaxy_batch_")
    paths: list[str] = []
    root = Path(tmp.name)
    for i, (_name, raw) in enumerate(files):
        safe = Path(_name).name
        if not safe or safe.startswith("."):
            continue
        dest = root / f"{i:05d}_{safe}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        try:
            with Image.open(dest) as im:
                im.verify()
            with Image.open(dest) as im2:
                im2.load()
            paths.append(str(dest.resolve()))
        except OSError:
            with contextlib.suppress(OSError):
                dest.unlink(missing_ok=True)
    return tmp, paths


def predictions_to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b"image,predicted_class,confidence\n"
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
