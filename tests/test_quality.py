"""Dataset quality analysis tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from galaxy_morphology.data.quality import analyze_dataset


def test_analyze_dataset_counts(tmp_path: Path) -> None:
    for cls in ("spiral", "elliptical", "irregular"):
        d = tmp_path / cls
        d.mkdir()
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(d / f"{cls}_a.png")
    r = analyze_dataset(str(tmp_path))
    assert r["total_valid_images"] == 3
    assert r["per_class_counts"]["spiral"] == 1
