"""Multi-task manifest loading."""

from __future__ import annotations

from pathlib import Path

from galaxy_morphology.data.multitask_manifest import (
    load_multitask_manifest,
    multitask_targets_for_paths,
)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "galaxies"
    (root / "spiral").mkdir(parents=True)
    img = root / "spiral" / "a.png"
    img.write_bytes(b"fake")  # not a real PNG; manifest path resolution only

    csv = tmp_path / "m.csv"
    csv.write_text("path,merger,bar,asymmetry\nspiral/a.png,1,0,0.4\n", encoding="utf-8")
    m = load_multitask_manifest(csv, root)
    aux, mask = multitask_targets_for_paths([str(img.resolve())], m)
    assert aux.shape == (1, 3) and mask.shape == (1, 3)
    assert mask[0, 0] == 1.0 and float(aux[0, 0]) == 1.0
    assert abs(float(aux[0, 2]) - 0.4) < 1e-5
