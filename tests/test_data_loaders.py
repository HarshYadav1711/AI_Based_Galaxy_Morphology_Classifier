"""Dataset loading tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from galaxy_morphology.data.loaders import load_dataset


def _write_dummy_images(root: Path) -> None:
    for cls in ("spiral", "elliptical", "irregular"):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(10 + i, 20, 30))
            img.save(d / f"{cls}_{i}.png")


def test_load_dataset(tmp_path: Path) -> None:
    _write_dummy_images(tmp_path)
    train_loader, val_loader, names, train_labels, _tp, _vp = load_dataset(
        data_dir=str(tmp_path),
        train_split=0.67,
        image_size=32,
        batch_size=2,
        num_workers=0,
        seed=0,
    )
    assert names == ["spiral", "elliptical", "irregular"]
    assert len(train_labels) == len(train_loader.dataset)
    assert len(train_loader) >= 1 and len(val_loader) >= 1
    batch = next(iter(train_loader))
    assert batch[0].shape[0] <= 2
    assert batch[0].shape[1:] == (3, 32, 32)
