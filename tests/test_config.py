"""Configuration loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from galaxy_morphology.utils.config import deep_update, load_yaml_config


def test_load_train_config(project_root: Path) -> None:
    cfg = load_yaml_config(project_root / "configs" / "train.yaml")
    assert "training" in cfg
    assert cfg["data"]["dir"] == "data/galaxies"


def test_deep_update() -> None:
    base = {"a": 1, "nested": {"x": 1}}
    deep_update(base, {"nested": {"y": 2}, "b": 3})
    assert base["nested"]["x"] == 1 and base["nested"]["y"] == 2 and base["b"] == 3


def test_load_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "nope.yaml")
