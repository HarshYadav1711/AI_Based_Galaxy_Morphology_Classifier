"""YAML configuration loading."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a nested dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration mapping.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Root of YAML must be a mapping, got {type(data)}")
    return data


def deep_update(
    base: MutableMapping[str, Any], overrides: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively update ``base`` with values from ``overrides`` (mutates ``base``)."""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_update(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base
