#!/usr/bin/env python3
"""Train launcher: ensures ``src`` is on ``sys.path`` without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from galaxy_morphology.training.cli import main

if __name__ == "__main__":
    main()
