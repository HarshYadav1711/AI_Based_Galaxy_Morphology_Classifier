#!/usr/bin/env python3
"""Sample data launcher (adds ``src`` to ``sys.path``)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from galaxy_morphology.data.sample_download import main

if __name__ == "__main__":
    main()
