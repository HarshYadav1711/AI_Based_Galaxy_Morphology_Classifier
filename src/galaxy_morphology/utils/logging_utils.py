"""Structured logging compatible with tqdm progress bars."""

from __future__ import annotations

import logging
import sys

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Emit log records via ``tqdm.write`` so output does not break progress bars."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:  # noqa: BLE001 — logging must not raise from emit
            self.handleError(record)


def setup_logging(
    level: str | int = logging.INFO,
    *,
    use_tqdm_handler: bool = True,
) -> None:
    """Configure root logging once (idempotent if handlers already present).

    Args:
        level: Logging level name or int.
        use_tqdm_handler: If True, attach :class:`TqdmLoggingHandler` for stderr.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    h = TqdmLoggingHandler(level) if use_tqdm_handler else logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    h.setLevel(level)
    root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger under the project namespace."""
    return logging.getLogger(name)
