"""Early stopping on validation metric."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

MonitorMode = Literal["min", "max"]


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: MonitorMode = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float | None = None
        self._wait = 0

    def reset(self) -> None:
        self._best = None
        self._wait = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.patience <= 0:
            return False
        if self._best is None:
            self._best = value
            return False
        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )
        if improved:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                logger.info("Early stopping triggered after patience=%d", self.patience)
                return True
        return False
