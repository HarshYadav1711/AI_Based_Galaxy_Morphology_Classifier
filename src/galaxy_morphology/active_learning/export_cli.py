"""CLI: export active-learning review CSV from a JSONL queue."""

from __future__ import annotations

import argparse
import logging

from galaxy_morphology.active_learning.export_review import export_review_csv
from galaxy_morphology.active_learning.queue import filter_low_confidence, load_records
from galaxy_morphology.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Export human-review CSV from active-learning JSONL queue.",
    )
    p.add_argument("--queue", type=str, default="outputs/active_learning/review_queue.jsonl")
    p.add_argument("--out", type=str, default="outputs/active_learning/review_export.csv")
    p.add_argument("--max-confidence", type=float, default=0.65)
    args = p.parse_args(argv)
    setup_logging("INFO")
    all_r = load_records(args.queue)
    low = filter_low_confidence(all_r, max_confidence=args.max_confidence)
    export_review_csv(low, args.out)
    logger.info("Queued total=%d low-conf=%d -> %s", len(all_r), len(low), args.out)


if __name__ == "__main__":
    main()
