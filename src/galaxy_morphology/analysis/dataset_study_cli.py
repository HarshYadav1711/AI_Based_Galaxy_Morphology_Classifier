"""CLI: write dataset study markdown under ``outputs/dataset_study/``."""

from __future__ import annotations

import argparse
import logging

from galaxy_morphology.analysis.dataset_study import write_dataset_study_report
from galaxy_morphology.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Dataset study: class balance, sharpness proxy, augmentation stats.",
    )
    p.add_argument("--data-dir", type=str, default="data/galaxies")
    p.add_argument("--out", type=str, default="outputs/dataset_study/report.md")
    args = p.parse_args(argv)
    setup_logging("INFO")
    write_dataset_study_report(args.data_dir, args.out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
