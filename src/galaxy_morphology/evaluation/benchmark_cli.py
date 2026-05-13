"""CLI to benchmark registered models on a local dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from galaxy_morphology.evaluation.benchmark import (
    benchmark_model,
    write_benchmark_csv,
    write_benchmark_markdown,
)
from galaxy_morphology.models.registry import list_model_names
from galaxy_morphology.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark model architectures on the same data split.")
    p.add_argument("--data-dir", type=str, default="data/galaxies", help="Dataset root.")
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all registered).",
    )
    p.add_argument("--epochs", type=int, default=1, help="Short training epochs per model.")
    p.add_argument("--batch-size", type=int, default=16, help="Loader batch size.")
    p.add_argument("--image-size", type=int, default=224, help="Resize side length.")
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random-init torchvision backbones (faster smoke test, worse metrics).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/benchmarks",
        help="Directory for benchmark_results.csv and benchmark_table.md.",
    )
    p.add_argument("--seed", type=int, default=42, help="Split / init seed.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging("INFO")
    names = list_model_names()
    if args.models:
        chosen = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in chosen:
            if m not in names:
                raise SystemExit(f"Unknown model '{m}'. Options: {names}")
    else:
        chosen = names

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in chosen:
        logger.info("Benchmarking %s ...", m)
        row = benchmark_model(
            m,
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            train_epochs=args.epochs,
            pretrained=not args.no_pretrained,
            seed=args.seed,
        )
        rows.append(row)

    write_benchmark_csv(rows, out / "benchmark_results.csv")
    write_benchmark_markdown(rows, out / "benchmark_table.md")
    logger.info("Done. Wrote %s and %s", out / "benchmark_results.csv", out / "benchmark_table.md")


if __name__ == "__main__":
    main()
