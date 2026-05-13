"""Command-line interface for training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from galaxy_morphology.training.run import run_training
from galaxy_morphology.utils.config import deep_update, load_yaml_config
from galaxy_morphology.utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Construct the training CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="galaxy-train",
        description=(
            "Train the galaxy morphology classifier. Hyperparameters are read from a YAML "
            "config; CLI flags override selected fields for quick experiments."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML training configuration (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data.data_dir / data.dir: root folder with class subdirectories.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override data.batch_size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override training.lr.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["lightweight", "efficient"],
        help="Override model.name.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override training.checkpoint.save_dir.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume (sets training.resume).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help="Override outputs.dir for metrics and CSV exports.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry: load config, apply overrides, run training."""
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path.resolve()}")

    cfg: dict[str, Any] = load_yaml_config(config_path)

    overrides: dict[str, Any] = {}
    if args.data_dir is not None:
        overrides.setdefault("data", {})["dir"] = args.data_dir
    if args.epochs is not None:
        if args.epochs < 1:
            raise SystemExit("--epochs must be >= 1")
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be >= 1")
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        if args.lr <= 0:
            raise SystemExit("--lr must be positive")
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.model is not None:
        overrides.setdefault("model", {})["name"] = args.model
    if args.save_dir is not None:
        overrides.setdefault("training", {}).setdefault("checkpoint", {})[
            "save_dir"
        ] = args.save_dir
    if args.resume is not None:
        overrides.setdefault("training", {})["resume"] = args.resume
    if args.outputs_dir is not None:
        overrides.setdefault("outputs", {})["dir"] = args.outputs_dir

    if overrides:
        deep_update(cfg, overrides)

    log_level = (cfg.get("logging") or {}).get("level", "INFO")
    setup_logging(str(log_level))
    run_training(cfg)


if __name__ == "__main__":
    main()
