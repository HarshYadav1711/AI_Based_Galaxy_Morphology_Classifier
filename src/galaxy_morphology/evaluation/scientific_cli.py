"""CLI: rotation / noise / low-resolution robustness metrics on the validation split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from galaxy_morphology.data.loaders import load_dataset
from galaxy_morphology.evaluation.scientific_robustness import (
    run_scientific_suite,
    write_scientific_report_json,
)
from galaxy_morphology.inference.predictor import load_model
from galaxy_morphology.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Scientific robustness metrics on the local validation split.",
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="data/galaxies")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", type=str, default="outputs/scientific_eval/robustness.json")
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = p.parse_args(argv)
    setup_logging("INFO")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, _classes, _name = load_model(args.checkpoint, device)
    use_mt = hasattr(model, "merger_head")

    train_loader, val_loader, _cn, _tl, _tp, _vp = load_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=0,
        seed=42,
        rotation_degrees=15,
        multitask_manifest_csv=None,
        use_multitask_dataset=use_mt,
    )
    _ = train_loader
    payload = run_scientific_suite(model, val_loader, device, use_multitask_batch=use_mt)
    payload["checkpoint"] = str(Path(args.checkpoint).resolve())
    payload["data_dir"] = args.data_dir
    write_scientific_report_json(args.out, payload)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
