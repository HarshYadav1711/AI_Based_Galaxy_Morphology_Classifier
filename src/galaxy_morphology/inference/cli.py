"""Inference CLI."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from galaxy_morphology.inference.predictor import (
    load_model,
    predict,
    predict_batch,
    preprocess_image,
)
from galaxy_morphology.utils.config import deep_update, load_yaml_config
from galaxy_morphology.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="galaxy-infer",
        description=(
            "Run galaxy morphology inference on one image, a directory of images, or "
            "paths implied by a YAML inference config."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML path (configs/inference.yaml). CLI flags override config values.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained .pth checkpoint (required unless set in config).",
    )
    parser.add_argument("--image", type=str, default=None, help="Single image path.")
    parser.add_argument(
        "--image-dir", type=str, default=None, help="Directory of images for batch mode."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["lightweight", "efficient"],
        help="Architecture name (must match training).",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Square resize side length.")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional CSV path for batch results."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg: dict = {}
    if args.config:
        cfg = load_yaml_config(args.config)

    overrides: dict = {}
    if args.checkpoint is not None:
        overrides["checkpoint"] = args.checkpoint
    if args.model is not None:
        overrides["model"] = args.model
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if args.image is not None:
        overrides["image"] = args.image
    if args.image_dir is not None:
        overrides["image_dir"] = args.image_dir
    if args.output is not None:
        overrides["output_csv"] = args.output
    if overrides:
        deep_update(cfg, overrides)

    log_level = (cfg.get("logging") or {}).get("level", "INFO")
    setup_logging(str(log_level))

    ckpt = cfg.get("checkpoint") or args.checkpoint
    if not ckpt:
        raise SystemExit("Missing checkpoint: pass --checkpoint or set checkpoint in YAML.")
    if not Path(ckpt).is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    model_name = str(cfg.get("model", "lightweight"))
    image_size = int(cfg.get("image_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model, class_names = load_model(ckpt, device, model_name=model_name)
    logger.info("Loaded classes: %s", class_names)

    image_path = cfg.get("image")
    image_dir = cfg.get("image_dir")
    output_csv = cfg.get("output_csv")

    if image_path:
        p = Path(str(image_path))
        if not p.is_file():
            raise SystemExit(f"Image not found: {p}")
        logger.info("Predicting: %s", p)
        tensor, _ = preprocess_image(str(p), image_size)
        label, conf, probs = predict(model, tensor, device, class_names)
        logger.info("Prediction: %s (confidence=%.4f)", label, conf)
        for name, pr in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            logger.info("  %s: %.4f", name, pr)
    elif image_dir:
        d = Path(str(image_dir))
        if not d.is_dir():
            raise SystemExit(f"Directory not found: {d}")
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        paths = [str(d / f) for f in os.listdir(d) if f.lower().endswith(exts)]
        if not paths:
            raise SystemExit(f"No images found in {d}")
        logger.info("Batch inference on %d files", len(paths))
        results = predict_batch(model, paths, device, class_names, image_size)
        for r in results:
            if "error" in r:
                logger.warning("%s ERROR %s", r["image_path"], r["error"])
            else:
                logger.info(
                    "%s -> %s (%.4f)",
                    os.path.basename(r["image_path"]),
                    r["predicted_class"],
                    r["confidence"],
                )
        if output_csv:
            rows = []
            for r in results:
                if "error" not in r:
                    row = {
                        "image": os.path.basename(r["image_path"]),
                        "predicted_class": r["predicted_class"],
                        "confidence": r["confidence"],
                    }
                    row.update({f"prob_{k}": v for k, v in r["probabilities"].items()})
                    rows.append(row)
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            logger.info("Wrote CSV: %s", output_csv)
    else:
        raise SystemExit("Provide --image or --image-dir (or set in inference YAML).")


if __name__ == "__main__":
    main()
