"""Inference CLI."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from galaxy_morphology.explainability.pipeline import (
    run_explainability_for_image,
    top_k_class_probs,
)
from galaxy_morphology.inference.onnx_export import export_onnx
from galaxy_morphology.inference.predictor import (
    benchmark_inference,
    load_model,
    predict,
    predict_paths_batched,
    preprocess_image,
)
from galaxy_morphology.models.registry import list_model_names
from galaxy_morphology.utils.config import deep_update, load_yaml_config
from galaxy_morphology.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    names = list_model_names()
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
        choices=names,
        help="Architecture name (must match training); overrides checkpoint if set.",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Square resize side length.")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional CSV path for batch results."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for directory inference (default: from config or 32).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="After inference, log batched throughput (images/sec) on the same files.",
    )
    parser.add_argument(
        "--onnx-export",
        type=str,
        default=None,
        metavar="PATH",
        help="If set, export the loaded model to this ONNX path and exit batch path early.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Single-image mode only: run Grad-CAM, top-3, and MC-dropout uncertainty.",
    )
    parser.add_argument(
        "--explain-out",
        type=str,
        default="outputs/visualizations/inference",
        help="Directory for explanation images when --explain is set.",
    )
    parser.add_argument(
        "--mc-dropout-samples",
        type=int,
        default=20,
        help="Stochastic forward passes for MC-dropout when --explain is set.",
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
    if args.batch_size is not None:
        overrides["inference_batch_size"] = args.batch_size
    if overrides:
        deep_update(cfg, overrides)

    log_level = (cfg.get("logging") or {}).get("level", "INFO")
    setup_logging(str(log_level))

    ckpt = cfg.get("checkpoint") or args.checkpoint
    if not ckpt:
        raise SystemExit("Missing checkpoint: pass --checkpoint or set checkpoint in YAML.")
    if not Path(ckpt).is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    model_name_cfg = cfg.get("model")
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("inference_batch_size", 32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model, class_names, resolved_model_name = load_model(
        ckpt,
        device,
        model_name=str(model_name_cfg) if model_name_cfg else None,
    )
    logger.info("Loaded classes: %s (architecture: %s)", class_names, resolved_model_name)

    if args.onnx_export:
        export_onnx(model, args.onnx_export, input_size=image_size)
        logger.info("ONNX export done.")
        return

    image_path = cfg.get("image")
    image_dir = cfg.get("image_dir")
    output_csv = cfg.get("output_csv")

    if image_path:
        p = Path(str(image_path))
        if not p.is_file():
            raise SystemExit(f"Image not found: {p}")
        logger.info("Predicting: %s", p)
        tensor, _ = preprocess_image(str(p), image_size)
        if args.explain:
            explain_dir = Path(args.explain_out)
            ex = run_explainability_for_image(
                model,
                tensor,
                device,
                class_names,
                resolved_model_name,
                stem=p.stem,
                out_dir=explain_dir,
                mc_dropout_samples=args.mc_dropout_samples,
            )
            mc = ex["mc_dropout"]
            top3 = ex["top3"]
            sep = "=" * 62
            print(sep)
            print(" Galaxy morphology inference (with explainability)")
            print(sep)
            print(f"  Image            : {p.resolve()}")
            print(f"  Architecture     : {resolved_model_name}")
            print(f"  Predicted class  : {ex['predicted_class']}")
            print(f"  Confidence       : {ex['confidence']:.4f}")
            print("  Top-3 classes    :")
            for row in top3:
                print(f"    - {row['class']}: {row['probability']:.4f}")
            print("  MC dropout       :")
            print(f"    - mean_confidence      : {mc['mean_confidence']:.4f}")
            print(f"    - uncertainty_score    : {mc['uncertainty_score']:.4f}")
            print(f"    - needs_human_review   : {mc['needs_human_review']}")
            print("  Grad-CAM outputs :")
            print(f"    - {ex['gradcam']['overlay_path']}")
            print(f"    - {ex['gradcam']['compare_path']}")
            print(sep)
        else:
            label, conf, probs = predict(model, tensor, device, class_names)
            top3 = top_k_class_probs(probs, 3)
            sep = "=" * 62
            print(sep)
            print(" Galaxy morphology inference")
            print(sep)
            print(f"  Image            : {p.resolve()}")
            print(f"  Architecture     : {resolved_model_name}")
            print(f"  Predicted class  : {label}")
            print(f"  Confidence       : {conf:.4f}")
            print("  Top-3 classes    :")
            for name, pr in top3:
                print(f"    - {name}: {pr:.4f}")
            print(sep)
            logger.info("Prediction: %s (confidence=%.4f)", label, conf)
    elif image_dir:
        d = Path(str(image_dir))
        if not d.is_dir():
            raise SystemExit(f"Directory not found: {d}")
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        paths = [str(d / f) for f in os.listdir(d) if f.lower().endswith(exts)]
        if not paths:
            raise SystemExit(f"No images found in {d}")
        logger.info("Batch inference on %d files (batch_size=%d)", len(paths), batch_size)
        results = predict_paths_batched(
            model,
            paths,
            device,
            class_names,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=0,
        )
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
        ok = [r for r in results if "error" not in r]
        if ok:
            mean_conf = sum(r["confidence"] for r in ok) / len(ok)
            logger.info(
                "Batch summary: %d images OK (mean confidence %.4f). Architecture: %s.",
                len(ok),
                mean_conf,
                resolved_model_name,
            )
        if args.benchmark:
            ok_paths = [r["image_path"] for r in results if "error" not in r]
            stats = benchmark_inference(
                model,
                ok_paths,
                device,
                image_size=image_size,
                batch_size=batch_size,
            )
            logger.info(
                "Inference benchmark: %.1f images/sec (n=%d)",
                stats["images_per_sec"],
                int(stats["num_images"]),
            )
    else:
        raise SystemExit("Provide --image or --image-dir (or set in inference YAML).")


if __name__ == "__main__":
    main()
