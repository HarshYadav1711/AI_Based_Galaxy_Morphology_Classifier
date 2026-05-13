"""Create dummy images or download SDSS cutouts for local testing."""

from __future__ import annotations

import argparse
import logging
import os
import time
from io import BytesIO

import requests
from PIL import Image

from galaxy_morphology.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def download_sdss_cutout(
    ra: float,
    dec: float,
    scale: float = 0.396,
    width: int = 224,
    height: int = 224,
    band: str = "g",
) -> Image.Image | None:
    """Download a JPEG cutout from the public SDSS web service."""
    _ = band
    base_url = "https://skyserver.sdss.org/dr16/SkyServerWS/SkyServerWS.asmx/getJpeg"
    params = {"ra": ra, "dec": dec, "scale": scale, "width": width, "height": height, "opt": "G"}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.warning("SDSS cutout failed: %s", exc)
    return None


def download_sample_galaxies(output_dir: str = "data/galaxies", num_per_class: int = 10) -> None:
    """Download a few example cutouts per class (best-effort; network required)."""
    os.makedirs(output_dir, exist_ok=True)
    sample_coords = {
        "spiral": [(146.714, 0.395), (150.123, 1.234), (145.567, 0.789)],
        "elliptical": [(200.123, 0.456), (201.234, 0.567), (199.890, 0.345)],
        "irregular": [(180.456, 0.234), (181.567, 0.345), (179.234, 0.123)],
    }
    for class_name, coords_list in sample_coords.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i, (ra, dec) in enumerate(coords_list[:num_per_class]):
            img = download_sdss_cutout(ra, dec)
            if img is not None:
                path = os.path.join(class_dir, f"{class_name}_{i + 1}.jpg")
                img.save(path)
                logger.info("Saved %s", path)
            else:
                logger.warning("Failed cutout RA=%s Dec=%s", ra, dec)
            time.sleep(0.5)


def create_dummy_dataset(output_dir: str = "data/galaxies", num_per_class: int = 5) -> None:
    """Create solid-color placeholder PNGs for each class (offline friendly)."""
    os.makedirs(output_dir, exist_ok=True)
    for class_name in ["spiral", "elliptical", "irregular"]:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_per_class):
            img = Image.new("RGB", (224, 224), color=(50 + i * 10, 100 + i * 5, 150 + i * 3))
            path = os.path.join(class_dir, f"{class_name}_{i + 1}.png")
            img.save(path)
            logger.info("Created %s", path)
    logger.info("Dummy dataset at %s (replace with real images for science use).", output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create dummy galaxy images or download SDSS cutouts.")
    p.add_argument(
        "--mode",
        type=str,
        default="dummy",
        choices=["dummy", "sdss"],
        help="dummy: local placeholders; sdss: download public cutouts (network).",
    )
    p.add_argument("--output-dir", type=str, default="data/galaxies", help="Output root directory.")
    p.add_argument("--num-per-class", type=int, default=5, help="Images per morphology class.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging("INFO")
    if args.mode == "dummy":
        create_dummy_dataset(args.output_dir, args.num_per_class)
    else:
        download_sample_galaxies(args.output_dir, args.num_per_class)


if __name__ == "__main__":
    main()
