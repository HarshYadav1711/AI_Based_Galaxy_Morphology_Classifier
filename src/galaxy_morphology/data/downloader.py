"""Optional SDSS / URL download helpers (astroquery is optional at runtime)."""

from __future__ import annotations

import logging
import os

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from PIL import Image

logger = logging.getLogger(__name__)


class SDSSDataDownloader:
    """Download or fetch SDSS cutouts (optional ``astroquery`` for FITS path)."""

    def __init__(self, output_dir: str = "data/galaxies", image_size: int = 224) -> None:
        self.output_dir = output_dir
        self.image_size = image_size
        os.makedirs(output_dir, exist_ok=True)
        for class_name in ["spiral", "elliptical", "irregular"]:
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    def download_sdss_image(
        self,
        ra: float,
        dec: float,
        objid: int,
        scale: float = 0.396,
        width: int = 224,
        height: int = 224,
    ) -> Image.Image | None:
        """Fetch SDSS image via astroquery when installed."""
        _ = objid  # reserved for naming saved files by callers
        try:
            from astroquery.sdss import SDSS  # type: ignore[import-untyped]

            coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            result = SDSS.get_images(coords, band="g", data_release=16)
            if len(result) > 0:
                hdu = result[0]
                data = hdu[0].data
                data = np.nan_to_num(data)
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                data = (data * 255).astype(np.uint8)
                img = Image.fromarray(data).convert("RGB")
                return img.resize((width, height), Image.Resampling.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SDSS download failed: %s", exc)
        return None

    def download_from_url(self, url: str, save_path: str) -> bool:
        """Download an image from HTTP URL to ``save_path``."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
                img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                img.save(save_path)
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("URL download failed: %s", exc)
        return False
