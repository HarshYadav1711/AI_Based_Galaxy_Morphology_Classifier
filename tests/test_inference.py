"""Inference pipeline tests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from galaxy_morphology.inference.predictor import load_model, predict, preprocess_image
from galaxy_morphology.models.cnn import get_model


def test_load_and_predict(tmp_path: Path) -> None:
    ck = tmp_path / "c.pth"
    model = get_model("lightweight", 3)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["spiral", "elliptical", "irregular"],
            "model_name": "lightweight",
        },
        ck,
    )
    device = torch.device("cpu")
    m2, classes, _name = load_model(str(ck), device, model_name="lightweight")
    img = tmp_path / "x.png"
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(img)
    tensor, _ = preprocess_image(str(img), image_size=32)
    label, conf, probs = predict(m2, tensor, device, classes)
    assert label in classes
    assert 0.0 <= conf <= 1.0
    assert set(probs.keys()) == set(classes)
