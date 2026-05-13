"""Streamlit-cached model bundle."""

from __future__ import annotations

import streamlit as st
import torch

from galaxy_morphology.inference.predictor import load_model


@st.cache_resource(show_spinner="Loading model…")
def load_model_bundle(checkpoint_path: str, use_cpu: bool) -> tuple:
    """Load checkpoint once per path + CPU flag (efficient for the session)."""
    want_cpu = use_cpu or not torch.cuda.is_available()
    device = torch.device("cpu" if want_cpu else "cuda")
    model, class_names, model_name = load_model(checkpoint_path, device)
    return model, class_names, model_name, device
