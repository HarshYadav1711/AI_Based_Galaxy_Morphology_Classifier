"""Session-scoped memoization for repeated identical inference requests."""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_cached_single_result(
    cache_ns: str,
    checkpoint_key: str,
    image_digest: str,
    image_size: int,
    mc_samples: int,
) -> Any | None:
    bucket = st.session_state.setdefault("_galaxy_infer_cache", {})
    key = (cache_ns, checkpoint_key, image_digest, image_size, mc_samples)
    return bucket.get(key)


def set_cached_single_result(
    cache_ns: str,
    checkpoint_key: str,
    image_digest: str,
    image_size: int,
    mc_samples: int,
    value: Any,
) -> None:
    bucket = st.session_state.setdefault("_galaxy_infer_cache", {})
    key = (cache_ns, checkpoint_key, image_digest, image_size, mc_samples)
    bucket[key] = value
