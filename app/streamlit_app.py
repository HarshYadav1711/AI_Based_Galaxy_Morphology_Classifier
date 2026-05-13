"""
Local Streamlit demo for the galaxy morphology classifier.

From the repository root (with the package installed and demo dependencies)::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_SRC,):
    s = str(_p)
    if _p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

import streamlit as st
from PIL import Image

from components.batch_export import (
    list_images_from_zip,
    predictions_to_csv_bytes,
    write_uploaded_images_to_temp,
)
from components.cache_state import (
    digest_bytes,
    get_cached_single_result,
    set_cached_single_result,
)
from components.inference_core import (
    predict_from_tensor,
    run_full_explain,
    tensor_from_uploaded_bytes,
)
from components.model_cache import load_model_bundle
from components.theme import inject_theme
from galaxy_morphology.explainability.pipeline import top_k_class_probs
from galaxy_morphology.inference.predictor import predict_paths_batched

APP_NS = "galaxy_demo_v1"


def _resolve_checkpoint_path(text_path: str, uploaded_ckpt: object | None) -> str | None:
    if uploaded_ckpt is not None:
        raw = uploaded_ckpt.getvalue()
        h = digest_bytes(raw)[:16]
        tmp_root = st.session_state.setdefault(
            "_galaxy_ckpt_tmp", tempfile.mkdtemp(prefix="galaxy_ckpt_")
        )
        d = Path(str(tmp_root))
        p = d / f"weights_{h}.pth"
        if not p.is_file():
            p.write_bytes(raw)
        return str(p.resolve())
    p = Path(text_path.strip())
    if p.is_file():
        return str(p.resolve())
    return None


def main() -> None:
    st.set_page_config(
        page_title="Galaxy Morphology Lab",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    st.title("Galaxy morphology lab")
    st.caption("Local inference · explainability · uncertainty · batch export")

    with st.sidebar:
        st.header("Model")
        ckpt_text = st.text_input(
            "Checkpoint path",
            value="checkpoints/best_model.pth",
            help="Trained .pth from this project, or upload below.",
        )
        ckpt_upload = st.file_uploader("Or upload .pth", type=["pth"])
        use_cpu = st.toggle(
            "Force CPU",
            value=True,
            help="Safer default; turn off to allow CUDA when available.",
        )
        image_size = st.slider("Input size", min_value=128, max_value=384, value=224, step=32)
        mc_samples = st.slider("MC dropout passes", min_value=5, max_value=40, value=12, step=1)
        st.divider()
        st.caption("Use a project checkpoint. Large batches on CPU can take a while.")

    ckpt_path = _resolve_checkpoint_path(ckpt_text, ckpt_upload)
    if not ckpt_path:
        st.warning("Provide a valid checkpoint path on disk or upload a `.pth` file.")
        return

    try:
        model, class_names, model_name, device = load_model_bundle(ckpt_path, use_cpu)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load checkpoint: {exc}")
        return

    st.success(f"Loaded **{model_name}** · {len(class_names)} classes · device `{device}`")

    tab_single, tab_batch = st.tabs(["Single image", "Batch (ZIP / files)"])

    with tab_single:
        up = st.file_uploader("Galaxy image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        run_explain = st.checkbox("Grad-CAM + MC uncertainty", value=True)
        go = st.button("Analyze", type="primary", use_container_width=True)

        if up is not None and go:
            data = up.getvalue()
            digest = digest_bytes(data)
            cache_mc = mc_samples if run_explain else -1
            cached = get_cached_single_result(APP_NS, ckpt_path, digest, image_size, cache_mc)
            if cached is not None:
                result = cached
                st.info("Cached result (same image and settings in this session).")
            else:
                with st.spinner("Running inference…"):
                    tensor = tensor_from_uploaded_bytes(data, image_size)
                    if run_explain:
                        tmp = tempfile.mkdtemp(prefix="galaxy_explain_")
                        stem = Path(up.name).stem[:64]
                        result = run_full_explain(
                            model,
                            tensor,
                            device,
                            class_names,
                            model_name,
                            Path(tmp),
                            stem,
                            mc_dropout_samples=mc_samples,
                        )
                    else:
                        label, conf, probs = predict_from_tensor(model, tensor, device, class_names)
                        t3 = top_k_class_probs(probs, 3)
                        result = {
                            "predicted_class": label,
                            "confidence": conf,
                            "top3": [{"class": n, "probability": float(p)} for n, p in t3],
                            "mc_dropout": None,
                            "gradcam": None,
                        }
                set_cached_single_result(APP_NS, ckpt_path, digest, image_size, cache_mc, result)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Predicted class", result["predicted_class"])
            with c2:
                st.metric("Confidence", f"{float(result['confidence']):.1%}")
            with c3:
                mc = result.get("mc_dropout") or {}
                st.metric("Uncertainty", f"{float(mc.get('uncertainty_score', 0.0)):.3f}")

            st.caption("Confidence")
            st.progress(min(1.0, max(0.0, float(result["confidence"]))))

            if result.get("mc_dropout"):
                mc = result["mc_dropout"]
                if mc.get("needs_human_review"):
                    st.warning(
                        "**Human review recommended** — higher uncertainty or lower mean "
                        "confidence under MC dropout."
                    )
                else:
                    st.success("Uncertainty is in a typical range for automated screening.")

            st.subheader("Top-3 probabilities")
            for row in result["top3"]:
                st.write(f"**{row['class']}** — {float(row['probability']):.2%}")

            pil_orig = Image.open(io.BytesIO(data)).convert("RGB")
            gc = result.get("gradcam")
            if gc:
                st.subheader("Explainability · Grad-CAM")
                ca, cb = st.columns(2)
                with ca:
                    st.image(pil_orig, caption="Input", use_container_width=True)
                with cb:
                    st.image(
                        gc["overlay_path"], caption="Grad-CAM overlay", use_container_width=True
                    )
                st.image(
                    gc["compare_path"],
                    caption="Side-by-side: input · heatmap · overlay",
                    use_container_width=True,
                )
            else:
                st.image(pil_orig, caption="Input", use_container_width=True)

    with tab_batch:
        st.markdown(
            "Upload a **ZIP** of images and/or **multiple image files**, then run inference."
        )
        zip_up = st.file_uploader("ZIP archive", type=["zip"])
        multi_up = st.file_uploader(
            "Image files (multi-select)",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
        )
        batch_bs = st.number_input("Batch size", min_value=1, max_value=64, value=8, step=1)
        run_batch = st.button("Run batch inference", type="primary", key="batch_go")

        pairs: list[tuple[str, bytes]] = []
        if zip_up is not None:
            pairs.extend(list_images_from_zip(zip_up.getvalue()))
        if multi_up:
            for f in multi_up:
                pairs.append((f.name, f.getvalue()))

        if run_batch:
            if not pairs:
                st.warning("Add a ZIP or select one or more image files.")
            else:
                tmp_ctx, paths = write_uploaded_images_to_temp(pairs)
                with tmp_ctx:
                    with st.spinner(f"Predicting {len(paths)} images…"):
                        rows_out = predict_paths_batched(
                            model,
                            paths,
                            device,
                            class_names,
                            image_size=int(image_size),
                            batch_size=int(batch_bs),
                            num_workers=0,
                        )
                    ok_rows: list[dict] = []
                    for r in rows_out:
                        if "error" in r:
                            st.caption(f"Skip {Path(r.get('image_path', '')).name}: {r['error']}")
                        else:
                            row = {
                                "image": Path(r["image_path"]).name,
                                "predicted_class": r["predicted_class"],
                                "confidence": r["confidence"],
                            }
                            for k, v in r["probabilities"].items():
                                row[f"prob_{k}"] = v
                            ok_rows.append(row)
                    if ok_rows:
                        st.dataframe(ok_rows, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download predictions.csv",
                            data=predictions_to_csv_bytes(ok_rows),
                            file_name="galaxy_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    else:
                        st.error("No valid images produced predictions.")


if __name__ == "__main__":
    main()
