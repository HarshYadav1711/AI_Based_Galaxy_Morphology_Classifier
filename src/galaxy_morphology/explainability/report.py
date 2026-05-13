"""Markdown evaluation report (metrics, calibration, examples)."""

from __future__ import annotations

from pathlib import Path


def write_evaluation_report_md(
    path: str | Path,
    *,
    title: str,
    metrics_lines: list[str],
    ece: float,
    rel_diagram_rel: str,
    cm_rel: str,
    gradcam_examples_rel: list[str],
    failure_montages_rel: list[str],
) -> None:
    """Write a concise markdown report; paths are relative for portability."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    gc_imgs = "\n".join(f"![Grad-CAM]({x})" for x in gradcam_examples_rel)
    gc_block = gc_imgs if gc_imgs else "_No Grad-CAM examples._"
    fail_imgs = "\n".join(f"![Failure analysis]({x})" for x in failure_montages_rel)
    fail_block = fail_imgs if fail_imgs else "_None._"
    body = f"""# {title}

## Metrics

{chr(10).join(metrics_lines)}

## Calibration

- **Expected Calibration Error (ECE, top-class):** {ece:.4f}

![Reliability diagram]({rel_diagram_rel})

## Confusion matrix

![Confusion matrix]({cm_rel})

## Example Grad-CAM explanations

{gc_block}

## Failure analysis montages

{fail_block}

---
_Report generated automatically._
"""
    p.write_text(body, encoding="utf-8")
