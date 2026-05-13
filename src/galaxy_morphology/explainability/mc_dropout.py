"""Monte Carlo dropout: keep dropout stochastic while BatchNorm stays in eval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from galaxy_morphology.utils.model_outputs import morph_logits


def _set_bn_eval_dropout_train(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
            m.eval()
        elif isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_stats(
    model: nn.Module,
    x: torch.Tensor,
    *,
    num_samples: int = 20,
    device: torch.device,
) -> dict[str, float | bool]:
    """Run multiple stochastic forward passes (dropout on, BN eval).

    Returns mean max-probability, predictive entropy-based uncertainty, and review flag.
    """
    if num_samples <= 0:
        with torch.inference_mode():
            logits = morph_logits(model(x.to(device)))
            probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return {
            "mean_confidence": float(conf.item()),
            "uncertainty_score": 0.0,
            "needs_human_review": False,
        }

    was_training = model.training
    model.train()
    _set_bn_eval_dropout_train(model)

    probs_stack: list[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(num_samples):
            logits = morph_logits(model(x.to(device)))
            probs_stack.append(F.softmax(logits, dim=1))

    model.train(was_training)

    p = torch.stack(probs_stack, dim=0).mean(dim=0).clamp(min=1e-8)
    mean_conf = float(p.max(dim=1).values.item())
    entropy = float((-(p * p.log()).sum(dim=1)).item())
    max_entropy = float(torch.log(torch.tensor(p.shape[1], device=p.device, dtype=p.dtype)))
    uncertainty = entropy / max(max_entropy, 1e-8)
    # High entropy or low mean confidence → review
    needs_review = uncertainty > 0.55 or mean_conf < 0.55
    return {
        "mean_confidence": mean_conf,
        "uncertainty_score": uncertainty,
        "needs_human_review": bool(needs_review),
    }
