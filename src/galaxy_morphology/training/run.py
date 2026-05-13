"""High-level training orchestration."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from galaxy_morphology.data.loaders import load_dataset
from galaxy_morphology.data.quality import analyze_dataset, save_dataset_statistics
from galaxy_morphology.evaluation.classification_metrics import (
    precision_recall_curve_data,
    roc_curve_data,
)
from galaxy_morphology.evaluation.metrics import (
    save_classification_report_json,
    save_metrics_json,
    write_training_history_csv,
)
from galaxy_morphology.models.registry import build_model, count_parameters
from galaxy_morphology.training.early_stopping import EarlyStopping
from galaxy_morphology.training.experiment import append_metrics_jsonl, create_experiment_dir
from galaxy_morphology.training.loops import train_epoch, validate
from galaxy_morphology.training.mt_loops import train_epoch_multitask, validate_multitask
from galaxy_morphology.utils.seed import set_seed
from galaxy_morphology.utils.torch_io import load_checkpoint
from galaxy_morphology.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_history,
)

logger = logging.getLogger(__name__)

SchedulerKind = Literal["plateau", "cosine"]


def _device_from_config(cfg: dict[str, Any]) -> torch.device:
    if cfg.get("device") == "cpu":
        return torch.device("cpu")
    if cfg.get("device") == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau | CosineAnnealingLR | None,
    best_val_acc: float,
    history: dict[str, list[float]],
    class_names: list[str],
    scaler: GradScaler | None,
    *,
    model_name: str,
    scheduler_kind: SchedulerKind,
    pretrained: bool,
    multitask: bool = False,
) -> dict[str, Any]:
    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "class_names": class_names,
        "model_name": model_name,
        "scheduler_kind": scheduler_kind,
        "pretrained": pretrained,
        "multitask": bool(multitask),
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    return ckpt


def _class_weights_tensor(
    train_labels: list[int], num_classes: int, device: torch.device
) -> torch.Tensor:
    counts = np.bincount(np.asarray(train_labels, dtype=int), minlength=num_classes).astype(
        np.float64
    )
    counts = np.maximum(counts, 1.0)
    weights = len(train_labels) / (num_classes * counts)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_training(cfg: dict[str, Any], *, config_path: Path | None = None) -> None:
    """Run full train/validate loop from nested config dict."""
    seed = int(cfg.get("seed", 42))
    repro = cfg.get("reproducibility", {}) or {}
    set_seed(seed, deterministic_cudnn=bool(repro.get("deterministic_cudnn", True)))

    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    sched_cfg = cfg.get("scheduler", {}) or {}
    out_cfg = cfg.get("outputs", {}) or {}
    ckpt_cfg = train_cfg.get("checkpoint", {}) or {}
    loss_cfg = train_cfg.get("loss", {}) or {}
    mix_cfg = train_cfg.get("mixup", {}) or {}
    exp_cfg = cfg.get("experiment", {}) or {}
    mt_cfg = cfg.get("multitask", {}) or {}
    aug_cfg = data_cfg.get("augmentation", {}) or {}

    data_dir = str(data_cfg.get("dir", "data/galaxies"))
    train_split = float(data_cfg.get("train_split", 0.8))
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    model_name = str(model_cfg.get("name", "lightweight"))
    use_mt = model_name == "lightweight_multitask"
    rotation_degrees = int(aug_cfg.get("rotation_degrees", 15))
    mcsv = mt_cfg.get("manifest_csv")
    multitask_manifest: str | None = str(Path(mcsv)) if mcsv else None

    device = _device_from_config(cfg)
    logger.info("Using device: %s", device)

    outputs_dir = Path(str(out_cfg.get("dir", "outputs")))
    save_dir = Path(str(ckpt_cfg.get("save_dir", "checkpoints")))
    exp_dir: Path | None = None
    if exp_cfg.get("enabled"):
        slug = str(exp_cfg.get("slug", model_cfg.get("name", "run")))
        exp_root = Path(str(exp_cfg.get("root", "experiments")))
        exp_dir = create_experiment_dir(
            exp_root,
            slug=slug,
            config_path=config_path,
            resolved_config=cfg,
        )
        outputs_dir = exp_dir / "outputs"
        save_dir = exp_dir / "checkpoints"
        figures_dir = exp_dir / "figures"
    else:
        figures_dir = outputs_dir

    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    quality_cfg = data_cfg.get("quality", {}) or {}
    if quality_cfg.get("enabled", True):
        qreport = analyze_dataset(data_dir)
        save_dataset_statistics(outputs_dir / "dataset_statistics.json", qreport)

    train_loader, val_loader, class_names, train_labels, _train_paths, _val_paths = load_dataset(
        data_dir=data_dir,
        train_split=train_split,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        rotation_degrees=rotation_degrees,
        multitask_manifest_csv=multitask_manifest,
        use_multitask_dataset=use_mt,
    )
    label_counts = Counter(train_labels)
    train_class_counts = {
        class_names[i]: int(label_counts.get(i, 0)) for i in range(len(class_names))
    }
    plot_class_distribution(train_class_counts, figures_dir / "class_distribution_train.png")

    pretrained = bool(model_cfg.get("pretrained", True))
    model = build_model(model_name, len(class_names), pretrained=pretrained)
    logger.info("Model: %s | parameters: %s", model_name, f"{count_parameters(model):,}")
    model = model.to(device)

    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    use_weighted = bool(loss_cfg.get("weighted", False))
    weight_tensor = None
    if use_weighted:
        weight_tensor = _class_weights_tensor(train_labels, len(class_names), device)
    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=label_smoothing,
    )
    lw = mt_cfg.get("loss_weights", {}) or {}
    w_merger = float(lw.get("merger", 0.5))
    w_bar = float(lw.get("bar", 0.5))
    w_asym = float(lw.get("asymmetry", 0.1))

    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(train_cfg.get("epochs", 50))
    scheduler_kind: SchedulerKind = str(sched_cfg.get("type", "plateau")).lower()  # type: ignore[assignment]
    if scheduler_kind not in ("plateau", "cosine"):
        scheduler_kind = "plateau"

    scheduler: ReduceLROnPlateau | CosineAnnealingLR | None
    if scheduler_kind == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=float(sched_cfg.get("eta_min", 1e-6))
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(sched_cfg.get("mode", "min")),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
        )

    max_grad_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler: GradScaler | None = (
        GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    )

    use_mixup = bool(mix_cfg.get("enabled", False)) and not use_mt
    if use_mt and bool(mix_cfg.get("enabled", False)):
        logger.warning("Mixup is disabled for multi-task training.")
    mixup_alpha = float(mix_cfg.get("alpha", 0.2))

    periodic = int(ckpt_cfg.get("periodic_every_epochs", 10))
    resume_path = train_cfg.get("resume") or ckpt_cfg.get("resume")

    start_epoch = 0
    best_val_acc = float("-inf")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    history_rows: list[dict[str, Any]] = []

    if resume_path:
        rp = Path(str(resume_path))
        logger.info("Resuming from %s", rp)
        checkpoint = load_checkpoint(rp, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ck_sched_kind = checkpoint.get("scheduler_kind", scheduler_kind)
        if ck_sched_kind == scheduler_kind and "scheduler_state_dict" in checkpoint:
            sd = checkpoint.get("scheduler_state_dict")
            if sd is not None and scheduler is not None:
                try:
                    scheduler.load_state_dict(sd)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load scheduler state: %s", exc)
        if scaler is not None and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", float("-inf")))
        history = checkpoint.get("history") or history

    es_cfg = train_cfg.get("early_stopping") or {}
    es_monitor = str(es_cfg.get("monitor", "val_loss"))
    es_mode = "min" if es_monitor in ("val_loss", "loss") else "max"
    early = EarlyStopping(
        patience=int(es_cfg.get("patience", 0) or 0),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        mode=es_mode,
    )

    metrics_path = outputs_dir / "metrics.json"
    report_path = outputs_dir / "classification_report.json"
    extended_metrics_path = outputs_dir / "extended_metrics.json"
    csv_path = outputs_dir / "training_history.csv"

    for epoch in range(start_epoch, epochs):
        logger.info("Epoch %d / %d", epoch + 1, epochs)
        if use_mt:
            train_loss, train_acc = train_epoch_multitask(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                use_amp=use_amp,
                max_grad_norm=max_grad_norm,
                w_merger=w_merger,
                w_bar=w_bar,
                w_asym=w_asym,
            )
            val_loss, val_acc, report, cm, extended, y_true, y_pred, y_score = validate_multitask(
                model,
                val_loader,
                criterion,
                device,
                class_names,
                use_amp=use_amp,
                w_merger=w_merger,
                w_bar=w_bar,
                w_asym=w_asym,
            )
        else:
            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                use_amp=use_amp,
                max_grad_norm=max_grad_norm,
                use_mixup=use_mixup,
                mixup_alpha=mixup_alpha,
            )
            val_loss, val_acc, report, cm, extended, y_true, y_pred, y_score = validate(
                model, val_loader, criterion, device, class_names, use_amp=use_amp
            )

        if scheduler_kind == "plateau" and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler_kind == "cosine" and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "macro_f1": extended.get("macro_f1"),
            "weighted_f1": extended.get("weighted_f1"),
            "balanced_accuracy": extended.get("balanced_accuracy"),
            "roc_auc_ovr": extended.get("roc_auc_ovr"),
        }
        history_rows.append(row)
        write_training_history_csv(csv_path, history_rows)

        if exp_dir is not None:
            append_metrics_jsonl(exp_dir, row)

        logger.info(
            "Train loss=%.4f acc=%.4f | Val loss=%.4f acc=%.4f | lr=%.6f | macro_f1=%.4f",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            float(extended.get("macro_f1", 0.0)),
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = _build_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_acc,
                history,
                class_names,
                scaler,
                model_name=model_name,
                scheduler_kind=scheduler_kind,
                pretrained=pretrained,
                multitask=use_mt,
            )
            torch.save(ckpt, save_dir / "best_model.pth")
            logger.info("Saved new best model (val_acc=%.4f)", val_acc)

        ckpt_full = _build_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_acc,
            history,
            class_names,
            scaler,
            model_name=model_name,
            scheduler_kind=scheduler_kind,
            pretrained=pretrained,
            multitask=use_mt,
        )
        if periodic > 0 and (epoch + 1) % periodic == 0:
            path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(ckpt_full, path)
            logger.info("Periodic checkpoint: %s", path)

        stop_metric = val_loss if es_monitor in ("val_loss", "loss") else val_acc
        if early.step(float(stop_metric)):
            break

    best_path = save_dir / "best_model.pth"
    if best_path.is_file():
        best = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(best["model_state_dict"])
    else:
        logger.warning("No best_model.pth found; skipping reload for final eval.")

    if use_mt:
        val_loss, val_acc, report, cm, extended, y_true, y_pred, y_score = validate_multitask(
            model,
            val_loader,
            criterion,
            device,
            class_names,
            use_amp=use_amp,
            w_merger=w_merger,
            w_bar=w_bar,
            w_asym=w_asym,
        )
    else:
        val_loss, val_acc, report, cm, extended, y_true, y_pred, y_score = validate(
            model, val_loader, criterion, device, class_names, use_amp=use_amp
        )
    logger.info("Final validation accuracy: %.4f", val_acc)

    save_metrics_json(
        metrics_path,
        {
            "best_val_acc": best_val_acc,
            "final_val_loss": val_loss,
            "final_val_acc": val_acc,
            "epochs_ran": len(history["train_loss"]),
            "seed": seed,
            "class_names": class_names,
            "model_name": model_name,
            "extended": extended,
        },
    )
    save_classification_report_json(report_path, report)
    with extended_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(extended, f, indent=2)

    roc_data = roc_curve_data(y_true, y_score, class_names)
    with (outputs_dir / "roc_curve_data.json").open("w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2)
    pr_data = precision_recall_curve_data(y_true, y_score, class_names)
    with (outputs_dir / "pr_curve_data.json").open("w", encoding="utf-8") as f:
        json.dump(pr_data, f, indent=2)

    if history["train_loss"]:
        plot_training_history(history, figures_dir / "training_history.png")
        plot_confusion_matrix(cm, class_names, figures_dir / "confusion_matrix.png")
        plot_roc_curves(roc_data, class_names, figures_dir / "roc_curves.png")
        plot_pr_curves(pr_data, class_names, figures_dir / "pr_curves.png")
    else:
        logger.warning("No training epochs completed; skipping plot exports.")

    logger.info(
        "Training complete. Best val acc: %.4f | artifacts under %s and %s",
        best_val_acc,
        save_dir,
        outputs_dir,
    )
