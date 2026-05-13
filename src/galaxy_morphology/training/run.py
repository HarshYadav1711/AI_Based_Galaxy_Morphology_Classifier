"""High-level training orchestration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from galaxy_morphology.data.loaders import load_dataset
from galaxy_morphology.evaluation.metrics import (
    save_classification_report_json,
    save_metrics_json,
    write_training_history_csv,
)
from galaxy_morphology.models.cnn import count_parameters, get_model
from galaxy_morphology.training.early_stopping import EarlyStopping
from galaxy_morphology.training.loops import train_epoch, validate
from galaxy_morphology.utils.seed import set_seed
from galaxy_morphology.utils.torch_io import load_checkpoint
from galaxy_morphology.visualization.plots import plot_confusion_matrix, plot_training_history

logger = logging.getLogger(__name__)


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
    scheduler: ReduceLROnPlateau | None,
    best_val_acc: float,
    history: dict[str, list[float]],
    class_names: list[str],
    scaler: GradScaler | None,
) -> dict[str, Any]:
    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "class_names": class_names,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    return ckpt


def run_training(cfg: dict[str, Any]) -> None:
    """Run full train/validate loop from nested config dict.

    Expected keys mirror ``configs/train.yaml`` (see repo default file).
    """
    seed = int(cfg.get("seed", 42))
    repro = cfg.get("reproducibility", {}) or {}
    set_seed(seed, deterministic_cudnn=bool(repro.get("deterministic_cudnn", True)))

    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    sched_cfg = cfg.get("scheduler", {}) or {}
    out_cfg = cfg.get("outputs", {}) or {}
    ckpt_cfg = train_cfg.get("checkpoint", {}) or {}

    data_dir = str(data_cfg.get("dir", "data/galaxies"))
    train_split = float(data_cfg.get("train_split", 0.8))
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    device = _device_from_config(cfg)
    logger.info("Using device: %s", device)

    train_loader, val_loader, class_names = load_dataset(
        data_dir=data_dir,
        train_split=train_split,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    logger.info("Classes: %s", class_names)
    logger.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    model_name = str(model_cfg.get("name", "lightweight"))
    model = get_model(model_name=model_name, num_classes=len(class_names))
    logger.info("Model: %s | parameters: %s", model_name, f"{count_parameters(model):,}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=str(sched_cfg.get("mode", "min")),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 5)),
    )

    epochs = int(train_cfg.get("epochs", 50))
    max_grad_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler: GradScaler | None = (
        GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    )

    save_dir = str(ckpt_cfg.get("save_dir", "checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    outputs_dir = str(out_cfg.get("dir", "outputs"))
    os.makedirs(outputs_dir, exist_ok=True)

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
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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

    metrics_path = Path(outputs_dir) / "metrics.json"
    report_path = Path(outputs_dir) / "classification_report.json"
    csv_path = Path(outputs_dir) / "training_history.csv"

    for epoch in range(start_epoch, epochs):
        logger.info("Epoch %d / %d", epoch + 1, epochs)
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
        )
        val_loss, val_acc, report, cm = validate(
            model, val_loader, criterion, device, class_names, use_amp=use_amp
        )

        scheduler.step(val_loss)
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
        }
        history_rows.append(row)
        write_training_history_csv(csv_path, history_rows)

        logger.info(
            "Train loss=%.4f acc=%.4f | Val loss=%.4f acc=%.4f | lr=%.6f",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = _build_checkpoint(
                epoch, model, optimizer, scheduler, best_val_acc, history, class_names, scaler
            )
            torch.save(ckpt, os.path.join(save_dir, "best_model.pth"))
            logger.info("Saved new best model (val_acc=%.4f)", val_acc)

        ckpt_full = _build_checkpoint(
            epoch, model, optimizer, scheduler, best_val_acc, history, class_names, scaler
        )
        if periodic > 0 and (epoch + 1) % periodic == 0:
            path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(ckpt_full, path)
            logger.info("Periodic checkpoint: %s", path)

        stop_metric = val_loss if es_monitor in ("val_loss", "loss") else val_acc
        if early.step(float(stop_metric)):
            break

    best_path = Path(save_dir) / "best_model.pth"
    if best_path.is_file():
        best = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(best["model_state_dict"])
    else:
        logger.warning("No best_model.pth found; skipping reload for final eval.")

    val_loss, val_acc, report, cm = validate(
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
        },
    )
    save_classification_report_json(report_path, report)

    if history["train_loss"]:
        plot_training_history(history, Path(save_dir) / "training_history.png")
        plot_confusion_matrix(cm, class_names, Path(save_dir) / "confusion_matrix.png")
    else:
        logger.warning("No training epochs completed; skipping plot exports.")

    logger.info(
        "Training complete. Best val acc: %.4f | artifacts under %s and %s",
        best_val_acc,
        save_dir,
        outputs_dir,
    )
