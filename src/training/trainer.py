"""Training engine with AdamW, CrossEntropyLoss, early stopping, and history plotting."""

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import load_split_datasets
from src.models.model import build_model
from src.utils.paths import resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger("smart_waste")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one full epoch across training dataloader batches."""
    model.train()
    running_loss = 0.0
    correct_count = 0
    total_samples = 0

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct_count += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = (correct_count / max(1, total_samples)) * 100.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on validation dataloader without computing gradients."""
    model.eval()
    running_loss = 0.0
    correct_count = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct_count += torch.sum(preds == labels).item()
            total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = (correct_count / max(1, total_samples)) * 100.0
    return epoch_loss, epoch_acc


def plot_training_history(history: List[Dict[str, Any]], save_path: Path) -> None:
    """Generate and save dual-panel loss and accuracy curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss panel
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, "s-", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy panel
    ax2.plot(epochs, train_acc, "o-", label="Train Acc", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, val_acc, "s-", label="Val Acc", color="#d62728", linewidth=2)
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Training curves saved to: {save_path}")


def save_history_csv(history: List[Dict[str, Any]], save_path: Path) -> None:
    """Save epoch-by-epoch training metrics to CSV file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_sec"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({k: round(row[k], 4) if isinstance(row[k], float) else row[k] for k in fields})
    logger.info(f"Training history CSV saved to: {save_path}")


def train_model(config: Dict[str, Any], quick_mode: bool = False) -> Dict[str, Any]:
    """
    Execute end-to-end model training, validation, early stopping, and artifact generation.
    """
    set_seed(config["project"].get("seed", 42))

    train_cfg = config["training"]
    epochs = train_cfg["quick_epochs"] if quick_mode else train_cfg["epochs"]
    batch_size = train_cfg["quick_batch_size"] if quick_mode else train_cfg["batch_size"]
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    patience = train_cfg.get("patience", 3)
    num_workers = train_cfg.get("num_workers", 0)

    logger.info("=" * 60)
    logger.info(f"Starting Training: {'[QUICK DEVELOPMENT MODE]' if quick_mode else '[FULL EXPERIMENT]'}")
    logger.info(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Weight Decay: {weight_decay}")
    logger.info("=" * 60)

    # Load datasets
    train_ds, val_ds, test_ds, classes = load_split_datasets(config, quick_mode=quick_mode)
    logger.info(f"Dataset summary: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, Classes={len(classes)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model
    model, device = build_model(config, num_classes=len(classes))

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    # Destination paths
    model_dir = resolve_path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = resolve_path(config["paths"]["best_model_path"])
    classes_path = resolve_path(config["paths"]["classes_path"])
    metadata_path = resolve_path(config["paths"]["metadata_path"])

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        ep_duration = time.time() - ep_start
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "epoch_time_sec": ep_duration,
        }
        history.append(record)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {ep_duration:.1f}s"
        )

        # Check for best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f" -> Best model checkpoint saved to {best_model_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and not quick_mode:
                logger.info(f"Early stopping triggered: No validation loss improvement for {patience} consecutive epochs.")
                break

    total_time = time.time() - total_start_time
    logger.info(f"Training completed in {total_time:.1f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # Save class names mapping
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    logger.info(f"Saved class names to {classes_path}")

    # Save training metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_architecture": config["model"]["architecture"],
        "num_classes": len(classes),
        "classes": classes,
        "parameters": model.get_parameter_summary(),
        "quick_mode": quick_mode,
        "total_training_time_sec": round(total_time, 2),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "final_epoch_metrics": history[-1] if history else {},
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")

    # Save plots and CSV
    save_history_csv(history, resolve_path(config["paths"]["history_csv"]))
    plot_training_history(history, resolve_path(config["paths"]["curves_plot"]))

    return metadata
