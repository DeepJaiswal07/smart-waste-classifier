"""Rigorous test set evaluation computing accuracy, precision, recall, F1, and confusion matrix."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import load_split_datasets
from src.models.model import build_model
from src.utils.paths import resolve_path

logger = logging.getLogger("smart_waste")


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: Path,
    normalize: bool = True
) -> None:
    """Plot and save confusion matrix heatmap with counts and percentages."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_display = cm_norm if normalize else cm

    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Test Set Confusion Matrix" + (" (Normalized)" if normalize else ""),
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)" if normalize else f"{cm[i, j]}"
            ax.text(
                j, i, val_str,
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {save_path}")


def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the best checkpointed model strictly on the held-out test split.
    """
    best_model_path = resolve_path(config["paths"]["best_model_path"])
    classes_path = resolve_path(config["paths"]["classes_path"])

    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {best_model_path}. "
            "Please train the model first by executing: python -m src.main train"
        )

    # Load classes
    if classes_path.is_file():
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = config["data"]["classes"]

    # Load test split
    _, _, test_ds, _ = load_split_datasets(config, quick_mode=False)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    # Instantiate model and load checkpoint
    model, device = build_model(config, num_classes=len(classes))
    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded weights from {best_model_path}. Evaluating on {len(test_ds)} test samples...")

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())

    # Compute classification metrics
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report_str = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save classification report text
    report_path = resolve_path(config["paths"]["report_txt"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== SMART WASTE CLASSIFIER TEST SET EVALUATION REPORT ===\n")
        f.write(f"Evaluated Samples: {len(y_true)}\n\n")
        f.write(report_str)
        f.write("\nSummary Metrics:\n")
        f.write(f"Accuracy:          {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Precision (Macro): {prec_macro:.4f}\n")
        f.write(f"Recall (Macro):    {rec_macro:.4f}\n")
        f.write(f"F1-Score (Macro):  {f1_macro:.4f}\n")
        f.write(f"F1-Score (Weight): {f1_weighted:.4f}\n")
    logger.info(f"Classification report saved to: {report_path}")

    # Save metrics JSON
    metrics_path = resolve_path(config["paths"]["metrics_json"])
    metrics_data = {
        "test_samples_count": len(y_true),
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100.0, 2),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_score_macro": round(f1_macro, 4),
        "f1_score_weighted": round(f1_weighted, 4),
        "per_class": {
            cls_name: {
                "precision": round(report_dict[cls_name]["precision"], 4),
                "recall": round(report_dict[cls_name]["recall"], 4),
                "f1_score": round(report_dict[cls_name]["f1-score"], 4),
                "support": int(report_dict[cls_name]["support"]),
            }
            for cls_name in classes if cls_name in report_dict
        },
        "confusion_matrix": cm.tolist(),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Metrics JSON saved to: {metrics_path}")

    # Save Confusion Matrix Plot
    cm_path = resolve_path(config["paths"]["confusion_matrix_png"])
    plot_confusion_matrix(cm, classes, cm_path)

    # Print summary to console
    logger.info("=" * 60)
    logger.info(f"Test Accuracy:         {acc * 100:.2f}%")
    logger.info(f"Macro Precision:       {prec_macro:.4f}")
    logger.info(f"Macro Recall:          {rec_macro:.4f}")
    logger.info(f"Macro F1-Score:        {f1_macro:.4f}")
    logger.info(f"Weighted F1-Score:     {f1_weighted:.4f}")
    logger.info("=" * 60)

    return metrics_data
