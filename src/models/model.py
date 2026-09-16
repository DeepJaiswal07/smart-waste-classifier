"""MobileNetV3-Small Transfer Learning architecture with customized classification head."""

import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

logger = logging.getLogger("smart_waste")


class WasteClassifier(nn.Module):
    """
    Transfer learning classifier based on MobileNetV3-Small.

    Replaces standard ImageNet 1000-class head with a tailored multi-layer
    perceptron designed for waste classification.
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Load MobileNetV3-Small backbone
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)

        # Backbone feature extractor
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # In MobileNetV3-Small, feature dimension after pooling is 576
        in_features = base_model.classifier[0].in_features

        # Freeze feature extractor layers if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Replace classification head with specialized multi-class head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting convolutional features and computing class logits."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """
        Unfreeze the last n convolutional blocks for fine-tuning.

        Args:
            n: Number of final feature blocks to unfreeze.
        """
        total_blocks = len(self.features)
        start_idx = max(0, total_blocks - n)
        for i in range(start_idx, total_blocks):
            for param in self.features[i].parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {n} feature blocks ({start_idx} to {total_blocks - 1}) for fine-tuning.")

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return counts of total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


def get_device() -> torch.device:
    """Return CUDA device if GPU is available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU (CUDA not available)")
    return device


def build_model(
    config: Dict[str, Any],
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[WasteClassifier, torch.device]:
    """
    Factory function to instantiate, configure, and device-allocate WasteClassifier.

    Args:
        config: Loaded project configuration dictionary
        num_classes: Number of target classes (defaults to config['data']['num_classes'])
        device: Device to place the model on (defaults to automatic device detection)
    """
    model_cfg = config["model"]
    if num_classes is None:
        num_classes = config["data"].get("num_classes", len(config["data"]["classes"]))

    dev = device if device is not None else get_device()

    model = WasteClassifier(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )

    if model_cfg.get("fine_tune", False):
        layers_to_unfreeze = int(model_cfg.get("fine_tune_layers", 2))
        model.unfreeze_last_n_blocks(layers_to_unfreeze)

    model = model.to(dev)
    summary = model.get_parameter_summary()
    logger.info(f"Model initialized: Total params: {summary['total']:,}, Trainable: {summary['trainable']:,}, Frozen: {summary['frozen']:,}")
    return model, dev
