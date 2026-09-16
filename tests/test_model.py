"""Unit tests for model architecture, backbone freezing, and forward pass."""

import torch
from src.models.model import WasteClassifier, build_model
from src.utils.config import load_config


def test_waste_classifier_initialization():
    """Verify WasteClassifier instantiates with correct head output dimensions."""
    model = WasteClassifier(num_classes=6, pretrained=False, freeze_backbone=True)
    summary = model.get_parameter_summary()

    assert summary["total"] > 0
    assert summary["trainable"] > 0
    # With backbone frozen, trainable parameters should be significantly less than total
    assert summary["trainable"] < summary["total"]
    assert summary["frozen"] > summary["trainable"]


def test_forward_pass():
    """Verify forward pass on synthetic image batch yields correct logits shape."""
    model = WasteClassifier(num_classes=6, pretrained=False, freeze_backbone=True)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (4, 6)


def test_unfreeze_fine_tuning():
    """Verify unfreezing layers increases trainable parameter count."""
    model = WasteClassifier(num_classes=6, pretrained=False, freeze_backbone=True)
    initial_trainable = model.get_parameter_summary()["trainable"]

    model.unfreeze_last_n_blocks(2)
    after_trainable = model.get_parameter_summary()["trainable"]

    assert after_trainable > initial_trainable


def test_build_model_from_config():
    """Verify build_model factory configures the model correctly."""
    config = load_config()
    model, device = build_model(config, num_classes=6)
    assert isinstance(model, WasteClassifier)
    assert device.type in ("cpu", "cuda")
