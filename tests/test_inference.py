"""Unit tests for inference, input validation, and Top-K logic."""

from pathlib import Path
import pytest
from PIL import Image
import torch

from src.inference.predictor import Predictor
from src.utils.config import load_config
from src.models.model import build_model


@pytest.fixture
def dummy_model_checkpoint(tmp_path):
    """Fixture creating a temporary dummy model checkpoint and class list for testing predictor."""
    config = load_config()
    model, _ = build_model(config, num_classes=6)
    ckpt_path = tmp_path / "dummy_model.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def test_predictor_missing_image(dummy_model_checkpoint):
    """Verify inference fails gracefully when image file does not exist."""
    config = load_config()
    predictor = Predictor(config, model_path=dummy_model_checkpoint)
    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        predictor.predict("non_existent_image_12345.jpg")


def test_predictor_unsupported_extension(dummy_model_checkpoint, tmp_path):
    """Verify inference rejects unsupported file extensions."""
    config = load_config()
    predictor = Predictor(config, model_path=dummy_model_checkpoint)
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image extension"):
        predictor.predict(bad_file)


def test_predictor_corrupted_image(dummy_model_checkpoint, tmp_path):
    """Verify inference rejects corrupted image files."""
    config = load_config()
    predictor = Predictor(config, model_path=dummy_model_checkpoint)
    corrupted_file = tmp_path / "corrupted.jpg"
    corrupted_file.write_bytes(b"corrupted binary data that cannot be parsed as JPEG")

    with pytest.raises(ValueError, match="corrupted or unreadable"):
        predictor.predict(corrupted_file)


def test_predictor_valid_image(dummy_model_checkpoint, tmp_path):
    """Verify inference returns valid predictions and Top-K rankings for a valid image."""
    config = load_config()
    predictor = Predictor(config, model_path=dummy_model_checkpoint)

    # Create a small valid test JPEG
    valid_file = tmp_path / "sample.jpg"
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    img.save(valid_file, "JPEG")

    result = predictor.predict(valid_file, top_k=3)

    assert "predicted_class" in result
    assert result["predicted_class"] in config["data"]["classes"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["top_predictions"]) == 3
    # Top predictions should be ordered by probability descending
    probs = [p["probability"] for p in result["top_predictions"]]
    assert probs == sorted(probs, reverse=True)
