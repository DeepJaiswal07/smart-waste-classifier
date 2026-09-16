"""Unit tests for configuration loading and validation."""

import pytest
from src.utils.config import load_config, validate_config


def test_load_config_valid():
    """Verify that default config.yaml loads properly and contains all required sections."""
    config = load_config()
    assert isinstance(config, dict)
    assert "project" in config
    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert "paths" in config


def test_config_values():
    """Verify required data parameters and split proportions."""
    config = load_config()
    assert config["data"]["image_size"] == 224
    assert len(config["data"]["classes"]) == 6
    assert "cardboard" in config["data"]["classes"]
    assert "plastic" in config["data"]["classes"]

    splits = (
        config["data"]["train_split"]
        + config["data"]["val_split"]
        + config["data"]["test_split"]
    )
    assert abs(splits - 1.0) < 1e-4


def test_validate_config_missing_section():
    """Verify that an invalid config missing required sections raises KeyError."""
    bad_config = {"project": {"name": "test"}}
    with pytest.raises(KeyError):
        validate_config(bad_config)


def test_validate_config_invalid_splits():
    """Verify that invalid train/val/test splits raise ValueError."""
    config = load_config()
    corrupted_config = config.copy()
    corrupted_config["data"] = config["data"].copy()
    corrupted_config["data"]["train_split"] = 0.9
    corrupted_config["data"]["val_split"] = 0.9
    with pytest.raises(ValueError):
        validate_config(corrupted_config)
