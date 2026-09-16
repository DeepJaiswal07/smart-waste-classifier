"""Configuration loading and schema validation."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

from src.utils.paths import resolve_path


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load and validate the project YAML configuration file."""
    full_path = resolve_path(config_path)
    if not full_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML format in {full_path}. Expected a top-level dictionary.")

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate that required sections and parameters are present and well-formed."""
    required_sections = ["project", "data", "model", "training", "paths"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required configuration section: '{section}'")

    # Validate data configuration
    data_cfg = config["data"]
    required_data_keys = ["image_size", "train_split", "val_split", "test_split", "classes"]
    for k in required_data_keys:
        if k not in data_cfg:
            raise KeyError(f"Missing required data configuration key: '{k}'")

    if not (0.0 < data_cfg["train_split"] < 1.0):
        raise ValueError("data.train_split must be between 0.0 and 1.0")
    if not (0.0 < data_cfg["val_split"] < 1.0):
        raise ValueError("data.val_split must be between 0.0 and 1.0")
    if not (0.0 < data_cfg["test_split"] < 1.0):
        raise ValueError("data.test_split must be between 0.0 and 1.0")

    split_sum = data_cfg["train_split"] + data_cfg["val_split"] + data_cfg["test_split"]
    if abs(split_sum - 1.0) > 1e-4:
        raise ValueError(f"Data splits must sum to 1.0, got {split_sum}")

    if not isinstance(data_cfg["classes"], list) or len(data_cfg["classes"]) < 2:
        raise ValueError("data.classes must be a list of at least 2 class names")

    # Validate model configuration
    model_cfg = config["model"]
    if "architecture" not in model_cfg:
        raise KeyError("Missing required key: 'model.architecture'")

    # Validate training configuration
    train_cfg = config["training"]
    for k in ["epochs", "batch_size", "learning_rate"]:
        if k not in train_cfg:
            raise KeyError(f"Missing required training configuration key: '{k}'")
        if train_cfg[k] <= 0:
            raise ValueError(f"training.{k} must be positive")
