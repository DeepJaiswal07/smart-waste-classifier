"""Single-image inference with rigorous input validation and top-k probabilistic output."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F

from src.data.preprocessing import get_transforms
from src.models.model import build_model, get_device
from src.utils.paths import resolve_path

logger = logging.getLogger("smart_waste")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Predictor:
    """Production-grade image predictor with file validation and confidence reporting."""

    def __init__(self, config: Dict[str, Any], model_path: Optional[str | Path] = None):
        self.config = config
        self.device = get_device()

        # Resolve paths
        self.model_path = resolve_path(model_path or config["paths"]["best_model_path"])
        self.classes_path = resolve_path(config["paths"]["classes_path"])

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path}. "
                "Please train the model first by executing: python -m src.main train"
            )

        # Load classes
        if self.classes_path.is_file():
            with open(self.classes_path, "r", encoding="utf-8") as f:
                self.classes = json.load(f)
        else:
            self.classes = config["data"]["classes"]

        # Build model and load weights
        self.model, _ = build_model(config, num_classes=len(self.classes), device=self.device)
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = get_transforms(
            image_size=config["data"]["image_size"],
            is_training=False
        )

    def validate_image_path(self, image_path: str | Path) -> Path:
        """
        Validate file existence, extension, and decodability.
        Raises ValueError or FileNotFoundError if invalid.
        """
        path = resolve_path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{path.suffix}'. "
                f"Supported extensions are: {sorted(list(SUPPORTED_EXTENSIONS))}"
            )

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Image file is corrupted or unreadable: {path} ({e})")

        return path

    def predict(self, image_path: str | Path, top_k: int = 1) -> Dict[str, Any]:
        """
        Run inference on an image and return class predictions and confidence scores.

        Args:
            image_path: Path to target image.
            top_k: Number of highest ranking classes to return.
        """
        validated_path = self.validate_image_path(image_path)

        with Image.open(validated_path) as img:
            rgb_image = img.convert("RGB")

        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        # Top-k
        k = max(1, min(top_k, len(self.classes)))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_predictions = [
            {
                "rank": i + 1,
                "class": self.classes[idx.item()],
                "probability": float(prob.item()),
                "confidence_pct": round(float(prob.item()) * 100.0, 2),
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
        ]

        top_1 = top_predictions[0]
        low_confidence_warning = top_1["confidence_pct"] < 50.0

        return {
            "image_path": str(validated_path),
            "predicted_class": top_1["class"],
            "confidence": top_1["confidence_pct"],
            "top_predictions": top_predictions,
            "low_confidence_warning": low_confidence_warning,
            "model_architecture": self.config["model"]["architecture"],
            "device": "CUDA" if self.device.type == "cuda" else "CPU",
        }
