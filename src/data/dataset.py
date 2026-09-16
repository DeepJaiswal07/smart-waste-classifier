"""Dataset downloading, extraction, verification, stratified splitting, and PyTorch Dataset."""

import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import get_transforms
from src.utils.paths import resolve_path

logger = logging.getLogger("smart_waste")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class WasteDataset(Dataset):
    """Memory-efficient PyTorch Dataset for waste images loading from disk on demand."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform=None,
        base_dir: Optional[Path] = None,
    ):
        """
        Args:
            samples: List of (relative_or_absolute_image_path, class_index)
            transform: torchvision callable transform pipeline
            base_dir: Base directory to resolve relative sample paths against
        """
        self.samples = samples
        self.transform = transform
        self.base_dir = base_dir or Path.cwd()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        image_path = Path(rel_path)
        if not image_path.is_absolute():
            image_path = self.base_dir / image_path

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read image at {image_path}: {e}")

        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            tensor = img_rgb

        return tensor, label


def download_file(url: str, dest_path: Path) -> None:
    """Download file from URL with chunked streaming and logging."""
    logger.info(f"Downloading dataset from: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_length = response.headers.get("Content-Length")
        total_bytes = int(total_length) if total_length else 0
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_bytes > 0:
                percent = downloaded * 100 / total_bytes
                if downloaded % (5 * chunk_size) < chunk_size or downloaded >= total_bytes:
                    logger.info(f"Download progress: {downloaded / (1024*1024):.1f}MB / {total_bytes / (1024*1024):.1f}MB ({percent:.1f}%)")
    logger.info(f"Downloaded successfully to: {dest_path}")


def prepare_dataset(config: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """
    Download, extract, validate images, and generate deterministic stratified train/val/test split.

    Returns the split metadata dictionary.
    """
    raw_dir = resolve_path(config["data"]["raw_dir"])
    split_file = resolve_path(config["data"]["split_file"])
    split_file.parent.mkdir(parents=True, exist_ok=True)

    if split_file.is_file() and not force:
        logger.info(f"Dataset split metadata already exists at: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = raw_dir / "dataset-resized.zip"
    extracted_folder = raw_dir / "dataset-resized"

    # Step 1: Download if needed
    if not extracted_folder.is_dir() or force:
        if not zip_dest.is_file() or force:
            download_file(config["data"]["dataset_url"], zip_dest)

        logger.info(f"Extracting dataset archive to: {raw_dir}")
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        logger.info("Extraction complete.")

    # Locate class directories
    # Usually extracted as dataset-resized/cardboard, glass, metal, paper, plastic, trash
    search_dir = extracted_folder if extracted_folder.is_dir() else raw_dir
    found_classes = [d.name for d in search_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    found_classes.sort()

    logger.info(f"Discovered classes in dataset: {found_classes}")
    expected_classes = config["data"]["classes"]
    # Check that all expected classes are present
    for c in expected_classes:
        if c not in found_classes:
            raise ValueError(f"Expected class '{c}' not found among directories in {search_dir}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(expected_classes)}

    # Step 2: Validate image files and collect paths
    logger.info("Validating image file integrity and structure...")
    valid_samples_by_class: Dict[str, List[str]] = {cls_name: [] for cls_name in expected_classes}
    corrupted_count = 0

    for cls_name in expected_classes:
        cls_dir = search_dir / cls_name
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            # Validate can open with PIL
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Relative path from project root
                rel_path = img_path.relative_to(resolve_path(".")).as_posix()
                valid_samples_by_class[cls_name].append(rel_path)
            except Exception as e:
                logger.warning(f"Corrupted or unreadable image skipped: {img_path} ({e})")
                corrupted_count += 1

    total_valid = sum(len(v) for v in valid_samples_by_class.values())
    logger.info(f"Validation complete: {total_valid} valid images found. ({corrupted_count} corrupted/skipped)")

    # Step 3: Stratified split
    rng = np.random.RandomState(config["project"]["seed"])
    train_split_pct = config["data"]["train_split"]
    val_split_pct = config["data"]["val_split"]

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []
    class_distribution: Dict[str, Dict[str, int]] = {}

    for cls_name in expected_classes:
        paths = valid_samples_by_class[cls_name]
        paths_array = np.array(paths)
        rng.shuffle(paths_array)

        n_total = len(paths_array)
        n_train = int(round(n_total * train_split_pct))
        n_val = int(round(n_total * val_split_pct))
        # remainder to test to ensure exact coverage
        n_test = n_total - n_train - n_val

        train_p = paths_array[:n_train].tolist()
        val_p = paths_array[n_train:n_train + n_val].tolist()
        test_p = paths_array[n_train + n_val:].tolist()

        c_idx = class_to_idx[cls_name]
        train_samples.extend([(p, c_idx) for p in train_p])
        val_samples.extend([(p, c_idx) for p in val_p])
        test_samples.extend([(p, c_idx) for p in test_p])

        class_distribution[cls_name] = {
            "total": n_total,
            "train": len(train_p),
            "val": len(val_p),
            "test": len(test_p),
        }

    # Deterministically sort samples
    train_samples.sort(key=lambda x: x[0])
    val_samples.sort(key=lambda x: x[0])
    test_samples.sort(key=lambda x: x[0])

    split_metadata = {
        "dataset_name": config["data"]["dataset_name"],
        "num_classes": len(expected_classes),
        "classes": expected_classes,
        "class_to_idx": class_to_idx,
        "total_samples": total_valid,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "class_distribution": class_distribution,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    logger.info(f"Split metadata successfully saved to: {split_file}")
    logger.info(f"Split counts: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    return split_metadata


def load_split_datasets(
    config: Dict[str, Any],
    quick_mode: bool = False
) -> Tuple[WasteDataset, WasteDataset, WasteDataset, List[str]]:
    """
    Load PyTorch WasteDatasets for train, val, and test splits according to metadata.

    Args:
        config: Loaded project configuration
        quick_mode: If True, uses a small subset of samples for quick development/testing.
    """
    split_file = resolve_path(config["data"]["split_file"])
    if not split_file.is_file():
        logger.info("Split metadata not found. Automatically triggering dataset preparation...")
        metadata = prepare_dataset(config)
    else:
        with open(split_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    image_size = config["data"]["image_size"]
    train_transform = get_transforms(image_size=image_size, is_training=True)
    eval_transform = get_transforms(image_size=image_size, is_training=False)

    train_samples = metadata["train_samples"]
    val_samples = metadata["val_samples"]
    test_samples = metadata["test_samples"]
    classes = metadata["classes"]
    root_dir = resolve_path(".")

    if quick_mode:
        limit = config["training"].get("quick_sample_limit", 120)
        # Stratified slice for quick mode
        num_cls = len(classes)
        per_cls = max(2, limit // num_cls)
        q_train, q_val, q_test = [], [], []

        for c_idx in range(num_cls):
            c_tr = [s for s in train_samples if s[1] == c_idx][:per_cls]
            c_va = [s for s in val_samples if s[1] == c_idx][:max(2, per_cls // 2)]
            c_te = [s for s in test_samples if s[1] == c_idx][:max(2, per_cls // 2)]
            q_train.extend(c_tr)
            q_val.extend(c_va)
            q_test.extend(c_te)

        train_samples = q_train
        val_samples = q_val
        test_samples = q_test
        logger.info(f"[QUICK MODE] Reduced dataset to {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")

    train_dataset = WasteDataset(train_samples, transform=train_transform, base_dir=root_dir)
    val_dataset = WasteDataset(val_samples, transform=eval_transform, base_dir=root_dir)
    test_dataset = WasteDataset(test_samples, transform=eval_transform, base_dir=root_dir)

    return train_dataset, val_dataset, test_dataset, classes
