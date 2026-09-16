"""Unit tests for image preprocessing and augmentations."""

from PIL import Image
import torch
from src.data.preprocessing import get_transforms, IMAGENET_MEAN, IMAGENET_STD


def test_eval_transforms_shape():
    """Verify evaluation transforms output standard 3x224x224 normalized float tensors."""
    transform = get_transforms(image_size=224, is_training=False)
    dummy_img = Image.new("RGB", (400, 300), color=(100, 150, 200))
    tensor = transform(dummy_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transforms_shape():
    """Verify training transforms with augmentation output 3x224x224 tensors."""
    transform = get_transforms(image_size=224, is_training=True)
    dummy_img = Image.new("RGB", (512, 512), color=(50, 80, 120))
    tensor = transform(dummy_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_imagenet_constants():
    """Verify standard ImageNet normalisation statistics."""
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3
    assert IMAGENET_MEAN[0] == 0.485
    assert IMAGENET_STD[0] == 0.229
