"""Image preprocessing and data augmentation transforms."""

from typing import Tuple
import torchvision.transforms as T

# Standard ImageNet statistics required by pretrained PyTorch torchvision backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: int = 224,
    is_training: bool = False
) -> T.Compose:
    """
    Construct torchvision transform pipeline for training or evaluation.

    Args:
        image_size: Target height and width (default 224 for MobileNetV3).
        is_training: If True, applies data augmentations (flip, rotation, color jitter).
                     If False, applies strictly deterministic resize, center crop, and normalization.
    """
    if is_training:
        return T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
