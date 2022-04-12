"""Resize"""
# pylint: disable=E0401
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
from nptyping import NDArray
from torchvision import transforms


class CropMode(Enum):
    """Crop modes"""

    RESIZE = None
    CROP_CENTER = "center"
    CROP_RANDOM = "random"


def resize_no_crop(
    image: NDArray[(Any, Any), np.int32], new_size: Tuple[int, int]
) -> NDArray[(Any, Any), np.int32]:
    """
    Resize an image without cropping it
    """
    return transforms.Resize(new_size)(image)


def resize_crop_center(
    image: NDArray[(Any, Any), np.int32], new_size: Tuple[int, int]
) -> NDArray[(Any, Any), np.int32]:
    """
    Resize an image after cropping it, keeping its center part
    """
    if isinstance(image, np.ndarray):
        min_side = min(image.shape)
    else:
        min_side = min(image.size)
    ratio = new_size[1] / new_size[0]

    image = transforms.CenterCrop((min_side, int(ratio * min_side)))(image)
    return transforms.Resize(new_size)(image)


def resize_crop_random(
    image: NDArray[(Any, Any), np.int32], new_size: Tuple[int, int]
) -> NDArray[(Any, Any), np.int32]:
    """
    Resize an image after cropping it, keeping a random part
    """
    min_side = min(image.size)
    ratio = new_size[1] / new_size[0]
    image = transforms.RandomCrop((min_side, int(ratio * min_side)))(image)
    return transforms.Resize(new_size)(image)


def resize(
    image: NDArray[(Any, Any), np.int32],
    new_size: Optional[Tuple[int, int]],
    crop_mode: CropMode,
) -> NDArray[(Any, Any), np.int32]:
    """
    Resize an image with a specific mode
    """
    if new_size is None:
        return image

    if crop_mode == CropMode.RESIZE.value:
        return resize_no_crop(image, new_size)
    if crop_mode == CropMode.CROP_CENTER.value:
        return resize_crop_center(image, new_size)
    if crop_mode == CropMode.CROP_RANDOM.value:
        return resize_crop_random(image, new_size)
    return image
