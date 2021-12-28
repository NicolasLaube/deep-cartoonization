from torchvision import transforms
import numpy as np
from nptyping import NDArray
from typing import Any, NewType, Tuple
from enum import Enum


class CropMode(Enum):
    RESIZE = None
    CROP_CENTER = "center"
    CROP_RANDOM = "random"


CropModes = NewType("CropModes", CropMode)


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
    new_size: Tuple[int, int],
    crop_mode: CropModes,
) -> NDArray[(Any, Any), np.int32]:
    """
    Resize an image with a specific mode
    """
    if new_size == None:
        image = image
    elif crop_mode == CropMode.RESIZE.value:
        image = resize_no_crop(image, new_size)
    elif crop_mode == CropMode.CROP_CENTER.value:
        image = resize_crop_center(image, new_size)
    elif crop_mode == CropMode.CROP_RANDOM.value:
        image = resize_crop_random(image, new_size)
    return image
