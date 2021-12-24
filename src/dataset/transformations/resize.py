import cv2
import numpy as np
from nptyping import NDArray
from typing import Any


def resize(
    image: NDArray[(Any, Any), np.int32], new_size: tuple[int, int]
) -> NDArray[(Any, Any), np.int32]:
    """Resize an image"""
    if new_size != None:
        image = cv2.resize(image, new_size)
    return image
