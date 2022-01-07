from torchvision import transforms
import numpy as np
from nptyping import NDArray
from typing import Any


def normalize(image: NDArray[(Any, Any), np.int32]) -> NDArray[(Any, Any), np.int32]:
    """
    Normalize images
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )(image)


def fixed_architecture_normalization(
    image: NDArray[(Any, Any), np.int32]
) -> NDArray[(Any, Any), np.int32]:
    """
    Normalize images for new architecture.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(image)
