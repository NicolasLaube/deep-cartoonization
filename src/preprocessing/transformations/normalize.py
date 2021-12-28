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
