"""Normalization transformations"""
# pylint: disable=E0401
from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray
from torchvision import transforms


class Normalizer:
    """Normalizer"""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.mean = mean
        self.std = std
        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def normalize(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Normalize one image"""
        return self.transformation(image)

    def normalize_images(
        self, images: List[NDArray[(Any, Any), np.int32]]
    ) -> NDArray[(Any, Any), np.int32]:
        """Normalize multiple images"""
        return self.transformation(images)
