"""Normalization transformations"""
# pylint: disable=E0401
from typing import Any, List, Tuple

import numpy as np
import torch
from nptyping import NDArray
from torchvision import transforms


class Normalizer:
    """Normalizer"""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        device: str = "cpu",
    ) -> None:
        self.mean = mean
        self.std = std
        self.device = device
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
        return self.transformation(image).to(self.device)

    def normalize_images(
        self, images: List[NDArray[(Any, Any), np.int32]]
    ) -> NDArray[(Any, Any), np.int32]:
        """Normalize multiple images"""
        return self.transformation(images).to(self.device)

    def inv_normalize(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """To inverse the normalization"""
        tensor_img = torch.Tensor(image).to(self.device)
        mean = torch.Tensor(self.mean).to(self.device)
        std = torch.Tensor(self.std).to(self.device)

        tensor_img = tensor_img * std.view(3, 1, 1) + mean.view(3, 1, 1)
        tensor_img = tensor_img.clamp(0, 1)
        return tensor_img.to(self.device)
