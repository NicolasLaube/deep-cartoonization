from torchvision import transforms
import numpy as np
from nptyping import NDArray
from typing import Any, Tuple


class Normalizer:
    def __init__(
        self, mean: Tuple[float] = (0.5, 0.5, 0.5), std: Tuple[float] = (0.5, 0.5, 0.5)
    ) -> None:
        self.mean = mean
        self.std = std

    def normalize(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )(image)
