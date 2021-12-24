from typing import Any
import numpy as np
from nptyping import NDArray
from torchvision import transforms


class Preprocessor:
    def __init__(self, size: int) -> None:
        self.size = size
    
    def cartoon_preprocessor(self) -> NDArray[(3, Any, Any), np.int32]:
        """Preprocesses cartoons"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def picture_preprocessor(self) -> NDArray[(3, Any, Any), np.int32]:
        """Preprocess pictures"""
        return transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])