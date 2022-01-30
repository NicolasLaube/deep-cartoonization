"""Transformations"""
from typing import Any, Optional, Tuple

import numpy as np
from nptyping import NDArray

from src.models.utils.parameters import Architecture
from src.preprocessing.transformations import resize
from src.preprocessing.transformations.normalize import Normalizer


class Transform:
    """A class for all transformations on pictures to process"""

    def __init__(
        self,
        architecture: Architecture,
        new_size: Optional[Tuple[int, int]] = (256, 256),
        crop_mode: resize.CropMode = resize.CropMode.RESIZE,
    ) -> None:
        self.architecture = architecture
        self.new_size = new_size
        self.crop_mode = crop_mode.value
        self.normalizer = self.__init_normalizer()

    def __init_normalizer(self) -> Normalizer:
        """Initialize normalizer"""
        if self.architecture == Architecture.GANFixed:
            return Normalizer(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if self.architecture == Architecture.GANModular:
            return Normalizer(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        raise NotImplementedError("Normalizer wasn't implemented for this architecture")

    def cartoon_transform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform cartoon frames"""
        return self.__main_filter(image)

    def picture_transform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform pictures"""
        return self.__main_filter(image)

    def __main_filter(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform images (functions that are common to both frame and picture preprocessing)"""
        image = resize.resize(image, self.new_size, self.crop_mode)

        return self.normalizer.normalize(image)
