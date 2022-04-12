"""Transformations"""
from typing import Any, Optional, Tuple

import numpy as np
from nptyping import NDArray

from src.models.utils.parameters import Architecture
from src.preprocessing.transformations import resize, smooth
from src.preprocessing.transformations.normalize import Normalizer


class Transform:
    """A class for all transformations on pictures to process"""

    def __init__(
        self,
        architecture: Architecture,
        new_size: Optional[Tuple[int, int]] = (256, 256),
        crop_mode: resize.CropMode = resize.CropMode.RESIZE,
        smoothing_kernel_size: int = 0,
        device: str = "cpu",
    ) -> None:
        self.architecture = architecture
        self.new_size = new_size
        self.crop_mode = crop_mode
        self.smoothing_kernel_size = smoothing_kernel_size
        self.device = device
        self.normalizer = self.__init_normalizer()

    def __init_normalizer(self) -> Normalizer:
        """Initialize normalizer"""
        if self.architecture == Architecture.GANFixed:
            return Normalizer(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                device=self.device,
            )
        if self.architecture == Architecture.GANModular:
            return Normalizer(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device=self.device
            )
        if self.architecture == Architecture.GANAnime:
            return Normalizer(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device=self.device
            )

        raise NotImplementedError("Normalizer wasn't implemented for this architecture")

    def cartoon_transform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform cartoon frames"""
        return self.__main_transformer(image)

    def picture_transform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform pictures"""
        return self.__main_transformer(image)

    def cartoon_untransform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Untransform cartoon frames"""
        return self.__main_untransformer(image)

    def picture_untransform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Untransform pictures"""
        return self.__main_untransformer(image)

    def __main_transformer(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform images (functions that are common to both frame and picture preprocessing)"""
        image = resize.resize(image, self.new_size, self.crop_mode)
        if self.smoothing_kernel_size > 0:
            image = smooth.edge_promoting(image, kernel_size=self.smoothing_kernel_size)

        return self.normalizer.normalize(image)

    def __main_untransformer(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Untransform images (functions that are common to both frame and picture preprocessing)"""
        return np.moveaxis(self.normalizer.inv_normalize(image).numpy(), 0, 2)
