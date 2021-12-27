from typing import Any
import numpy as np
from nptyping import NDArray

from src.preprocessing.transformations import resize
from src.preprocessing.transformations import normalize


class Transform:
    """A class for all transformations on pictures to process"""

    def __init__(
        self,
        new_size: tuple[int, int] = (256, 256),
        crop_mode: resize.CropModes = resize.CropMode.RESIZE.value,
    ) -> None:
        self.new_size = new_size
        self.crop_mode = crop_mode

    def cartoon_transform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform cartoon frames"""
        return self._main_filter(image)

    def picture_transform(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform pictures"""
        return self._main_filter(image)

    def _main_filter(
        self, image: NDArray[(Any, Any), np.int32]
    ) -> NDArray[(Any, Any), np.int32]:
        """Transform images (functions that are common to both frame and picture preprocessing)"""
        image = resize.resize(image, self.new_size, self.crop_mode)
        image = normalize.normalize(image)
        return image
