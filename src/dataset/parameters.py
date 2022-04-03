"""Dataset parameters"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src import config
from src.preprocessing.filters import ratio_filter
from src.preprocessing.transformations import resize


@dataclass
class ImageDatasetParameters:
    """Image dataset parameters"""

    new_size: Optional[Tuple[int, int]] = None
    ratio_filter_mode: ratio_filter.RatioFilterMode = (
        ratio_filter.RatioFilterMode.NO_FILTER
    )
    crop_mode: resize.CropMode = resize.CropMode.RESIZE
    nb_images: int = -1
    smoothing_kernel_size: Optional[int] = None


@dataclass
class CartoonsDatasetParameters(ImageDatasetParameters):
    """Cartoons dataset parameters"""

    selected_movies: List[config.Movie] = field(default_factory=lambda: config.MOVIES)


# No additional parameters required for pictures parameters
PicturesDatasetParameters = ImageDatasetParameters
