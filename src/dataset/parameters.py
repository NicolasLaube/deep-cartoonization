from dataclasses import dataclass, field
from typing import Tuple, List
from src import config
from src.dataset.utils import Movie
from src.preprocessing.filters import ratio_filter
from src.preprocessing.transformations import resize


@dataclass
class ImageDatasetParameters:
    new_size: Tuple[int, int] = None
    ratio_filter_mode: ratio_filter.RatioFilterModes = (
        ratio_filter.RatioFilterMode.NO_FILTER.value
    )
    crop_mode: resize.CropModes = resize.CropMode.RESIZE.value
    nb_images: int = -1


@dataclass
class CartoonsDatasetParameters(ImageDatasetParameters):
    selected_movies: List[Movie] = field(default_factory=lambda: config.MOVIES)


# No additional parameters required for pictures parameters
PicturesDatasetParameters = ImageDatasetParameters
