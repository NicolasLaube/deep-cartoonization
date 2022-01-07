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


@dataclass
class CartoonsDatasetParameters(ImageDatasetParameters):
    selected_movies: List[Movie] = field(default_factory=lambda: config.MOVIES)


PicturesDatasetParameters = ImageDatasetParameters
