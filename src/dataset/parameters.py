from dataclasses import dataclass, field
from typing import Tuple, List
from src import config
from src.preprocessing.filters import ratio_filter
from src.preprocessing.transformations import resize


@dataclass
class ImageDatasetParameters:
    new_size: Tuple[int, int] = None
    ratio_filter_mode: ratio_filter.RatioFilterModes = (
        ratio_filter.RatioFilterMode.NO_FILTER
    )

    crop_mode: resize.CropModes = resize.CropMode.RESIZE
    nb_images: int = -1


@dataclass
class CartoonsDatasetParameters(ImageDatasetParameters):
    selected_movies: List[config.Movie] = field(default_factory=lambda: config.MOVIES)


# No additional parameters required for pictures parameters
PicturesDatasetParameters = ImageDatasetParameters
