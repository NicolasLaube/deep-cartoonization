import pandas as pd
from typing import List
from src.dataset.utils import Movie
from src import config

from src.preprocessing.filters import low_quality_filter
from src.preprocessing.filters import ratio_filter
from src.preprocessing.filters import movie_filter


class Filter:
    """A class for all the filters to apply on images"""

    def __init__(
        self,
        new_size: tuple[int, int] = None,
        selected_movies: List[Movie] = config.MOVIES,
        ratio_filter_mode: ratio_filter.RatioFilterModes = ratio_filter.RatioFilterMode.NO_FILTER.value,
    ) -> None:
        self.new_size = new_size
        self.movies = selected_movies
        self.ratio_filter_mode = ratio_filter_mode

    def cartoon_filter(self, df_images: pd.DataFrame) -> pd.DataFrame:
        """Filter cartoons"""
        df_images = movie_filter.filter_movies(df_images, self.movies)
        return self._main_filter(df_images)

    def picture_filter(self, df_images: pd.DataFrame) -> pd.DataFrame:
        """Filter pictures"""
        return self._main_filter(df_images)

    def _main_filter(self, df_images: pd.DataFrame) -> pd.DataFrame:
        """Filter images (functions that are common to both frame and picture preprocessing)"""
        df_images = low_quality_filter.filter_low_quality(df_images, self.new_size)
        df_images = ratio_filter.filter_ratio(df_images, self.ratio_filter_mode)
        return df_images
