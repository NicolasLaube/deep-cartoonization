"""Filter"""
# pylint: disable=W0102
from typing import List, Optional, Tuple

import pandas as pd

from src import config
from src.preprocessing.filters.low_quality_filter import filter_low_quality
from src.preprocessing.filters.movie_filter import filter_movies
from src.preprocessing.filters.ratio_filter import (RatioFilterMode,
                                                    filter_ratio)


class Filter:
    """A class for all the filters to apply on images"""

    def __init__(
        self,
        new_size: Optional[Tuple[int, int]] = None,
        selected_movies: List[config.Movie] = config.MOVIES,
        ratio_filter_mode: RatioFilterMode = RatioFilterMode.NO_FILTER,
    ) -> None:
        self.new_size = new_size
        self.movies = selected_movies
        self.ratio_filter_mode = ratio_filter_mode.value

    def cartoon_filter(self, df_images: pd.DataFrame) -> pd.DataFrame:
        """Filter cartoons"""
        df_images = filter_movies(df_images, self.movies)
        return self.__main_filter(df_images)

    def picture_filter(self, df_images: pd.DataFrame) -> pd.DataFrame:
        """Filter pictures"""
        return self.__main_filter(df_images)

    def __main_filter(self, df_images: pd.DataFrame) -> pd.DataFrame:
        """Filter images (functions that are common to both frame and picture preprocessing)"""
        df_images = filter_low_quality(df_images, self.new_size)
        df_images = filter_ratio(df_images, self.ratio_filter_mode)
        return df_images
