"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.loader import ImageLoader
from src.dataset.utils import Movie


class CartoonDatasetLoader(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, train: bool = True, movies: List[Movie] = config.MOVIES, **kwargs
    ) -> None:
        self.movies = movies
        self.__load_specific_frames()
        ImageLoader.__init__(self, config.FRAMES_TRAIN_CSV if train else config.FRAMES_TEST_CSV, **kwargs)

    def __load_specific_frames(self) -> None:
        """Loads the correct list of frames"""
        self.df_images = self.df_images[
            self.df_images["movie"].isin([movie.name for movie in self.movies])
        ]
