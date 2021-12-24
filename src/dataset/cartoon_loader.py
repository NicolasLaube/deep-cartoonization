"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.image_loader import ImageLoader
from src.dataset.utils import Movie


class CartoonDatasetLoader(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, train: bool = True, movies: List[Movie] = config.MOVIES, **kwargs
    ) -> None:
        self.movies = movies
        self.train = train
        if train:
            csv_path = config.FRAMES_TRAIN_CSV
        else:
            csv_path = config.FRAMES_TEST_CSV
        ImageLoader.__init__(self, csv_path, **kwargs)
        self._load_specific_frames()

    def _load_specific_frames(self) -> None:
        """Loads the correct list of frames"""
        self.df_images = self.df_images[
            self.df_images["movie"].isin([movie.name for movie in self.movies])
        ]


if __name__ == "__main__":
    loader = CartoonDatasetLoader()
    print(loader[0])
