"""Cartoon dataset Loader"""
from typing import List, Optional

from torchvision import transforms


from src import config
from src.dataset.loader import ImageLoader
from src.dataset.utils import Movie


class CartoonDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, transform: Optional[callable] = None, train: bool = True, movies: List[Movie] = config.MOVIES, **kwargs
    ) -> None:
        self.movies = movies
        ImageLoader.__init__(self, transform, config.FRAMES_TRAIN_CSV if train else config.FRAMES_TEST_CSV, **kwargs)
        self.__load_specific_frames()


    def __load_specific_frames(self) -> None:
        """Loads the correct list of frames"""
        self.df_images = self.df_images[
            self.df_images["movie"].isin([movie.name for movie in self.movies])
        ]


if __name__ == "__main__":
    dd = CartoonDataset(train=True)

    print(len(dd))