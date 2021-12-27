"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.image_loader import ImageLoader
from src.dataset.utils import Movie


class CartoonDatasetLoader(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, train: bool = True, filter: callable = None, transform: callable = None
    ) -> None:
        self.train = train
        if train:
            csv_path = config.FRAMES_TRAIN_CSV
        else:
            csv_path = config.FRAMES_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter, transform)


if __name__ == "__main__":
    loader = CartoonDatasetLoader()
    print(loader[0])
