"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.image_loader import ImageLoader
from src.dataset.utils import Movie


class CartoonDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, filter_data: callable, transform: callable, train: bool = True, **args
    ) -> None:
        self.train = train
        if train:
            csv_path = config.FRAMES_TRAIN_CSV
        else:
            csv_path = config.FRAMES_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter_data, transform, **args)


if __name__ == "__main__":
    loader = CartoonDataset()
    print(loader[0])
