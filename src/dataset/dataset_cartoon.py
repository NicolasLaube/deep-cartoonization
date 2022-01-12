"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.image_loader import ImageLoader


class CartoonDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self,
        filter_data: callable,
        transform: callable,
        nb_images: int = -1,
        train: bool = True,
        test: bool = False,
    ) -> None:
        self.train = train
        if train:
            csv_path = config.CARTOONS_TRAIN_CSV
        else:
            csv_path = config.CARTOONS_VALIDATION_CSV
        if test:
            csv_path = config.CARTOONS_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter_data, transform, nb_images)
