"""Cartoon dataset Loader"""
from typing import Callable

import pandas as pd
from nptyping import NDArray
from typing_extensions import Literal

from src import config
from src.dataset.image_loader import ImageLoader


class CartoonDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self,
        filter_data: Callable[[pd.DataFrame], pd.DataFrame],
        transform: Callable[[NDArray], NDArray],
        nb_images: int = -1,
        mode: Literal["train", "validation", "test"] = "train",
    ) -> None:
        self.train = mode == "train"
        if mode == "train":
            csv_path = config.CARTOONS_TRAIN_CSV
        elif mode == "validation":
            csv_path = config.CARTOONS_VALIDATION_CSV
        elif mode == "test":
            csv_path = config.CARTOONS_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter_data, transform, nb_images)
