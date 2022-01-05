"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.image_loader import ImageLoader
from src.dataset.parameters import CartoonDatasetParameters
from src.preprocessing.filters import Filter
from src.preprocessing.transformations import Transform


class CartoonDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, filter_data: callable, transform: callable, train: bool = True
    ) -> None:
        self.train = train
        if train:
            csv_path = config.FRAMES_TRAIN_CSV
        else:
            csv_path = config.FRAMES_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter_data, transform)


def init_cartoon_dataset(
    parameters: CartoonDatasetParameters, train: bool = True
) -> CartoonDataset:
    data_filter = Filter(
        new_size=parameters.new_size,
        selected_movies=parameters.selected_movies,
        ratio_filter_mode=parameters.ratio_filter_mode,
    )
    transform = Transform(new_size=parameters.new_size, crop_mode=parameters.crop_mode)
    return CartoonDataset(data_filter, transform, train)
