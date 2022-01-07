"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.image_loader import ImageLoader
from src.dataset.parameters import CartoonsDatasetParameters
from src.preprocessing.filters import Filter
from src.preprocessing.transformations import Transform


class CartoonDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self,
        filter_data: callable,
        transform: callable,
        nb_images: int = -1,
        train: bool = True,
    ) -> None:
        self.train = train
        if train:
            csv_path = config.CARTOONS_TRAIN_CSV
        else:
            csv_path = config.CARTOONS_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter_data, transform, nb_images)


def init_cartoon_dataset(
    parameters: CartoonsDatasetParameters, train: bool = True
) -> CartoonDataset:
    data_filter = Filter(
        new_size=parameters.new_size,
        selected_movies=parameters.selected_movies,
        ratio_filter_mode=parameters.ratio_filter_mode,
    )
    transform = Transform(new_size=parameters.new_size, crop_mode=parameters.crop_mode)
    return CartoonDataset(
        data_filter.cartoon_filter,
        transform.cartoon_transform,
        parameters.nb_images,
        train,
    )
