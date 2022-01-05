"""Flickr pictures dataset Loader"""

from src import config
from src.dataset.image_loader import ImageLoader
from src.dataset.parameters import PicturesDatasetParameters
from src.preprocessing.filters import Filter
from src.preprocessing.transformations import Transform


class PicturesDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, filter_data: callable, transform: callable, train: bool = True
    ) -> None:
        self.train = train
        if train:
            csv_path = config.PICTURES_TRAIN_CSV
        else:
            csv_path = config.PICTURES_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter_data, transform)


def init_pictures_dataset(
    parameters: PicturesDatasetParameters, train: bool = True
) -> PicturesDataset:
    data_filter = Filter(
        new_size=parameters.new_size, ratio_filter_mode=parameters.ratio_filter_mode
    )
    transform = Transform(new_size=parameters.new_size, crop_mode=parameters.crop_mode)
    return PicturesDataset(data_filter, transform, train)
