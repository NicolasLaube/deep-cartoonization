"""Flickr pictures dataset Loader"""

from src import config
from src.dataset.image_loader import ImageLoader


class PicturesDatasetLoader(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self,
        train: bool = True,
        filter_data: callable = None,
        transform: callable = None,
    ) -> None:
        self.train = train
        if train:
            csv_path = config.PICTURES_TRAIN_CSV
        else:
            csv_path = config.PICTURES_TEST_CSV
        ImageLoader.__init__(self, csv_path, filter_data, transform)
