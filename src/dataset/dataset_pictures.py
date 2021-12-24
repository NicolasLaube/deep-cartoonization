"""Flickr pictures dataset Loader"""

from src import config
from src.dataset.loader import ImageLoader


class PicturesDatasetLoader(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(self, train: bool = True, **kwargs) -> None:
        ImageLoader.__init__(self, config.IMAGES_TRAIN_CSV if train else config.IMAGES_TEST_CSV, **kwargs)
