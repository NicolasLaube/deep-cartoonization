"""Flickr pictures dataset Loader"""
from typing import Optional
from src import config
from src.dataset.loader import ImageLoader


class PicturesDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(self, transform: Optional[callable] = None, train: bool = True, **kwargs) -> None:
        self.train = train
        ImageLoader.__init__(self, transform, config.IMAGES_TRAIN_CSV if train else config.IMAGES_TEST_CSV, **kwargs)


if __name__ == "__main__":
    dd = PicturesDataset()

    print(len(dd))