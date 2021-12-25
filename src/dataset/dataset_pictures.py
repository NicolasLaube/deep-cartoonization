"""Flickr pictures dataset Loader"""
from typing import Optional
from src import config
from src.dataset.loader import ImageLoader


class PicturesDataset(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(self, transform: Optional[callable] = None, train: bool = True, **kwargs) -> None:
        self.train = train
        ImageLoader.__init__(
            self, 
            transform=transform,
            csv_path=config.IMAGES_TRAIN_CSV if train else config.IMAGES_TEST_CSV, 
            folder=config.PICTURES_FOLDER,
            **kwargs
        )


if __name__ == "__main__":
    from src.preprocessing.preprocessor import Preprocessor

    p = Preprocessor(size=256)



    dd = PicturesDataset(
        train=True,
        transform=p.picture_preprocessor()
    )

    print(dd[0])