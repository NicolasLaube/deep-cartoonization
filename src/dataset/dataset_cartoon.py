"""Cartoon dataset Loader"""
# pylint: disable=R0913
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
        smooth_and_gray: bool = False,
    ) -> None:
        self.train = mode == "train"
        if mode == "train":
            csv_path = config.CARTOONS_TRAIN_CSV
        elif mode == "validation":
            csv_path = config.CARTOONS_VALIDATION_CSV
        elif mode == "test":
            csv_path = config.CARTOONS_TEST_CSV
        ImageLoader.__init__(
            self, csv_path, filter_data, transform, nb_images, smooth_and_gray
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def show_image(image: NDArray) -> None:
        """Show image"""
        # plot image with matplotlib
        plt.imshow(image)
        plt.show()

    # Test the dataset loader
    dataset = CartoonDataset(
        lambda df: df,
        lambda image: image,
        nb_images=10,
        mode="train",
        smooth_and_gray=True,
    )

    for _ in dataset:
        show_image(_[1].transpose(1, 2, 0))
    # print(dataset.df_images)
    # print(len(dataset))
    # show_image(dataset[0][0])
    # show_image(dataset[0][1])
    # show_image(dataset[0][2])
