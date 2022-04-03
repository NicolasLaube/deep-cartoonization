"""Generic Image Loader"""
# pylint: disable=E1101, R0913
import os
from typing import Callable, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from nptyping import NDArray
from numpy import asarray
from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.anime.dataset_preprocessing import OfflineDataProcessing


class ImageLoader(Dataset):
    """Generic image loader class"""

    def __init__(
        self,
        csv_path: str,
        filter_data: Callable[[pd.DataFrame], pd.DataFrame],
        transform: Callable[[NDArray], NDArray],
        nb_images: int = -1,
        smooth_and_gray: bool = False,
        anime_mode: bool = False,
    ) -> None:
        self.smooth_and_gray = smooth_and_gray
        self.anime_mode = anime_mode
        self.filter_data = filter_data
        self.transform = transform
        self.df_images: pd.DataFrame = self.filter_data(
            pd.read_csv(csv_path, index_col=0)
        )
        self.set_nb_images(nb_images)

    def set_nb_images(self, nb_images: int) -> None:
        """Set the number of images"""
        # Get the real nb of images to reset...
        nb_images = min(  # type: ignore
            nb_images if nb_images > 0 else np.inf,
            len(self.df_images),
        )
        # ...and reset the dataset
        self.df_images = self.df_images.sample(n=nb_images)
        self.df_images = self.df_images.reset_index(drop=True)

    def __len__(self) -> int:
        """Length"""
        return len(self.df_images)

    def __getitem__(
        self, index: int
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Get an item"""
        image_path = self.df_images["path"][index]
        image = Image.open(image_path)
        image_transformed = self.transform(image)

        if not self.smooth_and_gray and not self.anime_mode:

            return image_transformed

        if not self.smooth_and_gray and self.anime_mode:
            image = cv2.imread(image_path)
            image_transformed = asarray(self.transform(Image.fromarray(image)))

            return torch.from_numpy(image_transformed.copy())

        image_gray = self.transform(Image.fromarray(self.get_gray_image(image_path)))

        image_smooth_path = image_path.replace("cartoon_frames", "cartoon_edged")
        if os.path.exists(image_smooth_path):
            image_smooth = self.transform(
                Image.fromarray(cv2.imread(image_smooth_path))
            )
        else:
            image_smooth = OfflineDataProcessing.edge_image(cv2.imread(image_path))
            image_smooth = self.transform(Image.fromarray(image_smooth))
        return (
            image_transformed,
            image_gray,
            image_smooth,
        )

    @staticmethod
    def get_gray_image(image_path: str):
        """Get the gray image"""
        image = cv2.imread(image_path)

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)

        return image_gray
