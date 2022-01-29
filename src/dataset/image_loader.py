"""Generic Image Loader"""
from typing import Callable

import numpy as np
import pandas as pd
from nptyping import NDArray
from PIL import Image
from torch.utils.data import Dataset


class ImageLoader(Dataset):
    """Generic image loader class"""

    def __init__(
        self,
        csv_path: str,
        filter_data: Callable[[pd.DataFrame], pd.DataFrame],
        transform: Callable[[NDArray], NDArray],
        nb_images: int = -1,
    ) -> None:
        self.filter_data = filter_data
        self.transform = transform
        self.df_images: pd.DataFrame = self.filter_data(
            pd.read_csv(csv_path, index_col=0)
        )
        self.set_nb_images(nb_images)

    def set_nb_images(self, nb_images: int) -> None:
        """Set the number of images"""
        # Get the real nb of images to reset...
        nb_images = min(
            nb_images if nb_images > 0 else np.inf,
            len(self.df_images),
        )
        # ...and reset the dataset
        self.df_images = self.df_images.sample(n=nb_images)
        self.df_images = self.df_images.reset_index(drop=True)

    def __len__(self) -> int:
        """Length"""
        return len(self.df_images)

    def __getitem__(self, index: int) -> NDArray:
        """Get an item"""
        image = Image.open(self.df_images["path"][index])
        return self.transform(image)
