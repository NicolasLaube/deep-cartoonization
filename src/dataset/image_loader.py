"""Generic Image Loader"""
from typing import Any, Text
import numpy as np
from nptyping import NDArray
from torch.utils.data import Dataset
import cv2
import pandas as pd

from src.dataset.transformations import resize


class ImageLoader(Dataset):
    """Generic image loader class"""

    def __init__(
        self,
        csv_path: str,
        new_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.csv_path = csv_path
        self.df_images = None
        self.new_size = new_size
        self._load_images()

    def _load_images(self) -> None:
        """Loads the list of images"""
        self.df_images = pd.read_csv(self.csv_path, index_col=0)

    def __len__(self) -> int:
        """Length"""
        return len(self.df_images)

    def __getitem__(self, index: int) -> NDArray[(Any, Any), np.int32]:
        """Get an item"""
        image = cv2.imread(self.df_images["path"][index])
        return self._transform(image)

    def _transform(
        self,
        image: NDArray[(Any, Any), np.int32],
    ) -> NDArray[(Any, Any), np.int32]:
        """To transform the image"""
        image = resize.resize(image, self.new_size)
        return image
