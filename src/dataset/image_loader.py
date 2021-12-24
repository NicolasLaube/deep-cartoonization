"""Generic Image Loader"""
from typing import Any, Tuple
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
        transform: callable
    ) -> None:
        self.df_images = pd.read_csv(csv_path, index_col=0)
        self.__load_images()
        self.transform = transform

    def __len__(self) -> int:
        """Length"""
        return len(self.df_images)

    def __getitem__(self, index: int) -> NDArray[(Any, Any), np.int32]:
        """Get an item"""
        return self.transform(cv2.imread(self.df_images["path"][index]))
