"""Generic Image Loader"""
from typing import Any, Optional, Tuple
import numpy as np
from nptyping import NDArray
from torch.utils.data import Dataset
import cv2
import pandas as pd

class ImageLoader(Dataset):
    """Generic image loader class"""

    def __init__(
        self,
        transform: Optional[callable],
        csv_path: str,
        size: Optional[int] = None
    ) -> None:
        self.size = size
        self.transform = transform
        self.csv_path = csv_path
        self.df_images = pd.read_csv(csv_path, index_col=0)

    def __len__(self) -> int:
        """Length"""
        if self.size is not None:
            return len(self.df_images[:self.size])
        return len(self.df_images)

    def __getitem__(self, index: int) -> NDArray[(Any, Any), np.int32]:
        """Get an item"""
        if self.transform is not None:
            return self.transform(cv2.imread(self.df_images["path"][index]))
        return cv2.imread(self.df_images["path"][index])
    
    def get_path(self, index: int) -> str:
        return self.df_images["name"][index]