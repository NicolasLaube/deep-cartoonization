"""Generic Image Loader"""
from typing import Any
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class ImageLoader(Dataset):
    """Generic image loader class"""

    def __init__(
        self, csv_path: str, filter_data: callable, transform: callable
    ) -> None:
        self.csv_path = csv_path
        self.df_images = None
        self.filter_data = filter_data
        self.transform = transform
        self.__load_images()
        self.df_images = self.filter_data(self.df_images)

    def __load_images(self) -> None:
        """Loads the list of images"""
        self.df_images = pd.read_csv(self.csv_path, index_col=0)

    def __len__(self) -> int:
        """Length"""
        return len(self.df_images)

    def __getitem__(self, index: int) -> Any:
        """Get an item"""
        image = Image.open(self.df_images["path"][index])
        return self.transform(image)
