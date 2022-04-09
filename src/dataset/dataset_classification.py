"""Cartoon classification dataset"""
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """Custom image loader class"""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join("cartoongan", self.df.iloc[idx]["path"])
        image = Image.open(img_path)
        label = self.df.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label
