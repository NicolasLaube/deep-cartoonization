

"""Flickr pictures dataset Loader"""
import os
from typing import Any
import numpy as np
from nptyping import NDArray
from torch.utils.data import Dataset
import cv2


from src import config


class PicturesDatasetLoader(Dataset):
    """Cartoon dataset loader class"""
  
    def __init__(self) -> None:
        self.pictures = []
        self.__load_pictures()

    def __load_pictures(self) -> None:
        """Loads the list of frames"""
        with open(config.PICTURES_TXT) as pictures_txt:
            lines = pictures_txt.readlines()
            for line in lines:
                picture_name = line.split(",")[0]
                if ".jpg" in picture_name:
                    self.pictures.append(os.path.join(
                        config.PICTURES_FOLDER,
                        picture_name
                    ))

    def __len__(self) -> int:
        """Length"""
        return len(self.frames)

    def __getitem__(self, index: int) -> NDArray[(Any, Any), np.int32]:
        """Get an item"""
        return cv2.imread(self.pictures[index])
