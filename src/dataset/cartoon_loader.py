import os
import csv
from typing import List, Any
from enum import Enum
import numpy as np
from nptyping import NDArray
from torch.utils.data import Dataset
import cv2


from src import config


class Movie(Enum):
    """All movies"""
    Soul = "Soul"
    Cars3 = "Cars3"
    BabyBoss = "BabyBoss"
    Coco = "Coco"
    InsideOut = "InsideOut"
    Luca = "Luca"
    Onward = "Onward"
    TheIncredibles = "TheIncredibles"
    TheSecretLifeOfPets = "TheSecretLifeOfPets"
    ToyStory4 = "ToyStory4"
    Zootopia = "Zootopia"


class CartoonDatasetLoader(Dataset):
  
    def __init__(self, movies: List[Movie]) -> None:
        self.movies = movies
        self.frames = []
        self.__load_frames()

    def __load_frames(self):
        """Loads the list of frames"""
        for movie in self.movies:
            with open(os.path.join(config.CSV_FOLDER, \
                movie.value + ".csv"), "r", encoding="utf-8") as csv_f:
                reader = csv.reader(csv_f)
                for frame_path in reader:
                    self.frames.append(
                        os.path.join(
                            movie.value, 
                            frame_path[0]
                        )
                    ) 

    def __len__(self):
        """Length"""
        return len(self.frames)

    def __getitem__(self, index: int) -> NDArray[(Any, Any), np.int32]:
        """Get an item"""
        return cv2.imread(os.path.join(config.FRAMES_FOLDER, self.frames[index]))
