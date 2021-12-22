import os
from src.dataset.utils import Movie

ROOT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

FRAMES_FOLDER = os.path.join(ROOT_FOLDER, "data", "cartoon_frames")
FRAMES_CSV = os.path.join(ROOT_FOLDER, "data", "cartoon_csv")

PICTURES_FOLDER = os.path.join(ROOT_FOLDER, "data", "flickr", "Images")
PICTURES_TXT = os.path.join(ROOT_FOLDER, "data" "flickr" "captions.txt")

MOVIES = [
    Movie.BabyBoss,
    Movie.Cars3,
    Movie.Coco,
    Movie.InsideOut,
    Movie.Luca,
    Movie.Onward,
    Movie.Soul,
    Movie.TheIncredibles,
    Movie.TheSecretLifeOfPets,
    Movie.ToyStory4,
    Movie.Zootopia
]

VGG_WEIGHTS = os.path.join(ROOT_FOLDER, "weights", "pretrained", "vgg19.pth")
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3