import os
from src.dataset.utils import Movie

#####################
### About folders ###
#####################

ROOT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")

FRAMES_FOLDER = os.path.join(DATA_FOLDER, "cartoon_frames")
FRAMES_CSV = os.path.join(DATA_FOLDER, "cartoon_csv")

PICTURES_FOLDER = os.path.join(DATA_FOLDER, "flickr", "Images")
PICTURES_CSV = os.path.join(DATA_FOLDER, "flickr", "captions.csv")

FRAMES_ALL_CSV = os.path.join(DATA_FOLDER, "frames_all.csv")
FRAMES_FILTERED_CSV = os.path.join(DATA_FOLDER, "frames_all.csv")
FRAMES_TRAIN_CSV = os.path.join(DATA_FOLDER, "frames_train.csv")
FRAMES_TEST_CSV = os.path.join(DATA_FOLDER, "frames_test.csv")

PICTURES_ALL_CSV = os.path.join(DATA_FOLDER, "pictures_all.csv")
PICTURES_FILTERED_CSV = os.path.join(DATA_FOLDER, "pictures_all.csv")
PICTURES_TRAIN_CSV = os.path.join(DATA_FOLDER, "pictures_train.csv")
PICTURES_TEST_CSV = os.path.join(DATA_FOLDER, "pictures_test.csv")

####################
### About movies ###
####################

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
    Movie.Zootopia,
]

#######################
### Training params ###
#######################

TEST_SIZE = 0.2
RANDOM_STATE = 42
