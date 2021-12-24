import os
from src.dataset.utils import Movie

#####################
### About folders ###
#####################

ROOT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")

FRAMES_FOLDER = os.path.join(DATA_FOLDER, "cartoon_frames")
FRAMES_CSV = os.path.join(DATA_FOLDER, "cartoon_csv")

FLICKR_FOLDER = os.path.join(DATA_FOLDER, "flickr")
PICTURES_FOLDER = os.path.join(FLICKR_FOLDER, "Images")
PICTURES_TXT = os.path.join(FLICKR_FOLDER, "captions.txt")

FRAMES_ALL_CSV = os.path.join(DATA_FOLDER, "frames_all.csv")
FRAMES_FILTERED_CSV = os.path.join(DATA_FOLDER, "frames_all.csv")
FRAMES_TRAIN_CSV = os.path.join(DATA_FOLDER, "frames_train.csv")
FRAMES_TEST_CSV = os.path.join(DATA_FOLDER, "frames_test.csv")

IMAGES_ALL_CSV = os.path.join(DATA_FOLDER, "images_all.csv")
IMAGES_FILTERED_CSV = os.path.join(DATA_FOLDER, "images_all.csv")
IMAGES_TRAIN_CSV = os.path.join(DATA_FOLDER, "images_train.csv")
IMAGES_TEST_CSV = os.path.join(DATA_FOLDER, "images_test.csv")

####################
### About movies ###
####################

MOVIES = [
    Movie.BabyBoss.name,
    Movie.Cars3.name,
    Movie.Coco.name,
    Movie.InsideOut.name,
    Movie.Luca.name,
    Movie.Onward.name,
    Movie.Soul.name,
    Movie.TheIncredibles.name,
    Movie.TheSecretLifeOfPets.name,
    Movie.ToyStory4.name,
    Movie.Zootopia.name,
]

#######################
### Training params ###
#######################

TEST_SIZE = 0.2
RANDOM_STATE = 42
