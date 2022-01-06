import os
from src.dataset.utils import Movie

#####################
### About folders ###
#####################

ROOT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")

CARTOONS_FOLDER = os.path.join(DATA_FOLDER, "cartoon_frames")
CARTOONS_CSV = os.path.join(DATA_FOLDER, "cartoon_csv")

PICTURES_FOLDER = os.path.join(DATA_FOLDER, "flickr", "Images")
PICTURES_CSV = os.path.join(DATA_FOLDER, "flickr", "captions.csv")

CARTOONS_ALL_CSV = os.path.join(DATA_FOLDER, "cartoons_all.csv")
CARTOONS_FILTERED_CSV = os.path.join(DATA_FOLDER, "cartoons_all.csv")
CARTOONS_TRAIN_CSV = os.path.join(DATA_FOLDER, "cartoons_train.csv")
CARTOONS_TEST_CSV = os.path.join(DATA_FOLDER, "cartoons_test.csv")

PICTURES_ALL_CSV = os.path.join(DATA_FOLDER, "pictures_all.csv")
PICTURES_FILTERED_CSV = os.path.join(DATA_FOLDER, "pictures_all.csv")
PICTURES_TRAIN_CSV = os.path.join(DATA_FOLDER, "pictures_train.csv")
PICTURES_TEST_CSV = os.path.join(DATA_FOLDER, "pictures_test.csv")


WEIGHTS_FOLDER = os.path.join(ROOT_FOLDER, "weights")

LOGS_FOLDER = os.path.join(ROOT_FOLDER, "logs")
ALL_PARAMS_CSV = os.path.join(LOGS_FOLDER, "all_params.csv")

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

VGG_WEIGHTS = os.path.join(ROOT_FOLDER, "weights", "vgg", "vgg19.pth")
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3

#######################
### Training params ###
#######################

TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_WORKERS = 2
SAVE_EVERY_MIN = 15
