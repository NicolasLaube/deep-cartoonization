# pylint: disable=invalid-name
import os
from enum import Enum


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


#####################
### About folders ###
#####################

ROOT_FOLDER = "/usr/users/gpupro/gpu_stutz/cartoongan"

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")

CARTOONS_FOLDER = os.path.join(DATA_FOLDER, "cartoon_frames")
CARTOONS_CSV = os.path.join(DATA_FOLDER, "cartoon_csv")

PICTURES_FOLDER = os.path.join(DATA_FOLDER, "flickr", "Images")
PICTURES_CSV = os.path.join(DATA_FOLDER, "flickr", "captions.csv")

CARTOONS_ALL_CSV = os.path.join(DATA_FOLDER, "cartoons_all.csv")
CARTOONS_FILTERED_CSV = os.path.join(DATA_FOLDER, "cartoons_all.csv")
CARTOONS_TRAIN_CSV = os.path.join(DATA_FOLDER, "cartoons_train.csv")
CARTOONS_VALIDATION_CSV = os.path.join(DATA_FOLDER, "cartoons_validation.csv")
CARTOONS_TEST_CSV = os.path.join(DATA_FOLDER, "cartoons_test.csv")

PICTURES_ALL_CSV = os.path.join(DATA_FOLDER, "pictures_all.csv")
PICTURES_FILTERED_CSV = os.path.join(DATA_FOLDER, "pictures_all.csv")
PICTURES_TRAIN_CSV = os.path.join(DATA_FOLDER, "pictures_train.csv")
PICTURES_VALIDATION_CSV = os.path.join(DATA_FOLDER, "pictures_validation.csv")
PICTURES_TEST_CSV = os.path.join(DATA_FOLDER, "pictures_test.csv")


WEIGHTS_FOLDER = os.path.join(ROOT_FOLDER, "weights")

LOGS_FOLDER = os.path.join(ROOT_FOLDER, "logs")
TENSORBOARD_FOLDER = os.path.join(LOGS_FOLDER, "tensorboard")
ALL_PARAMS_CSV = os.path.join(LOGS_FOLDER, "all_params.csv")
ALL_PARAMS_EXAMPLE_CSV = os.path.join(LOGS_FOLDER, "all_params_example.csv")

SCRIPTS_FOLDER = os.path.join(ROOT_FOLDER, "scripts")
SCP_BASH_SCRIPT = os.path.join(SCRIPTS_FOLDER, "download_folder.sh")
PASSWORD_PATH = os.path.join(SCRIPTS_FOLDER, "password.txt")
REMOTE_PARAMS_PATH = os.path.join("logs", "all_params_remote.csv")

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
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
NUM_WORKERS = 2
SAVE_EVERY_MIN = 15
DEFAULT_BATCH_SIZE = 16
