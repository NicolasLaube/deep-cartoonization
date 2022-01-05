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


WEIGHTS_FOLDER = os.path.join(ROOT_FOLDER, "weights", "pretrained")
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


## NETWORK ARCHITECTURE

NB_RESNET_BLOCKS = (8,)
NB_CHANNELS_PICTURE = (3,)
NB_CHANNELS_CARTOON = (3,)
NB_CHANNELS_1st_HIDDEN_LAYER_GEN = (64,)
NB_CHANNELS_1st_HIDDEN_LAYER_DISC = 32
