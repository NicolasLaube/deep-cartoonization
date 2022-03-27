"""A script to predict images"""

import argparse
import os

import matplotlib.pyplot as plt

from src import dataset, models, preprocessing
from src.pipelines.pipeline import Cartoonizer

if __name__ == "__main__":

    ###################################
    ### First we load the arguments ###
    ###################################

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        type=str,
        help="Model architecture (Fixed, Modular or Anime)",
        choices=["Fixed", "Modular", "Anime"],
        default="Fixed",
    )

    parser.add_argument(
        "--gen-path", type=str, help="Path for generator weights", default=None
    )

    parser.add_argument(
        "--nb-images", type=int, help="Nb of images to infer", default=50
    )

    parser.add_argument(
        "--new-size", type=int, help="Size of the images to transform", default=256
    )

    parser.add_argument(
        "--crop-mode",
        type=str,
        help="How to crop the images (Resize, Center or Random)",
        choices=["Resize", "Center", "Random"],
        default="Center",
    )

    parser.add_argument(
        "--ratio-filter",
        type=str,
        help="If images of a certain ratio should be filtered (None, Portrait or Landscape)",
        choices=["None", "Portrait", "Landscape"],
        default="None",
    )

    parser.add_argument(
        "--batch-size", type=int, help="Batch size for infering", default=10
    )

    parser.add_argument(
        "--save-path", type=str, help="Where to save the images", default="./images"
    )

    args = parser.parse_args()

    ###########################
    ### Then we format them ###
    ###########################

    # About architecture
    architecture_types = {
        "Fixed": models.Architecture.GANFixed,
        "Modular": models.Architecture.GANModular,
        "Anime": models.Architecture.GANAnime,
    }
    ARCHITECTURE = architecture_types[args.architecture]
    ARCHITECTURE_PARAMS = models.ArchitectureParamsNULL()

    # Preload a model
    GEN_PATH = args.gen_path

    # About preprocessing
    NB_IMAGES = args.nb_images
    NEW_SIZE = (args.new_size, args.new_size)
    crop_modes = {
        "Resize": preprocessing.CropMode.RESIZE,
        "Center": preprocessing.CropMode.CROP_CENTER,
        "Random": preprocessing.CropMode.CROP_RANDOM,
    }
    CROP_MODE = crop_modes[args.crop_mode]
    ratio_filter_modes = {
        "None": preprocessing.RatioFilterMode.NO_FILTER,
        "Portrait": preprocessing.RatioFilterMode.FILTER_PORTRAIT,
        "Landscape": preprocessing.RatioFilterMode.FILTER_LANDSCAPE,
    }
    RATIO_FILTER_MODE = ratio_filter_modes[args.ratio_filter]

    # About training
    BATCH_SIZE = args.batch_size

    #####################################
    ### Then we can build the objects ###
    #####################################

    PICTURES_DATASET_PARAMETERS = dataset.PicturesDatasetParameters(
        new_size=NEW_SIZE,
        crop_mode=CROP_MODE,
        ratio_filter_mode=RATIO_FILTER_MODE,
        nb_images=NB_IMAGES,
    )

    cartoonizer = Cartoonizer(
        infering_parameters=models.InferingParams(batch_size=BATCH_SIZE),
        architecture=ARCHITECTURE,
        architecture_params=ARCHITECTURE_PARAMS,
        pictures_dataset_parameters=PICTURES_DATASET_PARAMETERS,
        gen_path=GEN_PATH,
    )

    ######################################
    ### Finally we can train the model ###
    ######################################
    cartoonized_images = cartoonizer.get_cartoonized_images()

    os.mkdir(args.save_path)

    WEIGHT = 8
    HEIGHT = 8
    fig = plt.figure(figsize=(WEIGHT, HEIGHT))
    for i, cartoonized_image in enumerate(cartoonized_images):
        picture = cartoonized_image["picture"]
        cartoon = cartoonized_image["cartoon"]
        plt.imshow(picture)
        plt.savefig(os.path.join(args.save_path, f"image_{i}_picture.png"))
        plt.imshow(cartoon)
        plt.savefig(os.path.join(args.save_path, f"image_{i}_cartoon.png"))
