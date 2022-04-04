"""A script to run easily some train"""

import argparse

from src import dataset, models, preprocessing
from src.pipelines.pipeline import ModelPathsParameters, Pipeline

if __name__ == "__main__":

    ###################################
    ### First we load the arguments ###
    ###################################

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        help="Training mode (Pretrain or Train)",
        choices=["Pretrain", "Train"],
        default="Train",
    )

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
        "--disc-path", type=str, help="Path for discriminator weights", default=None
    )

    parser.add_argument(
        "--nb-images", type=int, help="Nb of images to train on", default=-1
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
        "--batch-size", type=int, help="Batch size for training", default=16
    )

    parser.add_argument("--lr", type=float, help="The learning rate", default=1e-4)

    parser.add_argument("--epochs", type=int, help="The number of epochs", default=60)

    parser.add_argument(
        "--content-loss-weight",
        type=float,
        help="The weight of the content loss for the generator loss",
        default=1,
    )

    parser.add_argument(
        "--smoothing-kernel-size",
        type=int,
        help="The kernel size for the smoothing parameter (0 for no smoothing)",
        default=0,
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
    DISC_PATH = args.disc_path

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
    SMOOTHING_KERNEL_SIZE = (
        None if args.smoothing_kernel_size == 0 else args.smoothing_kernel_size
    )

    # About training
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    CONTENT_LOSS_WEIGHT = args.content_loss_weight

    #####################################
    ### Then we can build the objects ###
    #####################################

    CARTOONS_DATASET_PARAMETERS = dataset.CartoonsDatasetParameters(
        new_size=NEW_SIZE,
        crop_mode=CROP_MODE,
        ratio_filter_mode=RATIO_FILTER_MODE,
        nb_images=NB_IMAGES,
        smoothing_kernel_size=SMOOTHING_KERNEL_SIZE,
    )
    PICTURES_DATASET_PARAMETERS = dataset.PicturesDatasetParameters(
        new_size=NEW_SIZE,
        crop_mode=CROP_MODE,
        ratio_filter_mode=RATIO_FILTER_MODE,
        nb_images=NB_IMAGES,
        smoothing_kernel_size=SMOOTHING_KERNEL_SIZE,
    )
    PRETRAINING_PARAMETERS = models.PretrainingParams(
        batch_size=BATCH_SIZE, gen_lr=LEARNING_RATE, disc_lr=LEARNING_RATE
    )
    TRAINING_PARAMETERS = models.TrainingParams(
        batch_size=BATCH_SIZE,
        gen_lr=LEARNING_RATE,
        disc_lr=LEARNING_RATE,
        weight_generator_content_loss=CONTENT_LOSS_WEIGHT,
    )
    INIT_MODEL_PATH = (
        None
        if GEN_PATH is None
        else ModelPathsParameters(gen_path=GEN_PATH, disc_path=DISC_PATH)
    )

    pipeline = Pipeline(
        architecture=ARCHITECTURE,
        architecture_params=ARCHITECTURE_PARAMS,
        cartoons_dataset_parameters=CARTOONS_DATASET_PARAMETERS,
        pictures_dataset_parameters=PICTURES_DATASET_PARAMETERS,
        init_models_paths=INIT_MODEL_PATH,
        training_parameters=TRAINING_PARAMETERS,
        pretraining_parameters=PRETRAINING_PARAMETERS,
    )

    ######################################
    ### Finally we can train the model ###
    ######################################
    if args.mode == "Pretrain":
        pipeline.pretrain(args.epochs)
    else:
        pipeline.train(args.epochs)
