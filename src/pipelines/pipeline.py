from abc import abstractmethod, ABC
import enum
import os
import sys
from dataclasses import asdict, dataclass, replace
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from nptyping import NDArray
from functools import reduce
from numpy import logical_and
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from src import config
from src.models.utils.parameters import (
    CartoonGanBaseParameters,
    CartoonGanModelParameters,
    NULLArhcitectureParams,
)
from src.extraction import *
from src.dataset.parameters import CartoonsDatasetParameters, PicturesDatasetParameters
from src.dataset.dataset_cartoon import init_cartoon_dataset
from src.dataset.dataset_pictures import init_pictures_dataset
from src.pipelines.predictor import Predictor
from src.pipelines.trainer import Trainer


@dataclass
class ModelPathsParameters:
    generator_path: str
    discriminator_path: str


class Architecture(enum.Enum):
    GANStyle = "Style Gan"
    GANUNet = "UNet GAN"
    GANFixed = "Fixed GAN"
    GANModular = "Modular GAN"


@dataclass
class ArchitectureParams:
    pass


class Pipeline(ABC):
    def __init__(
        self,
        *,
        architecture: Architecture = Architecture.UnetGAN,
        architecture_params: ArchitectureParams = NULLArhcitectureParams(),
        cartoons_dataset_parameters: CartoonsDatasetParameters,
        pictures_dataset_parameters: PicturesDatasetParameters,
        training_parameters: CartoonGanBaseParameters,
        pretraining_parameters: CartoonGanBaseParameters = None,
        init_models_paths: Optional[ModelPathsParameters] = None,
    ):
        # Initialize parameters
        self.device = init_device()
        self.trainer = init_trainer(architecture)
        self.inferer = Predictor(architecture, architecture_params=architecture_params)
        self.architecture = architecture
        self.architecture_params = architecture_params
        self.cartoons_dataset_parameters = cartoons_dataset_parameters
        self.pictures_dataset_parameters = pictures_dataset_parameters
        self.training_params = training_parameters

        self.pretraining_params = (
            pretraining_parameters
            if pretraining_parameters != None
            else training_parameters
        )

        self.init_models_paths = (
            init_models_paths
            if init_models_paths != None
            else ModelPathsParameters(generator_path=None, discriminator_path=None)
        )

        # Initialize logs
        self.__init_logs()

    def pretrain(self, nb_epochs: int) -> None:
        """To pretrain the model"""
        self.__init_train_data_loaders(pretraining=True)
        pretraining_parameters = replace(self.pretraining_params)
        pretraining_parameters.epochs = nb_epochs
        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()
        if models_to_load_paths != None:
            self.cartoon_gan.load_model(**models_to_load_paths)

        self.trainer.pretrain(
            pictures_loader=self.train_pictures_loader,
            cartoon_loader=self.train_cartoons_loader,
            batch_callback=None,
            epoch_start=self.params["epochs_pretrained_nb"] + 1,
        )
        # parameters=pretraining_parameters,
        # saved_weights_path=self.weights_path,

        self.params["epochs_pretrained_nb"] += nb_epochs
        # self.params["pretrain_reconstruction_loss"] = loss
        self.__save_params()

    def train(self, nb_epochs: int) -> None:
        """To train the model"""
        self.__init_train_data_loaders(pretraining=False)
        training_parameters = replace(self.training_params)
        training_parameters.epochs = nb_epochs
        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()
        if models_to_load_paths != None:
            self.cartoon_gan.load_model(**models_to_load_paths)
        # Then we can train the model
        self.trainer.train(
            pictures_loader=self.train_pictures_loader,
            cartoon_loader=self.train_cartoons_loader,
            batch_callback=None,
            epoch_start=self.params["epochs_pretrained_nb"] + 1,
        )
        # parameters=training_parameters,
        # saved_weights_path=self.weights_path,

        self.params["epochs_trained_nb"] += nb_epochs
        # self.params["train_discriminator_loss"] = losses["discriminator_loss"]
        # self.params["train_generator_loss"] = losses["generator_loss"]
        # self.params["train_conditional_loss"] = losses["conditional_loss"]
        self.__save_params()

    def cartoonize_images(
        self, nb_images: int = -1
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """To show some cartoonized images"""

        models_to_load_paths = self.__get_model_to_load()

        if models_to_load_paths != None:
            self.inferer.load_model(**models_to_load_paths)

        test_pictures_loader = DataLoader(
            dataset=self.pictures_dataset,
            batch_size=self.training_params.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.NUM_WORKERS,
        )

        return self.cartoon_gan.cartoonize_dataset(
            pictures_loader=test_pictures_loader, nb_images=nb_images
        )

    ###################
    ### About inits ###
    ###################

    def __init_logs(self):
        """To init the logs folder or to load the training model"""
        # Import all fixed params & parameters of all runs
        global_params = {
            "cartoon_gan_architecture": self.cartoon_gan_architecture.value,
            **Trainer.__format_dataclass(
                self.cartoon_gan_model_params, "cartoon_gan_model"
            ),
            **Trainer.__format_dataclass(
                self.cartoons_dataset_parameters, "cartoon_dataset"
            ),
            **Trainer.__format_dataclass(
                self.pictures_dataset_parameters, "pictures_dataset"
            ),
            **Trainer.__format_dataclass(self.pretraining_params, "pretraining"),
            **Trainer.__format_dataclass(self.training_params, "training"),
            **Trainer.__format_dataclass(self.init_models_paths, "init"),
        }
        df_all_params = pd.read_csv(config.ALL_PARAMS_CSV, index_col=0)
        # Format some fields so they can be comparable with the ones from the csv file
        global_params["cartoon_dataset_selected_movies"] = sorted(
            [movie.value for movie in global_params["cartoon_dataset_selected_movies"]]
        )
        global_params = {
            key: (
                "None"
                if val == None
                else (str(val) if type(val) == tuple or type(val) == list else val)
            )
            for (key, val) in global_params.items()
        }
        # Check if the model was already created
        checking_df = df_all_params[global_params.keys()] == global_params.values()
        matching_values = checking_df[
            checking_df.apply(lambda x: reduce(logical_and, x), axis=1)
        ].index.values
        if len(matching_values) > 0:
            self.__import_logs(df_all_params.iloc[matching_values[0]])
        else:
            self.__create_logs(global_params)
        # We can now create the log file
        log_path = os.path.join(
            self.folder_path,
            "logs_{}.log".format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")),
        )
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    def __import_logs(self, params):
        # Load the parameters
        self.params = params
        self.folder_path = os.path.join(config.LOGS_FOLDER, self.params["run_id"])
        self.weights_path = os.path.join(config.WEIGHTS_FOLDER, self.params["run_id"])
        # Load the nb of epochs trained
        max_epoch_training = self.params["epochs_trained_nb"]
        max_epoch_pretraining = self.params["epochs_pretrained_nb"]
        for file_name in os.listdir(self.weights_path):
            if file_name[:10] == "pretrained":
                epoch_nb = int(file_name.split(".")[0].split("_")[-1]) - 1
                max_epoch_pretraining = max(max_epoch_pretraining, epoch_nb)
            if file_name[:7] == "trained":
                epoch_nb = int(file_name.split(".")[0].split("_")[-1]) - 1
                max_epoch_training = max(max_epoch_training, epoch_nb)
        self.params["epochs_pretrained_nb"] = max_epoch_pretraining
        self.params["epochs_trained_nb"] = max_epoch_training

    def __create_logs(self, global_params):
        """To create the logs if they aren't"""
        # Create the parameters
        self.params = pd.Series(global_params)
        # Create run id
        self.params["run_id"] = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # Create nb of epochs trained
        self.params["epochs_pretrained_nb"] = 0
        self.params["epochs_trained_nb"] = 0
        # Save new global params
        self.__save_params()
        # Create folder for this run
        self.folder_path = os.path.join(config.LOGS_FOLDER, self.params["run_id"])
        self.weights_path = os.path.join(config.WEIGHTS_FOLDER, self.params["run_id"])
        os.mkdir(self.folder_path)
        os.mkdir(self.weights_path)

    #######################################
    ### About (pre)training and testing ###
    #######################################

    def __init_datasets(self, train: bool, with_cartoons: bool) -> None:
        """To init the picture and cartoons datasets"""
        # Pictures dataset
        self.pictures_dataset = init_pictures_dataset(
            self.pictures_dataset_parameters,
            nb_images=self.nb_images_pictures,
            train=train,
        )
        if with_cartoons:
            # Cartoon dataset
            self.cartoons_dataset = init_cartoon_dataset(
                self.cartoons_dataset_parameters,
                nb_images=self.nb_images_cartoons,
                train=train,
            )
            # Then we must limit the nb of images
            nb_images = min(len(self.pictures_dataset), len(self.cartoons_dataset))
            self.pictures_dataset.set_nb_images(nb_images)
            self.cartoons_dataset.set_nb_images(nb_images)

    def __init_train_data_loaders(self, pretraining: bool) -> None:
        """To init the train data loaders"""

        if not pretraining:
            self.__init_datasets(train=True, with_cartoons=True)
            self.train_cartoons_loader = DataLoader(
                dataset=self.cartoons_dataset,
                batch_size=self.training_params.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=config.NUM_WORKERS,
            )
        else:
            self.__init_datasets(train=True, with_cartoons=False)

        self.train_pictures_loader = DataLoader(
            dataset=self.pictures_dataset,
            batch_size=self.pretraining_params.batch_size
            if pretraining
            else self.training_params.batch_size,
            shuffle=True,
            # drop last incomplete batch
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

    def __get_model_to_load(self) -> Optional[ModelPathsParameters]:
        """To load the (pre)trained model"""

        def load_model(pretrain):
            gen_name, disc_name = "{}trained_gen_{}.pkl".format(
                "pre" if pretrain else "", self.params["epochs_trained_nb"] + 1
            ), "{}trained_disc_{}.pkl".format(
                "pre" if pretrain else "", self.params["epochs_trained_nb"] + 1
            )
            if gen_name in os.listdir(self.weights_path):
                return {
                    "generator_path": os.path.join(self.weights_path, gen_name),
                    "discriminator_path": os.path.join(self.weights_path, disc_name),
                }
            else:
                gen_name, disc_name = "{}trained_gen_{}.pkl".format(
                    "pre" if pretrain else "", self.params["epochs_trained_nb"]
                ), "{}trained_disc_{}.pkl".format(
                    "pre" if pretrain else "", self.params["epochs_trained_nb"]
                )
                if gen_name in os.listdir(self.weights_path):
                    return {
                        "generator_path": os.path.join(self.weights_path, gen_name),
                        "discriminator_path": os.path.join(
                            self.weights_path, disc_name
                        ),
                    }
                else:
                    return None

        # First we try to load a saved model
        model = load_model(pretrain=False)
        if model == None:
            # If that doesn't work, we try to load a pretrained model
            model = load_model(pretrain=True)
            if model == None:
                # If that doesn't work, we put nb trained and pretrained to 0...
                self.params["epochs_pretrained_nb"] = 0
                self.params["epochs_trained_nb"] = 0
                self.__save_params()
                # ...and we try to load the possible input model
                if self.init_models_paths.generator_path != None:
                    model = asdict(self.init_models_paths)
        if model == None:
            logging.info("No model found")
        else:
            logging.info("Model found: ", model)
        return model

    #############
    ### Other ###
    #############

    def __save_params(self):
        """To save the parameters in the csv file"""
        df_all_params = pd.read_csv(config.ALL_PARAMS_CSV, index_col=0)
        df_extract = df_all_params[df_all_params["run_id"] == self.params["run_id"]]
        if len(df_extract) > 0:
            df_extract = self.params
        else:
            df_all_params = df_all_params.append(self.params, ignore_index=True)
        df_all_params.to_csv(config.ALL_PARAMS_CSV)

    def __format_dataclass(dataclass: dataclass, prefix: str) -> Dict[str, Any]:
        """To add a prefix on all the fields of a dictionary"""
        return {"{}_{}".format(prefix, k): v for (k, v) in asdict(dataclass).items()}


def init_device() -> str:
    """To find the GPU if it exists"""
    cuda = torch.cuda.is_available()
    if cuda:
        logging.info("Nvidia card available, running on GPU")
        logging.info(torch.cuda.get_device_name(0))
    else:
        logging.info("Nvidia card unavailable, running on CPU")
    return "cuda" if cuda else "cpu"


def init_trainer():
    pass
