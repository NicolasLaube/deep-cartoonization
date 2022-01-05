import os
from dataclasses import asdict, dataclass
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from nptyping import NDArray
from functools import reduce
from numpy import logical_and
from typing import Any, Union, List
import torch
from torch.utils.data import DataLoader
from src import config
from src.models.parameters import CartoonGanBaseParameters, CartoonGanModelParameters
from src.extraction import *
from src.dataset.parameters import CartoonDatasetParameters, PicturesDatasetParameters
from src.dataset.dataset_cartoon import init_cartoon_dataset
from src.dataset.dataset_pictures import init_pictures_dataset
from src.models.cartoon_gan import CartoonGan


@dataclass
class ModelPathsParameters:
    generator_path: str
    discriminator_path: str


class CartoonGanProcessor:
    def __init__(
        self,
        *,
        cartoons_dataset_parameters: CartoonDatasetParameters,
        pictures_dataset_parameters: PicturesDatasetParameters,
        training_params: CartoonGanBaseParameters,
        pretraining_params: CartoonGanBaseParameters = None,
        cartoon_gan_model_parameters: CartoonGanModelParameters = asdict(
            CartoonGanModelParameters()
        ),
        init_models_paths: Union[ModelPathsParameters, None] = None,
        extraction: bool = False,
    ):
        # Extract all the information about the images if necessary
        if extraction:
            self.__extract_images()
        # Init parameters
        self.cartoon_gan_model_params = cartoon_gan_model_parameters
        self.cartoons_dataset_parameters = cartoons_dataset_parameters
        self.pictures_dataset_parameters = pictures_dataset_parameters
        self.training_params = training_params
        self.pretraining_params = (
            pretraining_params if pretraining_params != None else training_params
        )
        self.init_models_paths = init_models_paths
        # Init logs
        self.__init_logs()
        # Init device (CPU/GPU)
        self.__init_device()
        # Init gan model
        self.__init_gan_model()
        # Init datasets
        self.__init_datasets()

    def pretrain(self, nb_epochs: int) -> None:
        """To pretrain the model"""
        self.__init_train_data_loaders(pretraining=True)
        pretraining_parameters = {**self.pretraining_params}
        pretraining_parameters["epochs"] = nb_epochs
        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()
        if models_to_load_paths != None:
            self.cartoon_gan.load_model(**models_to_load_paths)
        loss = self.cartoon_gan.pretrain(
            picture_loader=self.train_pictures_loader,
            parameters=pretraining_parameters,
            saved_weights_path=self.weights_path,
            first_epoch_index=self.params["epochs_pretrained_nb"],
        )
        self.params["epochs_pretrained_nb"] += nb_epochs
        self.params["pretrain_reconstruction_loss"] = loss
        self.__save_params()

    def train(self, nb_epochs: int) -> None:
        """To train the model"""
        self.__init_train_data_loaders(pretraining=False)
        training_parameters = {**self.training_params}
        training_parameters["epochs"] = nb_epochs
        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()
        if models_to_load_paths != None:
            self.cartoon_gan.load_model(**models_to_load_paths)
        losses = self.cartoon_gan.train(
            picture_loader=self.train_pictures_loader,
            dataset_cartoon=self.train_cartoons_loader,
            parameters=training_parameters,
            saved_weights_path=self.weights_path,
            first_epoch_index=self.params["epochs_trained_nb"],
        )
        self.params["epochs_trained_nb"] += nb_epochs
        self.params["train_discriminator_loss"] = losses["discriminator_loss"]
        self.params["train_generator_loss"] = losses["generator_loss"]
        self.params["train_conditional_loss"] = losses["conditional_loss"]
        self.__save_params()

    def cartoonize_images(
        self, nb_images: int = -1
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """To show some cartoonized images"""
        self.__init_test_data_loaders()
        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()
        if models_to_load_paths != None:
            self.cartoon_gan.load_model(**models_to_load_paths)
        cartoons = self.cartoon_gan.cartoonize_dataset(
            pictures_loader=self.test_pictures_loader, nb_images=nb_images
        )
        return cartoons

    ###################
    ### About inits ###
    ###################

    def __extract_images(self) -> None:
        """To build dataframes with all frames and all pictures, and divide it into a train and a test dataset"""
        create_all_frames_csv()
        create_all_pictures_csv()
        create_train_test_frames()
        create_train_test_pictures()

    def __init_logs(self):
        """To init the logs folder or to load the training model"""
        # Import all fixed params & parameters of all runs
        global_params = {
            **self.__format_dict(self.cartoon_gan_model_params, "cartoon_gan_model"),
            **self.__format_dict(self.cartoons_dataset_parameters, "cartoon_dataset"),
            **self.__format_dict(self.pictures_dataset_parameters, "pictures_dataset"),
            **self.__format_dict(self.pretraining_params, "pretraining"),
            **self.__format_dict(self.training_params, "training"),
            **self.__format_dict(self.init_models_paths, "init"),
        }
        df_all_params = pd.read_csv(config.ALL_PARAMS_CSV, index_col=0)
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
            config.LOGS_FOLDER,
            "logs_{}.log".format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")),
        )
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )

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

    def __init_device(self) -> None:
        """To find the GPU if it exists"""
        cuda = torch.cuda.is_available()
        if cuda:
            print("Nvidia card available, running on GPU")
            print(torch.cuda.get_device_name(0))
        else:
            print("Nvidia card unavailable, running on CPU")
        self.device = "cuda" if cuda else "cpu"

    def __init_gan_model(self) -> None:
        """To init the model we will train"""
        self.cartoon_gan = CartoonGan(self.cartoon_gan_model_params)

    def __init_datasets(self) -> None:
        """To init the picture and cartoons datasets"""
        self.train_cartoons_dataset = init_cartoon_dataset(
            self.cartoons_dataset_parameters, train=True
        )
        self.train_pictures_dataset = init_pictures_dataset(
            self.pictures_dataset_parameters, train=True
        )
        self.test_cartoons_dataset = init_cartoon_dataset(
            self.cartoons_dataset_parameters, train=False
        )
        self.test_pictures_dataset = init_pictures_dataset(
            self.pictures_dataset_parameters, train=False
        )

    #######################################
    ### About (pre)training and testing ###
    #######################################

    def __init_train_data_loaders(self, pretraining: bool) -> None:
        """To init the train data loaders"""

        if not pretraining:
            self.train_cartoons_loader = DataLoader(
                dataset=self.train_cartoons_dataset,
                batch_size=self.training_params["batch_size"],
                shuffle=True,
                drop_last=True,
                num_workers=config.NUM_WORKERS,
            )

        self.train_pictures_loader = DataLoader(
            dataset=self.train_pictures_dataset,
            batch_size=self.pretraining_params["batch_size"]
            if pretraining
            else self.training_params["batch_size"],
            shuffle=True,
            # drop last incomplete batch
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

    def __init_test_data_loaders(self) -> None:
        """To init the test data loaders"""

        self.test_pictures_loader = DataLoader(
            dataset=self.train_pictures_dataset,
            batch_size=self.training_params["batch_size"],
            shuffle=True,
            # drop last incomplete batch
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

    def __get_model_to_load(self) -> Union[ModelPathsParameters, None]:
        """To load the (pre)trained model"""

        def load_model(pretrain):
            gen_name, disc_name = "{}trained_gen_{}.pkl".format(
                "pre" if pretrain else "", self.params["epochs_trained_nb"]
            ), "{}trained_disc_{}.pkl".format(
                "pre" if pretrain else "", self.params["epochs_trained_nb"]
            )
            if gen_name in os.listdir(self.weights_path):
                return {
                    "generator_path": os.path.join(self.weights_path, gen_name),
                    "discriminator_path": os.path.join(self.weights_path, disc_name),
                }
            else:
                gen_name, disc_name = "{}trained_gen_{}.pkl".format(
                    "pre" if pretrain else "", self.params["epochs_trained_nb"] - 1
                ), "{}trained_disc_{}.pkl".format(
                    "pre" if pretrain else "", self.params["epochs_trained_nb"] - 1
                )
                return {
                    "generator_path": os.path.join(self.weights_path, gen_name),
                    "discriminator_path": os.path.join(self.weights_path, disc_name),
                }

        if self.params["epochs_trained_nb"] > 0:
            # In that case we already began to train the model
            return load_model(pretrain=False)
        elif self.params["epochs_pretrained_nb"] > 0:
            # In that case we already began to pretrain the model
            return load_model(pretrain=True)
        elif self.params["init_models_paths"] != None:
            # In that case we can load the init model
            return self.init_models_paths
        else:
            # If no above option worked, we can't load any model
            return None

    #############
    ### Other ###
    #############

    def __save_params(self):
        """To save the parameters in the csv file"""
        df_all_params = pd.read_csv(config.ALL_PARAMS_CSV, index_col=0)
        df_all_params = df_all_params.append(self.params, ignore_index=True)
        df_all_params.to_csv(config.ALL_PARAMS_CSV)

    def __format_dict(dictionary: dict[str, Any], prefix: str) -> dict[str, Any]:
        """To add a prefix on all the fields of a dictionary"""
        return {"{}_{}".format(prefix, k): v for (k, v) in dictionary.items()}
