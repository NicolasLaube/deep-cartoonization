"""Pipelines launcher"""
# pylint: disable=R0902,E1102
import csv
import enum
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import reduce
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from nptyping import NDArray
from numpy import logical_and
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import config, dataset, models, preprocessing
from src.base.base_trainer import Trainer
from src.models.utils.parameters import ArchitectureParams, TrainerParams
from src.pipelines.utils import init_device
from src.preprocessing.transformations.transformations import Transform


@dataclass
class ModelPathsParameters:
    """Model's parameters"""

    gen_path: Optional[str]
    disc_path: Optional[str]


class Pipeline:
    """Training and inference pipeline."""

    def __init__(
        self,
        *,
        architecture: models.Architecture = models.Architecture.GANFixed,
        architecture_params: models.ArchitectureParams = models.ArchitectureParamsNULL(),
        cartoons_dataset_parameters: dataset.CartoonsDatasetParameters,
        pictures_dataset_parameters: dataset.PicturesDatasetParameters,
        training_parameters: models.TrainerParams,
        pretraining_parameters: Optional[models.TrainerParams] = None,
        init_models_paths: Optional[ModelPathsParameters] = None,
    ):
        # Initialize parameters
        self.device = init_device()
        self.architecture = architecture
        self.architecture_params = architecture_params
        self.cartoons_dataset_parameters = cartoons_dataset_parameters
        self.pictures_dataset_parameters = pictures_dataset_parameters
        self.training_params = training_parameters
        self.logs_path: str = None  # type: ignore
        self.losses_path: str = None  # type: ignore
        self.params: Dict[str, Any] = None  # type: ignore
        self.trainer = self.__init_trainer()

        self.pretraining_params = (
            pretraining_parameters
            if pretraining_parameters is not None
            else training_parameters
        )

        self.init_models_paths = (
            init_models_paths
            if init_models_paths is not None
            else ModelPathsParameters(gen_path=None, disc_path=None)
        )

        # Initialize logs
        self.__init_logs()
        self.folder_path = os.path.join(config.LOGS_FOLDER, self.params["run_id"])
        self.weights_path = os.path.join(config.WEIGHTS_FOLDER, self.params["run_id"])

    def pretrain(self, nb_epochs: int) -> None:
        """To pretrain the model"""

        pretraining_parameters = replace(self.pretraining_params)

        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()

        if models_to_load_paths is not None:
            self.trainer.load_model(**models_to_load_paths)

        train_pictures_dataset = self.__init_pictures_dataset(train=True)
        validation_pictures_dataset = self.__init_pictures_dataset(train=False)

        train_pictures_loader = DataLoader(
            dataset=train_pictures_dataset,
            batch_size=self.pretraining_params.batch_size,
            shuffle=True,
            # drop last incomplete batch
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )
        validation_pictures_loader = DataLoader(
            dataset=validation_pictures_dataset,
            batch_size=self.pretraining_params.batch_size,
            shuffle=True,
            # drop last incomplete batch
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

        self.trainer.pretrain(
            pictures_loader_train=train_pictures_loader,
            pictures_loader_validation=validation_pictures_loader,
            batch_callback=self.__get_batch_callback(pretrain=True),
            validation_callback=self.__get_validation_callback(pretrain=True),
            epoch_start=self.params["epochs_pretrained_nb"] + 1,
            weights_folder=self.weights_path,
            pretrain_params=pretraining_parameters,
            epochs=nb_epochs,
        )

    def train(self, nb_epochs: int) -> None:
        """To train the model"""

        training_parameters = replace(self.training_params)

        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()

        if models_to_load_paths is not None:
            self.trainer.load_model(**models_to_load_paths)

        train_pictures_dataset = self.__init_pictures_dataset(train=True)
        validation_pictures_dataset = self.__init_pictures_dataset(train=False)
        train_cartoons_dataset = self.__init_cartoons_dataset(train=True)
        validation_cartoons_dataset = self.__init_cartoons_dataset(train=False)

        train_cartoons_loader = DataLoader(
            dataset=train_cartoons_dataset,
            batch_size=self.training_params.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

        validation_cartoons_loader = DataLoader(
            dataset=validation_cartoons_dataset,
            batch_size=self.training_params.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

        train_pictures_loader = DataLoader(
            dataset=train_pictures_dataset,
            batch_size=self.training_params.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

        validation_pictures_loader = DataLoader(
            dataset=validation_pictures_dataset,
            batch_size=self.training_params.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

        nb_images_train = min(len(train_pictures_dataset), len(train_cartoons_dataset))
        nb_images_validation = min(
            len(validation_pictures_dataset), len(validation_cartoons_dataset)
        )
        # datasets should have the same size
        train_pictures_dataset.set_nb_images(nb_images_train)
        train_cartoons_dataset.set_nb_images(nb_images_train)
        validation_pictures_dataset.set_nb_images(nb_images_validation)
        validation_cartoons_dataset.set_nb_images(nb_images_validation)

        self.trainer.train(
            pictures_loader_train=train_pictures_loader,
            pictures_loader_validation=validation_pictures_loader,
            cartoons_loader_train=train_cartoons_loader,
            cartoons_loader_validation=validation_cartoons_loader,
            batch_callback=self.__get_batch_callback(pretrain=False),
            validation_callback=self.__get_validation_callback(pretrain=False),
            epoch_start=self.params["epochs_trained_nb"] + 1,
            weights_folder=self.weights_path,
            train_params=training_parameters,
            epochs=nb_epochs,
        )

    def get_cartoonized_images(
        self, nb_images: int = -1
    ) -> List[Dict[str, NDArray[(3, Any, Any), np.int32]]]:
        """To show some cartoonized images"""

        transformer = Transform(
            architecture=self.architecture,
            new_size=self.cartoons_dataset_parameters.new_size,
            crop_mode=self.cartoons_dataset_parameters.crop_mode,
            device=self.device,
        )

        predictor = models.Predictor(
            architecture=self.architecture,
            architecture_params=self.architecture_params,
            transformer=transformer,
            device=self.device,
        )

        models_to_load_paths = self.__get_model_to_load()

        if models_to_load_paths is not None:
            predictor.load_weights(gen_path=models_to_load_paths["gen_path"])

        pictures_dataset = self.__init_pictures_dataset(train=False)

        test_pictures_loader = DataLoader(
            dataset=pictures_dataset,
            batch_size=self.training_params.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.NUM_WORKERS,
        )

        return predictor.cartoonize_dataset(
            pictures_loader=test_pictures_loader, nb_images=nb_images
        )

    def cartoonize_images(
        self, pictures: List[NDArray[(3, Any, Any), np.int32]]
    ) -> List[Dict[str, NDArray[(3, Any, Any), np.int32]]]:
        """To show some cartoonized images"""

        transformer = Transform(
            architecture=self.architecture,
            new_size=self.cartoons_dataset_parameters.new_size,
            crop_mode=self.cartoons_dataset_parameters.crop_mode,
            device=self.device,
        )

        predictor = models.Predictor(
            architecture=self.architecture,
            architecture_params=self.architecture_params,
            transformer=transformer,
            device=self.device,
        )

        models_to_load_paths = self.__get_model_to_load()

        if models_to_load_paths is not None:
            predictor.load_weights(gen_path=models_to_load_paths["gen_path"])

        return predictor.cartoonize(pictures=pictures)

    def cartoonize_images_from_path(
        self, paths: List[str]
    ) -> List[Dict[str, NDArray[(3, Any, Any), np.int32]]]:
        """To show some cartoonized images from their path"""

        pictures = [Image.open(path) for path in paths]

        return self.cartoonize_images(pictures)

    ###########################
    ### About initilization ###
    ###########################

    def __init_cartoons_dataset(self, train: bool) -> dataset.CartoonDataset:
        """Initialize dataset"""
        data_filter = preprocessing.Filter(
            new_size=self.cartoons_dataset_parameters.new_size,
            selected_movies=self.cartoons_dataset_parameters.selected_movies,
            ratio_filter_mode=self.cartoons_dataset_parameters.ratio_filter_mode,
        )
        transform = preprocessing.Transform(
            architecture=self.architecture,
            new_size=self.cartoons_dataset_parameters.new_size,
            crop_mode=self.cartoons_dataset_parameters.crop_mode,
        )
        return dataset.CartoonDataset(
            data_filter.cartoon_filter,
            transform.cartoon_transform,
            self.cartoons_dataset_parameters.nb_images,
            "train" if train else "validation",
        )

    def __init_pictures_dataset(self, train: bool) -> dataset.PicturesDataset:
        """Initialize pictures dataset"""

        data_filter = preprocessing.Filter(
            new_size=self.pictures_dataset_parameters.new_size,
            ratio_filter_mode=self.pictures_dataset_parameters.ratio_filter_mode,
        )
        transform = preprocessing.Transform(
            new_size=self.pictures_dataset_parameters.new_size,
            crop_mode=self.pictures_dataset_parameters.crop_mode,
            architecture=self.architecture,
        )

        return dataset.PicturesDataset(
            data_filter.picture_filter,
            transform.picture_transform,
            self.pictures_dataset_parameters.nb_images,
            "train" if train else "validation",
        )

    def __init_trainer(self) -> Trainer:
        """Initialize trainer"""
        if self.architecture == models.Architecture.GANFixed:
            assert isinstance(
                self.architecture_params, models.ArchitectureParamsNULL
            ), "Fixed architecture requires null architecture parameters"
            return models.FixedCartoonGANTrainer(device=self.device)
        if self.architecture == models.Architecture.GANModular:
            assert isinstance(
                self.architecture_params, models.ArchitectureParamsModular
            ), "Modular architecture requires modular params"
            return models.ModularGANTrainer(
                device=self.device, architecture_params=self.architecture_params
            )

        raise NotImplementedError("Pipeline wasn't implemented")

    def __init_logs(self) -> None:
        """To init the logs folder or to load the training model"""

        # Import all fixed params & parameters of all runs
        global_params: Dict[str, Any] = {
            "cartoon_gan_architecture": self.architecture.value,
            **self.__format_dataclass(self.architecture_params, "architecture"),
            **self.__format_dataclass(
                self.cartoons_dataset_parameters, "cartoon_dataset"
            ),
            **self.__format_dataclass(
                self.pictures_dataset_parameters, "picture_dataset"
            ),
            **self.__format_dataclass(self.pretraining_params, "pretraining"),
            **self.__format_dataclass(self.training_params, "training"),
            **self.__format_dataclass(self.init_models_paths, "init"),
        }

        df_all_params = pd.read_csv(config.ALL_PARAMS_CSV, index_col=0)
        # Format some fields so they can be comparable with the ones from the csv file
        global_params["cartoon_dataset_selected_movies"] = sorted(
            [movie.value for movie in global_params["cartoon_dataset_selected_movies"]]
        )
        global_params = {
            key: (
                "None"
                if val is None
                else (
                    str(val)
                    if isinstance(val, (tuple, list))
                    else (val.name if isinstance(val, enum.Enum) else val)
                )
            )
            for (key, val) in global_params.items()
            if key not in ["cartoon_dataset_nb_images", "pictures_dataset_nb_images"]
        }

        # Check if the model was already created
        checking_df = df_all_params[global_params.keys()] == global_params.values()
        matching_values = checking_df[
            checking_df.apply(lambda x: reduce(logical_and, x), axis=1)
        ].index.values
        if len(matching_values) > 0:
            self.__import_logs(df_all_params.iloc[matching_values[0]])
        else:
            self.__create_logs(pd.Series(global_params))
        # We can now create the log file
        log_path = os.path.join(
            self.logs_path,
            f"logs_{Pipeline.__get_time_id()}.log",
        )
        for handler in logging.getLogger().handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    def __import_logs(self, params: Dict[str, Any]) -> None:
        """Import existing logs"""

        # Load the parameters
        self.folder_path = os.path.join(config.LOGS_FOLDER, params["run_id"])
        self.logs_path = os.path.join(self.folder_path, "logs")
        self.losses_path = os.path.join(self.folder_path, "losses")
        self.weights_path = os.path.join(config.WEIGHTS_FOLDER, params["run_id"])
        # Load the nb of epochs trained
        max_epoch_training = params["epochs_trained_nb"]
        max_epoch_pretraining = params["epochs_pretrained_nb"]
        for file_name in os.listdir(self.weights_path):
            if file_name[:10] == "pretrained":
                epoch_nb = int(file_name.split(".")[0].split("_")[-1]) - 1
                max_epoch_pretraining = max(max_epoch_pretraining, epoch_nb)
            if file_name[:7] == "trained":
                epoch_nb = int(file_name.split(".")[0].split("_")[-1]) - 1
                max_epoch_training = max(max_epoch_training, epoch_nb)
        params["epochs_pretrained_nb"] = max_epoch_pretraining
        params["epochs_trained_nb"] = max_epoch_training
        self.params = params

    def __create_logs(self, params: Dict[str, Any]) -> None:
        """To create the logs if they aren't"""
        self.params = params
        # Create run id
        self.params["run_id"] = Pipeline.__get_time_id()
        # Create nb of epochs trained
        self.params["epochs_pretrained_nb"] = 0
        self.params["epochs_trained_nb"] = 0
        # Save new global params
        self.__save_params()
        # Create folder for this run
        self.folder_path = os.path.join(config.LOGS_FOLDER, self.params["run_id"])
        self.weights_path = os.path.join(config.WEIGHTS_FOLDER, self.params["run_id"])
        self.logs_path = os.path.join(self.folder_path, "logs")
        self.losses_path = os.path.join(self.folder_path, "losses")
        os.mkdir(self.folder_path)
        os.mkdir(self.weights_path)
        os.mkdir(self.logs_path)
        os.mkdir(self.losses_path)

    #######################################
    ### About (pre)training and testing ###
    #######################################

    def __get_model_to_load(self) -> Any:
        """To load the (pre)trained model"""

        def load_model(pretrain):
            prefix = "pre" if pretrain else ""
            num_epochs_str = self.params["epochs_trained_nb"] + 1

            gen_name = f"{prefix}trained_gen_{num_epochs_str}.pkl"
            disc_name = f"{prefix}trained_disc_{num_epochs_str}.pkl"

            if gen_name in os.listdir(self.weights_path):
                return {
                    "gen_path": os.path.join(self.weights_path, gen_name),
                    "disc_path": os.path.join(self.weights_path, disc_name),
                }

            num_epochs = self.params["epochs_trained_nb"]
            gen_name, disc_name = (
                f"{prefix}trained_gen_{num_epochs}.pkl",
                f"{prefix}trained_disc_{num_epochs}.pkl",
            )
            if gen_name in os.listdir(self.weights_path):
                return {
                    "gen_path": os.path.join(self.weights_path, gen_name),
                    "disc_path": os.path.join(self.weights_path, disc_name),
                }

            return None

        # First we try to load a saved model
        model = load_model(pretrain=False)
        if model is None:
            # If that doesn't work, we try to load a pretrained model
            model = load_model(pretrain=True)
            if model is None:
                # If that doesn't work, we put nb trained and pretrained to 0...
                self.params["epochs_pretrained_nb"] = 0
                self.params["epochs_trained_nb"] = 0
                self.__save_params()
                # ...and we try to load the possible input model
                if self.init_models_paths.gen_path is not None:
                    model = asdict(self.init_models_paths)
        if model is None:
            logging.info("No model found")
        else:
            logging.info("Model found.")
        return model

    def __get_batch_callback(self, pretrain: bool = False):
        """Return batch callback for pretrain/train"""
        train_state = "pretrain" if pretrain else "train"
        writer = SummaryWriter(
            f"logs/tensorboard/{self.params['run_id']}_{train_state}"
        )

        def callback(epoch: int, step: int, losses: Dict[str, Any]):
            # Save losses in tensorboard
            for key, loss in losses.items():
                writer.add_scalar(key, loss, global_step=step)
            # Save losses in file
            str_losses = {k: loss.item() for k, loss in losses.items()}
            with open(
                os.path.join(self.losses_path, f"{train_state}_{epoch}.txt"),
                "a",
                encoding="utf-8",
            ) as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([Pipeline.__get_time_id(), str_losses])

        return callback

    def __get_validation_callback(self, pretrain: bool = False):
        """Return validation callback for pretrain/train"""
        train_state = "pretrain" if pretrain else "train"

        def callback(epoch: int, losses: Dict[str, Any]):
            # Save losses in file
            str_losses = {k: loss.item() for k, loss in losses.items()}
            to_write = {"epoch": epoch, **str_losses}
            with open(
                os.path.join(self.losses_path, f"{train_state}_validation.txt"),
                "a",
                encoding="utf-8",
            ) as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([Pipeline.__get_time_id(), to_write])
            # Save global loss in global params
            for key, loss in str_losses.items():
                self.params[f"{train_state}_{key}"] = loss
            self.params[f"epochs_{train_state}ed_nb"] = epoch
            self.__save_params()
            logging.info("Results for epoch %s saved.", epoch)

        return callback

    #############
    ### Other ###
    #############

    def __save_params(self):
        """To save the parameters in the csv file"""
        df_all_params = pd.read_csv(config.ALL_PARAMS_CSV, index_col=0)
        df_extract = df_all_params[df_all_params["run_id"] == self.params["run_id"]]
        if len(df_extract) > 0:
            df_all_params.loc[df_extract.index[0], :] = self.params[:]
        # df_all_params is considered to be a File reader, so we must ensure it is a dataframe
        elif isinstance(df_all_params, pd.DataFrame):
            df_all_params = df_all_params.append(self.params, ignore_index=True)
        df_all_params.to_csv(config.ALL_PARAMS_CSV)

    @staticmethod
    def __get_time_id() -> str:
        """Return an id based on the current time"""
        return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    @staticmethod
    def __format_dataclass(
        data_class: Union[
            ArchitectureParams,
            TrainerParams,
            ModelPathsParameters,
            dataset.ImageDatasetParameters,
        ],
        prefix: str,
    ) -> Dict[str, Any]:
        """To add a prefix on all the fields of a dictionary"""
        return {f"{prefix}_{k}": v for (k, v) in asdict(data_class).items()}


if __name__ == "__main__":
    pipeline = Pipeline(
        architecture=models.Architecture.GANModular,
        architecture_params=models.ArchitectureParamsModular(),
        cartoons_dataset_parameters=dataset.CartoonsDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
            nb_images=4,
        ),
        pictures_dataset_parameters=dataset.PicturesDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
            ratio_filter_mode=preprocessing.RatioFilterMode.NO_FILTER,
            nb_images=4,
        ),
        init_models_paths=None,
        training_parameters=models.TrainerParams(batch_size=2),
        pretraining_parameters=models.TrainerParams(batch_size=2),
    )

    pipeline.pretrain(2)
