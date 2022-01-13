import enum
import os
import sys
import json
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
from torch.utils.tensorboard import SummaryWriter

from src import config
import src.dataset as dataset
import src.models as models
import src.preprocessing as preprocessing


@dataclass
class ModelPathsParameters:
    generator_path: str
    discriminator_path: str


class Pipeline:
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
        self.device = Pipeline.__init_device()
        """self.predictor = models.Predictor(
            architecture=architecture, architecture_params=architecture_params
        )"""
        self.architecture = architecture
        self.architecture_params = architecture_params
        self.cartoons_dataset_parameters = cartoons_dataset_parameters
        self.pictures_dataset_parameters = pictures_dataset_parameters
        self.training_params = training_parameters
        self.trainer = self.__init_trainer()

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
        self.paramas = self.__init_logs()

    def pretrain(self, nb_epochs: int) -> None:
        """To pretrain the model"""
        self.__init_train_data_loaders(pretraining=True)
        pretraining_parameters = replace(self.pretraining_params)
        pretraining_parameters.epochs = nb_epochs
        # If there is a model to load we must load it before

        models_to_load_paths = self.__get_model_to_load()
        if models_to_load_paths != None:
            self.trainer.load_model(**models_to_load_paths)

        self.trainer.pretrain(
            pictures_loader=self.train_pictures_loader,
            cartoon_loader=self.train_cartoons_loader,
            batch_callback=self.__get_callback(pretrain=True),
            epoch_start=self.params["epochs_pretrained_nb"] + 1,
            pretrain_params=pretraining_parameters,
        )

    def train(self, nb_epochs: int) -> None:
        """To train the model"""
        self.__init_train_data_loaders(pretraining=False)
        training_parameters = replace(self.training_params)
        training_parameters.epochs = nb_epochs

        # If there is a model to load we must load it before
        models_to_load_paths = self.__get_model_to_load()

        if models_to_load_paths != None:
            self.trainer.load_model(**models_to_load_paths)

        self.trainer.train(
            pictures_loader=self.train_pictures_loader,
            cartoons_loader=self.train_cartoons_loader,
            batch_callback=self.__get_callback(pretrain=False),
            epoch_start=self.params["epochs_pretrained_nb"] + 1,
            train_params=training_parameters,
        )

    def cartoonize_images(
        self, nb_images: int = -1
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """To show some cartoonized images"""

        models_to_load_paths = self.__get_model_to_load()

        if models_to_load_paths != None:
            self.predictor.load_model(**models_to_load_paths)

        test_pictures_loader = DataLoader(
            dataset=self.pictures_dataset,
            batch_size=self.training_params.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.NUM_WORKERS,
        )

        return self.predictor.cartoonize_dataset(
            pictures_loader=test_pictures_loader, nb_images=nb_images
        )

    ###################
    ### About inits ###
    ###################

    def __init_logs(self):
        """To init the logs folder or to load the training model"""
        # Import all fixed params & parameters of all runs
        global_params = {
            "cartoon_gan_architecture": self.architecture.value,
            **Pipeline.__format_dataclass(
                self.architecture_params, "cartoon_gan_model"
            ),
            **Pipeline.__format_dataclass(
                self.cartoons_dataset_parameters, "cartoon_dataset"
            ),
            **Pipeline.__format_dataclass(
                self.pictures_dataset_parameters, "pictures_dataset"
            ),
            **Pipeline.__format_dataclass(self.pretraining_params, "pretraining"),
            **Pipeline.__format_dataclass(self.training_params, "training"),
            **Pipeline.__format_dataclass(self.init_models_paths, "init"),
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
                else (
                    str(val)
                    if type(val) == tuple or type(val) == list
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
            params = self.__import_logs(df_all_params.iloc[matching_values[0]])
        else:
            params = self.__create_logs(global_params)
        # We can now create the log file
        log_path = os.path.join(
            self.logs_path,
            "logs_{}.log".format(Pipeline.__get_time_id()),
        )
        for handler in logging.getLogger().handlers[:]:
            print(handler)
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        return params

    def __import_logs(self, params):
        self.params = params
        # Load the parameters
        self.folder_path = os.path.join(config.LOGS_FOLDER, params["run_id"])
        self.logs_path = os.path.join(self.folder_path, "logs")
        self.losses_path = os.path.join(self.folder_path, "losses")
        self.weights_path = os.path.join(config.WEIGHTS_FOLDER, params["run_id"])
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
        params["epochs_pretrained_nb"] = max_epoch_pretraining
        params["epochs_trained_nb"] = max_epoch_training
        return params

    def __create_logs(self, global_params):
        """To create the logs if they aren't"""
        # Create the parameters
        self.params = pd.Series(global_params)
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

    def __init_cartoons_dataset(self, train: bool = True):
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
            train,
        )

    def __init_pictures_dataset(self, train: bool = True):
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
            train,
        )

    def __init_datasets(self, train: bool, with_cartoons: bool) -> None:
        """To init the picture and cartoons datasets"""
        # Pictures dataset
        self.pictures_dataset = self.__init_pictures_dataset(train=train)

        if with_cartoons:
            self.cartoons_dataset = self.__init_cartoons_dataset(train=train)

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
        if model is None:
            # If that doesn't work, we try to load a pretrained model
            model = load_model(pretrain=True)
            if model is None:
                # If that doesn't work, we put nb trained and pretrained to 0...
                self.params["epochs_pretrained_nb"] = 0
                self.params["epochs_trained_nb"] = 0
                self.__save_params()
                # ...and we try to load the possible input model
                if self.init_models_paths.generator_path != None:
                    model = asdict(self.init_models_paths)
        if model is None:
            logging.warning("No model found")
        else:
            logging.warning("Model found: ", model)
        return model

    def __get_callback(self, pretrain: bool = False):
        """Return callback for pretrain/train"""
        train_state = "pretrain" if pretrain else "train"
        writer = SummaryWriter(
            f"logs/tensorboard/{self.params['run_id']}_{train_state}"
        )

        def callback(epoch: int, losses: Dict[str, float]):
            # Save losses in tensorboard
            for key, loss in losses.items():
                writer.add_scalar(key, loss, global_step=epoch)
            # Save losses in file
            str_losses = {k: loss.item() for k, loss in losses.items()}
            with open(
                os.path.join(self.losses_path, f"{train_state}_{epoch}.txt"), "a"
            ) as file:
                file.write(f"{Pipeline.__get_time_id()}; {json.dumps(str_losses)}\n")
            # Save global loss in global params
            for key, loss in str_losses.items():
                self.params[f"{train_state}_{key}"] = loss
            self.__save_params()

        return callback

    #############
    ### Other ###
    #############

    def __init_trainer(self):
        if self.architecture == models.Architecture.GANFixed:
            assert isinstance(
                self.architecture_params, models.ArchitectureParamsNULL
            ), "Fixed architecture requires null architecture parameters"
            return models.FixedCartoonGANTrainer(device=self.device)
        if self.architecture == models.Architecture.GANModular:
            assert isinstance(
                self.architecture_params, models.ArchitectureParamsModular
            ), "Modular architecture requires modular params"
            return models.FixedCartoonGANTrainer(self.architecture_params, self.device)

    def __save_params(self):
        """To save the parameters in the csv file"""
        df_all_params = pd.read_csv(config.ALL_PARAMS_CSV, index_col=0)
        df_extract = df_all_params[df_all_params["run_id"] == self.params["run_id"]]
        if len(df_extract) > 0:
            df_all_params.loc[df_extract.index[0], :] = self.params[:]
        else:
            df_all_params = df_all_params.append(self.params, ignore_index=True)
        df_all_params.to_csv(config.ALL_PARAMS_CSV)

    @staticmethod
    def __get_time_id() -> str:
        """Return an id based on the current time"""
        return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    @staticmethod
    def __format_dataclass(dataclass: dataclass, prefix: str) -> Dict[str, Any]:
        """To add a prefix on all the fields of a dictionary"""
        return {"{}_{}".format(prefix, k): v for (k, v) in asdict(dataclass).items()}

    @staticmethod
    def __init_device() -> str:
        """To find the GPU if it exists"""
        cuda = torch.cuda.is_available()
        if cuda:
            logging.info("Nvidia card available, running on GPU")
            logging.info(torch.cuda.get_device_name(0))
        else:
            logging.info("Nvidia card unavailable, running on CPU")
        return "cuda" if cuda else "cpu"


if __name__ == "__main__":
    pipeline = Pipeline(
        architecture=models.Architecture.GANFixed,
        architecture_params=models.ArchitectureParamsNULL(),
        cartoons_dataset_parameters=dataset.CartoonsDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
        ),
        pictures_dataset_parameters=dataset.PicturesDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
            ratio_filter_mode=preprocessing.RatioFilterMode.NO_FILTER,
        ),
        init_models_paths=None,
        training_parameters=models.TrainerParams(),
        pretraining_parameters=models.TrainerParams(),
    )

    pipeline.train(10)
