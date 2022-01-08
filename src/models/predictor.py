from abc import ABC
from typing import List, Any
import numpy as np
from nptyping import NDArray
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from src.models.generators import UNet, FixedGenerator, ModularGenerator
from src.pipelines.pipeline import Architecture


class Predictor(ABC):
    """Predictor"""

    def __init__(self, architecture: Architecture, architecture_params) -> None:
        """Initialize predictor class"""
        self.architecture = architecture
        self.architecture_parameters = architecture_params
        self.model = self.__load_model()

    def __load_model(self):
        """Loads model"""

        if self.architecture == Architecture.MODULAR:
            return ModularGenerator(
                (self.architecture_parameters.nb_channels_picture,),
                (self.architecture_parameters.nb_channels_cartoon,),
                (self.architecture_parameters.nb_channels_1st_hidden_layer_gen,),
                (self.architecture_parameters.nb_resnet_blocks,),
            )

        if self.architecture == Architecture.FIXED:
            return FixedGenerator()

        if self.architecture == Architecture.UNet:
            return UNet()

        raise ImportError("Model architecture doens't exist.")

    def cartoonize(
        self, pictures: List[NDArray[(3, Any, Any)]]
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        cartoons = []
        with torch.no_grad():
            self.generator.eval()
            for picture in tqdm(pictures):
                picture = picture.to(self.device)
                cartoons.extend(self.generator(picture))

        return cartoons

    def cartoonize_dataset(
        self, data_loader: DataLoader, nb_images: int = -1
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """Cartoonize pictures"""
        cartoons = []
        with torch.no_grad():
            self.generator.eval()
            for pictures_batch in tqdm(data_loader):
                pictures_batch = pictures_batch.to(self.device)
                cartoons.extend(self.generator(pictures_batch))
                if nb_images >= 0 and len(cartoons) >= nb_images:
                    break

        return cartoons
