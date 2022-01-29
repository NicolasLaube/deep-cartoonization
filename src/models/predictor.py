"""Predictor"""
from abc import ABC
from typing import Any, List

import numpy as np
import torch
from nptyping import NDArray
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.models.generators import FixedGenerator, ModularGenerator, UNet
from src.models.utils.parameters import Architecture, ArchitectureParams


class Predictor(ABC):
    """Predictor"""

    def __init__(
        self,
        architecture: Architecture,
        architecture_params: ArchitectureParams,
        device: str,
    ) -> None:
        """Initialize predictor class"""
        self.device = device
        self.architecture = architecture
        self.architecture_parameters = architecture_params
        self.model = self.__load_model()

    def load_weights(self, gen_path: str, disc_path: str):
        """Load weights"""
        self.model.load_state_dict(gen_path)
        print(disc_path)

    def __load_model(self):
        """Loads model"""

        if self.architecture == Architecture.GANModular:
            return ModularGenerator(
                (self.architecture_parameters.nb_channels_picture,),
                (self.architecture_parameters.nb_channels_cartoon,),
                (self.architecture_parameters.nb_channels_1st_hidden_layer_gen,),
                (self.architecture_parameters.nb_resnet_blocks,),
            )

        if self.architecture == Architecture.GANFixed:
            return FixedGenerator()

        if self.architecture == Architecture.GANUNet:
            return UNet(n_channels=10, n_classes=10)  # To change

        raise ImportError("Model architecture doens't exist.")

    def cartoonize(
        self, pictures: List[NDArray[(3, Any, Any)]]
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """Cartoonize images"""
        cartoons = []
        with torch.no_grad():
            self.model.eval()
            for picture in tqdm(pictures):
                picture = picture.to(self.device)
                cartoons.extend(self.model(picture))

        return cartoons

    def cartoonize_dataset(
        self, pictures_loader: DataLoader, nb_images: int = -1
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """Cartoonize pictures dataset"""
        cartoons = []
        with torch.no_grad():
            self.model.eval()
            for pictures_batch in tqdm(pictures_loader):
                pictures_batch = pictures_batch.to(self.device)
                cartoons.extend(self.model(pictures_batch))
                if len(cartoons) >= nb_images >= 0:
                    break

        return cartoons
