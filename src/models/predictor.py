"""Predictor"""
from abc import ABC
from typing import Any, Dict, List

import numpy as np
import torch
from nptyping import NDArray
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.models.generators import FixedGenerator, ModularGenerator, UNet
from src.models.generators.generator_anim import AnimeGenerator
from src.models.utils.parameters import Architecture, ArchitectureParams


class Predictor(ABC):
    """Predictor"""

    def __init__(
        self,
        architecture: Architecture,
        architecture_params: ArchitectureParams,
        transformer: Any,
        device: str,
    ) -> None:
        """Initialize predictor class"""
        self.device = device
        self.architecture = architecture
        self.architecture_parameters = architecture_params
        self.model = self.__load_model()
        self.transformer = transformer

    def load_weights(self, gen_path: str) -> None:
        """Load the model from weights"""
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(gen_path))
        else:
            self.model.load_state_dict(
                torch.load(gen_path, map_location=lambda storage, loc: storage)
            )

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
        if self.architecture == Architecture.GANAnime:
            return AnimeGenerator()

        raise ImportError("Model architecture doens't exist.")

    def cartoonize(
        self, pictures: List[NDArray[(3, Any, Any)]]
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """Cartoonize images"""
        cartoons = []
        with torch.no_grad():
            self.model.eval()
            for picture in tqdm(pictures):
                picture = self.transformer.picture_transform(picture)[None]
                cartoons.append(
                    self.transformer.cartoon_untransform(self.model(picture).squeeze())
                )

        return cartoons

    def cartoonize_dataset(
        self, pictures_loader: DataLoader, nb_images: int = -1
    ) -> List[Dict[str, NDArray[(3, Any, Any), np.int32]]]:
        """Cartoonize pictures dataset"""
        images_with_cartoons = []
        with torch.no_grad():
            self.model.eval()
            for pictures_batch in tqdm(pictures_loader):
                pictures_batch = pictures_batch.to(self.device)
                to_add = [
                    {
                        "picture": self.transformer.picture_untransform(image),
                        "cartoon": self.transformer.cartoon_untransform(cartoon),
                    }
                    for (image, cartoon) in zip(
                        pictures_batch, self.model(pictures_batch)
                    )
                ]
                images_with_cartoons.extend(to_add)
                if len(images_with_cartoons) >= nb_images >= 0:
                    break

        return images_with_cartoons
