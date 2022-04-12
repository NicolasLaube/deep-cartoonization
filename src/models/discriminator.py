"""Predictor"""
from abc import ABC
from typing import Any, Dict, List

import numpy as np
import torch
from nptyping import NDArray
from torch import sigmoid  # pylint: disable=no-name-in-module
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.models.discriminators import FixedDiscriminator, ModularDiscriminator
from src.models.utils.parameters import Architecture, ArchitectureParams


class Discriminator(ABC):
    """Discriminator"""

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

    def load_weights(self, disc_path: str) -> None:
        """Load the model from weights"""
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(disc_path))
        else:
            self.model.load_state_dict(
                torch.load(disc_path, map_location=lambda storage, loc: storage)
            )

    def __load_model(self):
        """Loads model"""

        if self.architecture == Architecture.GANModular:
            return ModularDiscriminator(
                in_nc=self.architecture_parameters.nb_channels_disc_input,
                out_nc=self.architecture_parameters.nb_channels_disc_output,
                nf=self.architecture_parameters.nb_channels_1st_hidden_layer_disc,
            )

        if self.architecture == Architecture.GANFixed:
            return FixedDiscriminator()

        if self.architecture == Architecture.GANUNet:
            # return UNet(n_channels=10, n_classes=10)  # To change
            pass

        raise ImportError("Model architecture doens't exist.")

    def discriminate(self, cartoons: List[NDArray[(3, Any, Any)]]) -> List[float]:
        """Discriminate images"""
        results = []
        to_pil = transforms.ToPILImage()
        with torch.no_grad():
            self.model.eval()
            for cartoon in tqdm(cartoons):
                cartoon = to_pil(np.array(cartoon).astype(np.uint8))
                cartoon = self.transformer.cartoon_transform(cartoon)
                cartoon = cartoon[None]
                results.append(sigmoid(self.model(cartoon)).mean().item())

        return results

    def discriminate_dataset(
        self, cartoons_loader: DataLoader, nb_images: int = -1
    ) -> List[Dict[str, float]]:
        """Discriminate pictures dataset"""
        images_with_cartoons = []
        with torch.no_grad():
            self.model.eval()
            for cartoons_batch in tqdm(cartoons_loader):
                cartoons_batch = cartoons_batch.to(self.device)
                to_add = [
                    {
                        "cartoon": self.transformer.cartoon_untransform(cartoon),
                        "result": sigmoid(result).mean().tolist(),  # type: ignore
                    }
                    for (cartoon, result) in zip(
                        cartoons_batch, self.model(cartoons_batch)
                    )
                ]
                images_with_cartoons.extend(to_add)
                if len(images_with_cartoons) >= nb_images >= 0:
                    break

        return images_with_cartoons
