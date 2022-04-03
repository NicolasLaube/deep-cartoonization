"""Cartoonizer pipeline launcher"""
# pylint: disable=R0902,E1102
from typing import Any, Dict, List

import numpy as np
from nptyping import NDArray
from PIL import Image
from torch.utils.data import DataLoader

from src import config, dataset, models, preprocessing
from src.pipelines.utils import init_device
from src.preprocessing.transformations.transformations import Transform


class Discriminator:
    """Inference pipeline."""

    def __init__(
        self,
        *,
        infering_parameters: models.InferingParams,
        architecture: models.Architecture = models.Architecture.GANFixed,
        architecture_params: models.ArchitectureParams = models.ArchitectureParamsNULL(),
        cartoons_dataset_parameters: dataset.CartoonsDatasetParameters,
        disc_path: str = None,
    ):
        # Initialize parameters
        self.device = init_device()
        self.infering_params = infering_parameters
        self.architecture = architecture
        self.architecture_params = architecture_params
        self.cartoons_dataset_parameters = cartoons_dataset_parameters
        self.disc_path = disc_path

    def set_weights(self, disc_path: str) -> None:
        """To set new weights"""
        self.disc_path = disc_path

    def get_discriminated_images(self, nb_images: int = -1) -> List[Dict[str, float]]:
        """To discriminate some images"""

        transformer = Transform(
            architecture=self.architecture,
            new_size=self.cartoons_dataset_parameters.new_size,
            crop_mode=self.cartoons_dataset_parameters.crop_mode,
            device=self.device,
        )

        discriminator = models.Discriminator(
            architecture=self.architecture,
            architecture_params=self.architecture_params,
            transformer=transformer,
            device=self.device,
        )

        if self.disc_path is None:
            raise Exception("Weights path is not defined")
        discriminator.load_weights(disc_path=self.disc_path)

        cartoons_dataset = self.__init_cartoons_dataset(train=False)

        test_cartoons_loader = DataLoader(
            dataset=cartoons_dataset,
            batch_size=self.infering_params.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.NUM_WORKERS,
        )

        return discriminator.discriminate_dataset(
            cartoons_loader=test_cartoons_loader, nb_images=nb_images
        )

    def discriminate_images(
        self, cartoons: List[NDArray[(3, Any, Any), np.int32]]
    ) -> List[Dict[str, NDArray[(3, Any, Any), np.int32]]]:
        """To discriminate some images"""

        transformer = Transform(
            architecture=self.architecture,
            new_size=self.cartoons_dataset_parameters.new_size,
            crop_mode=self.cartoons_dataset_parameters.crop_mode,
            device=self.device,
        )

        discriminator = models.Discriminator(
            architecture=self.architecture,
            architecture_params=self.architecture_params,
            transformer=transformer,
            device=self.device,
        )

        if self.disc_path is None:
            raise Exception("Weights path is not defined")
        discriminator.load_weights(disc_path=self.disc_path)

        return [
            {"cartoon": cartoon, "result": result}
            for (cartoon, result) in zip(
                cartoons, discriminator.discriminate(cartoons=cartoons)
            )
        ]

    def discriminate_images_from_path(self, paths: List[str]) -> List[Dict[str, float]]:
        """To discriminate images from their path"""

        cartoons = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            cartoons.append(image)

        return self.discriminate_images(cartoons)

    ###########################
    ### About initilization ###
    ###########################

    def __init_cartoons_dataset(self, train: bool) -> dataset.CartoonDataset:
        """Initialize cartoons dataset"""

        data_filter = preprocessing.Filter(
            new_size=self.cartoons_dataset_parameters.new_size,
            ratio_filter_mode=self.cartoons_dataset_parameters.ratio_filter_mode,
        )
        transform = preprocessing.Transform(
            new_size=self.cartoons_dataset_parameters.new_size,
            crop_mode=self.cartoons_dataset_parameters.crop_mode,
            architecture=self.architecture,
        )

        return dataset.CartoonDataset(
            data_filter.picture_filter,
            transform.picture_transform,
            self.cartoons_dataset_parameters.nb_images,
            "train" if train else "validation",
        )


if __name__ == "__main__":
    disc = Discriminator(
        infering_parameters=models.InferingParams(batch_size=2),
        architecture=models.Architecture.GANFixed,
        architecture_params=models.ArchitectureParamsNULL(),
        cartoons_dataset_parameters=dataset.CartoonsDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
            ratio_filter_mode=preprocessing.RatioFilterMode.NO_FILTER,
            nb_images=4,
        ),
        disc_path="weights/pretrained/trained_netD.pth",
    )
    print(
        [
            x["result"]
            for x in disc.discriminate_images_from_path(
                [
                    "logs/2022_03_27-20_51_29/pictures/epoch_5/image_0_cartoon.png",
                    "logs/2022_03_27-20_51_29/pictures/epoch_5/image_1_cartoon.png",
                ]
            )
        ]
    )
