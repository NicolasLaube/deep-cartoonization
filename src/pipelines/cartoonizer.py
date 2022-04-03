"""Cartoonizer pipeline launcher"""
# pylint: disable=R0902,E1102
from typing import Any, Dict, List

import numpy as np
from matplotlib import pyplot as plt
from nptyping import NDArray
from PIL import Image
from torch.utils.data import DataLoader

from src import config, dataset, models, preprocessing
from src.pipelines.utils import init_device
from src.preprocessing.transformations.transformations import Transform


class Cartoonizer:
    """Inference pipeline."""

    def __init__(
        self,
        *,
        infering_parameters: models.InferingParams,
        architecture: models.Architecture = models.Architecture.GANFixed,
        architecture_params: models.ArchitectureParams = models.ArchitectureParamsNULL(),
        pictures_dataset_parameters: dataset.PicturesDatasetParameters,
        gen_path: str = None,
    ):
        # Initialize parameters
        self.device = init_device()
        self.infering_params = infering_parameters
        self.architecture = architecture
        self.architecture_params = architecture_params
        self.pictures_dataset_parameters = pictures_dataset_parameters
        self.gen_path = gen_path

    def set_weights(self, gen_path: str) -> None:
        """To set new weights"""
        self.gen_path = gen_path

    def get_cartoonized_images(
        self, nb_images: int = -1
    ) -> List[Dict[str, NDArray[(3, Any, Any), np.int32]]]:
        """To show some cartoonized images"""

        transformer = Transform(
            architecture=self.architecture,
            new_size=self.pictures_dataset_parameters.new_size,
            crop_mode=self.pictures_dataset_parameters.crop_mode,
            device=self.device,
        )

        predictor = models.Predictor(
            architecture=self.architecture,
            architecture_params=self.architecture_params,
            transformer=transformer,
            device=self.device,
        )

        if self.gen_path is None:
            raise Exception("Weights path is not defined")
        predictor.load_weights(gen_path=self.gen_path)

        pictures_dataset = self.__init_pictures_dataset(train=False)

        test_pictures_loader = DataLoader(
            dataset=pictures_dataset,
            batch_size=self.infering_params.batch_size,
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
            new_size=self.pictures_dataset_parameters.new_size,
            crop_mode=self.pictures_dataset_parameters.crop_mode,
            device=self.device,
        )

        predictor = models.Predictor(
            architecture=self.architecture,
            architecture_params=self.architecture_params,
            transformer=transformer,
            device=self.device,
        )

        if self.gen_path is None:
            raise Exception("Weights path is not defined")
        predictor.load_weights(gen_path=self.gen_path)

        return [
            {"picture": picture, "cartoon": cartoon}
            for (picture, cartoon) in zip(
                pictures, predictor.cartoonize(pictures=pictures)
            )
        ]

    def cartoonize_images_from_path(
        self, paths: List[str]
    ) -> List[Dict[str, NDArray[(3, Any, Any), np.int32]]]:
        """To show some cartoonized images from their path"""

        pictures = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            pictures.append(image)

        return self.cartoonize_images(pictures)

    ###########################
    ### About initilization ###
    ###########################

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


if __name__ == "__main__":
    cartoonizer = Cartoonizer(
        infering_parameters=models.InferingParams(batch_size=2),
        architecture=models.Architecture.GANFixed,
        architecture_params=models.ArchitectureParamsNULL(),
        pictures_dataset_parameters=dataset.PicturesDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
            ratio_filter_mode=preprocessing.RatioFilterMode.NO_FILTER,
            nb_images=4,
        ),
        gen_path="weights/pretrained/trained_netG.pth",
    )
    results = cartoonizer.cartoonize_images_from_path(
        [
            "data/flickr/Images/3532593368_be10432e92.jpg",
            "data/flickr/Images/3534512991_f9fd66f165.jpg",
        ]
    )
    for result in results:
        plt.imshow(result["cartoon"])
        plt.show()
