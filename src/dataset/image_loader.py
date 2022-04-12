"""Generic Image Loader"""
# pylint: disable=E1101, R0913
from typing import Callable, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from nptyping import NDArray
from PIL import Image
from torch.utils.data import Dataset


class ImageLoader(Dataset):
    """Generic image loader class"""

    def __init__(
        self,
        csv_path: str,
        filter_data: Callable[[pd.DataFrame], pd.DataFrame],
        transform: Callable[[NDArray], NDArray],
        nb_images: int = -1,
        smooth: bool = False,
        gray: bool = False,
    ) -> None:
        self.smooth = smooth
        self.gray = gray
        self.filter_data = filter_data
        self.transform = transform
        self.df_images: pd.DataFrame = self.filter_data(
            pd.read_csv(csv_path, index_col=0)
        )
        self.set_nb_images(nb_images)

    def set_nb_images(self, nb_images: int) -> None:
        """Set the number of images"""
        # Get the real nb of images to reset...
        nb_images = min(  # type: ignore
            nb_images if nb_images > 0 else np.inf,
            len(self.df_images),
        )
        # ...and reset the dataset
        self.df_images = self.df_images.sample(n=nb_images)
        self.df_images = self.df_images.reset_index(drop=True)

    def __len__(self) -> int:
        """Length"""
        return len(self.df_images)

    def __getitem__(
        self, index: int
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Get an item"""
        image_path = self.df_images["path"][index]
        image = Image.open(image_path)
        image_transformed = self.transform(image)
        if not self.gray and not self.smooth:
            return image_transformed
        if self.gray and not self.smooth:
            return image_transformed, self.transform(self.gray_images(image_path))
        if self.smooth and not self.gray:
            return image_transformed, self.transform(self.smooth_image(image_path))

        return (
            image_transformed,
            self.transform(self.gray_images(image_path)),
            self.transform(self.smooth_image(image_path)),
        )

    @staticmethod
    def edge_image(image: NDArray) -> NDArray:
        """Edge image"""
        # transform image to grayscale
        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(image_gray, 100, 200)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation = cv2.dilate(edges, kernel, iterations=1)

        gauss = cv2.getGaussianKernel(kernel_size, 0)
        gauss = gauss * gauss.transpose(1, 0)

        pad_image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode="reflect")

        gauss_image = np.copy(image)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_image[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(
                    pad_image[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        0,
                    ],
                    gauss,
                )
            )
            gauss_image[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(
                    pad_image[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        1,
                    ],
                    gauss,
                )
            )
            gauss_image[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(
                    pad_image[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        2,
                    ],
                    gauss,
                )
            )

        return gauss_image

    @staticmethod
    def gray_images(image_path: str) -> Image.Image:
        """Image to gray"""
        image = cv2.imread(image_path)  # [:, :, ::-1]
        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        return Image.fromarray(image_gray)

    def smooth_image(self, image_path: str) -> Image.Image:
        """Smooth image"""
        image = cv2.imread(image_path)
        image = self.edge_image(image)
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image = np.stack([image, image, image], axis=-1)
        return Image.fromarray(image)
