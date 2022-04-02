"""Offline Preprocessing for anime dataset."""
# pylint: disable=E1101
import os

import cv2
import numpy as np
import pandas as pd
from nptyping import NDArray
from tqdm import tqdm


class OfflineDataProcessing:
    """Dataset preprocessing methods offline"""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.df_images = pd.read_csv(csv_path, index_col=0)
        self.nb_images = len(self.df_images)

    @staticmethod
    def edge_image(image: NDArray) -> NDArray:
        """Edge image"""

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

        gauss_image = cv2.cvtColor(gauss_image.copy(), cv2.COLOR_BGR2GRAY)
        gauss_image = np.stack([gauss_image, gauss_image, gauss_image], axis=-1)

        return gauss_image

    @staticmethod
    def __gray_image(image: NDArray) -> NDArray:
        """Gray image"""

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)

        return image_gray

    def edge_images(self):
        """Edge images"""
        create_folder(os.path.join("data", "cartoon_edged"))
        # add column for edge image
        self.df_images["edge_path"] = ""
        # 705

        for i in tqdm(range(self.nb_images)):
            if i > 5000:
                edge_folder = "/".join(
                    self.df_images["path"][i].split("/")[:-1]
                ).replace("cartoon_frames", "cartoon_edged")
                create_folder(edge_folder)
                image_path = self.df_images["path"][i]
                image = cv2.imread(image_path)

                image_edge = self.__edge_image(image)
                self.df_images["edge_path"][i] = image_path.replace(
                    "cartoon_frames", "cartoon_edged"
                )
                cv2.imwrite(self.df_images["edge_path"][i], image_edge)

        self.df_images.to_csv(self.csv_path)

    def gray_images(self):
        """Gray images"""
        create_folder(os.path.join("data", "cartoon_gray"))
        # add column for gray image
        self.df_images["gray_path"] = ""

        for i in tqdm(range(self.nb_images)):
            gray_folder = "/".join(self.df_images["path"][i].split("/")[:-1]).replace(
                "cartoon_frames", "cartoon_gray"
            )
            create_folder(gray_folder)
            image_path = self.df_images["path"][i]
            image = cv2.imread(image_path)

            image_gray = self.__gray_image(image)
            self.df_images["gray_path"][i] = image_path.replace(
                "cartoon_frames", "cartoon_gray"
            )
            cv2.imwrite(self.df_images["gray_path"][i], image_gray)

        self.df_images.to_csv(self.csv_path)


def create_folder(path: str):
    """Create folder"""
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    data_preprocessing = OfflineDataProcessing(os.path.join("data", "cartoons_all.csv"))
    data_preprocessing.edge_images()

    data_preprocessing.gray_images()
