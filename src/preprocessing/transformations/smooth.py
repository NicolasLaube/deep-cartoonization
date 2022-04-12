"""Resize"""
# pylint: disable=E0401
from typing import Any

import cv2
import numpy as np
from nptyping import NDArray
from PIL import Image, ImageOps


def edge_promoting(
    image: NDArray[(Any, Any), np.int32], kernel_size: int = 5
) -> NDArray[(Any, Any), np.int32]:
    """To blur the edges"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)  # pylint: disable=no-member
    gauss = gauss * gauss.transpose(1, 0)

    img_size = image.size
    rgb_img = np.array(image)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member

    rgb_img = np.array(
        ImageOps.fit(Image.fromarray(rgb_img), img_size, Image.ANTIALIAS)
    )
    pad_img = np.pad(rgb_img, ((3, 3), (3, 3), (0, 0)), mode="reflect")
    gray_img = np.array(
        ImageOps.fit(Image.fromarray(gray_img), img_size, Image.ANTIALIAS)
    )
    edges = cv2.Canny(  # pylint: disable=no-member
        gray_img, 150, 500
    )  # 200, 500 is good but maybe too little blur is applied
    dilation = cv2.dilate(edges, kernel)  # pylint: disable=no-member

    _gauss_img = np.copy(rgb_img)
    gauss_img = _fast_loop(_gauss_img, pad_img, kernel_size, gauss, dilation)

    rgb_img = cv2.resize(  # pylint: disable=no-member
        rgb_img, img_size, Image.ANTIALIAS
    )
    gauss_img = cv2.resize(gauss_img, img_size)  # pylint: disable=no-member
    return Image.fromarray(gauss_img)


def _fast_loop(gauss_img, pad_img, kernel_size, gauss, dilation):
    """A loop used in edge promoting function"""
    idx = np.where(dilation != 0)
    loops = int(np.sum(dilation != 0))
    for i in range(loops):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
            np.multiply(
                pad_img[
                    idx[0][i] : idx[0][i] + kernel_size,
                    idx[1][i] : idx[1][i] + kernel_size,
                    0,
                ],
                gauss,
            )
        )
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
            np.multiply(
                pad_img[
                    idx[0][i] : idx[0][i] + kernel_size,
                    idx[1][i] : idx[1][i] + kernel_size,
                    1,
                ],
                gauss,
            )
        )
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
            np.multiply(
                pad_img[
                    idx[0][i] : idx[0][i] + kernel_size,
                    idx[1][i] : idx[1][i] + kernel_size,
                    2,
                ],
                gauss,
            )
        )
    return gauss_img
