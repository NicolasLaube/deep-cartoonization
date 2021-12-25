import os
from typing import Any
import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt

from src import config


def show_array(array: NDArray[(3, Any, Any), np.int32]) -> None:
    """Show image from path"""
    if array.shape[0] == 3:
        array = np.stack([array[0], array[1], array[2]], axis=2)
    plt.imshow(array)
    plt.show()


def save_array(array: NDArray[(3, Any, Any), np.int32], save_path: str) -> None:
    """Saves an array as .npy"""
    np.save(os.path.join(config.PRETRAINED_FOLDER, save_path), array)
