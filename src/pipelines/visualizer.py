from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from nptyping import NDArray


def show_array(array: NDArray[(3, Any, Any), np.int32]) -> None:
    """Show image from path"""
    if array.shape[0] == 3:
        array = np.stack([array[0], array[1], array[2]], axis=2)
    plt.imshow(array)
    plt.show()


def save_array(array: NDArray[(3, Any, Any), np.int32], save_path: str) -> None:
    """Saves an array as .npy"""
    np.save(save_path, array)
