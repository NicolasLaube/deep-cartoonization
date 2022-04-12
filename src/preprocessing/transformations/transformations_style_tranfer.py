"""Transformations for style transfer."""
import numpy as np
import torch
from torchvision import transforms

from src.config import IMAGE_SIZE


def preprocess(image: np.ndarray):
    """Preprocessing for style transfer."""
    return transforms.Compose(
        [
            transforms.Scale(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
            transforms.Normalize(
                mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                std=[1, 1, 1],
            ),
            transforms.Lambda(lambda x: x.mul_(255)),
        ]
    )(image)


def postprocess(image: np.ndarray):
    """Postprocessing for style transfer."""
    result = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.mul_(1.0 / 255)),
            transforms.Normalize(
                mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                std=[1, 1, 1],
            ),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
        ]
    )(image)

    result[result > 1] = 1
    result[result < 0] = 0
    return transforms.Compose([transforms.ToPILImage()])(result)
