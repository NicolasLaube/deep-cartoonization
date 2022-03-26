"""Color loss for GANs."""
# pylint: disable=C0103
from torch import nn

from src.preprocessing.anime.gram import rgb_to_yuv


class ColorLoss(nn.Module):
    """Color loss for GANs."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, image, image_g):
        """Forward pass of the loss."""
        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)

        # After convert to yuv, both images have channel last

        return (
            self.l1(image[:, :, :, 0], image_g[:, :, :, 0])
            + self.huber(image[:, :, :, 1], image_g[:, :, :, 1])
            + self.huber(image[:, :, :, 2], image_g[:, :, :, 2])
        )
