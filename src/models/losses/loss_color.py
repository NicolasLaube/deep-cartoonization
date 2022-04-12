"""Color loss for GANs."""
# pylint: disable=C0103
import torch
from torch import nn


class ColorLoss(nn.Module):
    """Color loss for GANs."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        self.device = device

        self._rgb_to_yuv_kernel = (
            torch.tensor(
                [
                    [0.299, -0.14714119, 0.61497538],
                    [0.587, -0.28886916, -0.51496512],
                    [0.114, 0.43601035, -0.10001026],
                ]
            )
            .float()
            .to(self.device)
        )

    def forward(self, image, image_g):
        """Forward pass of the loss."""
        image = self.rgb_to_yuv(image).to(self.device)
        image_g = self.rgb_to_yuv(image_g).to(self.device)

        # After convert to yuv, both images have channel last

        return (
            self.l1(image[:, :, :, 0], image_g[:, :, :, 0])
            + self.huber(image[:, :, :, 1], image_g[:, :, :, 1])
            + self.huber(image[:, :, :, 2], image_g[:, :, :, 2])
        )

    def rgb_to_yuv(self, image):
        """
        https://en.wikipedia.org/wiki/YUV
        output: Image of shape (H, W, C) (channel last)
        """
        # -1 1 -> 0 1
        image = (image + 1.0) / 2.0

        yuv_img = torch.tensordot(
            image, self._rgb_to_yuv_kernel, dims=([image.ndim - 3], [0])
        )

        return yuv_img
