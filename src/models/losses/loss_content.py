"""Loss content"""
import gc  # garbage collector

from torch import nn
from torchvision.models import vgg16


class ContentLoss(nn.Module):
    """Content Loss"""

    def __init__(self, omega=10, device="cpu"):
        super().__init__()
        self.device = device
        self.base_loss = nn.L1Loss()
        self.omega = omega

        perception = list(vgg16(pretrained=True).features)[:25]
        self.perception = nn.Sequential(*perception).eval()

        for param in self.perception.parameters():
            param.requires_grad = False

        gc.collect()

    def forward(self, matrix_1, matrix_2):
        """Forward pass of the module."""
        matrix_1 = self.perception(matrix_1)
        matrix_2 = self.perception(matrix_2)

        return self.omega * self.base_loss(matrix_1, matrix_2)
