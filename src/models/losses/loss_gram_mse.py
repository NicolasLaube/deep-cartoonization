"""Loss MSE GRAM"""
# pylint: disable=R0201
import torch
from torch import nn


class GramMatrix(nn.Module):
    """Gram Matrix"""

    def forward(self, image):
        """Forward"""
        batch, channels, height, width = image.size()
        flatten = image.view(batch, channels, height * width)
        gram = torch.bmm(flatten, flatten.transpose(1, 2))
        gram.div_(height * width)
        return gram


class GramMSELoss(nn.Module):
    """Loss MSE GRAM"""

    def forward(self, image, target):
        """Forward"""
        out = nn.MSELoss()(GramMatrix()(image), target)
        return out
